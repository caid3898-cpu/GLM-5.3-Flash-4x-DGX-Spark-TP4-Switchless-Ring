"""GPU exactness + microbench for the glm-kernels-20260926 overlays (run inside the vLLM image on ONE worker GPU,
fleet stopped). Real GLM-5.3-Flash TP4 per-rank shapes: KDA 16 local heads x 128, merged in-projection [T, 6416],
Marlin W8A16 MXFP8 o_proj 2048 -> 4096 and in_proj 4096 -> 6416.

docker run --rm --gpus all --memory 16g --entrypoint python3 -e PYTHONPATH=/overlay/overlay \
    -v ~/glm53-flash-4x-spark:/overlay:ro <image> /overlay/tests/test_glm_kernels_0926_gpu.py [stash|flags|l2|all]

stash  GLM_KDA_STASH_NOCOPY: strided stash kernel vs overlay/kda_stash.py (with its .contiguous copies), multi-step
       (replay records carried across steps), random accepted counts and FULL flags, c1 k=3/7 rows, c16 k=3/7 rows,
       pure verify views (token stride 6416) and mixed-step views (index_select, stride 6144). Bitwise on outputs,
       state pool and aux records. Timing of 34 layers in a CUDA graph.
flags  GLM_KDA_FLAG_FUSED: Triton kernel vs the stash wrapper's torch chain (pure, mixed, padded graph rows,
       negative positions, several block sizes). Bitwise. Timing chain vs one launch per layer vs one per forward.
l2     GLM_L2_PREFETCH window A: 34 layers of [in_proj GEMM -> fork prefetch(o_proj, MB) -> stash kernel ->
       o_proj GEMM] in one CUDA graph (side-stream fork/join captured), cold weights. o_proj output bitwise equal
       with and without prefetch; replay time per MB budget.
Prints one JSON line per result; exit 1 on any exactness failure.
"""
import json
import math
import os
import sys

import torch

dev = torch.device("cuda")
H = HV = 16
D = 128
PROJ = 6416  # q|k|v 3 x 2048, beta 16, f_a 128, g_a 128
QKV = 3 * H * D
FAIL = []
OUT = []


def emit(**kw):
    OUT.append(kw)
    print(json.dumps(kw), flush=True)


def graph_us(fn, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(7):
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        t.append(e0.elapsed_time(e1) * 1000.0 / reps)
    t.sort()
    return t[3], g


# ------------------------------------------------------------------------------------------ stash
def kda_views(T, gen, mixed_extra=0):
    """(q, k, v, g1, beta) exactly as Glm5NextLinearAttention._forward hands them to the recurrent kernel."""
    n = T + mixed_extra
    proj = (torch.randn(n, PROJ, device=dev, generator=gen) * 0.5).bfloat16()
    g1 = (torch.randn(n, H * D, device=dev, generator=gen) * 2).bfloat16().reshape(1, -1, H, D)
    qkv = proj[:, :QKV]
    beta = proj[:, QKV:QKV + H].unsqueeze(0)
    if mixed_extra:
        idx = torch.randperm(n, device=dev, generator=gen)[:T].sort().values
        qkv = qkv.index_select(0, idx)
        g1 = g1.index_select(1, idx)
        beta = beta.index_select(1, idx)
    q, k, v = (x.reshape(1, -1, H, D) for x in qkv.split(H * D, dim=-1))
    return q, k, v, g1, beta


def make_pool(n_slots, gen, pad=4096):
    per = HV * D * D
    base = torch.randn(n_slots * (per + pad), device=dev, generator=gen) * 0.05
    return base.as_strided((n_slots, HV, D, D), (per + pad, D * D, D, 1))


def stash_exact():
    import kda_stash as ks
    import glm_kda_stash_fast as f
    gen = torch.Generator(device=dev).manual_seed(1)
    a_log = (torch.randn(1, 1, H, 1, device=dev, generator=gen) * 0.5).float()
    dt_bias = (torch.randn(H * D, device=dev, generator=gen) * 0.5).float()
    worst = 0
    cases = [(1, 4, 0), (1, 8, 0), (16, 4, 0), (16, 8, 0), (3, 4, 5), (8, 8, 7), (2, 8, 0), (8, 8, 7), (8, 4, 3),
             (16, 8, 9)]
    if os.environ.get("KX_AA") == "1":
        cases = [(8, 8, 7), (8, 8, 7), (16, 8, 9)]
    for N, T, mixed in cases:
        n_slots = N * T + 3
        ks._AUX.clear()  # get_aux caches by pointer: a freed pool's aux must not leak into the next case
        pool_a = make_pool(n_slots, gen)
        base_b = pool_a.as_strided((pool_a.untyped_storage().nbytes() // 4,), (1,)).clone()
        pool_b = base_b.as_strided(pool_a.shape, pool_a.stride())  # same bytes, same strided slot layout
        idx = (torch.arange(N * T, device=dev, dtype=torch.int32) + 1).view(N, T)
        idx = idx[torch.randperm(N, device=dev, generator=gen)].contiguous()
        cu = torch.arange(N + 1, device=dev, dtype=torch.int32) * T
        steps, eq = 6, True
        n_strided = 0
        for step in range(steps):
            q, k, v, g1, beta = kda_views(N * T, gen, mixed)
            nacc = torch.randint(1, T + 1, (N,), device=dev, dtype=torch.int32, generator=gen)
            if step == 0:
                nacc.fill_(1)
            full = None
            if step % 3 == 1:
                full = torch.randint(0, 2, (N,), device=dev, dtype=torch.int32, generator=gen)
            use_out = step % 2 == 0
            out_a = torch.empty(1, N * T, H, D, device=dev, dtype=torch.bfloat16) if use_out else None
            out_b = torch.empty(1, N * T, H, D, device=dev, dtype=torch.bfloat16) if use_out else None
            o_a, _ = ks.fused_recurrent_kda_stash(q, k, v, g1, beta, pool_a, cu, idx, nacc, a_log, dt_bias,
                                                  lower_bound=-5.0, out=out_a, full_mode=full)
            strides = f.token_strides(q, k, v, g1, beta)
            assert strides is not None, "real views must be eligible"
            n_strided += 1
            if os.environ.get("KX_AA") == "1":
                o_b, _ = ks.fused_recurrent_kda_stash(q, k, v, g1, beta, pool_b, cu, idx, nacc, a_log, dt_bias,
                                                      lower_bound=-5.0, out=out_b, full_mode=full)
            else:
                o_b, _ = f.fused_recurrent_kda_stash_strided(q, k, v, g1, beta, pool_b, cu, idx, nacc, a_log,
                                                             dt_bias, lower_bound=-5.0, out=out_b, full_mode=full,
                                                             _strides=strides)
            torch.cuda.synchronize()
            aux_a, aux_b = ks.get_aux(pool_a), ks.get_aux(pool_b)
            same = (beq(o_a, o_b)
                    and torch.equal(pool_a.view(torch.int32), pool_b.view(torch.int32))
                    and torch.equal(aux_a.view(torch.int32), aux_b.view(torch.int32)))
            if not same:
                emit(test="stash_diff", N=N, T=T, mixed=mixed, step=step, full=full is not None,
                     out_eq=bool(torch.equal(o_a, o_b)),
                     pool_eq=bool(torch.equal(pool_a.view(torch.int32), pool_b.view(torch.int32))),
                     aux_eq=bool(torch.equal(aux_a.view(torch.int32), aux_b.view(torch.int32))),
                     out_maxdiff=float((o_a.float() - o_b.float()).abs().max()))
                d = (o_a.float() - o_b.float()).abs().max().item()
                worst = max(worst, d if d > 0 else 1e-30)
                eq = False
        emit(test="stash_exact", N=N, T=T, mixed=mixed, steps=steps, strided_calls=n_strided,
             stride_q=q.stride(1), stride_beta=beta.stride(1), bit_exact=eq)
        if not eq:
            FAIL.append(f"stash N={N} T={T} mixed={mixed}")
    emit(test="stash_exact_summary", exact=not any(x.startswith("stash") for x in FAIL))


def bits(t):
    """Bitwise view for exact comparison (signed zeros and NaN payloads count)."""
    if t.element_size() == 2:
        return t.contiguous().view(torch.int16)
    if t.element_size() == 4:
        return t.contiguous().view(torch.int32)
    if t.element_size() == 1:
        return t.contiguous().view(torch.uint8)
    return t.contiguous().view(torch.int64)


def beq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(bits(a), bits(b)))


class _Batch:
    """A serving-like verify batch over several steps: per-request k in 3..7 (ragged recurrence), requests that
    finish and hand their slots to new requests (slot recycling, fresh prefill state in slot 0, n_acc = 1), and
    graph padding rows (zero-length, NULL slot) whose count changes from step to step."""

    MAXT = 8  # num_spec + 1 with k up to 7

    def __init__(self, n_req, gen, ragged):
        self.gen, self.ragged = gen, ragged
        self.n_slots = 40 * self.MAXT + 1
        self.free = list(range(1, self.n_slots))
        self.reqs = [self._new() for _ in range(n_req)]

    def _rand(self, lo, hi):
        return int(torch.randint(lo, hi + 1, (1,), generator=torch.Generator().manual_seed(
            int(torch.randint(0, 2**31 - 1, (1,), device=dev, generator=self.gen).item()))).item())

    def _new(self):
        slots = [self.free.pop(0) for _ in range(self.MAXT)]
        return {"slots": slots, "k": self._rand(3, 7), "fresh": True}

    def step(self, pools):
        """Advance membership; returns (T per request, slot table, n_acc, cu_seqlens, n_pad)."""
        for i, r in enumerate(self.reqs):
            if not r["fresh"] and self._rand(0, 9) == 0:  # finishes; a new request takes the slots
                self.free.extend(r["slots"])
                self.reqs[i] = self._new()
        ks = [r["k"] if self.ragged else self.reqs[0]["k"] for r in self.reqs]
        Ts = [k + 1 for k in ks]
        nacc = []
        for r, T in zip(self.reqs, Ts):
            if r["fresh"]:
                init = torch.randn(HV, D, D, device=dev, generator=self.gen) * 0.05
                for p in pools:  # prefill wrote the full state into slot 0
                    p[r["slots"][0]].copy_(init)
                nacc.append(1)
                r["fresh"] = False
            else:
                nacc.append(self._rand(1, r.get("lastT", T)))
            r["lastT"] = T
        n_pad = self._rand(0, 5)
        idx = [r["slots"] for r in self.reqs] + [[0] * self.MAXT] * n_pad
        cu = [0]
        for T in Ts:
            cu.append(cu[-1] + T)
        cu += [cu[-1]] * n_pad
        return (Ts, torch.tensor(idx, dtype=torch.int32, device=dev),
                torch.tensor(nacc + [1] * n_pad, dtype=torch.int32, device=dev),
                torch.tensor(cu, dtype=torch.int32, device=dev), n_pad)


def stash_matrix():
    """Strided stash kernel vs overlay/kda_stash.py over N = 1..32, k = 3..7 (uniform and ragged), completion and
    slot recycling, changing graph padding, FULL flags, pure and mixed views. Bitwise on output, pool and aux."""
    import kda_stash as ks
    import glm_kda_stash_fast as f
    gen = torch.Generator(device=dev).manual_seed(11)
    a_log = (torch.randn(1, 1, H, 1, device=dev, generator=gen) * 0.5).float()
    dt_bias = (torch.randn(H * D, device=dev, generator=gen) * 0.5).float()
    total = bad = 0
    for N in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32):
        for ragged in (False, True):
            ks._AUX.clear()
            b = _Batch(N, gen, ragged)
            pool_a = make_pool(b.n_slots, gen)
            base_b = pool_a.as_strided((pool_a.untyped_storage().nbytes() // 4,), (1,)).clone()
            pool_b = base_b.as_strided(pool_a.shape, pool_a.stride())
            ok = True
            for step in range(8):
                Ts, idx, nacc, cu, n_pad = b.step((pool_a, pool_b))
                ntok = sum(Ts)
                mixed = 5 if step % 4 == 3 else 0
                q, k, v, g1, beta = kda_views(ntok, gen, mixed)
                n_all = N + n_pad
                full = torch.randint(0, 2, (n_all,), device=dev, dtype=torch.int32, generator=gen) \
                    if step % 3 == 1 else None
                out_a = torch.empty(1, ntok, H, D, device=dev, dtype=torch.bfloat16) if step % 2 == 0 else None
                out_b = torch.empty(1, ntok, H, D, device=dev, dtype=torch.bfloat16) if step % 2 == 0 else None
                o_a, _ = ks.fused_recurrent_kda_stash(q, k, v, g1, beta, pool_a, cu, idx, nacc, a_log, dt_bias,
                                                      out=out_a, full_mode=full)
                o_b, _ = f.fused_recurrent_kda_stash_strided(q, k, v, g1, beta, pool_b, cu, idx, nacc, a_log,
                                                             dt_bias, out=out_b, full_mode=full,
                                                             _strides=f.token_strides(q, k, v, g1, beta))
                same = (beq(o_a, o_b) and beq(pool_a, pool_b)
                        and beq(ks.get_aux(pool_a), ks.get_aux(pool_b)))
                total += 1
                if not same:
                    bad += 1
                    ok = False
                    emit(test="stash_matrix_diff", N=N, ragged=ragged, step=step, Ts=Ts, n_pad=n_pad, mixed=mixed,
                         out=beq(o_a, o_b), pool=beq(pool_a, pool_b))
            del pool_a, pool_b, base_b
    emit(test="stash_matrix", cases=total, mismatches=bad, bit_exact=bad == 0)
    if bad:
        FAIL.append("stash_matrix")


def stash_dispatch():
    """The installed dispatchers (glm_kda_stash + glm_kda_stash_fast on a stand-in kda module) with
    GLM_KDA_STASH_NOCOPY=0 vs 1: the real eligibility checks and argument plumbing, bitwise."""
    import types
    import kda_stash as ks
    import glm_kda_stash as gks
    import glm_kda_stash_fast as f
    from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda as stock

    class Glm5NextLinearAttention:
        def forward(self, hidden_states, positions):
            return None

        def _forward(self, qkv_proj_states, g1, beta, core_attn_out):
            return None

    mod = types.SimpleNamespace(Glm5NextLinearAttention=Glm5NextLinearAttention, fused_recurrent_kda=stock)
    os.environ["GLM_KDA_STASH_NOCOPY"] = "1"
    gks.install(mod)
    f.install(mod)
    gen = torch.Generator(device=dev).manual_seed(12)
    a_log = (torch.randn(1, 1, H, 1, device=dev, generator=gen) * 0.5).float()
    dt_bias = (torch.randn(H * D, device=dev, generator=gen) * 0.5).float()
    total = bad = 0
    for N in (1, 4, 16, 32):
        ks._AUX.clear()
        b = _Batch(N, gen, True)
        pool_a = make_pool(b.n_slots, gen)
        base_b = pool_a.as_strided((pool_a.untyped_storage().nbytes() // 4,), (1,)).clone()
        pool_b = base_b.as_strided(pool_a.shape, pool_a.stride())
        for step in range(6):
            Ts, idx, nacc, cu, n_pad = b.step((pool_a, pool_b))
            ntok = sum(Ts)
            q, k, v, g1, beta = kda_views(ntok, gen, 4 if step % 3 == 2 else 0)
            outs = []
            for flag, pool in (("0", pool_a), ("1", pool_b)):
                os.environ["GLM_KDA_STASH_NOCOPY"] = flag
                gks._state["full"] = None
                before = f._state["strided"]
                o, _ = mod.fused_recurrent_kda(
                    q=q, k=k, v=v, g=g1, beta=beta, initial_state=pool, use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu, ssm_state_indices=idx, num_accepted_tokens=nacc, out=None, sigmoid_beta=True,
                    a_log=a_log, g_bias=dt_bias, compute_gate=True, lower_bound=-5.0)
                outs.append((o, f._state["strided"] - before))
            total += 1
            if not (outs[0][1] == 0 and outs[1][1] == 1 and beq(outs[0][0], outs[1][0]) and beq(pool_a, pool_b)
                    and beq(ks.get_aux(pool_a), ks.get_aux(pool_b))):
                bad += 1
                emit(test="stash_dispatch_diff", N=N, step=step, strided=(outs[0][1], outs[1][1]))
    os.environ["GLM_KDA_STASH_NOCOPY"] = "1"
    emit(test="stash_dispatch", cases=total, mismatches=bad, bit_exact=bad == 0)
    if bad:
        FAIL.append("stash_dispatch")


def stash_time():
    import kda_stash as ks
    import glm_kda_stash_fast as f
    gen = torch.Generator(device=dev).manual_seed(2)
    a_log = (torch.randn(1, 1, H, 1, device=dev, generator=gen) * 0.5).float()
    dt_bias = (torch.randn(H * D, device=dev, generator=gen) * 0.5).float()
    for N, T in [(1, 4), (1, 8), (4, 4), (16, 4), (16, 8)]:
        L = 34 if N <= 4 else 8
        pools = [make_pool(N * T + 1, gen) for _ in range(L)]
        ins = [kda_views(N * T, gen) for _ in range(L)]
        idx = (torch.arange(N * T, device=dev, dtype=torch.int32) + 1).view(N, T)
        cu = torch.arange(N + 1, device=dev, dtype=torch.int32) * T
        nacc = torch.full((N,), max(1, T // 2), device=dev, dtype=torch.int32)
        outs = [torch.empty(1, N * T, H, D, device=dev, dtype=torch.bfloat16) for _ in range(L)]
        row = dict(test="stash_time", N=N, T=T, layers=L)
        for name in ("contig", "strided"):
            def run(name=name):
                for l in range(L):
                    q, k, v, g1, beta = ins[l]
                    if name == "contig":
                        ks.fused_recurrent_kda_stash(q, k, v, g1, beta, pools[l], cu, idx, nacc, a_log, dt_bias,
                                                     out=outs[l])
                    else:
                        f.fused_recurrent_kda_stash_strided(q, k, v, g1, beta, pools[l], cu, idx, nacc, a_log,
                                                            dt_bias, out=outs[l],
                                                            _strides=f.token_strides(q, k, v, g1, beta))
            us, _ = graph_us(run)
            row[f"{name}_us_per_layer"] = round(us / L, 2)
        row["saved_ms_per_step_34"] = round((row["contig_us_per_layer"] - row["strided_us_per_layer"]) * 34 / 1000, 3)
        emit(**row)


# ------------------------------------------------------------------------------------------ flags
def flags_exact():
    import glm_kda_stash_fast as f
    gen = torch.Generator(device=dev).manual_seed(3)
    bad = 0
    cases = 0
    for trial in range(400):
        n = int(torch.randint(1, 33, (1,), generator=torch.Generator().manual_seed(trial)).item())
        T = [4, 6, 8][trial % 3]
        bs = [2304, 16, 64, 7, 4096][trial % 5]
        mixed = trial % 4 == 3 or trial % 7 == 1
        pad_rows = [0, 3, 17, 1, 0][trial % 5]
        n_spec_tok = n * T
        n_extra = (trial % 7) * 3 if mixed else 0
        num_actual = n_spec_tok + n_extra
        # positions: per-sequence runs of T consecutive positions starting anywhere, near block edges often
        starts = torch.randint(0, 3 * bs + 50, (n,), device=dev, generator=gen)
        if trial % 11 == 5:
            starts = starts - 2 * bs  # negative positions: exercise the floor semantics
        spec_pos = (starts[:, None] + torch.arange(T, device=dev)[None, :]).reshape(-1)
        positions = torch.zeros(num_actual + 29, dtype=torch.int64, device=dev) + 12345
        if mixed:
            perm = torch.randperm(num_actual, device=dev, generator=gen)
            spec_idx = perm[:n_spec_tok].sort().values
            nonspec_idx = perm[n_spec_tok:].sort().values
            positions[spec_idx] = spec_pos
            positions[nonspec_idx] = torch.randint(0, 10 * bs, (n_extra,), device=dev, generator=gen)
        else:
            spec_idx = torch.arange(n_spec_tok, device=dev)
            nonspec_idx = torch.empty(0, dtype=torch.int64, device=dev)
            positions[:n_spec_tok] = spec_pos
        if trial % 2:
            spec_idx = spec_idx.to(torch.int32)
        qsl = torch.arange(n + 1, dtype=torch.int32, device=dev) * T
        if pad_rows:  # graph padding: trailing entries repeat the last offset (zero-length rows)
            qsl = torch.cat([qsl, qsl[-1:].repeat(pad_rows)])
        n_arg = n + (pad_rows if trial % 3 == 2 else 0)  # sometimes num_spec_decodes covers padded rows
        ns = nonspec_idx if mixed else (None if trial % 2 else nonspec_idx)
        ref = f.full_flags_torch(positions, num_actual, spec_idx, ns, qsl, n_arg, bs)
        got = f.full_flags_fused(positions, num_actual, spec_idx, ns, qsl, n_arg, bs)
        cases += 1
        if not torch.equal(ref, got):
            bad += 1
            if bad <= 3:
                emit(test="flags_mismatch", trial=trial, ref=ref.tolist(), got=got.tolist())
    mixed_pad = sum(1 for t in range(400) if (t % 4 == 3 or t % 7 == 1) and [0, 3, 17, 1, 0][t % 5] and t % 3 == 2)
    emit(test="flags_exact", cases=cases, mismatches=bad, bit_exact=bad == 0, mixed_with_padded_n=mixed_pad)
    if bad:
        FAIL.append("flags")


def flags_time():
    import glm_kda_stash_fast as f
    for n, T in [(1, 4), (1, 8), (16, 4), (16, 8)]:
        positions = torch.arange(n * T, device=dev, dtype=torch.int64) + 1000
        qsl = torch.arange(n + 1, dtype=torch.int32, device=dev) * T
        spec_idx = torch.arange(n * T, device=dev)
        ns = torch.empty(0, dtype=torch.int64, device=dev)
        row = dict(test="flags_time", n=n, T=T)
        for name, fn in (("chain", f.full_flags_torch), ("fused", f.full_flags_fused)):
            def run(fn=fn):
                for _ in range(34):
                    fn(positions, n * T, spec_idx, ns, qsl, n, 2304)
            us, _ = graph_us(run)
            row[f"{name}_us_34"] = round(us, 1)

        def run1():
            f.full_flags_fused(positions, n * T, spec_idx, ns, qsl, n, 2304)
        us, _ = graph_us(run1)
        row["shared_us_1"] = round(us, 1)
        emit(**row)


# ------------------------------------------------------------------------------------------ L2 prefetch
FP8_MAX = 448.0


def quant_mxfp8(w):  # glm-kernels gpu/bench_dense.py (ours)
    N, Kd = w.shape
    blk = w.float().view(N, Kd // 32, 32)
    amax = blk.abs().amax(-1).clamp(min=2.0 ** -126)
    e = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(-127, 127)
    scale = torch.exp2(e)
    q = (blk / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(N, Kd)
    return q, (e + 127).to(torch.uint8)


class FakeLinear(torch.nn.Module):
    def __init__(self, N, Kd):
        super().__init__()
        self.input_size_per_partition = Kd
        self.output_size_per_partition = N
        self.input_size = Kd
        self.output_size = N
        self.output_partition_sizes = [N]
        self.params_dtype = torch.bfloat16
        self.orig_dtype = torch.bfloat16
        self.has_bias = False
        self.bias = None


def marlin_mx8(N, Kd, gen):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_mxfp8_marlin_linear, prepare_mxfp8_layer_for_marlin)
    w = (torch.randn(N, Kd, device=dev, generator=gen) * 0.02).bfloat16()
    q, su8 = quant_mxfp8(w)
    del w
    L = FakeLinear(N, Kd).to(dev)
    L.weight = torch.nn.Parameter(q, requires_grad=False)
    L.weight_scale = torch.nn.Parameter(su8, requires_grad=False)
    prepare_mxfp8_layer_for_marlin(L)
    return L, (lambda x, L=L: apply_mxfp8_marlin_linear(x, L.weight, L.weight_scale, L.workspace, N, Kd))


def l2_bench():
    os.environ.setdefault("GLM_L2_PREFETCH_CACHE", "/tmp/glm_l2pf")
    import glm_l2_prefetch as l2
    import kda_stash as ks
    gen = torch.Generator(device=dev).manual_seed(4)
    LAYERS = 34
    ins, outs = [], []
    for _ in range(LAYERS):
        ins.append(marlin_mx8(PROJ, 4096, gen))
        outs.append(marlin_mx8(4096, 2048, gen))
    emit(test="l2_setup", o_proj_params=[(n, tuple(p.shape), str(p.dtype)) for n, p in
                                         outs[0][0].named_parameters(recurse=False)],
         o_proj_bytes=sum(b for _, b in l2.module_tensors(outs[0][0])))
    a_log = (torch.randn(1, 1, H, 1, device=dev, generator=gen) * 0.5).float()
    dt_bias = (torch.randn(H * D, device=dev, generator=gen) * 0.5).float()
    for N, T in [(1, 4), (1, 8), (4, 4), (16, 4), (16, 8)]:
        M = N * T
        x = (torch.randn(M, 4096, device=dev, generator=gen) * 0.5).bfloat16()
        pools = [make_pool(N * T + 1, gen) for _ in range(LAYERS if N <= 4 else 4)]
        idx = (torch.arange(N * T, device=dev, dtype=torch.int32) + 1).view(N, T)
        cu = torch.arange(N + 1, device=dev, dtype=torch.int32) * T
        nacc = torch.full((N,), max(1, T // 2), device=dev, dtype=torch.int32)
        core = torch.empty(1, M, H, D, device=dev, dtype=torch.bfloat16)
        ys = [torch.empty(M, 4096, device=dev, dtype=torch.bfloat16) for _ in range(LAYERS)]
        row = dict(test="l2_window_a", N=N, T=T, M=M)
        ref = None
        pool_base = [p.as_strided((p.untyped_storage().nbytes() // 4,), (1,)) for p in pools]
        pool_snap = [b.clone() for b in pool_base]
        for mb in (0, 4, 6, 8):
            for b, sn in zip(pool_base, pool_snap):
                b.copy_(sn)
            for p in pools:
                ks.get_aux(p).zero_()
            tables = [l2._table(l2.take(l2.module_tensors(outs[i][0]), int(mb * 2**20))) if mb else None
                      for i in range(LAYERS)]

            def run(tables=tables):
                for i in range(LAYERS):
                    proj = ins[i][1](x)
                    if tables[i] is not None:
                        l2._fork(tables[i])
                    qkv = proj[:, :QKV]
                    q, k, v = (t.reshape(1, -1, H, D) for t in qkv.split(H * D, dim=-1))
                    beta = proj[:, QKV:QKV + H].unsqueeze(0)
                    g1 = proj[:, :H * D].reshape(1, -1, H, D)
                    ks.fused_recurrent_kda_stash(q, k, v, g1, beta, pools[i % len(pools)], cu, idx, nacc, a_log,
                                                 dt_bias, out=core)
                    ys[i].copy_(outs[i][1](core.reshape(M, H * D)))
                l2.join_all()
            us, g = graph_us(run)
            # exactness: one fresh replay from the same pool state for every variant
            for b, sn in zip(pool_base, pool_snap):
                b.copy_(sn)
            for p in pools:
                ks.get_aux(p).zero_()
            g.replay()
            torch.cuda.synchronize()
            got = torch.stack([y.clone() for y in ys])
            if ref is None:
                ref = got
            elif not beq(ref, got):
                FAIL.append(f"l2 output differs N={N} T={T} mb={mb}")
            row[f"mb{mb}_us_34"] = round(us, 1)
            del g
        row["saved_us_best"] = round(row["mb0_us_34"] - min(row[f"mb{m}_us_34"] for m in (4, 6, 8)), 1)
        emit(**row)
    emit(test="l2_exact", exact=not any(x.startswith("l2") for x in FAIL))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    torch.backends.cuda.matmul.allow_tf32 = False
    if what in ("stash", "all", "exact"):
        stash_exact()
        stash_matrix()
        stash_dispatch()
    if what == "exact":
        flags_exact()
        stash_time()
    if what in ("flags", "all"):
        flags_exact()
        flags_time()
    if what in ("l2", "all"):
        l2_bench()
    out = os.environ.get("KX_JSON")
    if out:
        with open(out, "w") as fh:
            json.dump(OUT, fh, indent=1)
    print("FAIL" if FAIL else "PASS", FAIL, flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
