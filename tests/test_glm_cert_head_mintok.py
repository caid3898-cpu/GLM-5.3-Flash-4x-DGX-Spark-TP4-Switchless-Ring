# SPDX-License-Identifier: Apache-2.0
"""CPU test: min_tokens inside the certified target head (overlay/glm_cert_head.py, GLM_CERT_HEAD_MINTOK).

    uv run --no-project --with numpy --with torch python tests/test_glm_cert_head_mintok.py [OVERLAY_DIR] [-v]

OVERLAY_DIR defaults to the overlay/ next to this tests/ directory.

1 stock semantics  a line-by-line numpy port of the min-tokens clause of vLLM 487ecf187's
                   v1/worker/gpu/sample/logit_bias.py _bias_kernel (row t, request r = expanded_idx_mapping[t]:
                   if num_stop_token_ids[r] > 0 and pos[t] + 1 < min_lens[r]: logits[t, stop_token_ids[r, :n]] = -inf)
                   on the FULL vocab row, vs glm_cert_head.mask_shard_ (overflow / check path) and
                   glm_levers.mask_stop_torch on every one of 4 vocab shards; verify-shaped batches (several rows per
                   request at consecutive positions straddling min_len), duplicate stop ids, ids on other shards,
                   inactive requests, the pos + 1 == min_len boundary
2 certificate      cert_math (the numpy model of the screen / select / final kernels, extended with the same mask) vs
                   the stock argmax AFTER the mask, per shard and after the 4-rank pair reduction; adversarial: the
                   stop ids are the unmasked winners (and runners-up) of the rows, so masking changes the answer;
                   a token masked in one row but a candidate through another row; exact ties; overflow path
                   (small CAP); stock arithmetics exact / worst / seq32 and screen exact / worst
3 dispatch         glm_cert_head.install on fake runner classes: which steps take the certified path with the mask,
                   which stay on the glm_levers path (switch off, too many stop ids, gumbel rows, a foreign
                   lm_head patch), that the glm_levers wrapper is accepted, and that the decision reads CPU state only
"""
from __future__ import annotations

import os
import sys
import time
import types

import numpy as np

ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
OVERLAY = os.path.abspath(ARGS[0] if ARGS else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "overlay"))
sys.path.insert(0, OVERLAY)
VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    if VERBOSE or not cond:
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)


# ------------------------------------------------------------------------------------------------
# 1. stock semantics
# ------------------------------------------------------------------------------------------------
def stock_bias_mintok(logits, eidx, pos, min_lens, nstop, stop):
    """_bias_kernel, min-tokens clause only (use_logit_bias via min_tokens; no allowed ids / logit_bias), on a copy
    of the full-vocab logits in fp32 (Sampler.apply_sampling_params copies to fp32 first)."""
    out = np.array(logits, dtype=np.float32, copy=True)
    for t in range(out.shape[0]):                     # one program per logits row
        r = int(eidx[t])                              # req_state_idx = expanded_idx_mapping[token_idx]
        n = int(nstop[r])
        p = int(pos[t])
        ml = int(min_lens[r])
        if n > 0 and p + 1 < ml:
            ids = stop[r, :n]                         # mask = block < num_stop_token_ids
            out[t, ids] = -np.inf
    return out


def random_state(rng, R, Vt, S=128, verify=True):
    """LogitBiasState-like arrays for R requests + a batch: each request has 1 (decode) or k+1 (verify) rows at
    consecutive positions; min_len chosen so that some rows are before and some at/after the boundary."""
    min_lens = np.zeros(R, np.int32)
    nstop = np.zeros(R, np.int32)
    stop = np.zeros((R, S), np.int32)
    eidx, pos = [], []
    for r in range(R):
        n = int(rng.choice([0, 1, 2, 3, 5, 8, 16]))
        nstop[r] = n
        ids = rng.integers(0, Vt, n)
        if n >= 2:
            ids[1] = ids[0]                            # duplicate stop id (stock writes -inf twice)
        stop[r, :n] = ids
        stop[r, n:] = rng.integers(0, Vt, S - n)       # garbage past n must be ignored
        rows = int(rng.integers(1, 9)) if verify else 1
        p0 = int(rng.integers(100, 400))
        min_lens[r] = p0 + int(rng.integers(-2, rows + 3))   # boundary inside / just outside the verify rows
        for j in range(rows):
            eidx.append(r)
            pos.append(p0 + j)
    return (np.array(eidx, np.int32), np.array(pos, np.int64), min_lens, nstop, stop)


def test_semantics():
    import torch
    import glm_cert_head as ch
    import glm_levers as lv
    rng = np.random.default_rng(11)
    tp, Vs = 4, 97
    Vt = tp * Vs
    bad_c = bad_l = n = active_rows = boundary = 0
    for trial in range(300):
        R = int(rng.integers(1, 6))
        eidx, pos, min_lens, nstop, stop = random_state(rng, R, Vt)
        M = len(eidx)
        full = rng.standard_normal((M, Vt)).astype(np.float32)
        ref = stock_bias_mintok(full, eidx, pos, min_lens, nstop, stop)
        mt = ch.mintok_spec(torch.from_numpy(eidx), torch.from_numpy(pos), torch.from_numpy(min_lens),
                            torch.from_numpy(nstop), torch.from_numpy(stop), int(nstop.max()))
        for r in range(tp):
            sh = torch.from_numpy(full[:, r * Vs:(r + 1) * Vs].copy())
            got = ch.mask_shard_(sh.clone(), r * Vs, mt).numpy()
            bad_c += not np.array_equal(got, ref[:, r * Vs:(r + 1) * Vs])
            got2 = sh.clone()
            lv.mask_stop_torch(got2, r * Vs, torch.from_numpy(eidx), torch.from_numpy(pos), torch.from_numpy(min_lens),
                               torch.from_numpy(nstop), torch.from_numpy(stop))
            bad_l += not np.array_equal(got2.numpy(), ref[:, r * Vs:(r + 1) * Vs])
            n += 1
        act = [(nstop[eidx[t]] > 0) and (pos[t] + 1 < min_lens[eidx[t]]) for t in range(M)]
        active_rows += sum(act)
        boundary += sum(int(pos[t] + 1 == min_lens[eidx[t]]) for t in range(M))
    check("mask_shard_ == stock min-tokens clause on every shard (300 batches x 4 shards)", bad_c == 0,
          f"mismatch {bad_c}/{n}; active rows {active_rows}, boundary rows (pos+1 == min_len, unmasked) {boundary}")
    check("glm_levers.mask_stop_torch == the same stock clause (the non-certified admission path)", bad_l == 0,
          f"mismatch {bad_l}/{n}")
    check("spec width NS = next power of two >= max stop count; 0 when no request has stop ids",
          [ch.mintok_spec(torch.zeros(1), torch.zeros(1), None, None, torch.zeros(1, 4), k)["ns"]
           for k in (0, 1, 2, 3, 5, 16)] == [0, 1, 2, 4, 8, 16])
    mt0 = ch.mintok_spec(torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int64), None, None,
                         torch.zeros(2, 4, dtype=torch.int32), 0)
    x = torch.randn(2, 5)
    check("ns == 0 spec leaves the shard untouched", torch.equal(ch.mask_shard_(x.clone(), 0, mt0), x)
          and torch.equal(ch.mask_shard_(x.clone(), 0, None), x))


# ------------------------------------------------------------------------------------------------
# 2. certificate with the mask (numpy model of the kernels)
# ------------------------------------------------------------------------------------------------
def test_certificate():
    import cert_math as cm
    from fpfmt import bf16_round
    rng = np.random.default_rng(12)
    tp, Vs, K, G = 4, 256, 512, 128
    Vt = tp * Vs
    bad = bad_rank = n = changed = fallbacks = cross = 0
    t0 = time.time()
    for trial in range(120):
        W = (rng.standard_normal((Vt, K)) * 0.02 * rng.uniform(0.6, 1.4, (Vt, 1))).astype(np.float32)
        W = bf16_round(W)
        stock_mode = ("exact", "worst", "seq32")[trial % 3]
        screen_mode = ("exact", "worst")[(trial // 3) % 2]
        cap = (8, 64, 1 << 20)[trial % 3]
        # verify-shaped batch: 1-3 requests x 2-6 rows
        R = int(rng.integers(1, 4))
        eidx, pos, min_lens, nstop, stop = random_state(rng, R, Vt, S=16)
        M = len(eidx)
        gamma = rng.uniform(0.2, 0.35, K)
        x = rng.standard_normal((M, K)) * gamma
        for m in range(M):                               # peaked rows, several rows share a top token
            t = int(rng.integers(Vt)) if m % 2 == 0 or m == 0 else top
            top = t
            x[m] += W[t] / np.linalg.norm(W[t]) * np.linalg.norm(x[m]) * rng.uniform(0.3, 0.6)
        x = bf16_round(x.astype(np.float32))
        if trial % 5 == 0:                               # exact tie: duplicate a row across shards
            a, b = int(rng.integers(Vs)), int(rng.integers(2 * Vs, 3 * Vs))
            W[b] = W[a]
        # adversarial stop ids: the unmasked winners (and runners-up) of the rows
        l0 = cm.stock_logits(x, W, None, stock_mode, trial)
        order = np.argsort(-l0.astype(np.float64), axis=1, kind="stable")
        for r in range(R):
            rows = np.nonzero(eidx == r)[0]
            k = int(rng.integers(1, 6))
            ids = list(dict.fromkeys(int(v) for v in order[rows, :2].ravel()))[:k]
            nstop[r] = len(ids)
            stop[r, :len(ids)] = ids
            min_lens[r] = pos[rows].min() + int(rng.integers(1, len(rows) + 2))
        mask_full = np.isneginf(stock_bias_mintok(np.zeros((M, Vt), np.float32), eidx, pos, min_lens, nstop, stop))
        # a token masked for some row that is still the winner of another row -> it enters the union via that row
        cross += int(any(mask_full[m, order[m2, 0]] and not mask_full[m2, order[m2, 0]]
                         for m in range(M) for m2 in range(M)))
        want_v, want_i = cm.stock_local(x, W, stock_mode, seed=trial, mask=mask_full)
        changed += int(np.any(want_i != order[:, 0]))
        vals, idxs, starts = [], [], []
        for r in range(tp):
            Ws = W[r * Vs:(r + 1) * Vs]
            D = cm.deq_mxint8(*cm.make_mxint8(Ws)) if trial % 2 == 0 else cm.deq_fp8_twin(*cm.make_fp8_twin(Ws))
            Rt, Nv, _ = cm.build_tables(Ws, D, G)
            mloc = mask_full[:, r * Vs:(r + 1) * Vs]
            res = cm.cert_local(x, Ws, D, Rt, Nv, G, cap, stock_mode, screen_mode, seed=trial, mask=mloc)
            sv, si = cm.stock_local(x, Ws, stock_mode, seed=trial, mask=mloc)
            # a fully-masked local row has value -inf in both; the id then only needs to be in-vocab
            finite = np.isfinite(sv)
            same = (np.array_equal(res.val.view(np.uint32), sv.view(np.uint32))
                    and np.array_equal(res.idx[finite], si[finite]))
            bad_rank += not same
            fallbacks += res.fallback
            vals.append(res.val)
            idxs.append(res.idx)
            starts.append(r * Vs)
        got = cm.reduce_ranks(vals, idxs, starts)
        bad += not np.array_equal(got, want_i)
        n += 1
    check("certified + mask == stock argmax after the min-tokens mask, per rank (value and id)", bad_rank == 0,
          f"mismatch {bad_rank}/{n * tp}")
    check("certified + mask, 4-rank pair reduction == global stock argmax after the mask", bad == 0,
          f"mismatch {bad}/{n}; steps where the mask changed a winner {changed}; cross-row masked candidates "
          f"{cross}; overflow fallbacks {fallbacks}/{n * tp} ({time.time() - t0:.1f} s)")
    check("the mask changed winners (the test is not vacuous)", changed > n // 3, f"{changed}/{n}")
    check("cross-row case exercised (masked for one row, candidate through another)", cross > 0, f"{cross}")
    check("overflow path exercised with the mask", fallbacks > 0)
    # without masking the final selection per row, the cross-row case goes wrong: the check has teeth
    wrong = 0
    for trial in range(40):
        Vs1, K1 = 256, 256
        W = bf16_round((rng.standard_normal((Vs1, K1)) * 0.02).astype(np.float32))
        t = int(rng.integers(Vs1))
        x = rng.standard_normal((2, K1)) * 0.3
        x += W[t] / np.linalg.norm(W[t]) * np.linalg.norm(x, axis=1, keepdims=True) * 0.6
        x = bf16_round(x.astype(np.float32))
        mask = np.zeros((2, Vs1), bool)
        mask[0, t] = True                                 # t masked on row 0 only; row 1 keeps it as a candidate
        D = cm.deq_mxint8(*cm.make_mxint8(W))
        Rt, Nv, _ = cm.build_tables(W, D, 128)
        s_hat = cm.screen_emulation(x, D)
        lo, hi = cm.interval(s_hat, cm.bounds(x, Rt, Nv, 128))
        lo = np.where(mask, -np.inf, lo)
        hi = np.where(mask, -np.inf, hi)
        _, union, _ = cm.candidates(lo, hi)
        ids = np.nonzero(union)[0]
        l = cm.stock_logits(x, W, ids)
        _, naive = cm.final_select(l[0], ids)            # screen mask only, no final mask
        wrong += naive == t
    check("negative control: screen-only masking would emit the masked token (final mask is required)", wrong > 0,
          f"{wrong}/40 steps")


# ------------------------------------------------------------------------------------------------
# 3. dispatch through the real install() on fakes
# ------------------------------------------------------------------------------------------------
class _Uva:
    def __init__(self, a):
        import torch
        self.np = a
        self.gpu = torch.from_numpy(a.copy())   # a device copy: separate memory from .np


def test_dispatch():
    import torch
    import glm_cert_head as ch
    import glm_target_argmax as ta
    calls = []
    ta.plan = lambda runner, ib, g: ("reject", 8)
    ta.fast_sample = lambda runner, hs, ib, how: calls.append("orig_fast") or ("orig", None, None)

    class GPUModelRunner:
        def load_model(self, *a, **k):
            return None

        def sample(self, hidden_states, input_batch, grammar_output):
            calls.append("stock_sample")
            return ("stock", None, None)
    GPUModelRunner.sample.__wrapped__ = GPUModelRunner.sample   # glm_target_argmax marks its hook this way
    mod = types.SimpleNamespace(GPUModelRunner=GPUModelRunner)
    ch.install(mod)
    seen = []

    def fake_certified(runner, st, hs, ib, kind, width, mt=None):
        seen.append(mt)
        calls.append("cert")
        return ("cert", None, None, None)
    ch._certified = fake_certified

    W = torch.randn(256, 256).to(torch.bfloat16)
    st = ch.build_from_weight(W, 0, 4, "mxint8")
    st.lp = types.SimpleNamespace()
    R, S = 4, 128

    def make(nst=(1, 2, 0, 3), flag=True, M=8):
        runner = GPUModelRunner()
        runner._glm_cert = st
        runner._glm_lv_mintok = flag
        nstop = np.array(nst, np.int32)
        lb = types.SimpleNamespace(num_stop_token_ids=_Uva(nstop), min_lens=_Uva(np.full(R, 300, np.int32)),
                                   stop_token_ids=_Uva(np.zeros((R, S), np.int32)),
                                   num_allowed_token_ids=_Uva(np.zeros(R, np.int32)),
                                   num_logit_bias=_Uva(np.zeros(R, np.int32)))
        runner.sampler = types.SimpleNamespace(logit_bias_state=lb)
        ib = types.SimpleNamespace(logits_indices=torch.arange(M), idx_mapping_np=np.array([0, 1, 3]),
                                   expanded_idx_mapping=torch.tensor([0] * 3 + [1] * 3 + [3] * (M - 6),
                                                                     dtype=torch.int32),
                                   positions=torch.arange(M, dtype=torch.int64) + 290)
        return runner, ib

    hs = torch.zeros(8, 256, dtype=torch.bfloat16)

    def run(runner, ib, how=("reject", 8)):
        calls.clear()
        seen.clear()
        ta.fast_sample(runner, hs, ib, how)
        return calls[-1], (seen[-1] if seen else "n/a")

    r, ib = make(flag=False)
    c, mt = run(r, ib)
    check("no min_tokens: certified, no mask (unchanged behaviour)", c == "cert" and mt is None, f"{c} {mt}")
    r, ib = make()
    c, mt = run(r, ib)
    check("min_tokens step (stop ids 1..3): certified WITH the mask, NS = 4", c == "cert" and mt is not None
          and mt["ns"] == 4 and mt["stop"] is r.sampler.logit_bias_state.stop_token_ids.gpu, f"{c}")
    check("mask spec reads the stock tensors: positions[logits_indices], expanded_idx_mapping, min_lens",
          mt is not None and torch.equal(mt["pos"], ib.positions[ib.logits_indices])
          and torch.equal(mt["eidx"], ib.expanded_idx_mapping) and mt["min_lens"] is r.sampler.logit_bias_state.min_lens.gpu)
    old = ch.MINTOK
    ch.MINTOK = False
    r, ib = make()
    c, _ = run(r, ib)
    ch.MINTOK = old
    check("GLM_CERT_HEAD_MINTOK=0: min_tokens step stays on the glm_levers path", c == "orig_fast", c)
    r, ib = make(nst=(1, 17, 0, 3))
    c, _ = run(r, ib)
    check("17 stop ids (> MAX_STOP 16) in a request of the batch: glm_levers path", c == "orig_fast", c)
    r, ib = make(nst=(1, 17, 0, 3))
    ib.idx_mapping_np = np.array([0, 3])                  # the 17-stop request is not in this batch
    c, mt = run(r, ib)
    check("stop count is taken over the batch's requests only", c == "cert" and mt["ns"] == 4, c)
    r, ib = make()
    c, _ = run(r, ib, ("gumbel", 1))
    check("gumbel row with a min_tokens flag: full stock sampler (mask never combined with Gumbel)",
          c == "stock_sample", c)

    def tagged(head, h, bias):
        return None
    tagged._glm_lv_mintok_mask = True
    st.lp._apply_head = tagged
    r, ib = make()
    c, mt = run(r, ib)
    check("glm_levers' tagged lm_head wrapper (hook order levers-outer) is accepted", c == "cert" and mt is not None, c)
    st.lp._apply_head = lambda head, h, bias: None
    r, ib = make()
    c, _ = run(r, ib)
    check("a foreign lm_head instance patch keeps the step on the non-certified path", c == "orig_fast", c)
    r, ib = make(flag=False)
    c, _ = run(r, ib)
    check("a foreign patch without min_tokens: non-certified path (as before)", c == "orig_fast", c)
    del st.lp._apply_head
    # rank invariance: decisions read CPU arrays + agreed switches; GPU tensor contents do not matter
    r1, ib1 = make()
    r2, ib2 = make()
    r2.sampler.logit_bias_state.stop_token_ids.gpu.fill_(7)
    r2.sampler.logit_bias_state.min_lens.gpu.fill_(0)
    r2.sampler.logit_bias_state.num_stop_token_ids.gpu.fill_(99)   # a (hypothetically) diverged device copy
    a1 = run(r1, ib1)
    a2 = run(r2, ib2)
    check("dispatch identical whatever the device tensors hold (host state only)",
          a1[0] == a2[0] and a1[1]["ns"] == a2[1]["ns"], f"{a1[0]} {a2[0]}")
    check("GLM_CERT_HEAD_MINTOK / MAX_STOP enter the TP-agreed config hash",
          "GLM_CERT_HEAD_MINTOK" in ch.CONFIG_KEYS and "GLM_CERT_HEAD_MINTOK_MAX_STOP" in ch.CONFIG_KEYS
          and ch.config_hash({"GLM_CERT_HEAD_MINTOK": "1"}) != ch.config_hash({"GLM_CERT_HEAD_MINTOK": "0"}))
    import glm_ab
    check("GLM_CERT_HEAD_MINTOK is an in-boot A/B key (glm_ab.KNOWN, bool)",
          glm_ab.KNOWN.get("GLM_CERT_HEAD_MINTOK") == "bool")


if __name__ == "__main__":
    os.environ.setdefault("GLM_CERT_HEAD", "1")
    os.environ.setdefault("GLM_CERT_HEAD_SLICEK_MAP", "1:0,2-32:1")
    t0 = time.time()
    print(f"overlay: {OVERLAY}")
    test_semantics()
    test_certificate()
    test_dispatch()
    npass = sum(ok for _, ok in RESULTS)
    print(f"{npass}/{len(RESULTS)} checks passed ({time.time() - t0:.1f} s)")
    print("ALL PASS" if npass == len(RESULTS) else "FAIL")
    sys.exit(0 if npass == len(RESULTS) else 1)
