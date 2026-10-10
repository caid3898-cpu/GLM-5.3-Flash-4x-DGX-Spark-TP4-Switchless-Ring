# SPDX-License-Identifier: Apache-2.0
"""GLM_DEVSELECT: device-side verify-shape selection for GLM_DRAFT_TRUNC. Default OFF.

c1 only (one request, 7 scheduled drafts). The host prepares the ordinary 8-row verify step during the draft and replays
one PARENT graph instead of the 8-row target graph. The parent holds:
  [select]                         fp64 single-thread mirror of glm_draft_trunc.Policy.choose -> pred[2..8], L history
  IF pred[M] (M = 2..7): [fixups for M] [MLA plan copy M] [child = clone of vLLM's captured target graph for M]
  IF pred[8]:            [child = clone of the 8-row graph]
Nothing waits for the draft; L never reaches the host (scheduler: unverified drafts roll back as rejections, as 2209).

Needs glm_draft_trunc (policy, table, confidence broadcast in propose, SpecProbeScheduler default_k), GLM_EARLY_PLAN
(the 8-row MLA plan is built during the draft) and FULL target graphs for M = 2..8 at one request (env.trunc*).

Knobs:
  GLM_DEVSELECT                 install gate (os.environ) and glm_ab raw key (per variant, read per step)
  GLM_DEVSELECT_FORCE=-1|1..7   force L on the device (tests); -1 = the policy
  GLM_DEVSELECT_DEBUG_STEPS=N   exactness gate G1 on the first N device steps (sync-heavy; logs DEVSELECT_G1 lines)
  GLM_DEVSELECT_DEBUG_EVERY=N   and every N-th device step after that (0 = never)
  GLM_DEVSELECT_RANKCHECK_EVERY=N  gate G4: every N runner steps all TP ranks compare their device L history (0 = off)
  GLM_DEVSELECT_LOG_EVERY=N     status line every N device steps (default 2000)
Log prefix "glm-devselect:". Failure lines: "DEVSELECT_BUILD_FAILED", "DEVSELECT_MISMATCH", "DEVSELECT_RANK_MISMATCH".
"""
from __future__ import annotations

import os
import sys
import time
import traceback

_OFF = ("", "0", "off", "false", "no")
KMAX = 7
K = KMAX + 1                      # scheduled verify rows
MS = list(range(2, K + 1))
SM90 = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"
CGU = "vllm.v1.worker.gpu.cudagraph_utils"
RUNNER = "vllm.v1.worker.gpu.model_runner"
PAD_SLOT_ID = -1
RING = 256


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-devselect: {msg}\n")
    sys.stderr.flush()


def _int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default)) or default))
    except ValueError:
        return default


def installed() -> bool:
    return str(os.environ.get("GLM_DEVSELECT", "0")).strip().lower() not in _OFF


def active() -> bool:
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        v = ab.env("GLM_DEVSELECT", "0")
    else:
        v = os.environ.get("GLM_DEVSELECT", "0")
    return str(v).strip().lower() not in _OFF


class _D:
    built = False
    failed = None                 # reason string once disabled
    capturing_target = False
    next_desc = None              # descriptor whose graph object is about to be constructed (target capture)
    plan_info_cap: dict = {}      # num_tokens -> plan_info seen while capturing the target graphs
    sets: dict = {}               # id(graphs dict) -> per-set entry
    shared = None                 # device tensors shared by every set
    stage = None                  # per-M MLA staging
    targets = None                # patch targets
    go = False                    # this step runs the parent (set by glm_draft_trunc.execute_model)
    so = None                     # the step's scheduler output (debug G1 only)
    in_ref = False
    runner = None
    steps = 0
    dbg_done = 0
    dbg_runs = 0
    exec_calls = 0
    t_stage: list = []
    t_launch: list = []
    fallbacks: dict = {}
    warned: set = set()


def _fallback(reason: str) -> None:
    _D.fallbacks[reason] = _D.fallbacks.get(reason, 0) + 1
    if reason not in _D.warned:
        _D.warned.add(reason)
        _log(f"fallback to the 8-row graph: {reason} (first occurrence)")


# ================================================================================= kernels (built lazily on GPU)
_KERN = {}


def _kernels():
    if _KERN:
        return _KERN
    import triton
    import triton.language as tl

    @triton.jit
    def select_kernel(conf_ptr, valid_ptr, force_ptr, g_ptr, edges_ptr, t1_ptr, lam_ptr, coef_ptr, pred_ptr,
                      ring_ptr, cnt_ptr, hist_ptr, KMAX_: tl.constexpr, NB: tl.constexpr, NE: tl.constexpr,
                      RING_: tl.constexpr):
        # fp64 mirror of glm_draft_trunc.Policy.choose for one request (same operation order; no FMA contraction:
        # launched with enable_fp_fusion=False; ALPHA from an fp64 tensor). CPU-verified in devselect/.
        lam = tl.load(lam_ptr)
        alpha = tl.load(coef_ptr)
        one_m_alpha = tl.load(coef_ptr + 1)
        valid = tl.load(valid_ptr)
        force = tl.load(force_ptr)
        E = lam * 0.0 + 1.0
        cum = lam * 0.0 + 1.0
        best = lam * 0.0 - 1e30
        bE = E
        bL = valid * 0 + KMAX_
        for s in tl.static_range(KMAX_):
            lq = tl.load(conf_ptr + s).to(tl.float64)
            b = valid * 0
            for e in tl.static_range(NE):
                b += (lq >= tl.load(edges_ptr + e)).to(tl.int32)
            cum = cum * tl.load(g_ptr + s * NB + b)
            E = E + cum
            v = E - lam * tl.load(t1_ptr + s + 2)
            take = v > best
            best = tl.where(take, v, best)
            bL = tl.where(take, s + 1, bL)
            bE = tl.where(take, E, bE)
        use = valid != 0
        sel = tl.where(use, bL, KMAX_)
        sel = tl.where(force >= 1, force, sel)
        newlam = one_m_alpha * lam + alpha * bE / tl.load(t1_ptr + bL + 1)
        tl.store(lam_ptr, tl.where(use, newlam, lam))
        for m in tl.static_range(2, KMAX_ + 2):
            tl.store(pred_ptr + m, sel + 1 == m)
        c = tl.load(cnt_ptr)
        tl.store(ring_ptr + c % RING_, sel)
        tl.store(cnt_ptr, c + 1)
        h = tl.load(hist_ptr + sel)
        tl.store(hist_ptr + sel, h + 1)

    _KERN["select"] = select_kernel

    @triton.jit
    def _dsf_fill(ptr, lo, hi, val, B: tl.constexpr):
        for s in range(lo, hi, B):
            o = s + tl.arange(0, B)
            tl.store(ptr + o, (tl.zeros([B], dtype=tl.int64) + val).to(ptr.dtype.element_ty), mask=o < hi)

    @triton.jit
    def _dsf_copy(src, dst, soff, doff, n, B: tl.constexpr):
        for s in range(0, n, B):
            o = s + tl.arange(0, B)
            m = o < n
            tl.store(dst + doff + o, tl.load(src + soff + o, mask=m), mask=m)

    globals()["_dsf_fill"], globals()["_dsf_copy"] = _dsf_fill, _dsf_copy   # jit helpers resolvable as globals

    @triton.jit
    def fix_kernel(qsl, n_qsl, seq, cu, g0, g1, g2, g3, pr, dsl, csm, slots, n_sg, s_stride, ctx, n_cg, c_stride,
                   src, dst, offs, qo_s, qo_d, n_qo, kv_s, kv_d, n_kv, ln_s, ln_d, n_ln,
                   M: tl.constexpr, K_: tl.constexpr, NG: tl.constexpr, HAS_IDX: tl.constexpr,
                   HAS_CSM: tl.constexpr, NCL: tl.constexpr, NSM: tl.constexpr, PAD: tl.constexpr,
                   B: tl.constexpr):
        # One node for every body-M fixup of _fix_ops_legacy (same values, same buffers), plus the MLA int-workspace
        # copy restricted to its valid region: work arrays [0, W) with W = work_indptr[NCL] read from the staged plan
        # (device), merge arrays [0, NSM), work_indptr [0, NCL + 1). Offsets (int32 units) in offs[0..13].
        _dsf_fill(qsl, 1, n_qsl, M, B)
        v = tl.load(seq)
        tl.store(seq, v - (K_ - M))
        _dsf_fill(cu, 1, 2, M, B)
        if NG > 0:
            _dsf_fill(g0, 1, 2, M, B)
        if NG > 1:
            _dsf_fill(g1, 1, 2, M, B)
        if NG > 2:
            _dsf_fill(g2, 1, 2, M, B)
        if NG > 3:
            _dsf_fill(g3, 1, 2, M, B)
        if HAS_IDX:
            _dsf_fill(pr, 0, 1, M, B)
            _dsf_fill(dsl, M, K_, 0, B)
            if HAS_CSM:
                _dsf_fill(csm, M, K_, PAD, B)
        for g in range(0, n_sg):
            _dsf_fill(slots + g * s_stride, M, K_, PAD, B)
        for g in range(0, n_cg):
            _dsf_fill(ctx + g * c_stride, M, K_, PAD, B)
        wi = tl.load(offs + 13)
        W = tl.load(src + wi + NCL)
        for a in tl.static_range(8):
            off = tl.load(offs + a)
            _dsf_copy(src, dst, off, off, W, B)
        for a in tl.static_range(8, 13):
            off = tl.load(offs + a)
            _dsf_copy(src, dst, off, off, NSM, B)
        _dsf_copy(src, dst, wi, wi, NCL + 1, B)
        _dsf_copy(qo_s, qo_d, 0, 0, n_qo, B)
        _dsf_copy(kv_s, kv_d, 0, 0, n_kv, B)
        _dsf_copy(ln_s, ln_d, 0, 0, n_ln, B)

    _KERN["fix"] = fix_kernel
    return _KERN


# ================================================================================= build (after capture_model)
def _dev_tensors(runner, policy):
    import torch
    dev = runner.device
    f64 = dict(dtype=torch.float64, device=dev)
    t1 = [0.0] * (K + 2)
    for m in MS:
        t1[m] = float(policy.t1[m])
    from glm_draft_trunc import ALPHA
    return {
        "conf": torch.zeros(KMAX, dtype=torch.float32, device=dev),
        "valid": torch.zeros((), dtype=torch.int32, device=dev),
        "force": torch.full((), _int("GLM_DEVSELECT_FORCE", -1), dtype=torch.int32, device=dev),
        "g": torch.tensor(policy.g, **f64).contiguous(),
        "edges": torch.tensor(policy.edges, **f64),
        "t1": torch.tensor(t1, **f64),
        "coef": torch.tensor([ALPHA, 1 - ALPHA], **f64),
        "lam0": 2.2 / policy.cost(1, 4),
    }


def _set_tensors(runner, sh):
    import torch
    dev = runner.device
    return {
        "lam": torch.tensor(sh["lam0"], dtype=torch.float64, device=dev),
        "pred": torch.zeros(K + 2, dtype=torch.bool, device=dev),
        "ring": torch.zeros(RING, dtype=torch.int32, device=dev),
        "cnt": torch.zeros((), dtype=torch.int32, device=dev),
        "hist": torch.zeros(K + 1, dtype=torch.int32, device=dev),
    }


def _select(sh, st):
    k = _kernels()["select"]
    k[(1,)](sh["conf"], sh["valid"], sh["force"], sh["g"], sh["edges"], sh["t1"], st["lam"], sh["coef"], st["pred"],
            st["ring"], st["cnt"], st["hist"], KMAX_=KMAX, NB=sh["g"].shape[1], NE=sh["edges"].numel(),
            RING_=RING, num_warps=1, enable_fp_fusion=False)


def _targets(runner):
    """Every buffer a body patches."""
    import torch
    sm = sys.modules[SM90]
    state = sm._SM90_STATE
    if state is None:
        raise RuntimeError("no FlashInfer SM90 MLA state")
    gdn = []
    for groups in runner.attn_groups:
        for g in groups:
            for b in getattr(g, "metadata_builders", []) or []:
                q = getattr(b, "spec_query_start_loc", None)
                if isinstance(q, torch.Tensor) and q.is_cuda and all(q.data_ptr() != x.data_ptr() for x in gdn):
                    gdn.append(q)
    idx = []                                         # DSA indexer builders (G1 of ds0218: L-dependent decode metadata)
    for groups in runner.attn_groups:
        for g in groups:
            for b in getattr(g, "metadata_builders", []) or []:
                if all(isinstance(getattr(b, a_, None), torch.Tensor) for a_ in
                       ("scheduler_metadata_buffer", "per_req_decode_lens_buffer", "decode_seq_lens_buffer")) \
                        and all(b is not x for x in idx):
                    idx.append(b)
    spec = runner.speculator
    ctx = getattr(spec, "_context_slot_mappings", None)
    if ctx is None:
        raise RuntimeError("speculator has no _context_slot_mappings (not DFlash)")
    return types_ns(
        qsl=runner.input_buffers.query_start_loc,
        seq_lens=runner.input_buffers.seq_lens,
        cu=torch.zeros(runner.input_buffers.query_start_loc.numel(), dtype=torch.int32, device=runner.device),
        gdn=gdn,
        indexers=idx,
        slots=runner.block_tables.slot_mappings,
        ctx=ctx,
        state=state,
        wrapper=state.wrapper,
    )


class types_ns:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _stage_alloc(runner, T):
    """Per-M staging: device + pinned int workspace, device indptr / len buffers, pinned host inputs, events."""
    import torch
    w, st = T.wrapper, T.state
    nbytes = w._int_workspace_buffer.numel()
    S = {}
    for M in MS[:-1]:
        qo = torch.clamp(torch.arange(st.max_tokens + 1, dtype=torch.int32), max=M).pin_memory()
        S[M] = types_ns(
            int_dev=torch.empty(nbytes, dtype=torch.uint8, device=runner.device),
            int_pin=torch.empty(nbytes, dtype=torch.uint8, pin_memory=True),
            qo_dev=torch.empty_like(w._qo_indptr_buf), kv_dev=torch.empty_like(w._kv_indptr_buf),
            len_dev=torch.empty_like(w._kv_len_arr_buf),
            qo_cpu=qo, kv_cpu=(qo * st.topk_width).pin_memory(),
            lens_cpu=torch.full((st.max_tokens,), st.topk_width, dtype=torch.int32).pin_memory(),
            ev=None, ext=nbytes, plan_info=None)
    return S


def _plan_into(T, slot, M, lens):
    """FlashInfer plan for M rows into the staging slot (current stream); canonical buffers restored after."""
    import torch
    w, st = T.wrapper, T.state
    slot.lens_cpu[:M].copy_(lens[:M])
    saved = (w._int_workspace_buffer, w._pin_memory_int_workspace_buffer, w._qo_indptr_buf, w._kv_indptr_buf,
             w._kv_len_arr_buf, w._plan_info)
    w._int_workspace_buffer, w._pin_memory_int_workspace_buffer = slot.int_dev, slot.int_pin
    w._qo_indptr_buf, w._kv_indptr_buf, w._kv_len_arr_buf = slot.qo_dev, slot.kv_dev, slot.len_dev
    try:
        w.plan(slot.qo_cpu, slot.kv_cpu, st.kv_indices, slot.lens_cpu, st.num_heads, st.kv_lora_rank,
               st.qk_rope_head_dim, 1, False, st.sm_scale, q_data_type=torch.bfloat16, kv_data_type=st.kv_dtype)
        info = w._plan_info
    finally:
        (w._int_workspace_buffer, w._pin_memory_int_workspace_buffer, w._qo_indptr_buf, w._kv_indptr_buf,
         w._kv_len_arr_buf, w._plan_info) = saved
    return list(info) if isinstance(info, (list, tuple)) else info


def _measure_extents(T, S):
    """Used int-workspace bytes per M: sentinel-fill the pinned slot, plan a few context patterns, find the last
    written byte; generous margin (the copy is ~1 us per 100 KiB)."""
    import numpy as np
    import torch
    for M, slot in S.items():
        last = 0
        for ctx in (5, 1500, 2100, 70000, 250000):
            slot.int_pin.fill_(0xA5)
            lens = torch.tensor([min(ctx + j + 1, T.state.topk_width) for j in range(M)], dtype=torch.int32)
            _plan_into(T, slot, M, lens)
            torch.cuda.synchronize()
            a = slot.int_pin.numpy()
            nz = np.nonzero(a != 0xA5)[0]
            if nz.size:
                last = max(last, int(nz[-1]) + 1)
        ext = min(slot.int_pin.numel(), ((int(last * 1.5) + 65536 + 4095) // 4096) * 4096)
        slot.ext = ext
    return {M: s.ext for M, s in S.items()}


def _fix_ops(T, S, M):
    """The body-M fixups (captured into the IF body; also run eagerly by the G1 debug gate and the probes).
    Fused (GLM_DEVSELECT_FUSED=1, default, after the boot self-check passed): one Triton node + the DeepGEMM
    scheduler-metadata recompute (+ its 800 B copy). Legacy: ~20 torch nodes incl. the 0.84 MB MLA copy."""
    F = getattr(_D, "fused", None)
    if F is None:
        return _fix_ops_legacy(T, S, M)
    _fused_launch(F, M)
    _sched_meta(T, M)


def _sched_meta(T, M):
    for ib in T.indexers:
        from vllm.v1.attention.backends.mla.indexer import get_paged_mqa_logits_metadata
        ib.scheduler_metadata_buffer.copy_(get_paged_mqa_logits_metadata(
            ib.decode_seq_lens_buffer[:M].view(M, 1), ib.kv_cache_spec.storage_block_size, ib.num_sms))


def _fused_prepare(T, S):
    """Static arguments of the fused fixup kernel per M (host ints + persistent device tensors). Raises when the
    buffers fall outside what the kernel handles (the caller then keeps the legacy fixups)."""
    import torch
    if len(T.gdn) > 4 or len(T.indexers) > 1:
        raise RuntimeError(f"fused fixups handle <= 4 GDN qsl buffers and <= 1 indexer ({len(T.gdn)}, "
                           f"{len(T.indexers)})")
    for name, t in (("slots", T.slots), ("ctx", T.ctx)):
        if not isinstance(t, torch.Tensor) or t.dim() != 2 or t.stride(1) != 1:
            raise RuntimeError(f"{name} is not a row-major 2-D tensor")
    for name, t in [("qsl", T.qsl), ("seq_lens", T.seq_lens), ("cu", T.cu)] + [(f"gdn{i}", q) for i, q in
                                                                              enumerate(T.gdn)]:
        if not t.is_contiguous():
            raise RuntimeError(f"{name} not contiguous")
    w = T.wrapper
    dev = T.qsl.device
    dummy = torch.zeros(4, dtype=torch.int32, device=dev)
    gd = list(T.gdn) + [dummy] * (4 - len(T.gdn))
    if T.indexers:
        ib = T.indexers[0]
        pr, dsl = ib.per_req_decode_lens_buffer, ib.decode_seq_lens_buffer
        csm = getattr(ib, "compressed_slot_mapping_buffer", None)
        for name, t in (("per_req", pr), ("decode_seq_lens", dsl)) + ((("csm", csm),) if csm is not None else ()):
            if not t.is_contiguous():
                raise RuntimeError(f"indexer {name} not contiguous")
    else:
        pr = dsl = csm = None
    dst = w._int_workspace_buffer.view(torch.int32)
    per_m = {}
    for M, slot in S.items():
        pi = [int(x) for x in slot.plan_info]
        byte_offs = pi[2:16]
        if any(o % 4 for o in byte_offs):
            raise RuntimeError(f"M={M}: plan offsets not 4-byte aligned {byte_offs}")
        (q_ind, kv_ind, part_ind, mps, mpe, mpps, mppe, mstride, q_len, kv_len, q_st, kv_st, kv_end,
         work_ind) = [o // 4 for o in byte_offs]
        offs = torch.tensor([q_ind, kv_ind, part_ind, q_len, kv_len, q_st, kv_st, kv_end,
                             mps, mpe, mpps, mppe, mstride, work_ind], dtype=torch.int32, device=dev)
        per_m[M] = dict(src=slot.int_dev.view(torch.int32), offs=offs, qo_s=slot.qo_dev, kv_s=slot.kv_dev,
                        ln_s=slot.len_dev, NCL=pi[1], NSM=pi[0] * pi[1])
    return dict(T=T, gd=gd, pr=pr if pr is not None else dummy, dsl=dsl if dsl is not None else dummy,
                csm=csm if csm is not None else dummy, has_idx=pr is not None, has_csm=csm is not None,
                dst=dst, per_m=per_m)


def _fused_launch(F, M):
    k = _kernels()["fix"]
    T, p, w = F["T"], F["per_m"][M], F["T"].wrapper
    g = F["gd"]
    k[(1,)](T.qsl, T.qsl.numel(), T.seq_lens, T.cu, g[0], g[1], g[2], g[3], F["pr"], F["dsl"], F["csm"],
            T.slots, T.slots.shape[0], T.slots.stride(0), T.ctx, T.ctx.shape[0], T.ctx.stride(0),
            p["src"], F["dst"], p["offs"], p["qo_s"], w._qo_indptr_buf, w._qo_indptr_buf.numel(),
            p["kv_s"], w._kv_indptr_buf, w._kv_indptr_buf.numel(), p["ln_s"], w._kv_len_arr_buf,
            w._kv_len_arr_buf.numel(),
            M=M, K_=K, NG=len(T.gdn), HAS_IDX=F["has_idx"], HAS_CSM=F["has_csm"], NCL=p["NCL"], NSM=p["NSM"],
            PAD=PAD_SLOT_ID, B=1024, num_warps=4)


def _fused_selfcheck(runner, T, S):
    """Boot unit test on the GPU, before any capture (it also JIT-compiles every fused specialisation with the exact
    arguments the capture uses): for every M, legacy fixups vs fused fixups from the same buffer image must give
    bit-identical buffers everywhere except the MLA int workspace, whose valid region must be identical
    (mla_valid_equal) against the staged plan. Returns (ok, detail)."""
    import torch
    reg = _registry(runner)
    reg = reg + [("x.slots", T.slots), ("x.ctx", T.ctx), ("x.cu", T.cu)] + [(f"x.gdn{i}", q) for i, q in
                                                                            enumerate(T.gdn)]
    A = _snap(reg)
    bad = []
    try:
        for M in S:
            _restore(reg, A)
            _fix_ops_legacy(T, S, M)
            torch.cuda.synchronize()
            Bs = _snap(reg)
            _restore(reg, A)
            _fused_launch(_D.fused_try, M)
            _sched_meta(T, M)
            torch.cuda.synchronize()
            Cs = _snap(reg)
            for (name, _), b, c in zip(reg, Bs, Cs):
                if name.endswith("_int_workspace_buffer"):
                    ok, W, field = mla_valid_equal(b.cpu(), c.cpu(), S[M].plan_info)
                    ok2, _, _ = mla_valid_equal(S[M].int_dev.cpu(), c.cpu(), S[M].plan_info)
                    if not (ok and ok2):
                        bad.append(f"M={M} {name} valid region ({W} works) differs at {field or 'staged'}")
                    continue
                if not torch.equal(b.reshape(-1).view(torch.uint8), c.reshape(-1).view(torch.uint8)):
                    bad.append(f"M={M} {name}")
    finally:
        _restore(reg, A)
        torch.cuda.synchronize()
    return (not bad), ("; ".join(bad[:12]) if bad else f"{len(S)} M x {len(reg)} buffers identical")


def _fix_ops_legacy(T, S, M):
    """The body-M fixups, one torch op per buffer (overlay3 behaviour)."""
    T.qsl[1:].fill_(M)
    T.seq_lens[0:1].sub_(K - M)
    T.cu[1:2].fill_(M)
    for q in T.gdn:
        q[1:2].fill_(M)
    for ib in T.indexers:                            # DSA indexer decode metadata for M rows (n = 1, native path)
        from vllm.v1.attention.backends.mla.indexer import get_paged_mqa_logits_metadata
        ib.per_req_decode_lens_buffer[0:1].fill_(M)
        ib.decode_seq_lens_buffer[M:K].zero_()
        # flatten path (uniform decode): M per-token rows of next_n = 1 -> context lens shape (M, 1), as the builder
        ib.scheduler_metadata_buffer.copy_(get_paged_mqa_logits_metadata(
            ib.decode_seq_lens_buffer[:M].view(M, 1), ib.kv_cache_spec.storage_block_size, ib.num_sms))
        csm = getattr(ib, "compressed_slot_mapping_buffer", None)
        if csm is not None:                          # dead rows must never write compressed KV into live slots
            csm[M:K].fill_(PAD_SLOT_ID)
    T.slots[:, M:K].fill_(PAD_SLOT_ID)
    T.ctx[:, M:K].fill_(PAD_SLOT_ID)
    slot, w = S[M], T.wrapper
    w._int_workspace_buffer[: slot.ext].copy_(slot.int_dev[: slot.ext])
    w._qo_indptr_buf.copy_(slot.qo_dev)
    w._kv_indptr_buf.copy_(slot.kv_dev)
    w._kv_len_arr_buf.copy_(slot.len_dev)


def _add_child_to_capture(child_ptr: int) -> None:
    import torch
    from cuda.bindings import runtime as cudart
    s = cudart.cudaStream_t(init_value=int(torch.cuda.current_stream().cuda_stream))
    info = cudart.cudaStreamGetCaptureInfo(s)
    if info[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamGetCaptureInfo: {info[0]}")
    graph, deps, ndeps = info[3], info[4], info[-1]
    err, node = cudart.cudaGraphAddChildGraphNode(graph, deps, ndeps,
                                                  cudart.cudaGraph_t(init_value=int(child_ptr)))
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphAddChildGraphNode: {err}")
    enum = (getattr(cudart, "cudaStreamUpdateCaptureDependenciesFlags", None)
            or getattr(cudart, "cudaStreamUpdateCaptureFlags"))
    flags = enum.cudaStreamSetCaptureDependencies
    try:
        (err,) = cudart.cudaStreamUpdateCaptureDependencies(s, [node], None, 1, flags)
    except TypeError:
        (err,) = cudart.cudaStreamUpdateCaptureDependencies(s, [node], 1, flags)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamUpdateCaptureDependencies: {err}")


def _node_types(graph_ptr: int) -> dict:
    from cuda.bindings import runtime as cudart
    g = cudart.cudaGraph_t(init_value=int(graph_ptr))
    err, _, n = cudart.cudaGraphGetNodes(g, 0)
    if err != cudart.cudaError_t.cudaSuccess:
        return {"error": str(err)}
    err, nodes, n = cudart.cudaGraphGetNodes(g, n)
    hist: dict = {}
    for nd in nodes[:n]:
        e, t = cudart.cudaGraphNodeGetType(nd)
        key = str(t).split(".")[-1].replace("cudaGraphNodeType", "")
        hist[key] = hist.get(key, 0) + 1
    return hist


def build(runner) -> None:
    """After GPUModelRunner.capture_model: one parent graph per target-graph set."""
    import torch
    from glm_draft_trunc import _W
    if _W.policy is None:
        raise RuntimeError("glm_draft_trunc policy not loaded")
    mgr = runner.cudagraph_manager
    sets = mgr.__dict__.get("_glm_ab_sets") or [mgr.graphs]
    descs = {}
    for M in MS:
        d = mgr.dispatch(1, M, M, 0, max_query_len=M)
        if d.cg_mode.name != "FULL" or d.num_tokens != M or any(d not in s for s in sets):
            raise RuntimeError(f"no FULL 1-request graph for M={M} (got {d}); needs env.trunc SPEC_TABLE/CAPTURE_SIZES")
        descs[M] = d
    for M in MS:
        if M not in _D.plan_info_cap:
            raise RuntimeError(f"no capture-time MLA plan_info recorded for M={M}")
    T = _targets(runner)
    S = _stage_alloc(runner, T)
    ext = _measure_extents(T, S)
    for M, slot in S.items():                 # the staged plan_info must equal the capture-time one
        info = _plan_into(T, slot, M, torch.full((M,), min(3000, T.state.topk_width), dtype=torch.int32))
        slot.plan_info = info
        if info != _D.plan_info_cap[M]:
            raise RuntimeError(f"plan_info mismatch M={M}: staged {info} vs capture {_D.plan_info_cap[M]}")
    torch.cuda.synchronize()
    _D.targets, _D.fused = T, None
    if str(os.environ.get("GLM_DEVSELECT_FUSED", "1")).strip().lower() not in _OFF:
        try:
            _D.fused_try = _fused_prepare(T, S)
            with torch.inference_mode():          # the runner's buffers are inference tensors (in-place writes)
                ok, detail = _fused_selfcheck(runner, T, S)
            if ok:
                _D.fused = _D.fused_try
                _log(f"fused fixups ON (one Triton node + DeepGEMM metadata per body): self-check PASS ({detail})")
            else:
                _log(f"DEVSELECT_FUSED_SELFCHECK_FAIL: {detail}; keeping the legacy fixups")
        except Exception as exc:  # noqa: BLE001
            _log(f"DEVSELECT_FUSED_SELFCHECK_FAIL: {type(exc).__name__}: {exc}; keeping the legacy fixups\n"
                 f"{traceback.format_exc()}")
            _D.fused = None
    else:
        _log("fused fixups OFF (GLM_DEVSELECT_FUSED=0): legacy torch fixups")
    sh = _dev_tensors(runner, _W.policy)
    types0 = _node_types(sets[0][descs[K]].raw_cuda_graph())
    _log(f"target graph M=8 node types {types0}; int-ws extents {ext}; GDN qsl buffers {len(T.gdn)}; "
         f"indexer builders {len(T.indexers)}; "
         f"slot groups {T.slots.shape[0]}; ctx groups {T.ctx.shape[0]}")
    with torch.inference_mode():
        # Every kernel the capture launches must be JIT-compiled / specialised BEFORE the capture: a first Triton launch
        # inside a capture loads a module there and invalidates it (prototype run 0012). The fixups are torch ops, run
        # once per M on the boot-time dummy buffers (every real step rewrites them in its own prep).
        for M in MS[:-1]:
            _fix_ops(T, S, M)
        torch.cuda.synchronize()
        for i, graphs in enumerate(sets):
            st = _set_tensors(runner, sh)
            _select(sh, st)                       # the exact tensors captured below (valid = 0: lam unchanged)
            torch.cuda.synchronize()
            for k in ("ring", "cnt", "hist", "pred"):
                st[k].zero_()
            torch.cuda.synchronize()
            parent = torch.cuda.CUDAGraph()
            t0 = time.perf_counter()
            with torch.cuda.graph(parent, pool=mgr.pool):
                _select(sh, st)
                for M in MS:
                    parent.begin_capture_to_if_node(st["pred"][M])
                    if M < K:
                        _fix_ops(T, S, M)
                    _add_child_to_capture(graphs[descs[M]].raw_cuda_graph())
                    parent.end_capture_to_conditional_node()
            _D.sets[id(graphs)] = dict(parent=parent, st=st, set_index=i, launched=0,
                                       done=torch.cuda.Event(),
                                       lam_pin=torch.zeros(1, dtype=torch.float64, pin_memory=True),
                                       lam_ev=torch.cuda.Event(), dirty=False)
            _log(f"set {i}: parent built + instantiated in {(time.perf_counter() - t0) * 1e3:.0f} ms "
                 f"(7 IF bodies, child graphs of M={MS})")
    _D.shared, _D.stage, _D.targets, _D.runner = sh, S, T, runner
    _D.side = torch.cuda.Stream(device=runner.device)
    if str(os.environ.get("GLM_DEVSELECT_PROBE", "0")) not in _OFF:
        with torch.inference_mode():
            _probe(runner, sets, descs, T, S)
    _D.built = True
    _log(f"armed: {len(sets)} parent(s); force {int(sh['force'])}; debug steps "
         f"{_int('GLM_DEVSELECT_DEBUG_STEPS', 0)} every {_int('GLM_DEVSELECT_DEBUG_EVERY', 0)}; rankcheck "
         f"{_int('GLM_DEVSELECT_RANKCHECK_EVERY', 0)}")



def _audit_graph(graph_ptr: int) -> str:
    """Memcpy node memory kinds and kernel-node launch attributes of one captured graph (probe mode)."""
    from cuda.bindings import runtime as cudart
    g = cudart.cudaGraph_t(init_value=int(graph_ptr))
    err, _, n = cudart.cudaGraphGetNodes(g, 0)
    err, nodes, n = cudart.cudaGraphGetNodes(g, n)
    mem, clus, prog, other = [], 0, 0, {}

    def ptype(ptr):
        try:
            e, a = cudart.cudaPointerGetAttributes(int(ptr))
            return str(a.type).split(".")[-1].replace("cudaMemoryType", "") if e == cudart.cudaError_t.cudaSuccess \
                else f"err{int(e)}"
        except Exception as exc:  # noqa: BLE001
            return f"?{type(exc).__name__}"

    for nd in nodes[:n]:
        e, t = cudart.cudaGraphNodeGetType(nd)
        ts = str(t)
        if "Memcpy" in ts:
            try:
                e, prm = cudart.cudaGraphMemcpyNodeGetParams(nd)
                w, h, d = prm.extent.width, prm.extent.height, prm.extent.depth
                mem.append(f"{str(prm.kind).split('.')[-1]}:{ptype(prm.srcPtr.ptr)}->{ptype(prm.dstPtr.ptr)}:{w}x{h}x{d}")
            except Exception as exc:  # noqa: BLE001
                mem.append(f"?{type(exc).__name__}:{str(exc)[:60]}")
        elif "Kernel" in ts:
            try:
                e, v = cudart.cudaGraphKernelNodeGetAttribute(
                    nd, cudart.cudaLaunchAttributeID.cudaLaunchAttributeClusterDimension)
                if e == cudart.cudaError_t.cudaSuccess and (v.clusterDim.x * v.clusterDim.y * v.clusterDim.z) > 1:
                    clus += 1
            except Exception:  # noqa: BLE001
                pass
            try:
                e, v = cudart.cudaGraphKernelNodeGetAttribute(
                    nd, cudart.cudaLaunchAttributeID.cudaLaunchAttributeProgrammaticStreamSerialization)
                if e == cudart.cudaError_t.cudaSuccess and int(v.programmaticStreamSerializationAllowed):
                    prog += 1
            except Exception:  # noqa: BLE001
                pass
        else:
            other[ts] = other.get(ts, 0) + 1
    return f"memcpy {mem}; cluster kernels {clus}; programmatic kernels {prog}; other {other}"


def _probe(runner, sets, descs, T, S) -> None:
    """GLM_DEVSELECT_PROBE=1: bisect the parent-graph launch at boot (every rank does the same, so the collectives in
    the replayed target graphs match). One 'PROBE <x> start/ok' line per step; a crash leaves the last 'start'."""
    import torch
    from cuda.bindings import runtime as cudart
    g2 = sets[0][descs[2]]

    def step(name, fn):
        _log(f"PROBE {name} start")
        fn()
        torch.cuda.synchronize()
        _log(f"PROBE {name} ok")

    _log(f"PROBE audit M=2: {_audit_graph(g2.raw_cuda_graph())}")
    step("P0 plain replay G2", g2.replay)

    def p1():
        err, g = cudart.cudaGraphCreate(0)
        err, nd = cudart.cudaGraphAddChildGraphNode(g, [], 0, cudart.cudaGraph_t(init_value=int(g2.raw_cuda_graph())))
        assert err == cudart.cudaError_t.cudaSuccess, err
        err, ex = cudart.cudaGraphInstantiate(g, 0)
        assert err == cudart.cudaError_t.cudaSuccess, err
        s = cudart.cudaStream_t(init_value=int(torch.cuda.current_stream().cuda_stream))
        err, = cudart.cudaGraphLaunch(ex, s)
        assert err == cudart.cudaError_t.cudaSuccess, err
    step("P1 child clone (no conditional)", p1)

    pred = torch.ones((), dtype=torch.bool, device=runner.device)
    par = torch.cuda.CUDAGraph()
    with torch.cuda.graph(par, pool=runner.cudagraph_manager.pool):
        par.begin_capture_to_if_node(pred)
        _add_child_to_capture(g2.raw_cuda_graph())
        par.end_capture_to_conditional_node()
    step("P2 IF body = child clone only (true)", par.replay)
    ent = _D.sets[id(sets[0])]
    _D.shared["force"].fill_(1)
    step("P3 real parent, forced L=1 (body M=2 with fixups)", ent["parent"].replay)
    _D.shared["force"].fill_(7)
    step("P4 real parent, forced L=7 (body M=8)", ent["parent"].replay)
    if len(sets) > 1:
        ent1 = _D.sets[id(sets[1])]
        _D.shared["force"].fill_(3)
        step("P5 set-1 parent, forced L=3 (body M=4)", ent1["parent"].replay)
    # P6: the live per-step staging (8-row plan, then 6 per-M plans on the side stream), then the parent
    st = T.state
    st.plan(K, torch.full((K,), min(1500, st.topk_width), dtype=torch.int32))
    torch.cuda.synchronize()
    _D.cur = ent
    reg_t = _registry(runner) + [("x.slots", T.slots), ("x.ctx", T.ctx), ("x.cu", T.cu)]
    snap_t = _snap(reg_t)                         # the P6 start image (8-row MLA plan), reused by the timing probe

    def p6():
        ok = _stage_step()
        assert ok, f"staging declined: {_D.fallbacks}"
    step("P6a per-step staging (side stream)", p6)
    _D.shared["force"].fill_(4)
    step("P6b parent after staging, forced L=4 (body M=5)", ent["parent"].replay)
    try:
        _probe_timing(runner, sets, descs, T, S, ent, reg_t, snap_t)
    except Exception as exc:  # noqa: BLE001  (deterministic on every rank: same code, same shapes)
        _log(f"PROBE TIMING error {exc!r}\n{traceback.format_exc()}")
    _D.shared["force"].fill_(_int("GLM_DEVSELECT_FORCE", -1))
    for e in _D.sets.values():
        for k in ("ring", "cnt", "hist"):
            e["st"][k].zero_()
    _log("PROBE all ok")

def _probe_timing(runner, sets, descs, T, S, ent, reg, A) -> None:
    """P7/P8 (GLM_DEVSELECT_PROBE=1): where does the device-select step lose time on the REAL graphs?
    Per M, median over GLM_DEVSELECT_PROBE_REPS replays (CUDA events, synced; identical sequence on every rank, so
    the collectives inside the target graphs pair up), each replay starting from the same 8-row buffer image:
      plain   G_M replayed on its own (after eager fixups for M, i.e. the M-row state 2209 would replay)
      parent  the parent graph forced to L = M - 1 (selector + IF chain + fixups + child body)
      fixF    a graph holding only the current fixups (fused if armed)   fixL  the same with the legacy torch fixups
      sel     a graph holding only the selector
    parent - plain - fix - sel = the IF-chain + child-body execution overhead on the real graph."""
    import torch
    reps = max(3, _int("GLM_DEVSELECT_PROBE_REPS", 12))
    torch.cuda.synchronize()

    def timeit(fn, pre=None):
        ts = []
        for i in range(reps + 2):
            _restore(reg, A)
            if pre is not None:
                pre()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            if i >= 2:
                ts.append(a.elapsed_time(b))
        return _med(ts)

    def graph_of(fn):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):                  # private pool: these graphs never run after the probe
            fn()
        return g

    st = ent["st"]
    gsel = graph_of(lambda: _select(_D.shared, st))
    t_sel = timeit(gsel.replay)
    fused = _D.fused
    rows = []
    for M in MS:
        plain = sets[0][descs[M]]
        t_plain = timeit(plain.replay, pre=(lambda M=M: _fix_ops(T, S, M)) if M < K else None)
        _D.shared["force"].fill_(M - 1)
        t_par = timeit(ent["parent"].replay)
        t_fixf = t_fixl = 0.0
        if M < K:
            gf = graph_of(lambda M=M: _fix_ops(T, S, M))
            t_fixf = timeit(gf.replay)
            _D.fused = None
            try:
                gl = graph_of(lambda M=M: _fix_ops_legacy(T, S, M))
            finally:
                _D.fused = fused
            t_fixl = timeit(gl.replay)
        rows.append((M, t_plain, t_par, t_fixf, t_fixl))
        _log(f"PROBE TIMING M={M}: plain {t_plain:.4f} parent {t_par:.4f} (+{t_par - t_plain:.4f}) fix "
             f"{'fused' if fused is not None else 'legacy'} {t_fixf:.4f} fix legacy {t_fixl:.4f} sel {t_sel:.4f} "
             f"-> IF/body overhead {t_par - t_plain - t_fixf - t_sel:.4f} ms (reps {reps})")
    _restore(reg, A)
    for k in ("ring", "cnt", "hist"):
        st[k].zero_()
    torch.cuda.synchronize()
    d = [r[2] - r[1] for r in rows]
    _log(f"PROBE TIMING summary: parent - plain mean {sum(d) / len(d):.4f} ms over M={MS}; fixups fused vs legacy "
         f"mean {sum(r[3] for r in rows[:-1]) / 6:.4f} vs {sum(r[4] for r in rows[:-1]) / 6:.4f} ms")


# ================================================================================= per step
def note_propose(conf, n: int) -> None:
    """From glm_draft_trunc's propose hook: the broadcast per-depth confidences [n, 7] (device)."""
    if _TL.cur is not None and _tl_on():
        _tl_draft_end()
    if not _D.built:
        return
    sh = _D.shared
    if n == 1:
        sh["conf"].copy_(conf[0])
        sh["valid"].fill_(1)
    else:
        sh["valid"].fill_(0)


def eligible(runner, scheduler_output, pend) -> bool:
    """Pure host (identical on every rank): c1, 7 scheduled drafts, the request drafted last step, no structured
    output, no logprobs, a parent exists for the current graph set."""
    if not _D.built or _D.failed or not active():
        return False
    ns = scheduler_output.num_scheduled_tokens
    if len(ns) != 1 or pend is None or pend[1] != 1:
        return False
    rid = next(iter(ns))
    if tuple(pend[2]) != (rid,) or getattr(scheduler_output, "has_structured_output_requests", False):
        return False
    spec = scheduler_output.scheduled_spec_decode_tokens or {}
    if len(spec.get(rid, ())) != KMAX or ns[rid] != K:
        return False
    try:
        idx = runner.req_states.req_id_to_index[rid]
        if int(runner.sampler.sampling_states.num_logprobs[idx]) != -1:
            return False
    except Exception:  # noqa: BLE001
        return False
    return id(runner.cudagraph_manager.graphs) in _D.sets


def begin_step(runner, scheduler_output, go: bool) -> None:
    _D.exec_calls += 1
    _D.go = bool(go)
    _D.so = scheduler_output if go and _debug_due() else None
    if _D.so is not None:
        _D.dbg_runs += 1
    every = _int("GLM_DEVSELECT_RANKCHECK_EVERY", 0)
    if _D.built and every and _D.exec_calls % every == 0:
        _rank_check()


def _debug_due() -> bool:
    n = _D.steps + 1
    first, every = _int("GLM_DEVSELECT_DEBUG_STEPS", 0), _int("GLM_DEVSELECT_DEBUG_EVERY", 0)
    cap = _int("GLM_DEVSELECT_DEBUG_MAX", 0)
    if cap and _D.dbg_runs >= cap:
        return False
    return n <= first or (every > 0 and n % every == 0)


def after_prepare_inputs(ib):
    """Persistent cu_num_logits (a captured fixup cannot address the per-step allocation)."""
    if not _D.go or _D.in_ref:
        return ib
    n = int(ib.num_reqs)
    cu = _D.targets.cu
    cu[: n + 1].copy_(ib.cu_num_logits[: n + 1])
    ib.cu_num_logits = cu[: n + 1]
    return ib


def _stage_step() -> bool:
    """Per-M MLA plans for this step on the side stream (host work during the draft)."""
    import torch
    T, S = _D.targets, _D.stage
    st = T.state
    qo = getattr(st, "_qo_cpu", None)
    lens = getattr(st, "_lens_cpu", None)
    if qo is None or lens is None or int(qo[K]) != K or int(qo[K - 1]) != K - 1 or int(qo[K + 1]) != K:
        _fallback("last MLA plan was not the 8-row plan")
        return False
    lens8 = lens[:K].clone()
    _D.stage_lens = lens8.tolist()
    side, main = _D.side, torch.cuda.current_stream()
    ent = _D.cur
    side.wait_event(ent["done"])                  # the previous parent of this set no longer reads the slots
    with torch.cuda.stream(side):
        for M, slot in S.items():
            if slot.ev is not None:
                slot.ev.synchronize()             # host rewrite of the pinned slot after its last upload
            info = _plan_into(T, slot, M, lens8)
            if info != slot.plan_info:
                _fallback(f"plan_info changed for M={M}")
                return False
            slot.ev = torch.cuda.Event()
            slot.ev.record(side)
    ev = torch.cuda.Event()
    ev.record(side)
    main.wait_event(ev)
    return True


def run_parent(mgr, desc) -> bool:
    """From the CudaGraphManager.run_fullgraph hook. True = the parent ran (the caller skips the 8-row replay)."""
    if not _D.go:
        return False
    _D.go = False
    if type(mgr).__name__ != "ModelCudaGraphManager" or desc.cg_mode.name != "FULL" or desc.num_tokens != K \
            or desc.num_reqs != 1:
        _fallback(f"dispatch {desc}")
        return False
    ent = _D.sets.get(id(mgr.graphs))
    if ent is None:
        _fallback("no parent for this graph set")
        return False
    _D.cur = ent
    live = _D.steps < _int("GLM_DEVSELECT_LIVELOG", 0)
    if live:
        _log(f"LIVE step {_D.steps + 1}: stage start (set {ent['set_index']})")
    t0 = time.perf_counter()
    try:
        ok = _stage_step()
    except Exception as exc:  # noqa: BLE001
        _fallback(f"staging error {exc!r}")
        ok = False
    if not ok:
        return False
    t1 = time.perf_counter()
    dbg = _D.so is not None
    if dbg:
        _debug_g1(mgr, ent)
    t2 = time.perf_counter()
    if live:
        _log(f"LIVE step {_D.steps + 1}: stage ok{' + G1 done' if dbg else ''}; replay start")
    ent["parent"].replay()
    ent["done"].record()
    _lam_publish(ent)
    t3 = time.perf_counter()
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):   # the parent replays the runtime set's target graphs
        try:
            cnt = ab._state["replays"].setdefault("target", [0] * (ab.N + 1))
            cnt[int(ab.current())] += 1
        except Exception:  # noqa: BLE001
            pass
    if live:
        import torch
        torch.cuda.synchronize()
        _log(f"LIVE step {_D.steps + 1}: replay ok (synced), L {int(ent['st']['ring'][(int(ent['st']['cnt']) - 1) % RING])}")
    _D.steps += 1
    ent["launched"] += 1
    if not dbg:
        _D.t_stage.append((t1 - t0) * 1e3)
        _D.t_launch.append((t3 - t2) * 1e3)
    else:
        _debug_after(ent)
    every = _int("GLM_DEVSELECT_LOG_EVERY", 2000)
    if every and _D.steps % every == 0:
        _status()
    return True


# ================================================================================= one lam per (set = variant)
# Parity fix (devselect-parity, 2026-09-29): the device lam of a graph set and the host Policy.lam[1] of the same
# glm_ab variant are ONE state. Device steps publish their lam (8 B D2H into pinned memory, stream-ordered after the
# parent); the next host c1 decision adopts it (its draft event is later on the same stream, so the copy is done);
# every host c1 decision writes its updated lam back to the device (fill_, ordered before the next parent).
# Both sides run the identical fp64 update, so the L sequence equals host trunc v3 driven by one shared lam.
_LSYNC = {"pull": 0, "push": 0}      # host c1 decisions that adopted / wrote back the device lam


def _lam_publish(ent) -> None:
    ent["lam_pin"].copy_(ent["st"]["lam"].view(1), non_blocking=True)
    ent["lam_ev"].record()
    ent["dirty"] = True


def cur_set():
    r = _D.runner
    if not _D.built or r is None:
        return None
    return _D.sets.get(id(r.cudagraph_manager.graphs))


def lam_pull(pol) -> None:
    """Before a host c1 choose: adopt the current set's device lam if its parent ran since the last sync."""
    ent = cur_set()
    if ent is not None and ent.get("dirty"):
        ent["lam_ev"].synchronize()
        pol.lam[1] = float(ent["lam_pin"][0])
        ent["dirty"] = False
        _LSYNC["pull"] += 1


def lam_push(pol) -> None:
    """After a host c1 choose: the current set's device lam takes the host value (before the next parent)."""
    ent = cur_set()
    if ent is not None and 1 in pol.lam:
        ent["st"]["lam"].fill_(pol.lam[1])
        _LSYNC["push"] += 1


def _med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


def _status() -> None:
    try:
        from vllm.distributed.parallel_state import get_tp_group
        if get_tp_group().rank_in_group != 0:
            _D.t_stage, _D.t_launch = [], []
            return
    except Exception:  # noqa: BLE001
        pass
    parts = []
    for ent in _D.sets.values():
        h = ent["st"]["hist"].tolist()
        parts.append(f"set{ent['set_index']} launched {ent['launched']} L hist {h[1:K]} lam "
                     f"{float(ent['st']['lam']):.4f}")
    _log(f"steps {_D.steps}: {'; '.join(parts)}; host stage ms median {_med(_D.t_stage):.3f} "
         f"parent launch ms median {_med(_D.t_launch):.3f}; fallbacks {_D.fallbacks}; lam sync pull {_LSYNC['pull']} "
         f"push {_LSYNC['push']}")
    _D.t_stage, _D.t_launch = [], []


# ================================================================================= gates
def _registry(runner):
    """Small CUDA tensors the prep writes (large ones: a leading slice). Excludes the drafter (its PAD rows are
    patched by design) and KV caches (too large, never touched by prep)."""
    import torch
    roots = [("input_buffers", runner.input_buffers), ("block_tables", runner.block_tables),
             ("model_state", runner.model_state), ("sm90", _D.targets.state), ("wrapper", _D.targets.wrapper)]
    for gi, groups in enumerate(runner.attn_groups):
        for j, g in enumerate(groups):
            for k, b in enumerate(getattr(g, "metadata_builders", []) or []):
                roots.append((f"builder{gi}.{j}.{k}:{type(b).__name__}", b))
    out, seen = [], set()

    def add(name, t):
        if not isinstance(t, torch.Tensor) or not t.is_cuda or t.numel() == 0:
            return
        key = (t.data_ptr(), t.numel(), t.dtype)
        if key in seen:
            return
        seen.add(key)
        if name.endswith("_int_workspace_buffer"):
            out.append((name, t))
        elif t.numel() * t.element_size() <= (2 << 20):
            out.append((name, t))
        elif t.dim() >= 2 and t.shape[0] > 16:
            out.append((name + "[:16]", t[:16]))
        else:
            out.append((name + "[:65536]", t.reshape(-1)[:65536]))

    for rname, obj in roots:
        for a, v in list(vars(obj).items()):
            if isinstance(v, (list, tuple)):
                for i, x in enumerate(v):
                    add(f"{rname}.{a}[{i}]", x)
            elif isinstance(v, dict):
                for i, x in v.items():
                    add(f"{rname}.{a}[{i}]", x)
            else:
                add(f"{rname}.{a}", v)
    return out


def _snap(reg):
    return [t.clone() for _, t in reg]


def _restore(reg, snap):
    for (_, t), s in zip(reg, snap):
        t.copy_(s)


def mla_valid_equal(buf_a, buf_b, plan_info):
    """Compare two FlashInfer MLA int-workspace images over their VALID region only (scheduler.cuh MLAPlan, 0.6.18):
    work arrays [W] (W = work_indptr[num_clusters]), merge arrays [num_sm], work_indptr [num_clusters + 1].
    Bytes beyond these are stale leftovers of earlier plans (the planner never clears them; the kernel never reads
    them). buf_*: 1-D uint8 CPU tensors. Returns (equal, W, first differing field or '')."""
    import torch
    pi = [int(x) for x in plan_info]
    nbx, nby = pi[0], pi[1]
    num_sm, ncl = nbx * nby, nby
    (q_ind, kv_ind, part_ind, mps, mpe, mpps, mppe, mstride, q_len, kv_len, q_st, kv_st, kv_end, work_ind) = pi[2:16]

    def ints(buf, off, n):
        return buf[off: off + 4 * n].view(torch.int32)

    wa, wb = ints(buf_a, work_ind, ncl + 1), ints(buf_b, work_ind, ncl + 1)
    if not torch.equal(wa, wb):
        return False, int(wa[-1]), "work_indptr"
    W = int(wa[-1])
    for name, off in (("q_indptr", q_ind), ("kv_indptr", kv_ind), ("partial_indptr", part_ind), ("q_len", q_len),
                      ("kv_len", kv_len), ("q_start", q_st), ("kv_start", kv_st), ("kv_end", kv_end)):
        if not torch.equal(ints(buf_a, off, W), ints(buf_b, off, W)):
            return False, W, name
    for name, off in (("merge_packed_start", mps), ("merge_packed_end", mpe), ("merge_partial_start", mpps),
                      ("merge_partial_end", mppe), ("merge_partial_stride", mstride)):
        if not torch.equal(ints(buf_a, off, num_sm), ints(buf_b, off, num_sm)):
            return False, W, name
    return True, W, ""


def _debug_g1(mgr, ent) -> None:
    """Gate G1: (8-row prep + eager fixups for the host-mirrored L) vs (a real L-row prep, 2209's path)."""
    import copy
    import torch
    from glm_draft_trunc import _W
    runner = _D.runner
    try:
        torch.cuda.synchronize()
        sh, st = _D.shared, ent["st"]
        pol = copy.deepcopy(_W.policy)
        pol.lam = {1: float(st["lam"])}
        conf = sh["conf"].tolist()
        force = int(sh["force"])
        L = force if force >= 1 else (pol.choose([conf], KMAX) if int(sh["valid"]) else KMAX)
        _D.dbg_L = L
        M = L + 1
        reg = _registry(runner)
        A = _snap(reg)
        if M < K:
            _fix_ops(_D.targets, _D.stage, M)
        torch.cuda.synchronize()
        B = _snap(reg)
        _restore(reg, A)
        if M < K:
            so = copy.deepcopy(_D.so)
            rid = next(iter(so.num_scheduled_tokens))
            d = len(so.scheduled_spec_decode_tokens[rid]) - L
            so.scheduled_spec_decode_tokens[rid] = so.scheduled_spec_decode_tokens[rid][:L]
            so.num_scheduled_tokens[rid] -= d
            so.total_num_scheduled_tokens -= d
            _D.in_ref = True
            try:
                brs, _ = runner.gather_batch_req_state(so, False)
                desc_m = mgr.dispatch(1, M, M, 0, max_query_len=M)
                ib = runner.prepare_inputs(so, brs, desc_m)
                bt, sm = runner.prepare_attn(ib)
                runner.model_state.prepare_attn(ib, desc_m.cg_mode, bt, sm, runner.attn_groups,
                                                runner.kv_cache_config)
            finally:
                _D.in_ref = False
            torch.cuda.synchronize()
            C = _snap(reg)
            _restore(reg, A)
            torch.cuda.synchronize()
            diffs = []
            mla_note = ""
            for (name, _), b, c in zip(reg, B, C):
                if name.endswith("_int_workspace_buffer"):
                    rp = getattr(_D, "ref_plan", None)
                    info = rp[2] if rp else _D.stage[M].plan_info
                    ok, W, field = mla_valid_equal(b.cpu(), c.cpu(), info)
                    raw = int((b != c).sum())
                    mla_note = (f"MLA plan valid region ({W} works) {'IDENTICAL' if ok else 'DIFFERS at ' + field}"
                                f" (raw stale-tail bytes differing {raw})")
                    if not ok:
                        diffs.append(f"{name} VALID-REGION DIFF at {field} ({W} works)")
                    continue
                if b.shape != c.shape:
                    diffs.append(f"{name} shape {tuple(b.shape)} vs {tuple(c.shape)}")
                    continue
                ne = (b.reshape(-1).view(torch.uint8) != c.reshape(-1).view(torch.uint8)) if b.dtype == torch.bool \
                    else (b.reshape(-1) != c.reshape(-1))
                if b.is_floating_point():
                    ne = ne & ~(torch.isnan(b.reshape(-1)) & torch.isnan(c.reshape(-1)))
                cnt = int(ne.sum())
                if cnt:
                    idx = torch.nonzero(ne).reshape(-1)
                    first = idx[:6].tolist()
                    diffs.append(f"{name} {cnt}/{ne.numel()} first idx {first} "
                                 f"patched {b.reshape(-1)[idx[:3]].tolist()} ref {c.reshape(-1)[idx[:3]].tolist()}")
            _D.dbg_done += 1
            rp = getattr(_D, "ref_plan", None)
            slot = _D.stage[M]
            mla = (f"MLA: stage lens {getattr(_D, 'stage_lens', [])[:M]} ref lens {rp[1] if rp else None} "
                   f"(ref rows {rp[0] if rp else None}); plan_info staged==ref "
                   f"{(slot.plan_info == rp[2]) if rp else None}")
            _log(f"DEVSELECT_G1 step {_D.steps + 1} L {L}: {len(diffs)} differing buffer(s) of {len(reg)}; {mla}; "
                 f"{mla_note}"
                 + ("" if not diffs else ": " + " | ".join(diffs[:40])))
        else:
            _log(f"DEVSELECT_G1 step {_D.steps + 1} L {L}: full shape, nothing patched")
    except Exception as exc:  # noqa: BLE001
        _log(f"DEVSELECT_G1 error {exc!r}\n{traceback.format_exc()}")


def _debug_after(ent) -> None:
    import torch
    torch.cuda.synchronize()
    st = ent["st"]
    c = int(st["cnt"])
    got = int(st["ring"][(c - 1) % RING])
    want = getattr(_D, "dbg_L", None)
    if want is not None and got != want:
        _log(f"DEVSELECT_MISMATCH device L {got} vs host-mirrored L {want} at step {_D.steps}")


def _rank_check() -> None:
    import torch
    import torch.distributed as dist
    try:
        from vllm.distributed.parallel_state import get_tp_group
        tp = get_tp_group()
        if tp.world_size <= 1:
            return
        torch.cuda.synchronize()
        mine = [(ent["set_index"], int(ent["st"]["cnt"]), ent["st"]["ring"].tolist(), ent["st"]["hist"].tolist())
                for ent in sorted(_D.sets.values(), key=lambda e: e["set_index"])]
        allv = [None] * tp.world_size
        dist.all_gather_object(allv, mine, group=tp.cpu_group)
        ok = all(x == allv[0] for x in allv)
        if not ok:
            _log(f"DEVSELECT_RANK_MISMATCH at runner call {_D.exec_calls}: " + " || ".join(str(x)[:300] for x in allv))
        elif tp.rank_in_group == 0:
            _log(f"rank check ok at runner call {_D.exec_calls} (device steps {_D.steps})")
    except Exception as exc:  # noqa: BLE001
        _log(f"rank check error {exc!r}")


# ================================================================================= live GPU timeline (rank 0)
class _TL:
    """GLM_DEVSELECT_TIMELINE=1 (default): per c1 verify step, GPU events at target start (T0), target end (T1) and
    draft end (D1, recorded in note_propose right after the draft + score broadcast were enqueued), both arms alike.
    Cycle = (T0, T1, D1, next T0): target = T0->T1 (parent or G_M), down = T1->D1 (sampler / cert head / rejection /
    post_update / propose incl. the draft), gap = D1->next T0 (prep + host latency before the next target).
    Split by (variant, parent ran). Only rank 0 records; nothing is synced (events are read once complete)."""
    on = None
    cur = None                    # [variant, parent_ran, T0, T1, D1]
    pend: list = []
    stats: dict = {}
    n = 0


def _tl_on() -> bool:
    if _TL.on is None:
        try:
            from vllm.distributed.parallel_state import get_tp_group
            r0 = get_tp_group().rank_in_group == 0
        except Exception:  # noqa: BLE001
            r0 = True
        _TL.on = r0 and str(os.environ.get("GLM_DEVSELECT_TIMELINE", "1")).strip().lower() not in _OFF
    return _TL.on


def _tl_event():
    import torch
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    return ev


def _tl_target_start_impl(parent: bool) -> None:
    ev = _tl_event()
    c = _TL.cur
    if c is not None and c[3] is not None and c[4] is not None:
        _TL.pend.append((c[0], c[1], c[2], c[3], c[4], ev))
    ab = sys.modules.get("glm_ab")
    v = int(ab.current()) if ab is not None and getattr(ab, "ACTIVE", False) else 0
    _TL.cur = [v, parent, ev, None, None]
    _tl_drain()


def _tl_target_end_impl(parent: bool) -> None:
    c = _TL.cur
    if c is not None and c[3] is None:
        c[1] = parent
        c[3] = _tl_event()


def _tl_draft_end_impl() -> None:
    c = _TL.cur
    if c is not None and c[3] is not None and c[4] is None:
        c[4] = _tl_event()


def _tl_other() -> None:
    _TL.cur = None                # any other target replay in between: the open cycle is not a c1 step pair


def _tl_drain() -> None:
    keep = []
    for item in _TL.pend:
        v, par, t0, t1, d1, t0n = item
        try:
            if not t0n.query():
                keep.append(item)
                continue
            rec = (t0.elapsed_time(t1), t1.elapsed_time(d1), d1.elapsed_time(t0n))
        except Exception:  # noqa: BLE001
            continue
        if rec[1] > 30.0 or rec[2] > 15.0:          # a prefill / idle / round switch inside the cycle: not a step
            continue
        lst = _TL.stats.setdefault((v, par), [])
        if len(lst) < 50000:
            lst.append(rec)
        _TL.n += 1
    _TL.pend = keep[-64:]
    every = _int("GLM_DEVSELECT_TIMELINE_EVERY", 1000)
    if every and _TL.n >= every:
        _TL.n = 0
        _tl_report()


def _tl_report() -> None:
    parts = []
    for (v, par), lst in sorted(_TL.stats.items()):
        if not lst:
            continue
        cols = list(zip(*lst))
        med = [sorted(c)[len(c) // 2] for c in cols]
        mean = [sum(c) / len(c) for c in cols]
        parts.append(f"v{v}{'P' if par else 'H'} n{len(lst)} target {mean[0]:.3f}/{med[0]:.3f} down {mean[1]:.3f}/"
                     f"{med[1]:.3f} gap {mean[2]:.3f}/{med[2]:.3f} step {sum(mean):.3f}")
    _log("TIMELINE (ms mean/median, cumulative; P = parent, H = host-chosen graph): " + " | ".join(parts))


def _tl_target_start(*a):
    try:
        _tl_target_start_impl(*a)
    except Exception as exc:  # noqa: BLE001  (logging only: never break a step)
        _TL.on = False
        _log(f"timeline disabled after {exc!r}")


def _tl_target_end(*a):
    try:
        _tl_target_end_impl(*a)
    except Exception as exc:  # noqa: BLE001  (logging only: never break a step)
        _TL.on = False
        _log(f"timeline disabled after {exc!r}")


def _tl_draft_end(*a):
    try:
        _tl_draft_end_impl(*a)
    except Exception as exc:  # noqa: BLE001  (logging only: never break a step)
        _TL.on = False
        _log(f"timeline disabled after {exc!r}")


# ================================================================================= hooks
def _wants_template(desc) -> bool:
    """Keep the cudaGraph_t template only for the graphs a parent clones: FULL, one request, uniform M = 2..8."""
    try:
        return (desc is not None and desc.cg_mode.name == "FULL" and desc.num_reqs == 1
                and desc.uniform_token_count == desc.num_tokens and 2 <= desc.num_tokens <= K)
    except Exception:  # noqa: BLE001
        return False


def _patch_cgu(mod) -> None:
    import torch
    base, model = mod.CudaGraphManager, mod.ModelCudaGraphManager
    o_run, o_cap, o_base_cap = base.run_fullgraph, model.capture, base.capture

    def run_fullgraph(self, desc):
        tl = False
        if type(self).__name__ == "ModelCudaGraphManager" and _tl_on():
            try:
                tl = desc.num_reqs == 1 and 2 <= desc.num_tokens <= K and desc.cg_mode.name == "FULL"
            except Exception:  # noqa: BLE001
                tl = False
            if tl:
                _tl_target_start(bool(_D.go))
            else:
                _tl_other()
        if _D.go and run_parent(self, desc):
            if tl:
                _tl_target_end(True)
            return None
        out = o_run(self, desc)
        if tl:
            _tl_target_end(False)
        return out

    class _KeepGraph(torch.cuda.CUDAGraph):
        """torch.cuda.CUDAGraph whose keep_graph is decided by the descriptor being captured (_D.next_desc).
        pybind builds the C++ object in __init__ (0159 boot: overriding __new__ alone left keep_graph false), so
        both __new__ and __init__ receive the flag."""
        def __new__(cls, keep_graph=False):
            return super().__new__(cls, bool(keep_graph) or _wants_template(_D.next_desc))

        def __init__(self, keep_graph=False):
            keep = bool(keep_graph) or _wants_template(_D.next_desc)
            super().__init__(keep)
            self._glm_keep = keep                    # torch 2.13 has no Python-side _keep_graph attribute

    def base_capture(self, create_forward_fn, *a, **kw):
        if not _D.capturing_target:
            return o_base_cap(self, create_forward_fn, *a, **kw)

        def cff(desc, warmup, *aa, **kk):
            _D.next_desc = None if warmup else desc   # the graph object is constructed right after this call
            return create_forward_fn(desc, warmup, *aa, **kk)

        return o_base_cap(self, cff, *a, **kw)

    def capture(self, *a, **kw):
        orig_cls = torch.cuda.CUDAGraph
        torch.cuda.CUDAGraph = _KeepGraph            # clone-able templates (route B), target M = 2..8 only
        _D.capturing_target = True
        _D.next_desc = None
        try:
            out = o_cap(self, *a, **kw)
        finally:
            torch.cuda.CUDAGraph = orig_cls
            _D.capturing_target = False
            _D.next_desc = None
        try:                                         # never fatal: a failure only disables device select
            sets = self.__dict__.get("_glm_ab_sets") or [self.graphs]
            kept = 0
            for s in sets:
                for g in s.values():
                    if getattr(g, "_glm_keep", False):
                        g.instantiate()
                        kept += 1
            _log(f"target graph templates kept + instantiated: {kept} (1-request M=2..{K}, all sets); "
                 f"other target graphs unchanged")
        except Exception as exc:  # noqa: BLE001
            _D.failed = f"post-capture instantiate: {type(exc).__name__}: {exc}"
            _log(f"DEVSELECT_BUILD_FAILED {_D.failed}\n{traceback.format_exc()}")
        return out

    base.run_fullgraph, base.capture, model.capture = run_fullgraph, base_capture, capture
    _log("cudagraph_utils hooked (keep_graph for the 1-request M=2..8 target graphs, run_fullgraph -> parent)")


def _patch_sm90(mod) -> None:
    cls = mod._SM90State
    o_plan = cls.plan

    def plan(self, num_tokens, kv_lens):
        out = o_plan(self, num_tokens, kv_lens)
        if _D.in_ref:
            _D.ref_plan = (int(num_tokens), [int(x) for x in kv_lens[: int(num_tokens)].tolist()],
                           list(self.wrapper._plan_info) if isinstance(self.wrapper._plan_info, (list, tuple))
                           else self.wrapper._plan_info)
        if _D.capturing_target:
            info = self.wrapper._plan_info
            _D.plan_info_cap[int(num_tokens)] = list(info) if isinstance(info, (list, tuple)) else info
        return out

    cls.plan = plan
    _log("SM90 plan hooked (capture-time plan_info per row count)")


def _patch_runner(mod) -> None:
    cls = mod.GPUModelRunner
    o_cap, o_prep = cls.capture_model, cls.prepare_inputs

    def capture_model(self, *a, **kw):
        out = o_cap(self, *a, **kw)
        try:
            build(self)
        except Exception as exc:  # noqa: BLE001
            _D.failed = f"{type(exc).__name__}: {exc}"
            _log(f"DEVSELECT_BUILD_FAILED {_D.failed}\n{traceback.format_exc()}")
        return out

    def prepare_inputs(self, *a, **kw):
        return after_prepare_inputs(o_prep(self, *a, **kw))

    cls.capture_model, cls.prepare_inputs = capture_model, prepare_inputs
    _log("GPUModelRunner.capture_model / prepare_inputs hooked")


TARGETS = {CGU: _patch_cgu, SM90: _patch_sm90, RUNNER: _patch_runner}


def register() -> None:
    import importlib.abc
    import importlib.util
    if not installed():
        return
    fns = dict(TARGETS)
    for name in [n for n in fns if n in sys.modules]:
        fns.pop(name)(sys.modules[name])
    pending = fns

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in pending:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module
            fn = pending.pop(name)

            def exec_module(module, _orig=orig_exec, _fn=fn):
                _orig(module)
                _fn(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
    _log(f"registered ({len(pending)} pending module hooks)")
