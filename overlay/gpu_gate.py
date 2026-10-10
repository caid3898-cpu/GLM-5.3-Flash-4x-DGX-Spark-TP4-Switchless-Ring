#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU gate for the vllm#50843 tile argmax clamp port (port-50843).

RUNS IN THE DEPLOYMENT WINDOW, NOT NOW: must be executed with python3 INSIDE
the deployed container (glm53-stash-boundary-r0 or its replacement) on a node
with the patched files mounted, after warm-up, per ../DEPLOY.md. It is the
deterministic FAIL-unfixed / PASS-fixed gate for the four clamp sites:

  1. vllm/v1/worker/gpu/sample/gumbel.py:_gumbel_sample_kernel        (bs 1024)
  2. .../spec_decode/rejection_sampler_utils.py:_compute_local_logits_stats_kernel (bs 8192)
  3. .../spec_decode/rejection_sampler_utils.py:_resample_kernel      (bs 1024)
  4. overlay/glm_target_argmax.py:_block_argmax_kernel                (bs 8192)

Method: an in-gate UNCLAMPED replica kernel (`_tile_argmax_unclamped`) is the
exact per-tile algorithm the four sites shared pre-fix (masked load
other=-inf, tl.max(..., return_indices=True), stored id = tile_start + idx).
At temperature 0 the gumbel / stats-greedy / bonus-resample paths reduce to
exactly this per-tile argmax over the target logits row, so:

  P1 (fix works)      every adversarial row (all -inf, all NaN, NaN last tile):
                      clamped ids < vocab_size AND clamped == min(unclamped, V-1)
                      elementwise. On a PRE-FIX deployment the clamped kernel is
                      absent and the raw ids ARE the replica ids, so P1 fails
                      whenever the defect is reachable on this build.
  P2 (no-op)          rows where no replica tile id is out of range: deployed
                      output bit-identical to the replica (values AND ids), and
                      final id == torch.argmax for rows with a unique finite max.
  P3 (defect seen)    reports how many unclamped tile ids were out of vocab on
                      adversarial rows (evidence the pre-fix build would fail).
                      NOTE: FAIL-on-unfixed is conditional on defect
                      reachability. If this counter is 0 on a PRE-FIX build,
                      the trigger (OOV tie-break / NaN lane selection) is not
                      reachable with that Triton build — record it as "defect
                      not reachable on this build", do NOT treat it as a broken
                      gate or fabricate a failure. The post-fix invariants (P1
                      ids < V, P2 bit-identity, P4 graph == eager) are the hard
                      assertions either way.
  P4 (graph)          each site re-run under CUDA-graph capture + replay after
                      warm-up (our decode kernels run in FULL_DECODE_ONLY
                      graphs); replay output must be bit-identical to eager.

Exit 0 = gate passed. Any assertion failure = gate FAILED (do not serve until
explained). Usage: python3 gpu_gate.py
"""
import os
import sys

os.environ.setdefault("GLM_TARGET_VOCAB_ARGMAX_LOCAL", "triton")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rsu  # noqa: E402

V = 154880  # GLM-5.3-Flash target vocab, as served (audit geometry)
DEV = "cuda"

failures: list[str] = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(f"{name} {detail}")


def cuda_graph(fn):
    """Warm up, capture, replay (kpool-gate convention: warm-up first)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    return out


@triton.jit
def _tile_argmax_unclamped(logits_ptr, logits_stride, ids_ptr, ids_stride,
                           vals_ptr, vocab, num_blocks, BLOCK_SIZE: tl.constexpr):
    # Exact replica of the pre-fix per-tile argmax shared by all four sites.
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=offs < vocab,
                other=float("-inf")).to(tl.float32)
    v, i = tl.max(x, axis=0, return_indices=True)
    tl.store(ids_ptr + row * ids_stride + blk, blk * BLOCK_SIZE + i)
    tl.store(vals_ptr + row * num_blocks + blk, v)


def replica(logits, block_size):
    """Unclamped per-tile (ids [R, nb] int64, values [R, nb] fp32)."""
    rows, vocab = logits.shape
    nb = triton.cdiv(vocab, block_size)
    ids = torch.empty(rows, nb, dtype=torch.int64, device=logits.device)
    vals = torch.empty(rows, nb, dtype=torch.float32, device=logits.device)
    _tile_argmax_unclamped[(rows, nb)](logits, logits.stride(0), ids, ids.stride(0),
                                       vals, vocab, nb, BLOCK_SIZE=block_size)
    return ids, vals


def final_from_replica(logits, block_size):
    """Deployed reduction applied to the unclamped tiles: argmax over blocks
    (first maximal; torch.argmax treats NaN as maximal), then gather."""
    ids, vals = replica(logits, block_size)
    b = vals.argmax(dim=-1, keepdim=True)
    return ids.gather(-1, b).squeeze(-1)


def rows_for(block_size):
    """[6, V] fp32: 3 healthy rows, 3 adversarial rows."""
    g = torch.Generator(device="cpu").manual_seed(50843)
    rows = []
    # healthy: finite, unique max (final selection unambiguous)
    for _ in range(2):
        r = torch.rand(V, generator=g) * 4.0 - 2.0
        r[torch.randint(0, V, (1,), generator=g)] = 10.0
        rows.append(r)
    # healthy: NaN-free but last tile has no valid elements (all -inf);
    # finite max elsewhere => final id must be unchanged (clamp binds nowhere
    # on the WINNING tile)
    r = torch.rand(V, generator=g) * 4.0 - 2.0
    r[V - block_size:] = float("-inf")
    r[0] = 10.0
    rows.append(r)
    # adversarial: all -inf (ties with the OOV pad lanes)
    rows.append(torch.full((V,), float("-inf")))
    # adversarial: all NaN
    rows.append(torch.full((V,), float("nan")))
    # adversarial: healthy except the last tile's in-vocab lanes are all NaN
    r = torch.rand(V, generator=g) * 4.0 - 2.0
    r[V - block_size:] = float("nan")
    rows.append(r)
    return torch.stack(rows).to(DEV)


def report_defect(ids_unclamped, adversarial_slice):
    oob = (ids_unclamped[adversarial_slice] >= V).sum().item()
    if oob:
        print(f"  info unclamped replica out-of-vocab tile ids on adversarial rows: {oob}"
              f" (this many made the pre-fix kernel return id >= V here; the gate FAILS unclamped)")
    else:
        print("  info unclamped replica out-of-vocab tile ids on adversarial rows: 0"
              " (defect NOT reachable with this Triton build: tie-break/NaN semantics never"
              " select an OOB pad lane; on a pre-fix build this means 'not reachable', record"
              " it as such - the post-fix invariants below are still the hard assertions)")
    return oob


def main():
    assert torch.cuda.is_available(), "gpu_gate must run on the target GPU node"
    torch.cuda.init()
    print(f"gpu_gate port-50843: V={V}, device={torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------------- site 1
    print("site 1: gumbel_sample (temp 0 -> pure argmax), BLOCK 1024")
    bs = 1024
    R = 6
    logits = rows_for(bs)
    adv = slice(3, 6)
    nb = triton.cdiv(V, bs)
    eim = torch.arange(R, dtype=torch.int32, device=DEV)
    temp = torch.zeros(R, dtype=torch.float32, device=DEV)
    seed = torch.zeros(R, dtype=torch.int64, device=DEV)
    pos = torch.zeros(R, dtype=torch.int64, device=DEV)
    out = gumbel_sample(logits, eim, temp, seed, pos, apply_temperature=True)
    rep_ids, rep_vals = replica(logits, bs)
    report_defect(rep_ids, adv)
    check("site1 adversarial ids < V", bool((out[adv] < V).all()),
          f"got {out[adv].tolist()}")
    check("site1 clamped == min(unclamped_final, V-1)",
          bool(torch.equal(out, torch.minimum(final_from_replica(logits, bs),
                                              torch.full_like(out, V - 1)))))
    healthy_finite = slice(0, 2)
    check("site1 healthy unique-max == torch.argmax",
          bool(torch.equal(out[healthy_finite],
                           logits.argmax(dim=-1)[healthy_finite])))
    # graph replay (after warm-up; decode kernels run in FULL decode graphs)
    g_out = cuda_graph(lambda: gumbel_sample(logits, eim, temp, seed, pos,
                                             apply_temperature=True))
    check("site1 graph replay bit-identical to eager", bool(torch.equal(g_out, out)))

    # ---------------------------------------------------------------- site 2
    print("site 2: _compute_local_logits_stats_kernel (greedy), BLOCK 8192")
    bs2 = 8192
    logits2 = rows_for(bs2)
    nb2 = triton.cdiv(V, bs2)
    tls = torch.empty(R, nb2, dtype=torch.float32, device=DEV)
    dlm = torch.empty(R, nb2, dtype=torch.float32, device=DEV)
    dls = torch.empty(R, nb2, dtype=torch.float32, device=DEV)
    one = torch.zeros(1, dtype=torch.float32, device=DEV)

    def drive_stats(tla, tlm):
        rsu._compute_local_logits_stats_kernel[(R, nb2)](
            tla, tla.stride(0), tlm, tlm.stride(0), tls, tls.stride(0),
            dlm, dlm.stride(0), dls, dls.stride(0),
            logits2, logits2.stride(0),
            one, 0, 0,  # draft_logits ptr + strides (HAS_DRAFT_LOGITS=False)
            eim, torch.zeros(R, dtype=torch.int32, device=DEV),
            temp, V, 1,  # num_speculative_steps=1, expanded_local_pos=0
            BLOCK_SIZE=bs2, HAS_DRAFT_LOGITS=False)
        return tla

    tla = torch.empty(R, nb2, dtype=torch.int64, device=DEV)
    tlm = torch.empty(R, nb2, dtype=torch.float32, device=DEV)
    drive_stats(tla, tlm)
    rep2_ids, rep2_vals = replica(logits2, bs2)
    report_defect(rep2_ids, adv)
    fully_in_range = (rep2_ids < V).all(dim=-1)  # rows the clamp provably binds nowhere
    check("site2 adversarial stored ids < V", bool((tla[adv] < V).all()))
    check("site2 clamped == min(unclamped, V-1) elementwise",
          bool(torch.equal(tla, torch.minimum(
              rep2_ids, torch.full_like(rep2_ids, V - 1)))))
    check("site2 NaN-free fully-in-range arrays bit-identical",
          bool(torch.equal(tla[fully_in_range], rep2_ids[fully_in_range]))
          and bool(torch.equal(tlm[fully_in_range], rep2_vals[fully_in_range])))
    tla_g = torch.empty_like(tla)  # second buffer: a real comparison, not self-comparison
    tlm_g = torch.empty_like(tlm)
    check("site2 graph replay bit-identical to eager",
          bool(torch.equal(cuda_graph(lambda: drive_stats(tla_g, tlm_g)), tla)))

    # ---------------------------------------------------------------- site 3
    print("site 3: _resample_kernel (bonus, temp 0 -> raw target argmax), BLOCK 1024")
    logits3 = rows_for(bs)
    rla = torch.empty(R, nb, dtype=torch.int64, device=DEV)
    rlm = torch.empty(R, nb, dtype=torch.float32, device=DEV)
    # cu_num_logits for the single-req call, hoisted OUT of the capture:
    # torch.tensor(<python list>, device=DEV) is an H2D copy from PAGEABLE
    # host memory, forbidden during CUDA-graph capture (fleet-verified
    # attempt-2 abort: "Cannot copy between CPU and CUDA tensors during CUDA
    # graph capture unless the CPU tensor is pinned", attempt 2 2026-09-27).
    # The torch.zeros(..., device=DEV) below are capture-safe fill-kernel
    # allocations and stay in place. Any future device-tensor created from a
    # python list must be hoisted to this scope too.
    CU = torch.tensor([0, 1], dtype=torch.int32, device=DEV)

    def drive_resample_once(lg, out_ids, out_vals):
        # single req: cu=[0,1], rejected_step=0, is_bonus=True -> proceeds at temp 0
        rsu._resample_kernel[(1, nb)](
            out_ids, out_ids.stride(0), out_vals, out_vals.stride(0),
            lg, lg.stride(0),
            one,                       # target_rejected_logsumexp (unused, bonus)
            one, 0, 0,                 # draft_logits ptr + strides (HAS_DRAFT_LOGITS=False)
            one,                       # draft_rejected_logsumexp (unused)
            torch.zeros(1, dtype=torch.int32, device=DEV),   # rejected_step
            CU,                        # cu_num_logits (hoisted, capture-safe)
            torch.zeros(1, dtype=torch.int32, device=DEV),   # expanded_idx_mapping
            torch.zeros(1, dtype=torch.int64, device=DEV),   # draft_sampled
            torch.zeros(1, dtype=torch.float32, device=DEV),  # temp = 0
            torch.zeros(1, dtype=torch.int64, device=DEV),   # seed
            torch.zeros(1, dtype=torch.int64, device=DEV),   # pos
            one,                       # cumulative_log_p (unused, no block verification)
            V, BLOCK_SIZE=bs, HAS_DRAFT_LOGITS=False,
            USE_FP64=False, USE_BLOCK_VERIFICATION=False)

    for i in range(R):
        drive_resample_once(logits3[i:i + 1], rla[i:i + 1], rlm[i:i + 1])
    rep3_ids, rep3_vals = replica(logits3, bs)
    report_defect(rep3_ids, adv)
    fir3 = (rep3_ids < V).all(dim=-1)
    check("site3 adversarial stored ids < V", bool((rla[adv] < V).all()))
    check("site3 clamped == min(unclamped, V-1) elementwise",
          bool(torch.equal(rla, torch.minimum(
              rep3_ids, torch.full_like(rep3_ids, V - 1)))))
    check("site3 NaN-free fully-in-range arrays bit-identical",
          bool(torch.equal(rla[fir3], rep3_ids[fir3]))
          and bool(torch.equal(rlm[fir3], rep3_vals[fir3])))
    rla_g = torch.empty_like(rla)
    rlm_g = torch.empty_like(rlm)

    def resample_all():
        for i in range(R):
            drive_resample_once(logits3[i:i + 1], rla_g[i:i + 1], rlm_g[i:i + 1])
        return rla_g

    check("site3 graph replay bit-identical to eager",
          bool(torch.equal(cuda_graph(resample_all), rla)))

    # ---------------------------------------------------------------- site 4
    print("site 4: overlay glm_target_argmax.local_argmax (triton), BLOCK 8192")
    import importlib.util
    mod = None
    for cand in ("/overlay/overlay/glm_target_argmax.py",
                 os.path.expanduser("~/glm-stash-boundary-20260927/overlay/glm_target_argmax.py")):
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("glm_target_argmax_gate", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            break
    if mod is None:
        check("site4 overlay module found", False, "glm_target_argmax.py not found")
    elif mod.LOCAL_IMPL != "triton":
        # LOCAL_IMPL is read at module import (glm_target_argmax.py:57); an env write now
        # cannot change it. Refuse silently-degraded evidence: the torch fallback
        # (local_argmax_torch) has no clamp and would make site 4 vacuous.
        check("site4 LOCAL_IMPL is triton (Triton kernel actually exercised)", False,
              f"LOCAL_IMPL={mod.LOCAL_IMPL!r}; run the gate with "
              "GLM_TARGET_VOCAB_ARGMAX_LOCAL=triton in the container environment")
    else:
        check("site4 LOCAL_IMPL is triton (Triton kernel actually exercised)", True)
        logits4 = rows_for(bs2).contiguous()
        vals, idxs = mod.local_argmax(logits4)
        rep4_ids, _ = replica(logits4, bs2)
        report_defect(rep4_ids, adv)
        check("site4 adversarial final ids < V", bool((idxs[adv] < V).all()),
              f"got {idxs[adv].tolist()}")
        check("site4 clamped final == min(unclamped final, V-1)",
              bool(torch.equal(idxs, torch.minimum(
                  final_from_replica(logits4, bs2),
                  torch.full_like(idxs, V - 1)))))
        check("site4 healthy unique-max == torch.argmax",
              bool(torch.equal(idxs[healthy_finite],
                               logits4.argmax(dim=-1)[healthy_finite])))
        g_idx = cuda_graph(lambda: mod.local_argmax(logits4)[1])
        check("site4 graph replay bit-identical to eager", bool(torch.equal(g_idx, idxs)))

    # ---------------------------------------------------------------- verdict
    if failures:
        print(f"gpu_gate: FAILED ({len(failures)} check(s))")
        return 1
    print("gpu_gate: PASSED (all sites: adversarial ids < V, "
          "healthy rows bit-identical to the unclamped kernel, "
          "graph replay == eager)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
