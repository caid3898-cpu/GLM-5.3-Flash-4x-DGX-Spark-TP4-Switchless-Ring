#!/usr/bin/env python3
"""GPU test for overlay/glm_gumbel_coupled.py inside the serving image (one GB10, ~1 min). TEST ONLY.

  docker run --rm --gpus all -v REPO:/overlay -v REPO/overlay/gumbel.py:<vllm>/v1/worker/gpu/sample/gumbel.py:ro \
      -e PYTHONPATH=/overlay/overlay IMAGE python3 /overlay/tests/gpu_test_gumbel_coupled.py

1. noise port: the stock gumbel_sample (fp32) on random full-vocab rows == argmax(logits + gumbel_noise_np) of the
   numpy Philox port, except where the top-2 margin is below 1e-4 (last-ulp log differences).
2. coupling identity: target rows whose logits are the 16 candidate scores (-inf elsewhere) at key pos = sample_pos
   - 1; the image's walk (SAMPLE_PROBABILISTIC=True) and ours (tau 1, no draft top-p) must both equal the target's
   gumbel_sample draw at EVERY step of every request (the draft and the target use the same noise).
3. walk parity: ours == the image's walk bit for bit (tau 1, top-p off), for T = 0 (greedy walk) and T > 0; with the
   draft nucleus every pick is inside the nucleus and a greedy row is unchanged.
4. verify kernel: glm_target_argmax.greedy_verify (Triton) == its torch reference, with -1 drafts.
5. cost at M = 4 / 8 rows, T = 1, top-p 0.95 (processed logits given): stock rejection_sample vs coupled
   (gumbel_sample + greedy_verify), median of 50 CUDA-event timings.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "overlay"))
import glm_gumbel_coupled as gc  # noqa: E402
import glm_target_argmax as ta  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p  # noqa: E402
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as df2  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample  # noqa: E402

dev = torch.device("cuda")
FAIL = []
V = 154880


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def test_noise_port():
    g = torch.Generator(device="cpu").manual_seed(1)
    R = 6
    logits = (torch.randn(R, V, generator=g) * 3).to(dev)
    seeds = torch.tensor([7, -3, 2 ** 40 + 1, 123456789, -(2 ** 62), 5], dtype=torch.int64, device=dev)
    temp = torch.ones(R, dtype=torch.float32, device=dev)
    pos = torch.tensor([0, 1, 77, 4096, 262143, 12], dtype=torch.int64, device=dev)
    idx = torch.arange(R, dtype=torch.int32, device=dev)
    got = gumbel_sample(logits, idx, temp, seeds, pos, apply_temperature=False).cpu().numpy()
    ok = amb = 0
    for r in range(R):
        z = logits[r].double().cpu().numpy() + gc.gumbel_noise_np(int(seeds[r]), int(pos[r]), np.arange(V))
        o = np.argsort(z)[::-1]
        if int(o[0]) == int(got[r]):
            ok += 1
        elif z[o[0]] - z[o[1]] < 1e-4:
            amb += 1
    check(ok + amb == R and ok >= R - 1, f"noise port: stock gumbel_sample == numpy Philox argmax on {ok}/{R} rows "
                                         f"({amb} near-ties)")


def build_walk_inputs(R, K, C, seed, same_prev=False, temp=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    scores = torch.randn(R, K, C, C, generator=g) * 2
    if same_prev:
        scores = scores[:, :, :1, :].expand(R, K, C, C).contiguous()
    cands = torch.stack([torch.randperm(V, generator=g)[: K * C].view(K, C) for _ in range(R)])
    base_pos = torch.randint(10, 200000, (R,), generator=g)
    sample_pos = (base_pos[:, None] + 1 + torch.arange(K)[None, :]).reshape(-1)   # draft j sits at B + j
    req_state = torch.arange(R, dtype=torch.int32).repeat_interleave(K)
    temps = torch.full((R,), temp, dtype=torch.float32)
    seeds = torch.randint(-(2 ** 62), 2 ** 62, (R,), generator=g, dtype=torch.int64)
    return (scores.float().to(dev), cands.long().to(dev), sample_pos.long().to(dev), req_state.to(dev),
            temps.to(dev), seeds.to(dev))


def run_stock_walk(scores, cands, sample_pos, req_state, temps, seeds, K, C, probabilistic):
    R = scores.shape[0]
    tokens = torch.zeros(R * K, dtype=torch.int64, device=dev)
    realized = torch.zeros(R, K, C, dtype=torch.float32, device=dev)
    df2._selector_walk_kernel[(R,)](scores.contiguous(), cands.contiguous(), sample_pos, req_state, temps, seeds,
                                    tokens, realized, num_steps=K, top_k=C, BLOCK_K=triton.next_power_of_2(C),
                                    SAMPLE_PROBABILISTIC=probabilistic, USE_FP64=False, num_warps=1)
    return tokens.view(R, K)


def run_our_walk(scores, cands, sample_pos, req_state, temps, seeds, K, C, top_p=None, tau=1.0):
    R = scores.shape[0]
    k = gc._kernels()
    tokens = torch.zeros(R * K, dtype=torch.int64, device=dev)
    realized = torch.zeros(R, K, C, dtype=torch.float32, device=dev)
    tp = torch.ones(R, dtype=torch.float32, device=dev) if top_p is None else top_p
    k["walk"][(R,)](scores.contiguous(), cands.contiguous(), sample_pos, req_state, temps, seeds, tp, tokens,
                    realized, 1.0 / tau, num_steps=K, top_k=C, BLOCK_K=triton.next_power_of_2(C),
                    USE_TOP_P=top_p is not None, USE_FP64=False, num_warps=1)
    return tokens.view(R, K)


def test_coupling_identity():
    R, K, C = 16, 7, 16
    sc, cd, sp, rs, tt, sd = build_walk_inputs(R, K, C, 3, same_prev=True)
    stock = run_stock_walk(sc, cd, sp, rs, tt, sd, K, C, True)
    ours = run_our_walk(sc, cd, sp, rs, tt, sd, K, C)
    # the target rows: one per (request, step) with key pos = sample_pos - 1 and logits = the step's scores
    rows = torch.full((R * K, V), -float("inf"), dtype=torch.float32, device=dev)
    flat_c = cd.view(R * K, C)
    rows.scatter_(1, flat_c, sc[:, :, 0, :].reshape(R * K, C))
    exp_idx = rs.to(torch.int32)
    draw = gumbel_sample(rows, exp_idx, tt, sd, sp - 1, apply_temperature=False).view(R, K)
    check(torch.equal(stock, draw), f"coupling: image walk (probabilistic) == target draw at key sample_pos-1 "
                                    f"({int((stock == draw).sum())}/{R * K})")
    check(torch.equal(ours, draw), f"coupling: our walk == target draw at key sample_pos-1 "
                                   f"({int((ours == draw).sum())}/{R * K})")
    off = gumbel_sample(rows, exp_idx, tt, sd, sp, apply_temperature=False).view(R, K)
    check(not torch.equal(ours, off), f"coupling control: key sample_pos (off by one) agrees only by chance "
                                      f"({int((ours == off).sum())}/{R * K})")


def test_walk_parity():
    R, K, C = 24, 7, 16
    for temp in (0.0, 0.6, 1.0):
        sc, cd, sp, rs, tt, sd = build_walk_inputs(R, K, C, 11, temp=temp)
        stock = run_stock_walk(sc, cd, sp, rs, tt, sd, K, C, temp > 0)
        ours = run_our_walk(sc, cd, sp, rs, tt, sd, K, C)
        check(torch.equal(stock, ours), f"walk parity T={temp}: ours == image walk bit for bit")
    sc, cd, sp, rs, tt, sd = build_walk_inputs(R, K, C, 12, temp=1.0)
    tp = torch.full((R,), 0.5, dtype=torch.float32, device=dev)
    ours = run_our_walk(sc, cd, sp, rs, tt, sd, K, C, top_p=tp).cpu()
    # reference: the torch walk with the same nucleus rule and the numpy noise (near-ties aside)
    agree = 0
    for r in range(R):
        ref = gc.walk_torch(sc[r].cpu(), cd[r].cpu(), (sp.view(R, K)[r] - 1).cpu(), 1.0, int(sd[r]), top_p=0.5,
                            noise=lambda s, p, t: torch.from_numpy(gc.gumbel_noise_np(s, p, t.numpy())))
        agree += ref == ours[r].tolist()
    check(agree >= R - 1, f"draft nucleus: our walk == torch reference walk on {agree}/{R} requests")
    sc, cd, sp, rs, tt, sd = build_walk_inputs(R, K, C, 13, temp=0.0)
    g0 = run_stock_walk(sc, cd, sp, rs, tt, sd, K, C, False)
    o0 = run_our_walk(sc, cd, sp, rs, tt, sd, K, C, top_p=torch.full((R,), 0.3, device=dev), tau=0.7)
    check(torch.equal(g0, o0), "greedy rows: nucleus and tau do not touch the T=0 walk")


def test_verify_kernel():
    g = torch.Generator(device="cpu").manual_seed(5)
    for _ in range(20):
        R = int(torch.randint(1, 9, (1,), generator=g))
        n = torch.randint(1, 9, (R,), generator=g)
        cu = torch.cat([torch.zeros(1, dtype=torch.int64), n.cumsum(0)])
        L = int(cu[-1])
        target = torch.randint(0, 4, (L,), generator=g)
        draft = torch.randint(-1, 4, (L,), generator=g)
        W = 9
        a_s, a_n = ta.greedy_verify(target.to(dev), draft.to(dev), cu.to(dev), R, W)
        b_s, b_n = ta.greedy_verify_torch(target, draft, cu, R, W)
        ok = torch.equal(a_n.cpu().long(), b_n.long())
        for r in range(R):
            m = int(b_n[r])
            ok &= torch.equal(a_s.cpu()[r, :m], b_s[r, :m])
        if not ok:
            check(False, "verify kernel: Triton greedy_verify == torch reference")
            return
    check(True, "verify kernel: Triton greedy_verify == torch reference (20 random batches, -1 drafts included)")


def timeit(fn, n=50):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def test_cost():
    for M in (4, 8):
        logits = (torch.randn(M, V, device=dev) * 3)
        top_p = torch.full((M,), 0.95, device=dev)
        processed = apply_top_k_top_p(logits.clone(), None, top_p)
        seeds = torch.arange(M, dtype=torch.int64, device=dev) + 99
        temp = torch.ones(M, dtype=torch.float32, device=dev)
        pos = torch.arange(M, dtype=torch.int64, device=dev) + 1000
        idx0 = torch.zeros(M, dtype=torch.int32, device=dev)
        draft = torch.randint(0, V, (M,), device=dev)
        cu = torch.tensor([0, M], dtype=torch.int32, device=dev)
        idx_map = torch.zeros(1, dtype=torch.int32, device=dev)
        local = torch.arange(M, dtype=torch.int32, device=dev)
        t_stock = timeit(lambda: rejection_sample(processed, None, draft, cu, pos, idx_map, idx0, local, temp,
                                                  seeds, M - 1))
        t_coup = timeit(lambda: ta.greedy_verify(gumbel_sample(processed, idx0, temp, seeds, pos,
                                                               apply_temperature=False), draft, cu, 1, M))
        t_topp = timeit(lambda: apply_top_k_top_p(logits.clone(), None, top_p))
        print(f"info cost M={M}: stock rejection_sample {t_stock * 1000:.0f} us, coupled gumbel_sample+verify "
              f"{t_coup * 1000:.0f} us (both after apply_top_k_top_p = {t_topp * 1000:.0f} us, shared)", flush=True)


if __name__ == "__main__":
    test_noise_port()
    test_coupling_identity()
    test_walk_parity()
    test_verify_kernel()
    test_cost()
    print("ALL GPU CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)
