# SPDX-License-Identifier: Apache-2.0
"""Reference (numpy) model of the certified target-head argmax: the math that glm_cert_head.py runs on the GPU.

Problem. Per decode step the target LM head reads its BF16 shard W [Vs = 38720, K = 4096] (317 MB per rank) to
produce bf16 logits l = bf16(acc(W x)) for M rows, of which greedy verification only needs argmax (and the
vocab-parallel reduction only needs each rank's (max, argmax)). The weights are not on an 8-bit grid, so an 8-bit
head changes outputs. Instead:

  1. screen   s^_v = <D_v, x>       with D a 1-byte copy of W (MXINT8 by default: int8 + one power-of-two
                                    scale per 32 weights of a row; or the drafter's 32x32 e4m3 twin)
  2. bound    |l_v - s^_v| <= B_v   rigorous, for ANY fp32 summation order of the stock GEMM (see below)
  3. prune    candidates of row m = { v : hi_mv >= max_u lo_mu }, lo/hi = bf16-rounded interval ends
  4. recompute the candidates' logits from the BF16 rows with qualified stock-equivalent arithmetic (the stock op
              on gathered rows, or our kernel), argmax with the lowest token id on ties (the stock tie order)
  5. fallback when the union of candidates over the M rows exceeds CAP: the stock op on the whole shard.
              Non-finite data (a non-finite input row or screen value) on ANY rank routes the whole step to the
              unmodified stock sampler on every rank (agreed through the status column of the pair all-gather);
              the certificate never emits an id for such a step, and the local placeholder id is in-vocab.

The bound, for token v, row m (x = the lm_head input row, W_v / D_v rows, R_v = W_v - D_v, groups g of G columns):

  |<W_v, x> - <D_v, x>| = |sum_g <R_vg, x_g>| <= sum_g ||R_vg|| * ||x_g||            (Cauchy-Schwarz per group)
  screen accumulation:  |s^_v - <D_v, x>|   <= C_ACC * sum_k |D_vk x_k| <= C_ACC * ||D_v|| * ||x||
  stock accumulation:   |acc_v - <W_v, x>|  <= C_ACC * sum_k |W_vk x_k| <= C_ACC * ||W_v|| * ||x||
  => acc_v in [s^_v - B_v, s^_v + B_v],  B_v = sum_g ||R_vg|| ||x_g|| + 2 C_ACC N_v ||x||,  N_v = max(||W_v||, ||D_v||)

and the stock logit is l_v = rn_bf16(acc_v). Round-to-nearest is monotone, so lo_v = rn_bf16(s^_v - B_v) <=
l_v <= rn_bf16(s^_v + B_v) = hi_v (the fp32 evaluation of s^ +- B carries a small slop term so the computed
ends are on the safe side). If w is the stock argmax of row m then l_w >= l_u >= lo_u for the u that maximises
lo, so hi_w >= l_w >= T_m = max_u lo_u: w is a candidate, and so is every token tied with it. Recomputing the
candidates exactly and taking the lowest id among the maxima reproduces the stock argmax bit for bit.

C_ACC (an ASSUMPTION, validated empirically). Products of bf16 values are exact in fp32. Plain fp32 summation of
n terms in any order (sequential, pairwise, split-K with an fp32 workspace, with or without FMA) errs by at most
gamma_(n-1) sum|p| with gamma_n = n u / (1 - n u), u = 2**-24 (Higham, Accuracy and Stability of Numerical
Algorithms, 2nd ed., sec. 4.2). Tensor-core MMA does not round like IEEE fp32: per Fasi, Higham, Mikaitis and
Pranesh ("Numerical behavior of NVIDIA tensor cores", PeerJ CS 2021) the k products of one instruction and the
accumulator are aligned to the largest exponent and truncated. We charge every one of the K/16 chained k=16 steps
with 17 terms of at most one fp32 ulp of the running magnitude each, which gives C_ACC = (K/16) * 17 * 2**-23
(5.2e-4 at K = 4096) and dominates gamma_(K-1) = 2.4e-4 as well. NVIDIA does not specify BF16 MMA accumulation
order, rounding or subnormal handling (PTX ISA, mma), and the 2021 study did not measure GB10, so this term is a
model of the hardware, not a theorem about it. The "worst" arithmetic in the CPU tests injects errors bounded by the
same constant and therefore cannot validate it; what does: gpu_test_cert.py section 2 (measured screen error vs
bound on real rows: median 0.010 vs 1.08 logits on the GLM head, 2026-09-28) and GLM_CERT_HEAD=check in a live
boot (certified vs stock on every eligible step, mismatch counters in the log). The rest of the argument (grouped
Cauchy-Schwarz, monotone rounding of the interval ends, inclusive candidate test, lowest-id ties) is proven
conditional on this term and on the qualified exact recompute.

Gumbel (temperature sampling). The V2 sampler draws argmax_v (l_v / T + g_v) with counter-based noise g_v that
depends only on (seed, position, token id) (gumbel.py, Gumbel-max trick). The noise is known before screening,
and v -> fl(fl(v / T) + g) is monotone up to the <= 2 ulp error of Triton's default fp32 division, so the same
interval, pushed through that map with a small slack, certifies the sampled token too.

min_tokens (a mask). The stock min-tokens rule sets some (row, token) logits to -inf before the argmax. The
certificate only needs the masked tokens removed from BOTH the threshold and the candidates of that row: with lo = hi
= -inf there, the masked argmax w still has hi_w >= l_w >= max over unmasked u of lo_u = T. A token masked for row
m can still enter the union through another row, so the final selection masks it again per row. `mask` below is
[M, V] bool (True = -inf), mirroring glm_cert_head's screen and final kernels.

Everything here is float64/fp32 numpy and mirrors the kernel's arithmetic where it matters (bf16 rounding of the
interval ends, the inflation factors), so the CPU tests exercise the real certificate.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from fpfmt import bf16_round, decode_e8m0, e4m3_round, round_up_f32  # noqa: E402

MX_GROUP = 32          # MXINT8 scale group along K (OCP MX block size)
TWIN_BLOCK = 32        # the drafter's fp8 twin: one exponent per 32 x 32 block
INFL_Q = 1.0 + 2.0 ** -18    # fp32 evaluation of sum_g R_g ||x_g|| (<= K/G terms, all >= 0)
INFL_B = 1.0 + 2.0 ** -20
INFL_X = 1.0 + 2.0 ** -16    # fp32 row-group norms of x
SLOP_REL = 2.0 ** -22
SLOP_ABS = 2.0 ** -126


def c_acc(K: int) -> float:
    return (K / 16.0) * 17.0 * 2.0 ** -23


# ------------------------------------------------------------------------------------------------
# screening copies (as the kernel dequantizes them)
# ------------------------------------------------------------------------------------------------
def make_mxint8(W: np.ndarray):
    """W [V, K] (bf16 values) -> (q int8 [V, K], eb uint8 [V, K/32]); D = q * 2**(eb - 127).
    Scale = the smallest power of two >= amax / 127, so |q| <= 127 and D is exact in bf16 (|q| < 2**7)."""
    V, K = W.shape
    assert K % MX_GROUP == 0
    Wg = W.reshape(V, K // MX_GROUP, MX_GROUP).astype(np.float64)
    amax = np.abs(Wg).max(-1)
    e = np.where(amax > 0, np.ceil(np.log2(np.where(amax > 0, amax, 1.0) / 127.0)), 0.0)
    eb = np.clip(e + 127.0, 1.0, 254.0)
    q = np.clip(np.rint(Wg / np.exp2(eb - 127.0)[..., None]), -127, 127)
    return q.astype(np.int8).reshape(V, K), eb.astype(np.uint8)


def deq_mxint8(q: np.ndarray, eb: np.ndarray) -> np.ndarray:
    V, K = q.shape
    s = decode_e8m0(eb)  # [V, K/32]
    return (q.reshape(V, K // MX_GROUP, MX_GROUP) * s[..., None]).reshape(V, K)


def make_fp8_twin(W: np.ndarray):
    """Mirror of glm_ds_draft.make_twin: e4m3 + one exponent byte per 32 x 32 block (e + 127)."""
    V, K = W.shape
    assert K % TWIN_BLOCK == 0
    pad = (-V) % TWIN_BLOCK
    Wp = np.pad(W.astype(np.float64), ((0, pad), (0, 0)))
    Wb = Wp.reshape((V + pad) // 32, 32, K // 32, 32)
    amax = np.maximum(np.abs(Wb).max(axis=(1, 3)), 2.0 ** -126)
    e = np.clip(np.ceil(np.log2(amax / 448.0)), -127, 127)
    q = e4m3_round(Wb / np.exp2(e)[:, None, :, None]).reshape(V + pad, K)[:V]
    return q, (e + 127).astype(np.uint8)


def deq_fp8_twin(q: np.ndarray, eb: np.ndarray) -> np.ndarray:
    """Kernel semantics: scale bits = eb << 23 (exponent byte 0 -> 0.0), value then rounded to bf16."""
    V, K = q.shape
    s = decode_e8m0(eb)  # [ceil(V/32), K/32]
    s = np.repeat(np.repeat(s, 32, axis=0)[:V], 32, axis=1)
    return bf16_round((q * s).astype(np.float32)).astype(np.float64)


# ------------------------------------------------------------------------------------------------
# tables (built once per rank after load)
# ------------------------------------------------------------------------------------------------
def build_tables(W: np.ndarray, D: np.ndarray, G: int):
    """Rt [K/G, V] fp32 (residual group norms, rounded up), Nv [V] fp32 (rounded up), unsafe [V] bool.
    A row is unsafe when its screening copy is not exactly representable as the kernel builds it (bf16); its
    residual norms are set to +inf, which makes it a candidate on every step."""
    V, K = W.shape
    assert K % G == 0
    W64 = W.astype(np.float64)
    D64 = D.astype(np.float64)
    unsafe = np.any(bf16_round(D64.astype(np.float32)).astype(np.float64) != D64, axis=1)
    R = W64 - D64
    Rg = np.sqrt((R.reshape(V, K // G, G) ** 2).sum(-1)) * (1.0 + 2.0 ** -40)
    Rt = round_up_f32(Rg).T.copy()
    Rt[:, unsafe] = np.inf
    Nv = round_up_f32(np.maximum(np.linalg.norm(W64, axis=1), np.linalg.norm(D64, axis=1)) * (1.0 + 2.0 ** -40))
    return Rt, Nv, unsafe


# ------------------------------------------------------------------------------------------------
# bound, interval, candidates
# ------------------------------------------------------------------------------------------------
def x_norms(x: np.ndarray, G: int):
    M, K = x.shape
    xg = np.sqrt((x.astype(np.float64).reshape(M, K // G, G) ** 2).sum(-1)) * INFL_X
    xn = np.sqrt((xg ** 2).sum(-1)) * INFL_X
    return xg, xn


def bounds(x: np.ndarray, Rt: np.ndarray, Nv: np.ndarray, G: int) -> np.ndarray:
    K = x.shape[1]
    xg, xn = x_norms(x, G)
    with np.errstate(invalid="ignore"):
        Q = xg @ Rt.astype(np.float64)
        B = (Q * INFL_Q + 2.0 * c_acc(K) * xn[:, None] * Nv.astype(np.float64)[None, :]) * INFL_B
    return np.where(np.isnan(B), np.inf, B)  # 0 * inf (unsafe row, zero x group) must stay conservative


def interval(s_hat: np.ndarray, B: np.ndarray):
    """bf16-rounded (lo, hi) as fp32; mirrors the kernel epilogue."""
    with np.errstate(invalid="ignore", over="ignore"):
        slop = (np.abs(s_hat) + B) * SLOP_REL + SLOP_ABS
        lo = bf16_round((s_hat - B - slop).astype(np.float32))
        hi = bf16_round((s_hat + B + slop).astype(np.float32))
    return lo, hi


def candidates(lo: np.ndarray, hi: np.ndarray):
    """-> (row mask [M, V], union [V], thresholds [M])."""
    T = lo.max(axis=1)
    rows = hi >= T[:, None]
    return rows, rows.any(axis=0), T


def select(values: np.ndarray, ids: np.ndarray):
    """Max value, lowest id among the maxima (the stock tie order)."""
    mx = values.max()
    return mx, int(ids[values == mx].min())


# ------------------------------------------------------------------------------------------------
# stock-arithmetic emulations (every element is a pure function of (m, v): subset == full)
# ------------------------------------------------------------------------------------------------
def exact_dot(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) @ W.astype(np.float64).T


def stock_logits(x, W, ids=None, mode="exact", seed=0):
    """bf16 logits of rows ids (default all) under an arithmetic model of the stock GEMM:
    exact    rn_bf16 of the exact dot product
    seq32    sequential fp32 accumulation (a real fp32 order)
    worst    exact + delta, delta = +-C_ACC sum|w x| with a sign fixed per (m, v): the extreme of the error model,
             which is what could make a certificate fail if it under-covered the stock rounding"""
    ids = np.arange(W.shape[0]) if ids is None else np.asarray(ids)
    Ws = W[ids]
    if mode == "exact":
        return bf16_round(exact_dot(x, Ws).astype(np.float32))
    if mode == "seq32":
        acc = np.zeros((x.shape[0], len(ids)), dtype=np.float32)
        x32 = x.astype(np.float32)
        W32 = Ws.astype(np.float32)
        for k in range(x.shape[1]):
            acc = (acc + x32[:, k:k + 1] * W32[None, :, k]).astype(np.float32)
        return bf16_round(acc)
    if mode == "worst":
        s = exact_dot(x, Ws)
        mag = np.abs(x.astype(np.float64)) @ np.abs(Ws.astype(np.float64)).T
        m_idx = np.arange(x.shape[0])[:, None]
        h = (m_idx * 1000003 + ids[None, :] * 7919 + seed) % 2
        sign = np.where(h == 0, 1.0, -1.0)
        return bf16_round((s + sign * c_acc(x.shape[1]) * mag * 0.999).astype(np.float32))
    raise ValueError(mode)


def screen_emulation(x, D, mode="exact", seed=0):
    """s^ with the screen kernel's error pushed to the extreme of the model (mode worst) or none (exact)."""
    s = exact_dot(x, D)
    if mode == "exact":
        return s.astype(np.float32).astype(np.float64)
    mag = np.abs(x.astype(np.float64)) @ np.abs(D.astype(np.float64)).T
    rng = np.random.default_rng(seed)
    sign = np.where(rng.random(s.shape) < 0.5, 1.0, -1.0)
    return (s + sign * c_acc(x.shape[1]) * mag * 0.999).astype(np.float32).astype(np.float64)


# ------------------------------------------------------------------------------------------------
# Gumbel-max
# ------------------------------------------------------------------------------------------------
def gumbel_noise(n, rng):
    u = rng.random(n).astype(np.float32)
    u = np.maximum(u, np.float32(4.6566127342e-10))
    return (-np.log(-np.log1p(-u.astype(np.float64)))).astype(np.float32)


def gumbel_value(l, t, g, div_ulps=0):
    """Stock value in fp32: T == 0 -> l; T == 1 -> l + g; else fl(l / t) (+- div_ulps ulp: div.full) + g."""
    l32 = np.asarray(l, dtype=np.float32)
    if t == 0.0:
        return l32
    g32 = np.asarray(g, dtype=np.float32)
    if t == 1.0:
        return (l32 + g32).astype(np.float32)
    q = (l32 / np.float32(t)).astype(np.float32)
    for _ in range(abs(div_ulps)):
        q = np.nextafter(q, np.float32(np.inf if div_ulps > 0 else -np.inf)).astype(np.float32)
    return (q + g32).astype(np.float32)


def gumbel_interval(lo, hi, t, g):
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    if t == 0.0:
        return lo, hi
    g32 = np.asarray(g, dtype=np.float32)
    if t == 1.0:
        return (lo + g32).astype(np.float32), (hi + g32).astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        qlo = (lo / np.float32(t)).astype(np.float32)
        qhi = (hi / np.float32(t)).astype(np.float32)
        qlo = (qlo - np.abs(qlo) * np.float32(2.0 ** -20) - np.float32(SLOP_ABS)).astype(np.float32)
        qhi = (qhi + np.abs(qhi) * np.float32(2.0 ** -20) + np.float32(SLOP_ABS)).astype(np.float32)
        return (qlo + g32).astype(np.float32), (qhi + g32).astype(np.float32)


# ------------------------------------------------------------------------------------------------
# the whole certified local selection (one rank)
# ------------------------------------------------------------------------------------------------
ST_OK, ST_LOCAL_STOCK, ST_NONFINITE = 0, 1, 2
INT_MAX = 2147483647


def final_select(values, ids):
    """Mirror of glm_cert_head._final_kernel: max value, lowest id among the maxima, NaN never matches, and an
    id that stayed at INT_MAX (no match: all -inf / NaN) is clamped to local id 0 (in-vocab)."""
    values = np.asarray(values, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64)
    best_v, best_i = np.float32(-np.inf), INT_MAX
    for c0 in range(0, len(values), 1024):
        v, i = values[c0:c0 + 1024], ids[c0:c0 + 1024]
        with np.errstate(invalid="ignore"):
            mx = np.max(v) if len(v) else np.float32(-np.inf)    # NaN-propagating, like tl.max
            hit = v == mx
        im = int(i[hit].min()) if hit.any() else INT_MAX
        if mx > best_v or (mx == best_v and im < best_i):
            best_v, best_i = mx, im
    if best_i == INT_MAX:
        best_i = 0
    return best_v, best_i


class Result:
    def __init__(self, val, idx, union, fallback, per_row, status=ST_OK):
        self.val, self.idx, self.union, self.fallback, self.per_row = val, idx, union, fallback, per_row
        self.status = status


def cert_local(x, W, D, Rt, Nv, G, cap, stock_mode="exact", screen_mode="exact", temps=None, noise=None,
               div_ulps=0, seed=0, mask=None):
    """Per row (value, local id) exactly as the stock arithmetic `stock_mode` would select on this shard, and the
    status glm_cert_head.cert_local reports (ST_NONFINITE: val / idx are in-vocab placeholders and the step must
    take the stock sampler). temps [M] (None = all greedy) and noise [M, V] select the Gumbel variant."""
    M, K = x.shape
    V = W.shape[0]
    with np.errstate(invalid="ignore", over="ignore"):
        s_hat = screen_emulation(x, D, screen_mode, seed)
    if not (np.all(np.isfinite(s_hat)) and np.all(np.isfinite(x))):
        return Result(np.full(M, -np.inf, np.float32), np.zeros(M, np.int64), 0, True, np.zeros(M, int),
                      ST_NONFINITE)
    B = bounds(x, Rt, Nv, G)
    lo, hi = interval(s_hat, B)
    if mask is not None:
        lo = np.where(mask, np.float32(-np.inf), lo).astype(np.float32)
        hi = np.where(mask, np.float32(-np.inf), hi).astype(np.float32)
    vlo, vhi = lo, hi
    if temps is not None:
        vlo = np.empty_like(lo)
        vhi = np.empty_like(hi)
        for m in range(M):
            vlo[m], vhi[m] = gumbel_interval(lo[m], hi[m], temps[m], noise[m])
    rows, union, _ = candidates(vlo, vhi)
    n = int(union.sum())
    fallback = n > cap or n == 0
    ids = np.arange(V) if fallback else np.nonzero(union)[0]
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(ids))      # the device compaction order is arbitrary
    ids = ids[perm]
    l = stock_logits(x, W, ids, stock_mode, seed)
    vals, idxs = np.empty(M, np.float32), np.empty(M, np.int64)
    for m in range(M):
        v = l[m] if temps is None else gumbel_value(l[m], temps[m], noise[m, ids], div_ulps)
        if mask is not None:
            v = np.where(mask[m, ids], np.float32(-np.inf), v).astype(np.float32)
        vals[m], idxs[m] = final_select(v, ids)
    return Result(vals, idxs, n, fallback, rows.sum(axis=1), ST_LOCAL_STOCK if fallback else ST_OK)


def step_tp(results, starts):
    """The per-step protocol over TP ranks (glm_cert_head._certified): every rank sees every rank's status in the
    gathered pairs; any ST_NONFINITE -> "stock" on every rank; else the pair reduction's global ids."""
    if any(r.status == ST_NONFINITE for r in results):
        return "stock"
    return reduce_ranks([r.val for r in results], [r.idx for r in results], starts)


def stock_local(x, W, stock_mode="exact", temps=None, noise=None, div_ulps=0, seed=0, mask=None):
    l = stock_logits(x, W, None, stock_mode, seed)
    if mask is not None:   # the stock min-tokens rule: -inf written into the (fp32 copy of the) logits
        l = np.where(mask, np.float32(-np.inf), l.astype(np.float32)).astype(np.float32)
    ids = np.arange(W.shape[0])
    out_v, out_i = [], []
    for m in range(x.shape[0]):
        v = l[m] if temps is None else gumbel_value(l[m], temps[m], noise[m], div_ulps)
        a, b = select(v, ids)
        out_v.append(a)
        out_i.append(b)
    return np.array(out_v, np.float32), np.array(out_i, np.int64)


def reduce_ranks(vals, idxs, starts):
    """Vocab-parallel pair reduction (glm_target_argmax.reduce_pairs): max value, first rank on ties."""
    vals = np.stack(vals, 1)
    ids = np.stack([i + s for i, s in zip(idxs, starts)], 1)
    r = vals.argmax(axis=1)
    return ids[np.arange(vals.shape[0]), r]
