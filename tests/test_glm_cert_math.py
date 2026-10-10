# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the certified-head math (numpy only).

    uv run --no-project --with numpy python tests/test_glm_cert_math.py [-v]

What is tested: the float-format helpers; the MXINT8 and fp8-twin screening copies; that the interval [lo, hi]
contains the stock bf16 logit for every (row, token) under three stock arithmetics (exact, sequential fp32, and the
worst case of the error model) and two screen arithmetics; that the certified selection equals the stock argmax bit
for bit (value and id) on peaked, flat, tied, outlier-channel and vocab-parallel inputs, through the candidate path
and the fallback path; the Gumbel-max variant with temperatures 0 / 1 / other and +-2 ulp division error; the
non-finite fallback; and that the stored tables are rounded up.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.join(os.path.dirname(HERE), "overlay")
sys.path.insert(0, OVERLAY)
import cert_math as cm  # noqa: E402
from fpfmt import (bf16_round, e2m1_decode, e2m1_encode, e4m3_decode, e4m3_encode, e4m3_round,  # noqa: E402
                   pack_nibbles, unpack_nibbles)

VERBOSE = "-v" in sys.argv
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    if VERBOSE or not cond:
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")


# ------------------------------------------------------------------------------------------------
# synthetic LM-head-like data
# ------------------------------------------------------------------------------------------------
def make_head(V, K, rng, outlier_cols=4):
    """Rows with varying norms, a few heavy columns, bf16 values."""
    W = rng.standard_normal((V, K)) * 0.02
    W *= rng.uniform(0.6, 1.4, size=(V, 1))
    cols = rng.choice(K, outlier_cols, replace=False)
    W[:, cols] *= 6.0
    return bf16_round(W.astype(np.float32))


def make_x(M, K, rng, W=None, peaked=True, massive=True):
    """Final-norm-like rows: gamma * n, optional massive-activation channels, optionally aligned with some
    head rows so the logits have a clear top (peaked) or not (flat)."""
    gamma = rng.uniform(0.2, 0.35, size=K)
    x = rng.standard_normal((M, K)) * gamma
    if massive:
        ch = rng.choice(K, 2, replace=False)
        x[:, ch] *= 25.0
    if peaked and W is not None:
        for m in range(M):
            t = rng.integers(W.shape[0])
            w = W[t] / np.linalg.norm(W[t])
            x[m] += w * np.linalg.norm(x[m]) * rng.uniform(0.25, 0.6)
    return bf16_round(x.astype(np.float32))


def twin(W, kind):
    if kind == "mxint8":
        q, eb = cm.make_mxint8(W)
        return cm.deq_mxint8(q, eb)
    q, eb = cm.make_fp8_twin(W)
    return cm.deq_fp8_twin(q, eb)


# ------------------------------------------------------------------------------------------------
def test_formats():
    one = np.float32(1.0)
    check("bf16 tie to even (down)", bf16_round(np.float32(1 + 2 ** -8)) == one)
    check("bf16 tie to even (up)", bf16_round(np.float32(1 + 3 * 2 ** -8)) == np.float32(1 + 2 ** -6))
    check("bf16 non-tie", bf16_round(np.float32(1 + 2 ** -8 + 2 ** -20)) == np.float32(1 + 2 ** -7))
    codes = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], dtype=np.uint8)
    vals = e4m3_decode(codes)
    rt = e4m3_encode(e4m3_round(vals))
    same = (rt == codes) | ((codes & 0x7F) == 0)  # +0 / -0 both encode as 0 or 0x80
    check("e4m3 decode/round/encode round trip (254 codes)", same.all())
    check("e4m3 max and tie", e4m3_round(np.array([448.0, 450.0, 463.9]))[1] == 448.0)
    check("e4m3 subnormal quantum", e4m3_round(np.array([2.0 ** -9 * 1.49]))[0] == 2.0 ** -9)
    x = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.26, -5.1, 0.0])
    got = e2m1_decode(e2m1_encode(x))
    want = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -0.5, -6.0, 0.0])
    check("e2m1 ties to even code, saturation", np.array_equal(got, want), f"{got}")
    c = np.random.default_rng(0).integers(0, 16, size=(3, 32)).astype(np.uint8)
    check("nibble pack/unpack", np.array_equal(unpack_nibbles(pack_nibbles(c)), c))
    check("nibble order (element 0 low)", pack_nibbles(np.array([[1, 2]], np.uint8))[0, 0] == 0x21)


def test_copies():
    rng = np.random.default_rng(1)
    W = make_head(512, 1024, rng)
    q, eb = cm.make_mxint8(W)
    D = cm.deq_mxint8(q, eb)
    check("mxint8 |q| <= 127, eb in [1, 254]", np.abs(q.astype(int)).max() <= 127 and eb.min() >= 1 and eb.max() <= 254)
    check("mxint8 dequant exact in bf16", np.array_equal(bf16_round(D.astype(np.float32)).astype(np.float64), D))
    eps = np.linalg.norm(W - D, axis=1) / np.linalg.norm(W, axis=1)
    check("mxint8 relative row error < 1.2 %", eps.mean() < 0.012, f"mean {eps.mean() * 100:.3f} %")
    Dt = twin(W, "fp8")
    eps8 = np.linalg.norm(W - Dt, axis=1) / np.linalg.norm(W, axis=1)
    check("fp8 twin relative row error ~2.7 %", 0.015 < eps8.mean() < 0.035, f"mean {eps8.mean() * 100:.3f} %")
    # a block of denormal-scale weights: dequant not bf16-exact or zero scale -> row flagged unsafe, R = inf
    W2 = W.copy()
    W2[0:32, 0:32] = bf16_round(np.full((32, 32), 3e-38, np.float32) * rng.uniform(0.5, 1, (32, 32)).astype(np.float32))
    q2, eb2 = cm.make_fp8_twin(W2)
    D2 = cm.deq_fp8_twin(q2, eb2)
    Rt, Nv, unsafe = cm.build_tables(W2, D2, 128)
    check("tiny block handled (unsafe rows get R = inf or the residual covers them)",
          (unsafe[:32].all() and np.isinf(Rt[:, :32]).all()) or not unsafe.any())
    Rt, Nv, unsafe = cm.build_tables(W, D, 128)
    R64 = np.sqrt(((W.astype(np.float64) - D).reshape(512, 8, 128) ** 2).sum(-1)).T
    check("residual table rounded up", np.all(Rt.astype(np.float64) >= R64))
    check("row-norm table rounded up", np.all(Nv.astype(np.float64) >= np.linalg.norm(W.astype(np.float64), axis=1)))


def test_interval_soundness():
    rng = np.random.default_rng(2)
    for kind in ("mxint8", "fp8"):
        for K, G in ((512, 128), (1024, 256), (4096, 128)):
            V = 384 if K < 4096 else 192
            W = make_head(V, K, rng)
            D = twin(W, kind)
            Rt, Nv, _ = cm.build_tables(W, D, G)
            x = make_x(6, K, rng, W, peaked=bool(rng.integers(2)))
            B = cm.bounds(x, Rt, Nv, G)
            for smode in ("exact", "worst"):
                s_hat = cm.screen_emulation(x, D, smode, seed=K)
                lo, hi = cm.interval(s_hat, B)
                for mode in ("exact", "seq32", "worst"):
                    if mode == "seq32" and K > 1024:
                        continue
                    l = cm.stock_logits(x, W, None, mode, seed=7)
                    ok = np.all((lo <= l) & (l <= hi))
                    check(f"interval contains stock logit [{kind} K={K} G={G} screen={smode} stock={mode}]", ok,
                          f"violations {(~((lo <= l) & (l <= hi))).sum()}")


def test_adversarial_alignment():
    """x aligned group by group with a row's residual makes Cauchy-Schwarz tight: the true screen error then equals
    the residual term of the bound, so any under-statement of the bound (e.g. a factor 2) would show up here."""
    rng = np.random.default_rng(9)
    worst_ratio = 0.0
    viol = 0
    for kind in ("mxint8", "fp8"):
        for G in (32, 128, 512):
            V, K = 256, 1024
            W = make_head(V, K, rng)
            D = twin(W, kind)
            Rt, Nv, _ = cm.build_tables(W, D, G)
            R = W.astype(np.float64) - D
            for v in rng.choice(V, 8, replace=False):
                c = rng.uniform(0.5, 2.0, size=K // G)
                Rg = R[v].reshape(K // G, G)
                x = (Rg / np.maximum(np.linalg.norm(Rg, axis=1, keepdims=True), 1e-300) * c[:, None]).reshape(1, K)
                x = bf16_round((x * 40.0).astype(np.float32))
                s_hat = cm.screen_emulation(x, D, "exact")
                B = cm.bounds(x, Rt, Nv, G)
                err = abs(cm.exact_dot(x, W)[0, v] - cm.exact_dot(x, D)[0, v])
                worst_ratio = max(worst_ratio, err / B[0, v])
                lo, hi = cm.interval(s_hat, B)
                for mode in ("exact", "worst"):
                    l = cm.stock_logits(x, W, None, mode, seed=int(v))
                    viol += int(not (lo[0, v] <= l[0, v] <= hi[0, v]))
    check("adversarially aligned x: interval still contains the stock logit", viol == 0, f"violations {viol}")
    check("adversarially aligned x: the residual term is tight (error/bound > 0.9)", worst_ratio > 0.9,
          f"max error/bound {worst_ratio:.4f}")


def _one_trial(rng, kind, V, K, G, M, cap, peaked, stock_mode, screen_mode, ties=False, temps=None, div_ulps=0):
    W = make_head(V, K, rng)
    if ties:
        # exact duplicates: rows a < b identical, x aligned with them -> a tie at the top
        a, b = sorted(rng.choice(V, 2, replace=False))
        W[b] = W[a]
    D = twin(W, kind)
    Rt, Nv, _ = cm.build_tables(W, D, G)
    x = make_x(M, K, rng, W, peaked=peaked)
    if ties:
        x[0] = bf16_round((W[a] / np.linalg.norm(W[a]) * np.linalg.norm(x[0]) * 0.8 + x[0] * 0.3).astype(np.float32))
    noise = None
    if temps is not None:
        noise = np.stack([cm.gumbel_noise(V, rng) for _ in range(M)])
    seed = int(rng.integers(1 << 30))
    r = cm.cert_local(x, W, D, Rt, Nv, G, cap, stock_mode, screen_mode, temps, noise, div_ulps, seed)
    sv, si = cm.stock_local(x, W, stock_mode, temps, noise, div_ulps, seed)
    same = np.array_equal(r.idx, si) and np.array_equal(r.val.view(np.uint32), sv.view(np.uint32))
    return same, r, (a if ties else None)


def test_selection_equals_stock():
    rng = np.random.default_rng(3)
    n = bad = fb = 0
    unions = []
    t0 = time.time()
    for trial in range(160):
        kind = ("mxint8", "fp8")[trial % 2]
        K, G = ((512, 128), (1024, 128), (1024, 512))[trial % 3]
        stock_mode = ("exact", "worst", "seq32")[trial % 3] if K <= 512 else ("exact", "worst")[trial % 2]
        screen_mode = ("exact", "worst")[(trial // 2) % 2]
        cap = (4, 64, 1 << 20)[trial % 3]
        same, r, _ = _one_trial(rng, kind, 1024, K, G, int(rng.integers(1, 9)), cap, bool(trial % 4), stock_mode,
                                screen_mode)
        n += 1
        bad += not same
        fb += r.fallback
        unions.append(r.union)
    check("certified selection == stock argmax (160 trials, both copies, 3 stock arithmetics)", bad == 0,
          f"mismatches {bad}/{n}, fallbacks {fb}, union median {np.median(unions):.0f} ({time.time() - t0:.1f} s)")
    check("fallback path exercised", fb > 0)
    check("candidate path exercised", fb < n)


def test_ties():
    rng = np.random.default_rng(4)
    bad = 0
    for trial in range(40):
        same, r, a = _one_trial(rng, "mxint8", 512, 512, 128, 2, (8, 1 << 20)[trial % 2], True, "exact", "exact",
                                ties=True)
        bad += not same
    check("exact ties resolve to the lowest id, both paths", bad == 0, f"mismatches {bad}/40")


def test_vocab_parallel():
    rng = np.random.default_rng(5)
    bad = 0
    for trial in range(30):
        V, K, G, tp, M = 1024, 512, 128, 4, 5
        W = make_head(V, K, rng)
        if trial % 3 == 0:  # cross-rank tie: same row on rank 0 and rank 2
            W[700] = W[100]
        x = make_x(M, K, rng, W, peaked=bool(trial % 2))
        if trial % 3 == 0:
            x[0] = bf16_round((W[100] / np.linalg.norm(W[100]) * np.linalg.norm(x[0]) + x[0] * 0.2).astype(np.float32))
        Vs = V // tp
        vals, idxs, starts = [], [], []
        for r in range(tp):
            Ws = W[r * Vs:(r + 1) * Vs]
            D = twin(Ws, "mxint8")
            Rt, Nv, _ = cm.build_tables(Ws, D, G)
            res = cm.cert_local(x, Ws, D, Rt, Nv, G, 64, "worst", "worst", seed=trial)
            vals.append(res.val)
            idxs.append(res.idx)
            starts.append(r * Vs)
        got = cm.reduce_ranks(vals, idxs, starts)
        _, want = cm.stock_local(x, W, "worst", seed=trial)
        bad += not np.array_equal(got, want)
    check("vocab-parallel (4 ranks) == global stock argmax, incl. cross-rank ties", bad == 0, f"mismatches {bad}/30")


def test_gumbel():
    rng = np.random.default_rng(6)
    bad = 0
    for trial in range(60):
        M = 4
        temps = np.array([0.0, 1.0, 0.7, 1.6])[rng.permutation(4)]
        du = (-2, 0, 2)[trial % 3]
        same, r, _ = _one_trial(rng, ("mxint8", "fp8")[trial % 2], 768, 512, 128, M, (16, 1 << 20)[trial % 2],
                                bool(trial % 2), ("exact", "worst")[trial % 2], "worst", temps=temps, div_ulps=du)
        bad += not same
    check("Gumbel-max: certified sample == stock sample (T in {0, 1, 0.7, 1.6}, +-2 ulp division)", bad == 0,
          f"mismatches {bad}/60")


def test_nonfinite():
    """Non-finite data never yields a certified id: the step goes to the stock sampler on EVERY rank, and even the
    placeholder / clamped ids are in-vocab."""
    rng = np.random.default_rng(7)
    V, K, G, tp, M = 1024, 256, 128, 4, 3
    Vs = V // tp
    W = make_head(V, K, rng)
    x = make_x(M, K, rng, W)

    def run(x, W):
        res, starts = [], []
        for r in range(tp):
            Ws = W[r * Vs:(r + 1) * Vs]
            D = twin(Ws, "mxint8") if np.all(np.isfinite(Ws)) else Ws.astype(np.float64)
            Rt, Nv, _ = cm.build_tables(Ws, D, G) if np.all(np.isfinite(Ws)) else (
                np.full((K // G, Vs), np.inf, np.float32), np.full(Vs, np.inf, np.float32), None)
            res.append(cm.cert_local(x, Ws, D, Rt, Nv, G, 1 << 20))
            starts.append(r * Vs)
        return res, starts

    for what, xx in (("NaN", np.nan), ("+inf", np.inf), ("-inf", -np.inf)):
        xb = x.copy()
        xb[1, 5] = xx
        res, starts = run(xb, W)
        check(f"{what} in a hidden row: every rank reports non-finite", all(r.status == cm.ST_NONFINITE for r in res))
        check(f"{what} in a hidden row: the step goes to the stock sampler", cm.step_tp(res, starts) == "stock")
        check(f"{what} in a hidden row: placeholder ids in-vocab", all(((r.idx >= 0) & (r.idx < Vs)).all() for r in res))
    W2 = W.copy()
    W2[2 * Vs + 17, 3] = np.inf  # one rank's shard only: that rank's screen goes non-finite, the others stay finite
    res, starts = run(x, W2)
    check("non-finite on one rank only: that rank flags it", [r.status == cm.ST_NONFINITE for r in res]
          == [False, False, True, False])
    check("non-finite on one rank only: ALL ranks take the stock sampler (agreed via the gathered status)",
          cm.step_tp(res, starts) == "stock")
    res, starts = run(x, W)
    out = cm.step_tp(res, starts)
    check("finite step: certified ids in-vocab and equal to the stock argmax",
          not isinstance(out, str) and ((out >= 0) & (out < V)).all()
          and np.array_equal(out, cm.stock_local(x, W)[1]))
    ids = np.arange(40, 40 + 3000)
    for name, vals in (("all NaN", np.full(3000, np.nan)), ("all -inf", np.full(3000, -np.inf)),
                       ("NaN mixed with finite", np.where(np.arange(3000) % 7 == 0, np.nan, 1.0)),
                       ("NaN in the first block, finite later", np.r_[np.full(1024, np.nan), np.ones(1976)])):
        _, bi = cm.final_select(vals, ids)
        check(f"final selection with {name}: id in-vocab (never INT_MAX)", 0 <= bi < 40 + 3000 and bi != cm.INT_MAX,
              f"id {bi}")


def _load_pure(names):
    """The torch-free helpers of glm_cert_head.py (parse_rows, refine_table, config_hash), extracted with ast so the
    test runs without torch."""
    import ast
    import hashlib
    path = os.path.join(OVERLAY, "glm_cert_head.py")
    src = open(path).read()
    tree = ast.parse(src)
    ns = {"MAX_M": 32, "os": os, "hashlib": hashlib}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "CONFIG_KEYS" for t in node.targets):
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    return ns


def test_fail_closed_map():
    ns = _load_pure({"parse_rows", "refine_table", "config_hash"})
    rt = ns["refine_table"]
    check("empty maps: no row count qualified (stock everywhere)", rt("", "") == {})
    t = rt("", "2-5:1,8:1,1:0")
    check("SLICEK map: only listed non-zero M qualify", sorted(t) == [2, 3, 4, 5, 8] and t[3] == ("kernel", 1),
          f"{sorted(t)}")
    check("SLICEK map: M=1:0, M=6, M=7, M=9..32 stay stock", all(m not in t for m in [1, 6, 7] + list(range(9, 33))))
    t = rt("2-4", "2-8:1")
    check("GEMM map wins over the kernel map, kernel for the rest", t[2] == ("gemm", 0) and t[5] == ("kernel", 1)
          and 9 not in t)
    t = rt("30-40", "")
    check("entries above MAX_M are ignored", sorted(t) == [30, 31, 32])
    try:
        rt("", "2-4")
        check("SLICEK map entry without a value is rejected", False)
    except ValueError:
        check("SLICEK map entry without a value is rejected", True)
    h1 = ns["config_hash"]({"GLM_CERT_HEAD": "1", "GLM_CERT_HEAD_GEMM_MAP": "2-32"})
    h2 = ns["config_hash"]({"GLM_CERT_HEAD": "1", "GLM_CERT_HEAD_GEMM_MAP": "2-31"})
    check("config hash differs when a rank's settings differ, fits int64", h1 != h2 and 0 <= h1 < 2 ** 62)


def report_tightness():
    """Not a test: how many candidates the certificate leaves on synthetic peaked / flat rows at K = 4096."""
    rng = np.random.default_rng(8)
    V, K, G = 4096, 4096, 128
    W = make_head(V, K, rng)
    for kind in ("mxint8", "fp8"):
        D = twin(W, kind)
        Rt, Nv, _ = cm.build_tables(W, D, G)
        for peaked in (True, False):
            x = make_x(8, K, rng, W, peaked=peaked)
            B = cm.bounds(x, Rt, Nv, G)
            s = cm.screen_emulation(x, D)
            lo, hi = cm.interval(s, B)
            rows, union, _ = cm.candidates(lo, hi)
            err = np.abs(cm.exact_dot(x, W) - cm.exact_dot(x, D))
            print(f"  [{kind:6s} {'peaked' if peaked else 'flat  '}] candidates/row median {np.median(rows.sum(1)):.0f} "
                  f"max {rows.sum(1).max()} union {union.sum()} of {V}; bound/true-error median "
                  f"{np.median(B / np.maximum(err, 1e-30)):.0f}x; B median {np.median(B):.3f} logits")


if __name__ == "__main__":
    t0 = time.time()
    test_formats()
    test_copies()
    test_interval_soundness()
    test_adversarial_alignment()
    test_selection_equals_stock()
    test_ties()
    test_vocab_parallel()
    test_gumbel()
    test_nonfinite()
    test_fail_closed_map()
    npass = sum(ok for _, ok in RESULTS)
    print(f"{npass}/{len(RESULTS)} checks passed ({time.time() - t0:.1f} s)")
    if "--tightness" in sys.argv:
        report_tightness()
    sys.exit(0 if npass == len(RESULTS) else 1)
