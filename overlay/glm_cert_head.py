# SPDX-License-Identifier: Apache-2.0
"""GLM_CERT_HEAD=1: certified target argmax from a 1-byte copy of the LM head. Output is the stock argmax.

Tony v11 image, vLLM 0.1.dev20051+g487ecf187, V2 model runner. Needs GLM_TARGET_VOCAB_ARGMAX=1 (overlay
glm_target_argmax.py): this module replaces its local step "full BF16 shard GEMM -> local (max, argmax)" and keeps
everything else (the 16-byte pair all-gather, the greedy rejection step, the dispatch rules).

Per step and rank, for the M target rows:
  xnorm    ||x_g|| per 128-column group and ||x|| of every row; flags a non-finite row
  screen   s^ = x . D^T from the 1-byte copy D (MXINT8: int8 + power-of-two scale per 32 weights of a row,
           163.6 MB instead of 317.2 MB), plus the bound B = sum_g R_g ||x_g|| + 2 C_ACC N_v ||x|| per (row, token)
           and the bf16-rounded interval ends lo / hi; writes hi and atomically max-reduces lo per row (threshold T);
           flags a non-finite screen value
  select   candidates = tokens with hi >= T in any row; compacted into a CAP-slot list with one atomic per CTA
  SYNC 1   the host reads (candidate count, non-finite flag)
  refine   count <= CAP: the candidates' bf16 logits, either from the STOCK op (the lm_head's own
           dispatch_unquantized_gemm, i.e. F.linear / cuBLAS) on the gathered candidate rows padded to a fixed bucket
           (64 .. CAP rows), or from our exact kernel (same MMA shape and K order as the stock cuBLAS kernel);
           whichever is qualified for this M (below)
           count > CAP: the stock op on the whole shard (lp._apply_head, exactly the glm_target_argmax path)
  final    per row: max value, lowest token id among the maxima -> (value, global id, status) for the pair reduction
  SYNC 2   after the pair all-gather the host reads every rank's status; if ANY rank saw a non-finite row or
           value, every rank discards the certified result and runs the unmodified runner sample() (full logits,
           stock sampler / rejection sampler), so the collective sequence stays identical on all ranks
The two host syncs cost a short GPU bubble each (the host launches the next kernels only after the GPU caught up);
gpu_test_cert.py section 4 times the whole path including them.

Fail-closed qualification. A row count M uses the certificate only if it is listed in GLM_CERT_HEAD_GEMM_MAP
(stock-op refinement) or GLM_CERT_HEAD_SLICEK_MAP with a non-zero value (exact-kernel refinement); every other M
takes the glm_target_argmax path untouched. Both maps come from gpu_test_cert.py, which compares every logit of
every refinement shape with the full-shard stock GEMM bit for bit, on every rank's shard, and prints maps built only
from row counts that passed everywhere. Note that the stock op on a different shape (gathered rows) is NOT exact by
construction either (cuBLAS picks its kernel per shape); it is qualified per (M, bucket) exactly like our kernel.
The overflow path and the non-finite path always use the stock op on the stock shape.

The bound (cert_math.py). The grouped Cauchy-Schwarz term, the monotone bf16 rounding of the interval ends, the
inclusive candidate test and the lowest-id tie rule are proven (conditional on the next sentence). The
accumulation term C_ACC = (K/16) * 17 * 2**-23 per unit of ||W_v|| ||x|| models tensor-core accumulation after
Fasi, Higham, Mikaitis and Pranesh (2021); NVIDIA does not specify BF16 MMA accumulation order, rounding or
subnormal handling (PTX ISA, mma), and that study predates GB10. The term is therefore an ASSUMPTION, validated
empirically: gpu_test_cert.py section 2 (the measured screen error is ~1 % of the bound on real rows) and
GLM_CERT_HEAD=check in a live boot (certified vs stock on every eligible step, counters in the log).

Rank invariance. Eligibility is decided from per-request CPU state (glm_target_argmax.plan plus, for sampled rows,
the checks below) and from TP-agreed state: after every rank built (or failed to build) its tables, the TP group
MAX-all-reduces (failed, config hash, -config hash) on the CPU group; if any rank failed or the configurations
differ, every rank disables the feature. A rank-local overflow uses the stock op on that rank only: the collectives
(one pair all-gather) do not change. Non-finite data is agreed through the status column of that same all-gather.

Sampled rows (GLM_CERT_HEAD_SAMPLED=1, off by default). Only the plain Sampler path (steps without drafts, e.g.
the first token after a prefill): temperature > 0 with no top-k / top-p / min-p / penalties / logit bias / bad words
/ active thinking budget / logprobs, fp32 Gumbel. The Gumbel noise of the V2 sampler (gumbel.py) is a pure function
of (seed, position, token id), so it is generated for the whole shard, the interval is pushed through
v -> fl(fl(v / T) + g) with a 2**-20 relative slack for Triton's approximate division, and the candidates'
values are recomputed with the stock expression. Rejection-sampled rows (T > 0 with drafts) need the full
softmax (normaliser, residual distribution): not covered, stock path.

min_tokens (GLM_CERT_HEAD_MINTOK=1, default on). sparkDash sends min_tokens = max_tokens + ignore_eos, so every
request carries stop ids (all_stop_token_ids keeps EOS) and LogitBiasState.use_logit_bias is set. The stock rule
(sample/logit_bias.py _bias_kernel, min-tokens clause, applied by Sampler.apply_sampling_params before the argmax in
both the Sampler and the RejectionSampler._verify path): for logits row j of request r = expanded_idx_mapping[j],
if num_stop_token_ids[r] > 0 and positions[logits_indices[j]] + 1 < min_lens[r], logits[j, stop_token_ids[r, :n]]
= -inf. glm_levers (GLM_LV_ARGMAX_MINTOK=1) admits such steps to the vocab-parallel greedy path when the ONLY logit
bias in the batch is this one and sets runner._glm_lv_mintok; the certified path then applies the SAME rule, read
from the same GPU state (min_lens / num_stop_token_ids / stop_token_ids / positions), inside the screen kernel (a
masked (row, token) gets lo = hi = -inf: it neither raises the threshold nor becomes a candidate for that row) and
inside the final kernel (a candidate that is masked for the row being reduced, e.g. one that entered the union through
another row, is -inf there), and on the shard logits of the overflow path. The result is argmax(mask(stock logits))
with the stock tie order. Masking only removes tokens, so the certificate argument is unchanged: the masked argmax w
has l_w >= l_u >= lo_u for every UNMASKED u, hence hi_w >= T. Steps with more than GLM_CERT_HEAD_MINTOK_MAX_STOP
(16) stop ids in some request stay on the glm_levers path. Other logit processors (allowed_token_ids, logit_bias,
bad words, penalties) never reach this module: glm_levers / glm_target_argmax route those steps to the stock path.

Modes
  GLM_CERT_HEAD=0|1|check        check: run both, count mismatches (local winners and final tokens), return STOCK
  GLM_CERT_HEAD_MINTOK=1|0       certified path also on glm_levers min_tokens steps (read per call; glm_ab key)
  GLM_CERT_HEAD_MINTOK_MAX_STOP=16  more stop ids than this in a request: that step stays on the glm_levers path
  GLM_CERT_HEAD_GEMM_MAP=        row counts qualified for stock-op refinement, e.g. "2-32" (gpu_test_cert.py)
  GLM_CERT_HEAD_SLICEK_MAP=      row counts qualified for exact-kernel refinement, e.g. "2-31:1" (M:SLICEK)
                                 an M in neither map (or mapped to 0) always takes the stock path
  GLM_CERT_HEAD_TWIN=mxint8|fp8  mxint8 (default): own int8 copy, eps ~0.8 %. fp8: reuse the drafter's
                                 GLM_DS_DRAFT_HEAD_FP8 twin (no extra memory, eps ~2.7 %: ~3x wider bound)
  GLM_CERT_HEAD_DRAFT=1          give the MXINT8 copy to the drafter's head too (needs glm_ds_draft_fp4 hooks) and
                                 drop the fp8 twin: one 1-byte copy serves both
  GLM_CERT_HEAD_GROUP=128        residual-norm group width (Rt is [K/G, V] fp16, rounded up)
  GLM_CERT_HEAD_CAP=2048         candidate slots per rank and step (union over rows); also the largest bucket
  GLM_CERT_HEAD_MAX_M=32         larger batches take the unmodified path
  GLM_CERT_HEAD_TILE=64,128,4,3  screen kernel BN,BK,warps,stages
  GLM_CERT_HEAD_SAMPLED=0|1
  GLM_CERT_HEAD_DUMP=DIR  GLM_CERT_HEAD_DUMP_ROWS=20000   save target lm_head input rows (bf16) per rank for
                                 gpu_test_cert.py --hidden (test boots only: it syncs)
  GLM_CERT_HEAD_LOG_EVERY=2000
The mode is read per call through overlay/glm_ab.py when the in-boot A/B harness is armed and knows the key (the
harness switches all ranks between the same two steps). All GLM_CERT_HEAD* variables must be identical on every
rank: they enter the agreed config hash, and a rank that does not register the module at all would leave the
others waiting in the agreement all-reduce (a loud boot hang, never a silent divergence).

Prior art. Certified top-k from a quantized first pass followed by exact re-ranking is the classic
filter-and-refine scheme of vector search (e.g. VA-File, Weber, Schek and Blott, VLDB 1998; the asymmetric
distance bounds of product quantization, Jegou, Douze and Schmid, TPAMI 2011); here the bound is Cauchy-Schwarz on
the quantization residual. Gumbel-max trick: Gumbel 1954, Maddison, Tarlow and Minka 2014. Tensor-core rounding:
Fasi, Higham, Mikaitis and Pranesh 2021. MXINT8 layout: OCP Microscaling Formats spec v1.0. vocab-parallel argmax:
zixi-qi, vllm#34049. Kernels are ours.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import torch

_OFF = ("", "0", "off", "false", "no")
MODE = os.environ.get("GLM_CERT_HEAD", "0").strip().lower()
TWIN = os.environ.get("GLM_CERT_HEAD_TWIN", "mxint8").strip().lower()
SHARE_DRAFT = os.environ.get("GLM_CERT_HEAD_DRAFT", "0").strip().lower() not in _OFF
GROUP = int(os.environ.get("GLM_CERT_HEAD_GROUP", "128"))
CAP = int(os.environ.get("GLM_CERT_HEAD_CAP", "2048"))
MAX_M = int(os.environ.get("GLM_CERT_HEAD_MAX_M", "32"))
_GEMM_MAP = os.environ.get("GLM_CERT_HEAD_GEMM_MAP", "").strip()
_SLICE_MAP = os.environ.get("GLM_CERT_HEAD_SLICEK_MAP", "").strip()
SAMPLED = os.environ.get("GLM_CERT_HEAD_SAMPLED", "0").strip().lower() not in _OFF
MINTOK = os.environ.get("GLM_CERT_HEAD_MINTOK", "1").strip().lower() not in _OFF
MINTOK_MAX_STOP = int(os.environ.get("GLM_CERT_HEAD_MINTOK_MAX_STOP", "16"))
DUMP = os.environ.get("GLM_CERT_HEAD_DUMP", "").strip()
DUMP_ROWS = int(os.environ.get("GLM_CERT_HEAD_DUMP_ROWS", "20000"))
LOG_EVERY = int(os.environ.get("GLM_CERT_HEAD_LOG_EVERY", "2000"))
_tile = [int(v) for v in os.environ.get("GLM_CERT_HEAD_TILE", "64,128,4,3").split(",")]
BN, BK, WARPS, STAGES = _tile
if MODE not in ("0", "1", "on", "true", "check") + _OFF:
    raise ValueError(f"GLM_CERT_HEAD={MODE!r}")
BUCKETS = tuple(b for b in (64, 128, 256, 512, 1024, 2048, 4096) if b <= CAP)
if TWIN not in ("mxint8", "fp8") or MAX_M > 64 or CAP not in BUCKETS or GROUP % 32 or not 1 <= MINTOK_MAX_STOP <= 128:
    raise ValueError("glm-cert-head: bad TWIN / MAX_M / CAP (a power of two in 64..4096) / GROUP / MINTOK_MAX_STOP")


def parse_rows(spec: str, with_value: bool) -> dict:
    """"2-5,8" -> {2: 1, ..., 8: 1};  with_value: "2-5:1,8:0" -> {2: 1, ..., 8: 0}."""
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        rng, _, val = part.partition(":")
        if with_value and not val:
            raise ValueError(f"glm-cert-head: map entry {part!r} needs M:VALUE")
        a, _, b = rng.partition("-")
        for m in range(int(a), int(b or a) + 1):
            out[m] = int(val) if with_value else 1
    return out


def refine_table(gemm_spec: str, slice_spec: str, max_m: int = MAX_M) -> dict:
    """M -> ("gemm", 0) | ("kernel", SLICEK) for qualified row counts only (fail closed: absent = stock)."""
    g = parse_rows(gemm_spec, False)
    s = parse_rows(slice_spec, True)
    if any(v not in (0, 1, 2, 4) for v in s.values()):
        raise ValueError("glm-cert-head: GLM_CERT_HEAD_SLICEK_MAP values must be 0, 1, 2 or 4")
    out = {}
    for m in range(1, max_m + 1):
        if g.get(m):
            out[m] = ("gemm", 0)
        elif s.get(m):
            out[m] = ("kernel", s[m])
    return out


REFINE_FOR = refine_table(_GEMM_MAP, _SLICE_MAP)
CONFIG_KEYS = ("GLM_CERT_HEAD", "GLM_CERT_HEAD_TWIN", "GLM_CERT_HEAD_DRAFT", "GLM_CERT_HEAD_GROUP", "GLM_CERT_HEAD_CAP",
               "GLM_CERT_HEAD_MAX_M", "GLM_CERT_HEAD_GEMM_MAP", "GLM_CERT_HEAD_SLICEK_MAP", "GLM_CERT_HEAD_SAMPLED",
               "GLM_CERT_HEAD_TILE", "GLM_CERT_HEAD_MINTOK", "GLM_CERT_HEAD_MINTOK_MAX_STOP")


def config_hash(env=None) -> int:
    env = os.environ if env is None else env
    s = "|".join(f"{k}={env.get(k, '')}" for k in CONFIG_KEYS)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:7], "little")  # < 2**56, fits int64 with sign

# status column of the pair all-gather
ST_OK, ST_LOCAL_STOCK, ST_NONFINITE = 0.0, 1.0, 2.0

# certificate constants (identical to cert_math.py)
INFL_Q = 1.0 + 2.0 ** -18
INFL_B = 1.0 + 2.0 ** -20
INFL_X = 1.0 + 2.0 ** -16
SLOP_REL = 2.0 ** -22
SLOP_ABS = 2.0 ** -126
X_ABS = 2.0 ** -59          # covers flushed-to-zero squares in the fp32 group norms
DIV_SLACK = 2.0 ** -20      # Triton fp32 '/' is div.full (<= 2 ulp)
INT_MAX = 2147483647


def c_acc(K: int) -> float:
    """ASSUMED tensor-core accumulation bound per unit of ||W_v|| ||x|| (see the module docstring)."""
    return (K / 16.0) * 17.0 * 2.0 ** -23


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-cert-head: {msg}\n")


def _mode() -> str:
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False) and "GLM_CERT_HEAD" in getattr(ab, "KNOWN", {}):
        return str(ab.norm_value("GLM_CERT_HEAD", ab.env("GLM_CERT_HEAD")))
    if MODE in _OFF:
        return "0"
    return "check" if MODE == "check" else "1"


def _mintok_on() -> bool:
    """GLM_CERT_HEAD_MINTOK, per call (the in-boot A/B harness switches it on every rank between the same steps)."""
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False) and "GLM_CERT_HEAD_MINTOK" in getattr(ab, "KNOWN", {}):
        return bool(ab.norm_value("GLM_CERT_HEAD_MINTOK", ab.env("GLM_CERT_HEAD_MINTOK", "1")))
    return MINTOK


# ---------------------------------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------------------------------
_K: dict = {}
_SQRT_RN = False  # set by _kernels(): Triton has tl.sqrt_rn (IEEE) or only the approximate tl.sqrt


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl
    from vllm.triton_utils import tldevice  # noqa: F401  (global for the jit functions)
    from vllm.v1.worker.gpu.sample.gumbel import tl_rand32  # noqa: F401
    globals()["tldevice"] = tldevice
    globals()["tl_rand32"] = tl_rand32
    sqrt_rn = getattr(tl, "sqrt_rn", None)
    globals()["_SQRT_RN"] = sqrt_rn is not None

    @triton.jit
    def _xnorm_kernel(X, sx, XG, XN, CTRL, K: tl.constexpr, G: tl.constexpr, INFL: tl.constexpr,
                      XABS: tl.constexpr, RN: tl.constexpr):
        m = tl.program_id(0)
        NG: tl.constexpr = K // G
        gi = tl.arange(0, NG)
        x = tl.load(X + m * sx + gi[:, None] * G + tl.arange(0, G)[None, :]).to(tl.float32)
        s = tl.sum(x * x, axis=1)
        if RN:
            xg = tl.sqrt_rn(s) * INFL + XABS
        else:
            xg = tl.sqrt(s) * INFL + XABS
        tl.store(XG + m * NG + gi, xg)
        t = tl.sum(xg * xg, axis=0)
        if RN:
            xn = tl.sqrt_rn(t) * INFL
        else:
            xn = tl.sqrt(t) * INFL
        tl.store(XN + m, xn)
        if (xn != xn) | (xn == float("inf")):  # NaN / inf in the row (or a sum of squares that overflowed)
            tl.atomic_max(CTRL + 1, 1)

    @triton.jit
    def _screen_kernel(X, sx, Q, E, RT, NV, XG, XN, HI, LO, THR, CTRL, M, V,
                       vstart, EIDX, POS, MINLEN, NSTOP, STOP, sstop,
                       K: tl.constexpr, G: tl.constexpr, FMT: tl.constexpr, STORE_LO: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, CACC2: tl.constexpr,
                       INFL_Q: tl.constexpr, INFL_B: tl.constexpr, SLOP_REL: tl.constexpr, SLOP_ABS: tl.constexpr,
                       NS: tl.constexpr):
        pid = tl.program_id(0)
        m = tl.arange(0, BM)
        n = pid * BN + tl.arange(0, BN)
        mok = m < M
        nok = n < V
        n64 = n.to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(X + m[:, None] * sx + k[None, :], mask=mok[:, None], other=0.0)
            if FMT == 0:  # MXINT8: int8 [V, K], exponent byte [V, K/32]
                w = tl.load(Q + n64[None, :] * K + k[:, None], mask=nok[None, :], other=0)
                eb = tl.load(E + n64[None, :] * (K // 32) + k[:, None] // 32, mask=nok[None, :], other=127)
            else:  # drafter fp8 twin: e4m3 [V, K], exponent byte per 32 x 32 block [ceil(V/32), K/32]
                w = tl.load(Q + n64[None, :] * K + k[:, None], mask=nok[None, :], other=0.0)
                eb = tl.load(E + (n64[None, :] // 32) * (K // 32) + k[:, None] // 32, mask=nok[None, :], other=127)
            sc = (eb.to(tl.int32) << 23).to(tl.float32, bitcast=True)  # 2**(eb - 127); byte 0 -> 0.0
            wd = (w.to(tl.float32) * sc).to(tl.bfloat16)
            acc = tl.dot(x, wd, acc)
        NG: tl.constexpr = K // G
        qb = tl.zeros((BM, BN), dtype=tl.float32)
        for g in range(0, NG):
            r = tl.load(RT + g * V + n, mask=nok, other=0.0).to(tl.float32)
            xg = tl.load(XG + m * NG + g, mask=mok, other=0.0)
            qb += xg[:, None] * r[None, :]
        nv = tl.load(NV + n, mask=nok, other=0.0)
        xn = tl.load(XN + m, mask=mok, other=0.0)
        B = (qb * INFL_Q + CACC2 * (xn[:, None] * nv[None, :])) * INFL_B
        B = tl.where(B != B, float("inf"), B)  # 0 * inf (unsafe row) stays conservative
        slop = (tl.abs(acc) + B) * SLOP_REL + SLOP_ABS
        lo32 = acc - B - slop
        hi32 = acc + B + slop
        if NS > 0:  # min_tokens: the stock _bias_kernel rule, (row, stop id) -> -inf before any selection
            req = tl.load(EIDX + m, mask=mok, other=0).to(tl.int64)
            ns = tl.load(NSTOP + req, mask=mok, other=0)
            ps = tl.load(POS + m, mask=mok, other=0)
            ml = tl.load(MINLEN + req, mask=mok, other=0)
            act = mok & (ns > 0) & (ps + 1 < ml)
            gid = vstart + n
            hit = tl.zeros((BM, BN), dtype=tl.int32)
            for j in tl.static_range(NS):
                sid = tl.load(STOP + req * sstop + j, mask=act & (j < ns), other=-1)
                hit = hit | (gid[None, :] == sid[:, None]).to(tl.int32)
            lo32 = tl.where(hit != 0, float("-inf"), lo32)
            hi32 = tl.where(hit != 0, float("-inf"), hi32)
        lo = lo32.to(tl.bfloat16)
        hi = hi32.to(tl.bfloat16)
        ok = mok[:, None] & nok[None, :]
        tl.store(HI + m[:, None] * V + n[None, :], hi, mask=ok)
        if STORE_LO:
            tl.store(LO + m[:, None] * V + n[None, :], lo, mask=ok)
        lof = tl.where(ok, lo.to(tl.float32), float("-inf"))
        tl.atomic_max(THR + m, tl.max(lof, axis=1), mask=mok)
        bad = ok & ((acc != acc) | (tl.abs(acc) == float("inf")))
        nbad = tl.sum(tl.sum(bad.to(tl.int32), axis=1), axis=0)
        if nbad > 0:
            tl.atomic_max(CTRL + 1, 1)

    @triton.jit
    def _gumbel_bounds_kernel(LO, HI, VHI, THR, M, V, vstart, TEMP, SEEDS, POS, IMAP,
                              BLOCK: tl.constexpr, SLACK: tl.constexpr, SLOP_ABS: tl.constexpr):
        m = tl.program_id(0)
        b = tl.program_id(1)
        o = b * BLOCK + tl.arange(0, BLOCK)
        ok = o < V
        lo = tl.load(LO + m * V + o, mask=ok, other=float("-inf")).to(tl.float32)
        hi = tl.load(HI + m * V + o, mask=ok, other=float("-inf")).to(tl.float32)
        req = tl.load(IMAP + m).to(tl.int64)
        t = tl.load(TEMP + req).to(tl.float32)
        vlo = lo
        vhi = hi
        if t != 0.0:
            seed = tl.load(SEEDS + req)
            pos = tl.load(POS + m)
            gseed = tl.randint(seed, pos)
            u = tl_rand32(gseed, vstart + o, False)
            g = -tl.log(-tldevice.log1p(-u))
            if t == 1.0:
                vlo = lo + g
                vhi = hi + g
            else:
                qlo = lo / t
                qhi = hi / t
                qlo = qlo - tl.abs(qlo) * SLACK - SLOP_ABS
                qhi = qhi + tl.abs(qhi) * SLACK + SLOP_ABS
                vlo = qlo + g
                vhi = qhi + g
        vlo = tl.where(ok, vlo, float("-inf"))
        tl.store(VHI + m * V + o, vhi, mask=ok)
        tl.atomic_max(THR + m, tl.max(vlo, axis=0))

    @triton.jit
    def _select_kernel(HI, THR, CTRL, IDS, M, V, CAP, BM: tl.constexpr, BS: tl.constexpr):
        pid = tl.program_id(0)
        m = tl.arange(0, BM)
        mok = m < M
        T = tl.load(THR + m, mask=mok, other=float("inf"))
        v = pid * BS + tl.arange(0, BS)
        vok = v < V
        h = tl.load(HI + m[:, None] * V + v[None, :], mask=mok[:, None] & vok[None, :],
                    other=float("-inf")).to(tl.float32)
        c = tl.max(((h >= T[:, None]) & mok[:, None] & vok[None, :]).to(tl.int32), axis=0)
        n = tl.sum(c, axis=0)
        if n > 0:
            base = tl.atomic_add(CTRL, n)
            p = base + tl.cumsum(c, axis=0) - 1
            tl.store(IDS + p, v, mask=(c > 0) & (p < CAP))

    @triton.jit
    def _exact_kernel(X, sx, W, IDS, OUT, so, M, n,
                      K: tl.constexpr, BV: tl.constexpr, BMX: tl.constexpr, BKX: tl.constexpr, SLICE: tl.constexpr):
        pid = tl.program_id(0)
        pm = tl.program_id(1)
        j = pid * BV + tl.arange(0, BV)
        ok = j < n
        ids = tl.load(IDS + j, mask=ok, other=0)
        mm = pm * BMX + tl.arange(0, BMX)
        mok = mm < M
        ids64 = ids.to(tl.int64)
        SK: tl.constexpr = BKX // SLICE
        acc0 = tl.zeros((BV, BMX), dtype=tl.float32)
        acc1 = tl.zeros((BV, BMX), dtype=tl.float32)
        acc2 = tl.zeros((BV, BMX), dtype=tl.float32)
        acc3 = tl.zeros((BV, BMX), dtype=tl.float32)
        for k0 in range(0, K, BKX):
            for s in tl.static_range(SLICE):
                k = k0 + s * SK + tl.arange(0, SK)
                w = tl.load(W + ids64[:, None] * K + k[None, :], mask=ok[:, None], other=0.0)
                xt = tl.load(X + mm[None, :] * sx + k[:, None], mask=mok[None, :], other=0.0)
                if s == 0:
                    acc0 = tl.dot(w, xt, acc0)
                elif s == 1:
                    acc1 = tl.dot(w, xt, acc1)
                elif s == 2:
                    acc2 = tl.dot(w, xt, acc2)
                else:
                    acc3 = tl.dot(w, xt, acc3)
        acc = acc0
        if SLICE > 1:
            acc = acc + acc1
        if SLICE > 2:
            acc = acc + acc2
            acc = acc + acc3
        out = acc.to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + mm[None, :] * so + j[:, None], out, mask=ok[:, None] & mok[None, :])

    @triton.jit
    def _final_kernel(CAND, sc, IDS, n, VAL, IDX, vstart, TEMP, SEEDS, POS, IMAP,
                      EIDX, MPOS, MINLEN, NSTOP, STOP, sstop,
                      HAS_IDS: tl.constexpr, GUMBEL: tl.constexpr, BLOCK: tl.constexpr, NS: tl.constexpr):
        """Row m: max over the first n columns of CAND (token ids IDS, or the column index), lowest id on ties.
        Callers pass finite values only (non-finite steps never get here); the clamp below still guarantees an
        in-vocab id (the shard start) if that contract is ever broken."""
        m = tl.program_id(0)
        t = 0.0
        gseed = 0
        if GUMBEL:
            req = tl.load(IMAP + m).to(tl.int64)
            t = tl.load(TEMP + req).to(tl.float32)
            seed = tl.load(SEEDS + req)
            pos = tl.load(POS + m)
            gseed = tl.randint(seed, pos)
        mreq = 0
        mns = 0
        mact = False
        if NS > 0:  # min_tokens (same rule as the screen): this row's stop ids are -inf among the candidates
            mreq = tl.load(EIDX + m).to(tl.int64)
            mns = tl.load(NSTOP + mreq)
            mact = (mns > 0) & (tl.load(MPOS + m) + 1 < tl.load(MINLEN + mreq))
        best_v = tl.full([1], float("-inf"), tl.float32)
        best_i = tl.full([1], 2147483647, tl.int32)
        for c0 in range(0, n, BLOCK):
            o = c0 + tl.arange(0, BLOCK)
            msk = o < n
            val = tl.load(CAND + m * sc + o, mask=msk, other=float("-inf"))
            if HAS_IDS:
                ids = tl.load(IDS + o, mask=msk, other=2147483647)
            else:
                ids = o
            if NS > 0:
                hit = tl.zeros([BLOCK], dtype=tl.int32)
                for j in tl.static_range(NS):
                    sid = tl.load(STOP + mreq * sstop + j, mask=mact & (j < mns), other=-1)
                    hit = hit | (vstart + ids == sid).to(tl.int32)
                val = tl.where(hit != 0, float("-inf"), val)
            if GUMBEL:
                if t != 0.0:
                    u = tl_rand32(gseed, vstart + ids, False)
                    g = -tl.log(-tldevice.log1p(-u))
                    if t == 1.0:
                        val = val + g
                    else:
                        val = val / t + g
                val = tl.where(msk, val, float("-inf"))
            mx = tl.max(val, axis=0)
            im = tl.min(tl.where(msk & (val == mx), ids, 2147483647), axis=0)
            take = (mx > best_v) | ((mx == best_v) & (im < best_i))
            best_v = tl.where(take, mx, best_v)
            best_i = tl.where(take, im, best_i)
        best_i = tl.where(best_i == 2147483647, 0, best_i)
        one = tl.arange(0, 1)
        tl.store(VAL + m + one, best_v)
        tl.store(IDX + m + one, best_i.to(tl.int64) + vstart)

    _K.update(triton=triton, xnorm=_xnorm_kernel, screen=_screen_kernel, gbounds=_gumbel_bounds_kernel,
              select=_select_kernel, exact=_exact_kernel, final=_final_kernel)
    return _K


# ---------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------
def _round_up_f32(x64: torch.Tensor) -> torch.Tensor:
    """float64 (>= 0 or inf) -> smallest fp32 >= x."""
    f = x64.float()
    bump = f.double() < x64
    return torch.where(bump, (f.view(torch.int32) + 1).view(torch.float32), f)


def _round_up_f16(x32: torch.Tensor) -> torch.Tensor:
    """fp32 (>= 0 or inf) -> smallest fp16 >= x (inf above the fp16 range)."""
    h = x32.half()
    bump = h.float() < x32
    return torch.where(bump, (h.view(torch.int16) + 1).view(torch.float16), h)


def make_mxint8(w: torch.Tensor, chunk_rows: int = 2048):
    """bf16 [V, K] -> (int8 [V, K], exponent byte uint8 [V, K/32]); scale 2**(e - 127) >= amax / 127."""
    V, K = w.shape
    q = torch.empty((V, K), dtype=torch.int8, device=w.device)
    e = torch.empty((V, K // 32), dtype=torch.uint8, device=w.device)
    for r0 in range(0, V, chunk_rows):
        r1 = min(V, r0 + chunk_rows)
        wf = w[r0:r1].float().view(r1 - r0, K // 32, 32)
        amax = wf.abs().amax(-1)
        ex = torch.where(amax > 0, torch.ceil(torch.log2(amax.clamp_min(1e-38) / 127.0)), torch.zeros_like(amax))
        eb = (ex + 127.0).clamp(1.0, 254.0)
        qq = torch.round(wf / torch.exp2(eb - 127.0)[..., None]).clamp(-127, 127)
        q[r0:r1] = qq.view(r1 - r0, K).to(torch.int8)
        e[r0:r1] = eb.to(torch.uint8)
    return q, e


def deq_rows(fmt: int, q: torch.Tensor, e: torch.Tensor, r0: int, r1: int) -> torch.Tensor:
    """Rows r0:r1 of the screening copy exactly as the screen kernel builds them (fp32 holding bf16 values)."""
    K = q.shape[1]
    qf = q[r0:r1].float()
    if fmt == 0:
        eb = e[r0:r1].to(torch.int32)
        sc = (eb << 23).view(torch.float32).repeat_interleave(32, dim=1)
    else:
        rows = torch.arange(r0, r1, device=q.device) // 32
        eb = e[rows].to(torch.int32)
        sc = (eb << 23).view(torch.float32).repeat_interleave(32, dim=1)
    return (qf * sc[:, :K]).to(torch.bfloat16).float(), (qf * sc[:, :K])


class State:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _torch_linear(x, w):
    return torch.nn.functional.linear(x, w)


@torch.inference_mode()
def build_from_weight(W: torch.Tensor, vstart: int, tp: int, fmt_name: str = TWIN, twin=None, group: int = GROUP,
                      stock_gemm=None, stock_full=None):
    """State for one rank's head shard W [V, K] bf16 (also used by gpu_test_cert.py without a runner).
    stock_gemm(x, w) is the stock op on arbitrary rows (default F.linear); stock_full(x) the stock op on the shard."""
    V, K = W.shape
    if K % group or K % 32 or K % BK or (K // group) & (K // group - 1):
        raise ValueError(f"K={K} does not fit GROUP={group} / BK={BK}")
    if fmt_name == "fp8":
        if twin is None:
            import glm_ds_draft
            twin = glm_ds_draft.make_twin(W)
        q, e = twin
        fmt = 1
    else:
        q, e = make_mxint8(W)
        fmt = 0
    NG = K // group
    rt = torch.empty((NG, V), dtype=torch.float16, device=W.device)
    nv = torch.empty((V,), dtype=torch.float32, device=W.device)
    n_unsafe = 0
    for r0 in range(0, V, 1024):
        r1 = min(V, r0 + 1024)
        d_bf, d_raw = deq_rows(fmt, q, e, r0, r1)
        unsafe = (d_bf != d_raw).any(dim=1)
        w64 = W[r0:r1].double()
        res = (w64 - d_bf.double()).view(r1 - r0, NG, group)
        rg = _round_up_f32(res.pow(2).sum(-1).sqrt() * (1.0 + 2.0 ** -40))
        rg[unsafe] = float("inf")
        rt[:, r0:r1] = _round_up_f16(rg).t()
        nrm = torch.maximum(w64.norm(dim=1), d_bf.double().norm(dim=1)) * (1.0 + 2.0 ** -40)
        nv[r0:r1] = _round_up_f32(nrm)
        n_unsafe += int(unsafe.sum())
    d0, _ = deq_rows(fmt, q, e, 0, min(V, 4096))
    w0 = W[:d0.shape[0]].float()
    eps = float(((w0 - d0).norm(dim=1) / w0.norm(dim=1).clamp_min(1e-30)).mean())
    ctrl_init = torch.zeros(4 + 2 * 64, dtype=torch.int32, device=W.device)
    ctrl_init[4:].view(torch.float32).fill_(float("-inf"))
    host_ctrl = torch.zeros(2, dtype=torch.int32, pin_memory=W.is_cuda)
    wbuf = torch.empty((max(BUCKETS), K), dtype=W.dtype, device=W.device)  # gathered candidate rows (16 MB)
    return State(W=W, q=q, e=e, fmt=fmt, rt=rt, nv=nv, V=V, K=K, vstart=int(vstart), tp=int(tp), group=group,
                 ctrl_init=ctrl_init, host_ctrl=host_ctrl, wbuf=wbuf, n_unsafe=n_unsafe, eps=eps,
                 stock_gemm=stock_gemm or _torch_linear, stock_full=stock_full or (lambda x: _torch_linear(x, W)),
                 lm_head=None, lp=None)


def _stock_gemm_for(lm_head):
    """The lm_head's own stock op (UnquantizedEmbeddingMethod.apply without VLLM_BATCH_INVARIANT) on other rows."""
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
    gemm = dispatch_unquantized_gemm()
    return lambda x, w: gemm(lm_head, x, w, None)


@torch.inference_mode()
def build_state(runner):
    import glm_target_argmax as ta
    import vllm.envs as envs
    head = ta.resolve_head(runner.model)
    if isinstance(head, str):
        return head
    _, lm_head, lp = head
    if getattr(envs, "VLLM_BATCH_INVARIANT", False):
        return "VLLM_BATCH_INVARIANT is set (the stock head uses linear_batch_invariant)"
    # Both stock methods apply dispatch_unquantized_gemm()(layer, x, weight, bias) (vocab_parallel_embedding.py:75,
    # linear.py:212); lossless8 builds the head with UnquantizedLinearMethod.
    if type(lm_head.quant_method).__name__ not in ("UnquantizedEmbeddingMethod", "UnquantizedLinearMethod"):
        return f"lm_head quant method is {type(lm_head.quant_method).__name__}"
    if getattr(lp, "head_dtype", None) not in (None, torch.bfloat16):
        return "head_dtype is not bf16"
    W = getattr(lm_head, "weight", None)
    if W is None or W.dtype != torch.bfloat16 or W.dim() != 2 or not W.is_contiguous():
        return "lm_head.weight is not a contiguous bf16 matrix"
    si = lm_head.shard_indices
    vstart = int(si.org_vocab_start_index)
    if int(si.org_vocab_end_index) - vstart != W.shape[0]:
        return "shard width != weight rows"
    if not REFINE_FOR:
        return "no row count is qualified (GLM_CERT_HEAD_GEMM_MAP / GLM_CERT_HEAD_SLICEK_MAP empty): stock everywhere"
    draft = getattr(getattr(runner, "speculator", None), "model", None)
    twin = getattr(draft, "_glm_ds_head_fp8_twin", None) if TWIN == "fp8" else None
    try:
        st = build_from_weight(W.data, vstart, int(lm_head.tp_size), TWIN, twin,
                               stock_gemm=_stock_gemm_for(lm_head),
                               stock_full=lambda x, _lp=lp, _h=lm_head: _lp._apply_head(_h, x, None))
    except ValueError as e:
        return str(e)
    st.lm_head, st.lp = lm_head, lp
    st.draft = draft
    mb = (st.q.numel() * st.q.element_size() + st.e.numel() + st.rt.numel() * 2 + st.nv.numel() * 4) / 1e6
    _log(f"built: {'MXINT8' if st.fmt == 0 else 'fp8 twin'} copy of the {st.V} x {st.K} head shard, {mb:.1f} MB read "
         f"per step (BF16: {st.V * st.K * 2 / 1e6:.1f} MB), eps {st.eps * 100:.3f} %, unsafe rows {st.n_unsafe}, "
         f"G={GROUP}, CAP={CAP}, qualified M: gemm {sorted(m for m, r in REFINE_FOR.items() if r[0] == 'gemm')}, "
         f"kernel {sorted(m for m, r in REFINE_FOR.items() if r[0] == 'kernel')}")
    return st


def tp_agree(ok: bool) -> tuple[bool, str]:
    """All TP ranks: MAX-all-reduce (failed, hash, -hash) on the TP CPU group. Every rank must call this exactly once
    per runner, whether its own build succeeded or not."""
    try:
        from vllm.distributed.parallel_state import get_tp_group
        grp = get_tp_group()
        if grp.world_size == 1:
            return ok, "tp=1"
        h = config_hash()
        t = torch.tensor([0 if ok else 1, h, -h], dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=grp.cpu_group)
        failed, hmax, hneg = (int(v) for v in t.tolist())
        if failed:
            return False, "a TP rank failed to build its tables"
        if hmax != -hneg:
            return False, "GLM_CERT_HEAD* settings differ between TP ranks"
        return ok, "agreed"
    except Exception as e:  # noqa: BLE001  cannot agree -> nobody enables (this rank at least; the others hang loud)
        return False, f"agreement failed: {e!r}"


# ---------------------------------------------------------------------------------------------------
# the certified local selection
# ---------------------------------------------------------------------------------------------------
def bucket_for(n: int) -> int:
    return next(b for b in BUCKETS if n <= b)


class LocalResult:
    __slots__ = ("val", "idx", "status", "count", "refine")

    def __init__(self, val, idx, status, count, refine):
        self.val, self.idx, self.status, self.count, self.refine = val, idx, status, count, refine


def mintok_spec(eidx, pos, min_lens, nstop, stop, max_nstop: int) -> dict:
    """The min_tokens mask of one step, exactly the arguments of the stock _bias_kernel's min-tokens clause:
    eidx [M] (expanded_idx_mapping), pos [M] (positions of the logits rows), and the per-request LogitBiasState
    tensors min_lens [R], num_stop_token_ids [R], stop_token_ids [R, S]. max_nstop (host int, from the CPU copy of
    num_stop_token_ids over the batch) sets the unrolled loop width NS = next power of two >= max_nstop."""
    ns = 1
    while ns < max_nstop:
        ns *= 2
    return dict(eidx=eidx.contiguous(), pos=pos.contiguous(), min_lens=min_lens, nstop=nstop, stop=stop,
                ns=ns if max_nstop > 0 else 0)


def mask_shard_(shard: torch.Tensor, vstart: int, mt: dict | None) -> torch.Tensor:
    """In place on this rank's [M, V] shard logits (overflow / check paths): the stock min-tokens rule restricted to
    the shard's vocab range [vstart, vstart + V). Same rule and tensors as the kernels; torch ops only."""
    if mt is None or not mt["ns"]:
        return shard
    M, V = shard.shape
    req = mt["eidx"].long()
    n = mt["nstop"][req].long()
    act = (mt["pos"].long() + 1 < mt["min_lens"][req].long()) & (n > 0)
    S = mt["stop"].shape[1]
    loc = mt["stop"][req].long() - vstart                                     # [M, S]
    col = torch.arange(S, device=shard.device)
    m = act[:, None] & (col[None, :] < n[:, None]) & (loc >= 0) & (loc < V)
    rows = torch.arange(M, device=shard.device)[:, None].expand_as(loc)
    shard[rows[m], loc[m]] = float("-inf")
    return shard


def cert_local(st, h: torch.Tensor, refine, gum: dict | None = None, cap: int = CAP, stats=None,
               mt: dict | None = None) -> LocalResult:
    """h [M, K] bf16 -> this rank's (value fp32 [M], global id int64 [M]) exactly as the stock head + local argmax
    (greedy) or the stock Gumbel-max (gum) would select on this shard, plus a status (mt: mintok_spec(...), the
    stock min_tokens mask applied to the logits before the selection; greedy only):
      ST_OK          certified: candidates refined with `refine` = ("gemm", 0) | ("kernel", SLICEK)
      ST_LOCAL_STOCK the candidate list overflowed: the stock op ran on the whole shard (still exact)
      ST_NONFINITE   a non-finite row or screen value: val / idx are placeholders (in-vocab), the caller must
                     route the whole step to the stock path on every rank
    One host sync (the candidate count)."""
    k = _kernels()
    triton = k["triton"]
    x = h.contiguous()
    M, K = x.shape
    if mt is not None and gum is not None:
        raise ValueError("glm-cert-head: min_tokens masking is certified for greedy rows only")
    NS = mt["ns"] if mt is not None else 0
    V = st.V
    dev = x.device
    G = st.group
    NG = K // G
    xg = torch.empty((M, NG), dtype=torch.float32, device=dev)
    xn = torch.empty((M,), dtype=torch.float32, device=dev)
    ctrl = st.ctrl_init.clone()
    thr = ctrl[4:4 + 64].view(torch.float32)
    thr2 = ctrl[4 + 64:].view(torch.float32)
    hi = torch.empty((M, V), dtype=torch.bfloat16, device=dev)
    lo = torch.empty((M, V), dtype=torch.bfloat16, device=dev) if gum is not None else hi
    BM = max(16, triton.next_power_of_2(M))
    dummy = ctrl
    if NS:
        margs = (mt["eidx"], mt["pos"], mt["min_lens"], mt["nstop"], mt["stop"], mt["stop"].stride(0))
    else:
        margs = (dummy, dummy, dummy, dummy, dummy, 0)
    k["xnorm"][(M,)](x, x.stride(0), xg, xn, ctrl, K=K, G=G, INFL=INFL_X, XABS=X_ABS, RN=_SQRT_RN)
    k["screen"][(triton.cdiv(V, BN),)](
        x, x.stride(0), st.q, st.e, st.rt, st.nv, xg, xn, hi, lo, thr, ctrl, M, V, st.vstart, *margs,
        K=K, G=G, FMT=st.fmt, STORE_LO=gum is not None, BM=BM, BN=BN, BK=BK, CACC2=2.0 * c_acc(K),
        INFL_Q=INFL_Q, INFL_B=INFL_B, SLOP_REL=SLOP_REL, SLOP_ABS=SLOP_ABS, NS=NS, num_warps=WARPS,
        num_stages=STAGES)
    sel_hi, sel_thr = hi, thr
    if gum is not None:
        vhi = torch.empty((M, V), dtype=torch.float32, device=dev)
        k["gbounds"][(M, triton.cdiv(V, 1024))](lo, hi, vhi, thr2, M, V, st.vstart, gum["temp"], gum["seeds"],
                                               gum["pos"], gum["imap"], BLOCK=1024, SLACK=DIV_SLACK,
                                               SLOP_ABS=SLOP_ABS)
        sel_hi, sel_thr = vhi, thr2
    ids = torch.zeros((cap,), dtype=torch.int32, device=dev)
    BS = max(128, 16384 // BM)
    k["select"][(triton.cdiv(V, BS),)](sel_hi, sel_thr, ctrl, ids, M, V, cap, BM=BM, BS=BS)
    st.host_ctrl.copy_(ctrl[:2], non_blocking=True)
    torch.cuda.current_stream().synchronize()                                           # SYNC 1
    count, flag = (int(v) for v in st.host_ctrl.tolist())
    val = torch.empty((M,), dtype=torch.float32, device=dev)
    idx = torch.empty((M,), dtype=torch.int64, device=dev)
    g = gum or {}

    def final(cand, ids_or_none, n):
        k["final"][(M,)](cand, cand.stride(0), ids_or_none if ids_or_none is not None else dummy, n, val, idx,
                         st.vstart, g.get("temp", dummy), g.get("seeds", dummy), g.get("pos", dummy),
                         g.get("imap", dummy), *margs, HAS_IDS=ids_or_none is not None, GUMBEL=gum is not None,
                         BLOCK=1024, NS=NS)

    if flag:
        val.fill_(float("-inf"))
        idx.fill_(st.vstart)
        return LocalResult(val, idx, ST_NONFINITE, count, None)
    if count > cap or count == 0:  # overflow (or no candidate at all, impossible for finite data): stock op, shard
        shard = mask_shard_(st.stock_full(x), st.vstart, mt)
        if gum is None:
            import glm_target_argmax as ta
            v, i = ta.local_argmax(shard)
            val.copy_(v)
            idx.copy_(i + st.vstart)
        else:
            final(shard.float(), None, V)
        return LocalResult(val, idx, ST_LOCAL_STOCK, count, "stock")
    kind, slice_k = refine
    if kind == "gemm":
        nb = bucket_for(count)
        wc = st.wbuf[:nb]
        torch.index_select(st.W, 0, ids[:nb].to(torch.int64), out=wc)  # ids past `count` are 0: a valid row
        cand = st.stock_gemm(x, wc).float()
        final(cand, ids, count)
    else:
        cand = torch.empty((M, count), dtype=torch.float32, device=dev)
        k["exact"][(triton.cdiv(count, 16), triton.cdiv(M, 16))](
            x, x.stride(0), st.W, ids, cand, cand.stride(0), M, count, K=K, BV=16, BMX=16, BKX=128,
            SLICE=slice_k, num_warps=4)
        final(cand, ids, count)
    return LocalResult(val, idx, ST_OK, count, kind)


# ---------------------------------------------------------------------------------------------------
# runner integration (wraps glm_target_argmax.plan / fast_sample)
# ---------------------------------------------------------------------------------------------------
class Stats:
    cert = 0          # steps that ran the certified path
    stock = 0         # eligible-shape steps passed through untouched (unqualified M, mode 0, ...)
    gumbel = 0
    mintok = 0        # certified steps with the min_tokens mask armed
    mintok_skip = 0   # min_tokens steps left on the glm_levers path (switch off / too many stop ids / patched head)
    local_stock = 0   # candidate overflow on this rank: stock op on the shard
    nonfinite = 0     # steps routed to the full stock sample() because some rank saw non-finite data
    union_sum = 0
    union_max = 0
    checked = 0
    mismatch_local = 0
    mismatch_out = 0
    dumped = 0


def _rank():
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001
        return 0


def _plan_gumbel(runner, input_batch, grammar_output, st):
    """("gumbel", 1) when every row is plain Gumbel-max (or greedy) on the Sampler path; decided from CPU state."""
    s = runner.sampler
    if grammar_output is not None or s is None or type(s).__name__ != "Sampler" or getattr(s, "compute_nans", False):
        return None
    if getattr(s, "use_fp64_gumbel", False) or "_requires_logits_processing" in vars(s):
        return None
    if input_batch.num_draft_tokens != 0 and runner.rejection_sampler is not None:
        return None
    idx = input_batch.idx_mapping_np
    if idx.size == 0 or (idx < 0).any() or idx.size > MAX_M or int(idx.size) not in REFINE_FOR:
        return None
    if int(input_batch.logits_indices.shape[0]) != int(input_batch.num_reqs):
        return None
    ss = s.sampling_states
    temp = ss.temperature.np[idx]
    if not np.all(np.isfinite(temp)) or np.any(temp < 0):
        return None
    if np.any(s.logit_bias_state.use_logit_bias[idx]) or np.any(s.penalties_state.use_penalty[idx]):
        return None
    if np.any(s.bad_words_state.num_bad_words.np[idx] > 0):
        return None
    tb = s.thinking_budget_state
    if getattr(tb, "enabled", False) and np.any(tb.use_thinking_budget[idx]):
        return None
    if np.any(ss.min_p.np[idx] != 0.0) or np.any(ss.top_k.np[idx] != ss.vocab_size) or np.any(ss.top_p.np[idx] != 1.0):
        return None
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
    if ss.max_num_logprobs(idx) != NO_LOGPROBS or s.logprob_token_ids_state.max_num_token_ids(idx) > 0:
        return None
    return ("gumbel", 1)


def plan_mintok(runner, input_batch):
    """None when the step has no glm_levers min_tokens admission; else (spec_or_None, why). Decided from TP-agreed
    config and per-request CPU state only (rank-invariant). spec is None when the certified path must not take the
    step (it then stays on the glm_levers masked stock-head path, which is exact)."""
    if not getattr(runner, "_glm_lv_mintok", False):
        return None
    if not _mintok_on():
        return None, "GLM_CERT_HEAD_MINTOK=0"
    lb = runner.sampler.logit_bias_state
    idx = input_batch.idx_mapping_np
    nmax = int(lb.num_stop_token_ids.np[idx].max()) if idx.size else 0
    if nmax > MINTOK_MAX_STOP:
        return None, f"{nmax} stop ids > GLM_CERT_HEAD_MINTOK_MAX_STOP"
    if bool((lb.num_allowed_token_ids.np[idx] > 0).any()) or bool((lb.num_logit_bias.np[idx] > 0).any()):
        return None, "allowed_token_ids / logit_bias"   # glm_levers.mintok_only never admits these; belt and braces
    pos = input_batch.positions[input_batch.logits_indices]
    return mintok_spec(input_batch.expanded_idx_mapping, pos, lb.min_lens.gpu, lb.num_stop_token_ids.gpu,
                       lb.stop_token_ids.gpu, nmax), "armed"


def _certified(runner, st, hidden_states, input_batch, kind, width, mt=None):
    """-> (SamplerOutput, num_sampled, num_rejected, local) or None when the step must take the stock sample()
    (decided identically on every rank from the gathered status column)."""
    import glm_target_argmax as ta
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    h = hidden_states[input_batch.logits_indices]
    M = int(h.shape[0])
    gum = None
    if kind == "gumbel":
        ss = runner.sampler.sampling_states
        gum = dict(temp=ss.temperature.gpu, seeds=ss.seeds.gpu,
                   pos=input_batch.positions[input_batch.logits_indices].contiguous(),
                   imap=input_batch.expanded_idx_mapping.contiguous())
    loc = cert_local(st, h, REFINE_FOR[M], gum, mt=mt)
    Stats.union_sum += loc.count
    Stats.union_max = max(Stats.union_max, loc.count)
    Stats.local_stock += loc.status == ST_LOCAL_STOCK
    pairs = ta.pack_pairs(loc.val, loc.idx)
    pairs[:, 2] = loc.status
    if st.tp > 1:
        gathered = tensor_model_parallel_all_gather(pairs, dim=-1)
    else:
        gathered = pairs
    g = gathered.view(M, st.tp, ta.PAIR_WIDTH)
    if bool((g[:, :, 2] >= ST_NONFINITE).any()):                                        # SYNC 2
        Stats.nonfinite += 1
        return None
    target = ta.reduce_pairs(gathered, st.tp) if st.tp > 1 else loc.idx
    num_reqs = input_batch.num_reqs
    if kind in ("sampler", "gumbel"):
        sampled = target.view(-1, 1)
        num_sampled = input_batch.seq_lens.new_ones(num_reqs)
    else:
        draft = input_batch.input_ids[input_batch.logits_indices]
        dead = ta._dead_rows(input_batch)
        if dead is not None:
            draft = draft.masked_fill(dead, -1)
        sampled, num_sampled = ta.greedy_verify(target, draft, input_batch.cu_num_logits, num_reqs, width)
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        num_sampled, input_batch.seq_lens, input_batch.cu_num_logits, input_batch.idx_mapping,
        runner.sampler.req_states.prefill_len.gpu)
    out = SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                        num_sampled=num_sampled, num_rejected=num_rejected)
    return out, num_sampled, num_rejected, (h, loc)


def _dump(h):
    if not DUMP or Stats.dumped >= DUMP_ROWS:
        return
    os.makedirs(DUMP, exist_ok=True)
    rows = h.detach().to("cpu")
    Stats.dumped += rows.shape[0]
    torch.save(rows, os.path.join(DUMP, f"rank{_rank()}-{Stats.dumped:08d}.pt"))


def _log_stats(rank0, force=False):
    n = Stats.cert + Stats.stock
    if not rank0 or not LOG_EVERY or (not force and n % LOG_EVERY):
        return
    s = Stats
    _log(f"steps certified={s.cert} (gumbel {s.gumbel}, min_tokens {s.mintok}) passed-through={s.stock} "
         f"(min_tokens left to glm_levers {s.mintok_skip}); local-stock (overflow) "
         f"{s.local_stock}, non-finite->stock {s.nonfinite}, union mean {s.union_sum / max(1, s.cert):.1f} max "
         f"{s.union_max}" + (f"; checked={s.checked} local-mismatch={s.mismatch_local} "
                             f"output-mismatch={s.mismatch_out}" if s.checked else ""))


# engine functions this module replaces or relies on: ast-extracted text, sha256[:16], Tony v11 image
# (vLLM 0.1.dev20051+g487ecf187; same scheme and helpers as glm_ds_hooks.py)
EXPECTED = {
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.load_model": "c95b75873f6c2797",
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.sample": "d58876dc84455870",
    "vllm.v1.worker.gpu.sample.sampler:Sampler.sample": "0a9891bf03af6e71",
    "vllm.v1.worker.gpu.sample.sampler:Sampler.apply_sampling_params": "56558b1afa204b45",
    "vllm.v1.worker.gpu.sample.sampler:Sampler._requires_logits_processing": "49aa83c34acadedd",
    "vllm.v1.worker.gpu.sample.states:SamplingStates.apply_temperature": "354a838bf3329f6b",
    "vllm.v1.worker.gpu.sample.gumbel:gumbel_block_argmax": "96bf9de0671d9e25",
    "vllm.v1.worker.gpu.sample.gumbel:tl_rand32": "13ea069c256c3051",
    "vllm.v1.worker.gpu.sample.gumbel:_temperature_kernel": "820268a50c07c039",
    "vllm.v1.worker.gpu.sample.thinking_budget:ThinkingBudgetState.apply": "8f00ad4f2c030e5f",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor._apply_head": "a3d5c2eec69e2448",
}


def check_sources() -> None:
    import glm_ds_hooks as h
    got = h.hashes_at(h.vllm_root(), EXPECTED)
    bad = {k: (v, got[k]) for k, v in EXPECTED.items() if got[k] != v}
    if bad and os.environ.get("GLM_CERT_HEAD_ALLOW_DRIFT", "0") != "1":
        lines = "\n".join(f"  {k}: expected {e}, image {g}" for k, (e, g) in sorted(bad.items()))
        raise RuntimeError("glm-cert-head: engine sources differ from the qualified image; refusing to patch "
                           f"(GLM_CERT_HEAD=0 runs stock, GLM_CERT_HEAD_ALLOW_DRIFT=1 overrides):\n{lines}")


def install(mod) -> None:
    """Order-independent: glm_target_argmax.install (glm_exact_hooks) may run before or after this; both only
    rebind module globals (ta.plan / ta.fast_sample) or wrap GPUModelRunner.sample, which reads them per call."""
    import glm_target_argmax as ta
    cls = mod.GPUModelRunner
    if getattr(cls, "_glm_cert_head", False):
        return

    def stock_sample(runner, *args):
        f = type(runner).sample
        return getattr(f, "__wrapped__", f)(runner, *args)

    orig_load = cls.load_model

    def load_model(self, *a, **kw):
        out = orig_load(self, *a, **kw)
        if MODE in _OFF:  # DUMP only: no tables, no agreement needed (dumping is rank-local and collective-free)
            self._glm_cert = "GLM_CERT_HEAD=0"
            return out
        st = None
        if not hasattr(type(self).sample, "__wrapped__"):
            st = "GPUModelRunner.sample is not hooked by glm_target_argmax (GLM_TARGET_VOCAB_ARGMAX=1 needed)"
        else:
            try:
                st = build_state(self)
            except Exception as e:  # noqa: BLE001  never take the boot down; the stock path stays
                st = f"build failed: {e!r}"
        ok, why = tp_agree(isinstance(st, State))  # every rank, success or not
        if not ok:
            if isinstance(st, State):
                st = f"disabled on every rank: {why}"
                torch.cuda.empty_cache()
            _log(f"disabled for this runner: {st}")
            self._glm_cert = st if isinstance(st, str) else "disabled"
            return out
        if (SHARE_DRAFT and st.fmt == 0 and st.draft is not None and hasattr(st.draft, "compute_candidates")
                and getattr(st.draft, "_glm_head_twin", None) is None):  # an NVFP4 draft head (glm_ds_draft_fp4) wins
            st.draft._glm_head_twin = ("mxint8", st.q, st.e)
            if hasattr(st.draft, "_glm_ds_head_fp8_twin"):
                del st.draft._glm_ds_head_fp8_twin
            _log("drafter head now reads the MXINT8 copy (fp8 twin dropped)")
        torch.cuda.empty_cache()
        _log(f"enabled on every TP rank ({why})")
        self._glm_cert = st
        return out

    orig_plan, orig_fast = ta.plan, ta.fast_sample

    def plan(runner, input_batch, grammar_output):
        res = orig_plan(runner, input_batch, grammar_output)
        st = getattr(runner, "_glm_cert", None)
        if res is None and SAMPLED and _mode() != "0" and isinstance(st, State):
            res = _plan_gumbel(runner, input_batch, grammar_output, st)
        return res

    def fast_sample(runner, hidden_states, input_batch, how):
        st = getattr(runner, "_glm_cert", None)
        mode = _mode()
        kind, width = how
        rank0 = _rank() == 0
        M = int(input_batch.logits_indices.shape[0])
        if DUMP:
            _dump(hidden_states[input_batch.logits_indices])
        # every input below is TP-agreed (st, mode, switches) or per-request CPU state: all ranks decide alike
        usable = (isinstance(st, State) and mode != "0" and M in REFINE_FOR and hidden_states.dtype == torch.bfloat16)
        mt = None
        if usable:
            mtp = plan_mintok(runner, input_batch)
            patched = vars(st.lp).get("_apply_head")
            if mtp is not None:
                mt, why = mtp
                # glm_levers may already have wrapped lp._apply_head with its stop mask (hook order): that wrapper
                # is the same rule we apply, so it is the only instance patch accepted
                if mt is not None and (kind == "gumbel" or (patched is not None
                                                            and not getattr(patched, "_glm_lv_mintok_mask", False))):
                    mt, why = None, "gumbel row / foreign lm_head patch"
                if mt is None:
                    Stats.mintok_skip += 1
                    usable = False
            elif patched is not None:
                usable = False
        if not usable:
            Stats.stock += 1
            if kind == "gumbel":  # plan admitted it, so this is a mode switch mid-run: the full stock sampler
                return stock_sample(runner, hidden_states, input_batch, None)
            return orig_fast(runner, hidden_states, input_batch, how)
        Stats.cert += 1
        Stats.gumbel += kind == "gumbel"
        Stats.mintok += mt is not None
        res = _certified(runner, st, hidden_states, input_batch, kind, width, mt)
        if mode == "check":
            ref = stock_sample(runner, hidden_states, input_batch, None)
            if res is not None:
                out, _, _, (h, loc) = res
                if kind != "gumbel":
                    shard = mask_shard_(st.lp._apply_head(st.lm_head, h, None), st.vstart, mt)
                    sv, si = ta.local_argmax(shard)
                    ok_local = torch.equal(si + st.vstart, loc.idx) and torch.equal(sv, loc.val)
                else:
                    ok_local = True
                a, b = out.sampled_token_ids, ref[0].sampled_token_ids
                live = torch.arange(b.shape[1], device=b.device)[None, :] < ref[1].to(torch.int64)[:, None]
                ok_out = (a.shape == b.shape and bool(((a == b) | ~live).all())
                          and torch.equal(out.num_sampled.to(torch.int64), ref[1].to(torch.int64)))
                Stats.checked += 1
                Stats.mismatch_local += not ok_local
                Stats.mismatch_out += not ok_out
                if (not ok_local or not ok_out) and rank0 and Stats.mismatch_local + Stats.mismatch_out <= 20:
                    _log(f"MISMATCH kind={kind} mintok={mt is not None} M={M} refine={loc.refine} count={loc.count} "
                         f"local_ok={ok_local} "
                         f"out_ok={ok_out} cert={a.tolist()} stock={b.tolist()}")
            _log_stats(rank0)
            return ref
        if res is None:  # non-finite somewhere: every rank takes the full stock sampler
            _log_stats(rank0)
            return stock_sample(runner, hidden_states, input_batch, None)
        _log_stats(rank0)
        return res[0], res[1], res[2]

    ta.plan, ta.fast_sample = plan, fast_sample
    cls.load_model = load_model
    cls._glm_cert_head = True
    _log(f"armed (mode={_mode()}, twin={TWIN}, sampled={SAMPLED}, mintok={_mintok_on()} (max stop "
         f"{MINTOK_MAX_STOP}), share_draft={SHARE_DRAFT}, qualified M "
         f"{sorted(REFINE_FOR) or 'none (stock everywhere)'})")


def register() -> None:
    """From sitecustomize, after the glm_ds_hooks block (reuses its after-import finder). Works in either order
    with glm_exact_hooks / glm_levers, which patch the same runner module."""
    if MODE in _OFF and not DUMP:
        return
    check_sources()
    import glm_ds_hooks
    glm_ds_hooks.after_import("vllm.v1.worker.gpu.model_runner", install)
