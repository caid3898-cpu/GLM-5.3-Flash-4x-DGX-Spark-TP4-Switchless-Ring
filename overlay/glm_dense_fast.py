# SPDX-License-Identifier: Apache-2.0
"""GLM_DENSE_FAST / GLM_DENSE_FAST_PREFILL: faster kernels for the dense 8-bit linears served by Marlin W8A16.

Image glm53-roce:rel0928 (vLLM 0.1.dev20051+g487ecf187, V2 model runner), SM121 (GB10), lossless8 weights: the
dense non-expert projections are 8-bit and weight-only on Marlin (bf16 activations, fp32 accumulation):

  shape            K     N(rank)  PN    fmt  calls/step  module (per rank, TP4)
  kda_in_proj      4096  6416     6464  mx   34          KDA self_attn.in_proj_qkvbfg_a  (MarlinMxfp8LinearKernel)
  kda_o            2048  4096     4096  mx   34          KDA self_attn.o_proj            (MarlinMxfp8LinearKernel)
  mla_qkv_a        4096  2048     2048  blk  11          MLA fused q_a | kv_a            (MarlinFP8ScaledMMLinearKernel)
  mla_q_b          1536  4096     4096  blk  11          MLA q_b_proj
  mla_o            4096  4096     4096  blk  11          MLA o_proj
  shared_gate_up   4096  1024     1024  blk  42          shared expert gate_up_proj
  shared_down      512   4096     4096  blk  42          shared expert down_proj
  (mx = MXFP8, e8m0 scales per 32 K; blk = block-128 FP8, bf16 scales carrying 2^120 per 128 K x 1 column)

Two paths, both dispatched per call by (shape, M = token rows) from overlay/glm_dense_fast_table.json:

  decode   (GLM_DENSE_FAST)          1 <= M <= 32, only the (shape, M bucket) entries of the table (M buckets
           <= 8 / <= 16 / <= 32, one tuned config each). Kernel: glmk dense_w8a16 (overlay/glm_dense_fast_csrc,
           copied unchanged from diagnostics/glm-cuda-kernels-20260928): reads the served Marlin buffers in place,
           the same MMA operands as Marlin (identical dequant bit construction + __hmul2 scale), fp32 accumulation,
           deterministic (fixed-order split-K, no data atomics), writes the unpadded [M, N] directly.
  prefill  (GLM_DENSE_FAST_PREFILL)  M >= the table's min_m for the shape (kda_in_proj, kda_o): unpack the Marlin
           buffers to one shared bf16 [K, PN] scratch with a Triton kernel (the exact Marlin operand, from
           diagnostics/glm-prefill-kernels-20260928/dense/w8a16_largem.py), then one cuBLAS bf16 GEMM with fp32
           accumulation (bf16 reduced-precision split-K reductions disabled for the call). The scratch is one buffer
           per device shared by every layer (the largest K*PN, 53 MB for kda_in_proj): a per-layer BF16 cache would
           cost 34 x 53 MB = 1.8 GB for kda_in_proj alone, to save the ~0.4 ms unpack per call. Never inside a CUDA
           graph (prefill steps are eager under FULL_DECODE_ONLY); a capturing call takes the stock path.
  Anything else (M in the gaps, a shape or M bucket not in the table, bias, non-bf16 input, K padding, a drafter
  layer, an unexpected scale dtype) calls the stock apply_weights.

Measured (tests/test_glm_dense_fast_gpu.py, GB10 Spark_04, real lossless8 rank-0 shards, 2026-09-28; decode = CUDA
graphs over cold weights, the whole stock apply incl. the unpad copy vs the whole fast apply):
  decode speedup 1.03-1.13x per covered (shape, M); ms/step saved over all covered layers (34/34/11/11/11/42/42
  calls): M=4 0.76, M=8 0.78, M=16 0.70, M=32 0.62. Prefill (eager): kda_in_proj 1.21x at M=1024, 1.49x at 2048,
  1.95x at 6912 (8.87 -> 4.56 ms per call); kda_o 1.14x at 4096, 1.23x at 6912 (below 4096 it is ~1.0x, stock);
  ms saved per prefill chunk: 29 at 2048 tokens, 157 at 6912. Load-time cost: 27 s first JIT build per node (then
  cached), 1.4 s self-test; memory: 50.5 MB prefill scratch + 0.8 MB split-K workspace per rank.

Numerics: NOT bitwise vs stock. The operands are identical; the fp32 summation order differs (split over warps /
CTAs here, Marlin's own order there; cuBLAS tiling for prefill), so ~0.1 % of outputs differ by one bf16 ulp
(rel L2 vs stock ~5e-5 to 1.3e-4, the same distance each has to an fp64 reference). Quality-gate it like any
numerics-changing lever: KLD probe + qeval, not a text hash.

Env (read once at install unless noted; identical on every rank):
  GLM_DENSE_FAST=0|1|<shapes>          decode path; <shapes> = comma list of shape names (e.g. kda_in_proj,kda_o).
                                       Per call through overlay/glm_ab.py when armed (kind "raw").
  GLM_DENSE_FAST_PREFILL=0|1|<shapes>  prefill path, same syntax; per call through glm_ab.
  GLM_DENSE_FAST_DECODE_MAX_M=N        cap the decode path at M <= N (default 32 = the table)
  GLM_DENSE_FAST_PREFILL_MIN_M=N       override every prefill entry's min_m
  GLM_DENSE_FAST_TABLE=path            dispatch table (default: glm_dense_fast_table.json next to this file)
  GLM_DENSE_FAST_BUILD_DIR=dir         JIT build cache root (default /cache/glm_dense_fast when /cache is writable,
                                       i.e. the persistent $OVERLAY_REMOTE/cache mount of start.sh)
  GLM_DENSE_FAST_DRAFT=1               also switch the drafter's matching layers (default: target model only)

Load time (BaseModelLoader.load_model wrapper, every rank, target model): tag the Marlin 8-bit layers whose shape
is in the table, JIT-build the CUDA extension (cached under the build dir, keyed by a hash of the sources, flags,
torch and CUDA versions), then self-test every (shape, M bucket) and prefill entry on the first real layer of the
shape against the stock Marlin call (finite, rel L2 <= 1e-3, |diff| <= 2 bf16 ulp + 1e-3 rms, and bit-identical
across two runs). The pass flags are MIN all-reduced over the TP CPU group together with a hash of the table, so an
entry is on for every rank or for none: a build failure, a failed check or a table mismatch on any rank turns the
affected entries off everywhere and logs it loudly (fail closed to stock; the boot continues).

CUDA graphs: the decode kernels are compiled ahead of time into the extension and every instance the table uses is
launched once eagerly in the self-test (module load happens there, not in a capture). No host sync on the call
path; the output is a torch.empty on the current stream (the graph pool inside a capture, as for the stock path);
the split-K workspace and ticket counters are allocated once at load, per device, and every call leaves the
tickets at zero. They are shared by all layers: correct because every dense linear runs on the one forward stream
(no two of these kernels are ever in flight at once).

Credits: Marlin (Elias Frantar, Dan Alistarh / IST-DASLab; Neural Magic; vLLM contributors, incl. vllm#24722) for
the weight layout, dequantisation and small-M MMA the kernel reuses; the 2026-09-28 GLM kernel studies for the
kernels, sweep and the verified unpack formulas.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "glm_dense_fast_csrc")
CSRC_FILES = ("glmk_dense.cu", "glm_dense_fast_bind.cpp", "glmk_common.cuh")
DEFAULT_TABLE = os.path.join(HERE, "glm_dense_fast_table.json")
MXFP8_MOD = "vllm.model_executor.kernels.linear.mxfp8.marlin"
FP8_MOD = "vllm.model_executor.kernels.linear.scaled_mm.marlin"
LOADER_MOD = "vllm.model_executor.model_loader.base_loader"
_OFF = ("", "0", "off", "false", "no")
TAG = "_glm_dense_fast"
DECODE_MAX_M = 32
BUCKETS = ((8, "1"), (16, "2"), (32, "4"))   # M <= 8 -> mg 1, <= 16 -> mg 2, <= 32 -> mg 4 (kernel row groups)
BUCKET_RANGE = {"1": (1, 8), "2": (9, 16), "4": (17, 32)}
# self-test tolerances vs the stock Marlin call (measured: rel ~5e-5..1.3e-4, 1 ulp flips)
TOL_REL = 1e-3
TOL_ULP = 2
TOL_RMS = 1e-3

S = {"fast": {}, "stock": {}, "logged": set(), "installed": set(), "report": None}
_STATE = {"table": None, "ext": None, "ws": {}, "scratch": {}, "enabled_dec": {}, "enabled_pre": {},
          "prepared_models": 0, "pre_tagged": 0}
_LOCK = threading.Lock()


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-dense-fast: {msg}\n")
    sys.stderr.flush()


def _once(key: str, msg: str) -> None:
    if key not in S["logged"]:
        S["logged"].add(key)
        _log(msg)


def env(name: str, default=None):
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


# ------------------------------------------------------------------------------------------------------------------
# switches
# ------------------------------------------------------------------------------------------------------------------
_PARSED: dict = {}


def parse_mode(value, known=None):
    """'0'/off -> frozenset(); '1'/on/all -> None (every shape); 'a,b' -> frozenset({'a','b'}). Unknown names raise
    when `known` is given."""
    raw = "" if value is None else str(value).strip().lower()
    hit = _PARSED.get((raw, known))
    if hit is not None or (raw, known) in _PARSED:
        return hit
    if raw in _OFF:
        out = frozenset()
    elif raw in ("1", "on", "true", "all"):
        out = None
    else:
        names = frozenset(x.strip() for x in raw.replace(";", ",").split(",") if x.strip())
        if known is not None:
            bad = sorted(names - set(known))
            if bad:
                raise ValueError(f"unknown shape name(s) {bad}; known {sorted(known)}")
        out = names
    _PARSED[(raw, known)] = out
    return out


def mode_on(value) -> bool:
    m = parse_mode(value)
    return m is None or bool(m)


def shape_on(var: str, name: str) -> bool:
    """Per call (glm_ab-aware): is the path `var` on for shape `name`?"""
    m = parse_mode(env(var, "0"))
    return m is None or name in m


def bucket(M: int):
    for hi, b in BUCKETS:
        if M <= hi:
            return b
    return None


def _int_env(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


# ------------------------------------------------------------------------------------------------------------------
# table
# ------------------------------------------------------------------------------------------------------------------
def load_table(path=None) -> dict:
    """{'shapes': {name: {K, PN, fmt, decode: {bucket: {cfg, ...}}, prefill: {min_m, ...}}}} + '_key' index and
    '_hash' (sha256 of the file bytes, part of the rank agreement)."""
    path = path or os.environ.get("GLM_DENSE_FAST_TABLE") or DEFAULT_TABLE
    with open(path, "rb") as f:
        raw = f.read()
    t = json.loads(raw)
    t["_hash"] = hashlib.sha256(raw).hexdigest()[:16]
    t["_path"] = path
    t["_key"] = {}
    for name, e in t["shapes"].items():
        if e["fmt"] not in ("mx", "blk"):
            raise ValueError(f"{name}: fmt {e['fmt']!r}")
        for b, d in e.get("decode", {}).items():
            if b not in BUCKET_RANGE:
                raise ValueError(f"{name}: decode bucket {b!r} not in {sorted(BUCKET_RANGE)}")
            c = d["cfg"]
            for k in ("tpw", "ks", "gs", "d"):
                if not isinstance(c.get(k), int) or c[k] < 1:
                    raise ValueError(f"{name} bucket {b}: cfg {c}")
            if c.get("ldg", 0) != 0:
                raise ValueError(f"{name} bucket {b}: ldg != 0 needs the GLMK_ALL_LDG build (not shipped)")
        t["_key"][shape_key(e["K"], e["PN"], e["fmt"] == "mx")] = name
    return t


def shape_key(K: int, PN: int, mx: bool) -> str:
    return f"{K}x{PN}x{'mx' if mx else 'blk'}"


def entries(table: dict) -> list:
    """Fixed, ordered list of switchable entries: ('dec', name, bucket) and ('pre', name, None)."""
    out = []
    for name in sorted(table["shapes"]):
        e = table["shapes"][name]
        for b in sorted(e.get("decode", {})):
            out.append(("dec", name, b))
        if e.get("prefill"):
            out.append(("pre", name, None))
    return out


def table():
    if _STATE["table"] is None:
        _STATE["table"] = load_table()
    return _STATE["table"]


# ------------------------------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------------------------------
def build_root() -> str:
    root = os.environ.get("GLM_DENSE_FAST_BUILD_DIR", "").strip()
    if root:
        return root
    if os.path.isdir("/cache") and os.access("/cache", os.W_OK):
        return "/cache/glm_dense_fast"
    return os.path.join(os.path.expanduser("~"), ".cache", "glm_dense_fast")


def _nv_include_dirs() -> list:
    out = []
    try:
        import nvidia
        for base in getattr(nvidia, "__path__", []):
            for sub in ("cu13/include", "cu12/include"):
                p = os.path.join(base, sub)
                if os.path.isdir(p):
                    out.append(p)
    except ImportError:
        pass
    return out


def build_flags(after_dirs):
    # torch's ATen/cuda headers include cusparse.h, shipped in this image only in the pip nvidia wheels: search them
    # last so nvcc's own runtime headers still win. nvcc has no -idirafter: hand it to the host compiler.
    cflags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-lineinfo"]
    for d in after_dirs:
        cflags += ["-Xcompiler", "-idirafter," + d]
    cxx = ["-O3", "-std=c++17"]
    for d in after_dirs:
        cxx += ["-idirafter", d]
    return cflags, cxx


def build_tag(arch: str, cflags, cxx) -> str:
    import torch
    h = hashlib.sha256()
    for f in CSRC_FILES:
        with open(os.path.join(CSRC, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read())
    h.update(json.dumps([arch, cflags, cxx, torch.__version__, torch.version.cuda]).encode())
    return h.hexdigest()[:12]


def _clear_stale_lock(bdir: str, max_age_s: float = 900.0) -> None:
    """torch's FileBaton waits forever on a lock left by a build that was killed (container stop mid-boot)."""
    lock = os.path.join(bdir, "lock")
    try:
        age = time.time() - os.path.getmtime(lock)
    except OSError:
        return
    if age > max_age_s:
        try:
            os.remove(lock)
            _log(f"removed stale build lock {lock} ({age:.0f} s old)")
        except OSError as exc:
            _log(f"could not remove stale build lock {lock}: {exc!r}")


def load_ext(verbose: bool = False):
    if _STATE["ext"] is not None:
        return _STATE["ext"]
    with _LOCK:
        if _STATE["ext"] is not None:
            return _STATE["ext"]
        from torch.utils.cpp_extension import load
        srcs = [os.path.join(CSRC, f) for f in CSRC_FILES if not f.endswith(".cuh")]
        cflags, cxx = build_flags(_nv_include_dirs())
        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        last = None
        t0 = time.time()
        try:
            for arch in (os.environ.get("GLM_DENSE_FAST_ARCH", "12.1a"), "12.1"):
                bdir = os.path.join(build_root(), f"glm_dense_fast_ext-{build_tag(arch, cflags, cxx)}")
                os.makedirs(bdir, exist_ok=True)
                _clear_stale_lock(bdir)
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
                try:
                    ext = load(name="glm_dense_fast_ext", sources=srcs, extra_cuda_cflags=cflags, extra_cflags=cxx,
                               build_directory=bdir, verbose=verbose)
                    break
                except (ValueError, RuntimeError) as exc:  # arch string unknown to this torch -> plain 12.1
                    last = exc
                    if "arch" not in str(exc).lower():
                        raise
            else:
                raise last
        finally:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
        _log(f"extension ready in {time.time() - t0:.1f} s ({bdir})")
        _STATE["ext"] = ext
        return ext


# ------------------------------------------------------------------------------------------------------------------
# kernels
# ------------------------------------------------------------------------------------------------------------------
def mg_of(M: int) -> int:
    return 1 if M <= 8 else (2 if M <= 16 else 4)


def ws_need(cfg: dict, PN: int, mg: int):
    """(fp32 workspace elements, int32 ticket counters) a call needs; (0, 0) without split over CTAs."""
    if cfg["gs"] <= 1:
        return 0, 0
    return cfg["gs"] * 8 * mg * PN, -(-(PN // 64) // cfg["tpw"])


def _ws(device):
    return _STATE["ws"][str(device)]


def alloc_workspaces(device, need_ws: int, need_cnt: int) -> None:
    import torch
    key = str(device)
    cur = _STATE["ws"].get(key)
    if cur is not None and cur[0].numel() >= max(need_ws, 1) and cur[1].numel() >= max(need_cnt, 1):
        return
    ws = torch.empty(max(need_ws, 1), dtype=torch.float32, device=device)
    cnt = torch.zeros(max(need_cnt, 1), dtype=torch.int32, device=device)
    if cur is not None:
        _STATE.setdefault("retired", []).append(cur)  # a captured graph may still reference the old buffers
    _STATE["ws"][key] = (ws, cnt)


def decode_gemm(x2, weight, scales, N: int, mx: bool, cfg: dict):
    """x2 [M, K] bf16 (M <= 32) -> [M, N] bf16 via the glmk decode kernel."""
    import torch
    ext = _STATE["ext"]
    if x2.stride(-1) != 1 or x2.stride(0) % 2 or x2.data_ptr() % 4:
        x2 = x2.contiguous()
    out = torch.empty((x2.shape[0], N), dtype=torch.bfloat16, device=x2.device)
    ws, cnt = _ws(x2.device)
    ext.dense_w8a16(x2, weight, scales.view(torch.uint8) if mx else scales, out, ws, cnt, mx, N,
                    cfg["tpw"], cfg["ks"], cfg["gs"], cfg["d"], cfg.get("xp", 0), 0)
    return out


_TRITON = {}


def _unpack_kernel():
    if "k" in _TRITON:
        return _TRITON["k"]
    import triton
    import triton.language as tl

    @triton.jit
    def _e4m3(q):
        """byte (int32 0..255) -> fp32, Marlin's reading (0x7f / 0xff are finite 480 / -480)."""
        e = (q >> 3) & 15
        m = (q & 7).to(tl.float32)
        p2 = ((e + 117) << 23).to(tl.float32, bitcast=True)          # 2^(e-10), exact
        mag = tl.where(e == 0, m * 0.001953125, (8.0 + m) * p2)
        return tl.where((q & 128) != 0, -mag, mag)

    @triton.jit
    def unmarlin8_kernel(w_ptr, s_ptr, out_ptr, stride_ok,
                         PN: tl.constexpr, MX: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        """out[k, n] (bf16, [K, PN] row-major) = the Marlin mma operand of logical (k, n). Layout (8-bit, not a8,
        gptq_marlin_repack): a k16 row of [K, PN] is 4*PN int32 words; in a 64-column tile (k, n) -> word
        nt*256 + tc*32 + tr*8 + w*2 + hn, byte hk + 2*kb (tc = n%8, w = (n%64)//16, hn = (n%16)//8, tr = (k%8)//2,
        kb = k%2, hk = (k%16)//8). Scale of (group g, n): block g*PN + 64*(n//64) + 8*tc + 2*w + hn; MX
        8*tc + 4*(w//2) + 2*hn + w%2 in the tile (validated in diagnostics/glm-prefill-kernels-20260928)."""
        pid_k = tl.program_id(0)
        pid_n = tl.program_id(1)
        ROWW: tl.constexpr = 4 * PN
        k = pid_k * BK + tl.arange(0, BK)
        n = pid_n * BN + tl.arange(0, BN)
        kt = k // 16
        hk = (k // 8) % 2
        tr = (k // 2) % 4
        kb = k % 2
        nt = n // 64
        w = (n // 16) % 4
        hn = (n // 8) % 2
        tc = n % 8
        woff = (kt * ROWW + tr * 8)[:, None] + (nt * 256 + tc * 32 + w * 2 + hn)[None, :]
        byte = (tl.load(w_ptr + woff) >> (8 * (hk + 2 * kb))[:, None]) & 255
        if MX:
            col = nt * 64 + 8 * tc + 4 * (w // 2) + 2 * hn + (w % 2)
            sb = tl.load(s_ptr + ((kt // 2) * PN)[:, None] + col[None, :]).to(tl.int32)
            sv = (sb << 23).to(tl.float32, bitcast=True)                # 2^(e-127); e == 0 -> 0.0
        else:
            col = nt * 64 + 8 * tc + 2 * w + hn
            sv = tl.load(s_ptr + ((kt // 8) * PN)[:, None] + col[None, :]).to(tl.float32) * 7.52316384526264e-37
        tl.store(out_ptr + k[:, None].to(tl.int64) * stride_ok + n[None, :], (_e4m3(byte) * sv).to(tl.bfloat16))

    _TRITON["k"] = unmarlin8_kernel
    return unmarlin8_kernel


def unpack_marlin8(weight, scales, mx: bool, scratch):
    """Served Marlin int32 [K/16, 4*PN] + scales -> bf16 [K, PN] view of `scratch` (the exact Marlin operand)."""
    import torch
    kt, words = weight.shape
    K, PN = kt * 16, words // 4
    BK, BN = 64, (128 if PN % 128 == 0 else 64)
    wt = scratch[:K * PN].view(K, PN)
    s = scales.view(torch.uint8) if mx else scales
    _unpack_kernel()[(K // BK, PN // BN)](weight, s, wt, wt.stride(0), PN=PN, MX=mx, BK=BK, BN=BN, num_warps=4)
    return wt


def prefill_gemm(x2, weight, scales, N: int, mx: bool):
    import torch
    wt = unpack_marlin8(weight, scales, mx, _STATE["scratch"][str(x2.device)])
    m = torch.backends.cuda.matmul
    prev = m.allow_bf16_reduced_precision_reduction
    m.allow_bf16_reduced_precision_reduction = False
    try:
        return torch.matmul(x2, wt[:, :N])
    finally:
        m.allow_bf16_reduced_precision_reduction = prev


# ------------------------------------------------------------------------------------------------------------------
# layers
# ------------------------------------------------------------------------------------------------------------------
class Tag:
    """Per-shape dispatch record shared by every tagged layer of that shape."""
    __slots__ = ("name", "K", "N", "PN", "mx", "sattr", "dec", "dec_max_m", "pre_min_m")

    def __init__(self, name, K, N, PN, mx, sattr):
        self.name, self.K, self.N, self.PN, self.mx, self.sattr = name, K, N, PN, mx, sattr
        self.dec = {}          # bucket -> cfg (agreed-enabled entries only)
        self.dec_max_m = 0
        self.pre_min_m = 1 << 62

    def __repr__(self):
        return (f"Tag({self.name} K={self.K} N={self.N} PN={self.PN} {'mx' if self.mx else 'blk'} "
                f"dec={sorted(self.dec)} max_m={self.dec_max_m} pre_min_m={self.pre_min_m})")


def layer_operands(layer):
    """(weight, scales, mx, scale attribute name) of a Marlin 8-bit layer, or None. MX: weight_scale float8_e8m0fnu/uint8 [K/32, PN];
    block: weight_scale_inv (or weight_scale) bf16 [K/128, PN] carrying 2^120 (fp8_fused_exponent_bias_into_scales)."""
    import torch
    w = getattr(layer, "weight", None)
    if w is None or w.dtype != torch.int32 or w.dim() != 2:
        return None
    K, PN = w.shape[0] * 16, w.shape[1] // 4
    for attr in ("weight_scale", "weight_scale_inv"):
        s = getattr(layer, attr, None)
        if s is None or not isinstance(s, torch.Tensor) or s.dim() != 2 or s.shape[1] != PN:
            continue
        if s.element_size() == 1 and s.shape[0] * 32 == K:
            return w, s, True, attr
        if s.dtype == torch.bfloat16 and s.shape[0] * 128 == K:
            return w, s, False, attr
    return None


def classify(layer, tbl, require_cuda: bool = True) -> Tag | None:
    ops = layer_operands(layer)
    if ops is None or getattr(layer, "bias", None) is not None or (require_cuda and not ops[0].is_cuda):
        return None
    w, s, mx, sattr = ops
    K, PN = w.shape[0] * 16, w.shape[1] // 4
    N = getattr(layer, "output_size_per_partition", None)
    k_in = getattr(layer, "input_size_per_partition", None)
    if N is None or k_in != K or not 1 <= N <= PN or PN % 64:
        return None
    name = tbl["_key"].get(shape_key(K, PN, mx))
    if name is None:
        return None
    return Tag(name, K, N, PN, mx, sattr)


def scan(model, tbl, require_cuda: bool = True) -> list:
    """[(module name, module, Tag)] for the Marlin 8-bit linears whose shape is in the table."""
    out = []
    for mname, mod in model.named_modules():
        if TAG in mod.__dict__:
            continue
        t = classify(mod, tbl, require_cuda)
        if t is not None:
            out.append((mname, mod, t))
    return out


# ------------------------------------------------------------------------------------------------------------------
# self-test + rank agreement
# ------------------------------------------------------------------------------------------------------------------
def close_to(a, b) -> dict:
    """a (candidate) vs b (stock), bf16 tensors: finite, rel L2, and the per-element |a-b| <= TOL_ULP ulp(max) +
    TOL_RMS * rms(b) test. -> dict(ok, rel, bad, eq)"""
    import torch
    af, bf = a.float(), b.float()
    finite = bool(torch.isfinite(af).all())
    diff = (af - bf).abs()
    rms = float(bf.pow(2).mean().sqrt())
    mag = torch.maximum(af.abs(), bf.abs()).clamp_min(1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(mag)) - 7)            # bf16 ulp at the larger magnitude
    bad = int((diff > TOL_ULP * ulp + TOL_RMS * rms).sum())
    rel = float(diff.norm() / bf.norm().clamp_min(1e-30))
    eq = float((a.view(torch.int16) == b.view(torch.int16)).float().mean()) if a.shape == b.shape else 0.0
    return dict(ok=finite and bad == 0 and rel <= TOL_REL, rel=rel, bad=bad, eq=round(eq, 6), finite=finite)


def test_Ms(bucket_id: str, max_m: int) -> list:
    lo, hi = BUCKET_RANGE[bucket_id]
    hi = min(hi, max_m)
    return sorted({lo, hi}) if lo <= hi else []


def self_test_entry(kind, tag, layer, stock_fn, cfg=None, min_m=None, max_m=DECODE_MAX_M, seed=1234) -> tuple:
    """Run one entry on a real layer; -> (ok, detail str). stock_fn(x2) is the stock Marlin call for the layer."""
    import torch
    w, s, mx, _ = layer_operands(layer)
    g = torch.Generator(device="cpu").manual_seed(seed)
    Ms = test_Ms(kind[2], max_m) if kind[0] == "dec" else [min_m]
    worst = None
    for M in Ms:
        x = (torch.randn((M, tag.K), generator=g) * 0.5).to(torch.bfloat16).to(w.device)
        ref = stock_fn(x)
        if kind[0] == "dec":
            a = decode_gemm(x, w, s, tag.N, mx, cfg)
            b2 = decode_gemm(x, w, s, tag.N, mx, cfg)
        else:
            a = prefill_gemm(x, w, s, tag.N, mx)
            b2 = prefill_gemm(x, w, s, tag.N, mx)
        r = close_to(a, ref)
        det = torch.equal(a, b2)
        if not r["ok"] or not det:
            return False, f"M={M} {r} deterministic={det}"
        if worst is None or r["rel"] > worst[1]["rel"]:
            worst = (M, r)
    if w.is_cuda:
        torch.cuda.synchronize()
    return True, (f"M={Ms} worst rel {worst[1]['rel']:.2e} eq {worst[1]['eq']:.4f}" if worst else "no M")


def _tp():
    try:
        from vllm.distributed.parallel_state import get_tp_group
        g = get_tp_group()
        return g.cpu_group, g.world_size, g.rank_in_group
    except Exception:  # noqa: BLE001
        return None, 1, 0


def agree(flags: list, table_hash: str, group=None, world: int = 1) -> list:
    """Elementwise MIN of the pass flags over the TP CPU group, plus a table-hash equality check (a mismatch
    turns every entry off on every rank). Single process: returns flags."""
    if group is None or world <= 1:
        return list(flags)
    import torch
    import torch.distributed as dist
    h = int(table_hash, 16) & ((1 << 62) - 1)
    t = torch.tensor([h, -h] + [int(f) for f in flags], dtype=torch.int64)
    lo = t.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
    hi = t.clone()
    dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
    if int(lo[0]) != int(hi[0]):
        _log("RANKS DISAGREE on the dispatch table (hash differs): every entry OFF, stock everywhere")
        return [0] * len(flags)
    return [int(v) for v in lo[2:]]


def _stock_call(layer, mx):
    """The stock Marlin call for this layer's operands (the functions the stock apply_weights ends in)."""
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mu
    w, s, _, _ = layer_operands(layer)
    N, K = layer.output_size_per_partition, layer.input_size_per_partition
    ws = layer.workspace
    if mx:
        return lambda x: mu.apply_mxfp8_marlin_linear(input=x, weight=w, weight_scale=s, workspace=ws, size_n=N,
                                                      size_k=K)
    return lambda x: mu.apply_fp8_marlin_linear(input=x, weight=w, weight_scale=s, workspace=ws, size_n=N, size_k=K,
                                                bias=None)


def _self_test_all(ents, wanted, first, tbl, flags, details, max_m, pre_override, rank, stock_call, build):
    import torch
    build_ok = True
    if any(w and k[0] == "dec" and k[1] in first for w, k in zip(wanted, ents)):
        try:
            (build or load_ext)()
        except Exception as exc:  # noqa: BLE001
            build_ok = False
            _log(f"BUILD FAILED on rank {rank}, decode path OFF on every rank: {exc!r}"[:2000])
    for i, kind in enumerate(ents):
        if not wanted[i] or kind[1] not in first:
            continue
        mname, mod, tag = first[kind[1]]
        e = tbl["shapes"][kind[1]]
        dev = mod.weight.device
        try:
            if kind[0] == "dec":
                if not build_ok:
                    flags[i] = 0
                    details[kind] = "build failed"
                    continue
                cfg = e["decode"][kind[2]]["cfg"]
                nws, ncnt = ws_need(cfg, tag.PN, int(kind[2]))
                alloc_workspaces(dev, nws, ncnt)
                ok, det = self_test_entry(kind, tag, mod, stock_call(mod, tag.mx), cfg=cfg, max_m=max_m)
            else:
                min_m = pre_override or int(e["prefill"]["min_m"])
                key = str(dev)
                need = tag.K * tag.PN
                cur = _STATE["scratch"].get(key)
                if cur is None or cur.numel() < need:
                    _STATE["scratch"][key] = torch.empty(need, dtype=torch.bfloat16, device=dev)
                ok, det = self_test_entry(kind, tag, mod, stock_call(mod, tag.mx), min_m=min_m)
        except Exception as exc:  # noqa: BLE001
            ok, det = False, f"exception {exc!r}"[:600]
        flags[i] = 1 if ok else 0
        details[kind] = det
        if not ok:
            _log(f"SELF-TEST FAILED rank {rank} {kind} on {mname}: {det}")


def prepare(cands: list, *, group=None, world: int = 1, rank: int = 0, stock_call=_stock_call,
            build=None) -> dict:
    """Collective (all TP ranks, same model structure): build, self-test, agree, then tag the layers of the
    agreed entries. cands = scan(model). Returns a report."""
    tbl = table()
    dec_mode = parse_mode(os.environ.get("GLM_DENSE_FAST", "0"), tuple(tbl["shapes"]))
    pre_mode = parse_mode(os.environ.get("GLM_DENSE_FAST_PREFILL", "0"), tuple(tbl["shapes"]))
    max_m = min(_int_env("GLM_DENSE_FAST_DECODE_MAX_M", DECODE_MAX_M), DECODE_MAX_M)
    pre_override = _int_env("GLM_DENSE_FAST_PREFILL_MIN_M", 0)
    ents = entries(tbl)
    first = {}
    for mname, mod, tag in cands:
        first.setdefault(tag.name, (mname, mod, tag))
    wanted = []
    for kind in ents:
        mode = dec_mode if kind[0] == "dec" else pre_mode
        on = mode is None or kind[1] in mode
        if on and kind[0] == "dec" and not test_Ms(kind[2], max_m):
            on = False
        wanted.append(on)
    flags = [1] * len(ents)   # neutral for entries that are off or have no layer here
    details = {}
    try:
        _self_test_all(ents, wanted, first, tbl, flags, details, max_m, pre_override, rank, stock_call, build)
    except Exception as exc:  # noqa: BLE001  (never skip the collective below)
        _log(f"PREPARE FAILED on rank {rank}, every entry OFF on every rank: {exc!r}"[:2000])
        flags = [0] * len(ents)
    agreed = agree(flags, tbl["_hash"], group, world)
    # tag layers
    by_name: dict = {}
    for i, kind in enumerate(ents):
        if not wanted[i] or not agreed[i] or kind[1] not in first:
            continue
        e = tbl["shapes"][kind[1]]
        rec = by_name.setdefault(kind[1], {"dec": {}, "pre": None})
        if kind[0] == "dec":
            rec["dec"][kind[2]] = dict(e["decode"][kind[2]]["cfg"])
        else:
            rec["pre"] = pre_override or int(e["prefill"]["min_m"])
    counts: dict = {}
    shared: dict = {}
    for mname, mod, tag in cands:
        rec = by_name.get(tag.name)
        if rec is None:
            continue
        key = (tag.name, tag.N)
        t = shared.get(key)
        if t is None:
            t = Tag(tag.name, tag.K, tag.N, tag.PN, tag.mx, tag.sattr)
            t.dec = rec["dec"]
            t.dec_max_m = max((min(BUCKET_RANGE[b][1], max_m) for b in t.dec), default=0)
            t.pre_min_m = rec["pre"] if rec["pre"] else 1 << 62
            shared[key] = t
        mod.__dict__[TAG] = t
        counts[tag.name] = counts.get(tag.name, 0) + 1
    _STATE["prepared_models"] += 1
    _STATE["pre_tagged"] += sum(n for name, n in counts.items() if by_name[name]["pre"])
    if not _STATE["pre_tagged"]:
        _STATE["scratch"].clear()                 # no prefill entry survived anywhere: give the scratch back
    off = [k for i, k in enumerate(ents) if wanted[i] and not agreed[i] and k[1] in first]
    report = {"layers": counts, "tags": {k[0]: repr(v) for k, v in shared.items()}, "off": off,
              "details": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in details.items()}, "table": tbl["_hash"]}
    if rank == 0 or off:
        for name, n in sorted(counts.items()):
            t = next(v for k, v in shared.items() if k[0] == name)
            _log(f"rank {rank}: {name}: {n} layers, decode buckets {sorted(t.dec)} (M <= {t.dec_max_m}), "
                 f"prefill {'M >= %d' % t.pre_min_m if t.pre_min_m < (1 << 62) else 'off'}")
        if off:
            _log(f"rank {rank}: entries OFF after the self-test / rank agreement (stock path): {off}")
        _log(f"rank {rank}: table {tbl['_hash']} ({tbl['_path']}); self-test {report['details']}")
    S["report"] = report
    return report


# ------------------------------------------------------------------------------------------------------------------
# dispatch
# ------------------------------------------------------------------------------------------------------------------
def _count(path, name):
    d = S[path]
    d[name] = d.get(name, 0) + 1


def fast_apply(layer, x, bias):
    """The fast result for this call, or None for the stock path."""
    tag = layer.__dict__.get(TAG)
    if tag is None:
        if not _STATE["prepared_models"]:
            _once("noprep", "WARNING: a Marlin 8-bit call arrived but no model was prepared (the load_model hook did "
                            "not run?): every call stays on stock")
        return None
    if bias is not None:
        return None
    import torch
    if x.dtype != torch.bfloat16 or x.shape[-1] != tag.K:
        return None
    x2 = x.reshape(-1, tag.K)
    M = x2.shape[0]
    if 1 <= M <= tag.dec_max_m:
        cfg = tag.dec.get(bucket(M))
        if cfg is None or not shape_on("GLM_DENSE_FAST", tag.name):
            return None
        out = decode_gemm(x2, layer.weight, getattr(layer, tag.sattr), tag.N, tag.mx, cfg)
        _count("fast", tag.name)
        _once(f"dec:{tag.name}", f"decode kernel live: {tag.name} M={M} cfg={cfg}")
    elif M >= tag.pre_min_m:
        if not shape_on("GLM_DENSE_FAST_PREFILL", tag.name) or (x.is_cuda and torch.cuda.is_current_stream_capturing()):
            return None
        out = prefill_gemm(x2, layer.weight, getattr(layer, tag.sattr), tag.N, tag.mx)
        _count("fast", tag.name + ":prefill")
        _once(f"pre:{tag.name}", f"prefill unpack+cuBLAS live: {tag.name} M={M}")
    else:
        return None
    return out.reshape(x.shape[:-1] + (tag.N,))


def install_mxfp8(mod) -> None:
    cls = mod.MarlinMxfp8LinearKernel
    if getattr(cls, "_glm_dense_fast", False):
        return
    orig = cls.apply_weights

    def apply_weights(self, layer, x, bias=None):
        out = fast_apply(layer, x, bias)
        if out is not None:
            return out
        return orig(self, layer, x, bias)

    apply_weights.__wrapped__ = orig
    apply_weights.__doc__ = orig.__doc__
    cls.apply_weights = apply_weights
    cls._glm_dense_fast = True
    S["installed"].add("mxfp8")
    _log("MarlinMxfp8LinearKernel.apply_weights -> table dispatch (MXFP8 layers)")


def install_fp8(mod) -> None:
    cls = mod.MarlinFP8ScaledMMLinearKernel
    if getattr(cls, "_glm_dense_fast", False):
        return
    orig = cls.apply_weights

    def apply_weights(self, layer, x, bias=None):
        # block-FP8 only (bf16 scales carrying 2^120 per 128 K), weight-only (no W8A8 input quantisation)
        if getattr(self, "block_quant", False) and getattr(self, "marlin_input_dtype", None) is None:
            out = fast_apply(layer, x, bias)
            if out is not None:
                return out
        return orig(self, layer, x, bias)

    apply_weights.__wrapped__ = orig
    apply_weights.__doc__ = orig.__doc__
    cls.apply_weights = apply_weights
    cls._glm_dense_fast = True
    S["installed"].add("fp8")
    _log("MarlinFP8ScaledMMLinearKernel.apply_weights -> table dispatch (block-FP8 layers)")


def _is_target(model, vllm_config, model_config) -> bool:
    try:
        if model_config is vllm_config.model_config:
            return True
        if getattr(model_config, "model", None) != getattr(vllm_config.model_config, "model", None):
            return False
    except AttributeError:
        pass
    n = type(model).__name__.lower()
    return not any(s in n for s in ("dflash", "eagle", "draft", "mtp", "medusa"))


def install_loader(mod) -> None:
    cls = mod.BaseModelLoader
    if getattr(cls, "_glm_dense_fast", False):
        return
    orig = cls.load_model

    def load_model(self, vllm_config, model_config, prefix: str = ""):
        model = orig(self, vllm_config, model_config, prefix)
        target = _is_target(model, vllm_config, model_config)
        if not target and os.environ.get("GLM_DENSE_FAST_DRAFT", "0").strip().lower() in _OFF:
            _log(f"{type(model).__name__}: drafter, left on stock (GLM_DENSE_FAST_DRAFT=1 to include)")
            return model
        group, world, rank = _tp()
        t0 = time.time()
        # Collective: every rank loads the same model; an exception here would leave peers in the all-reduce,
        # so everything that can fail is caught inside prepare() and turned into an OFF flag.
        rep = prepare(scan(model, table()), group=group, world=world, rank=rank)
        if rank == 0:
            _log(f"{type(model).__name__}: prepared in {time.time() - t0:.1f} s: {rep['layers']}")
        return model

    load_model.__wrapped__ = orig
    cls.load_model = load_model
    cls._glm_dense_fast = True
    S["installed"].add("loader")


HOOKS = {MXFP8_MOD: install_mxfp8, FP8_MOD: install_fp8, LOADER_MOD: install_loader}


def register() -> None:
    import importlib.abc
    import importlib.util

    dec = os.environ.get("GLM_DENSE_FAST", "0")
    pre = os.environ.get("GLM_DENSE_FAST_PREFILL", "0")
    if not (mode_on(dec) or mode_on(pre)):
        return
    try:
        # Fatal before vLLM starts (an exception in sitecustomize would only print and run stock): a bad table,
        # an unknown shape name in the env or in any GLM_AB_V<i> spec, a non-integer cap.
        tbl = table()
        values = [dec, pre]
        ab = sys.modules.get("glm_ab")
        if ab is not None and getattr(ab, "ACTIVE", False):
            for i in range(ab.N):
                values += [ab.env_for(i, "GLM_DENSE_FAST", "0"), ab.env_for(i, "GLM_DENSE_FAST_PREFILL", "0")]
        for v in values:
            parse_mode(v, tuple(tbl["shapes"]))
        _int_env("GLM_DENSE_FAST_DECODE_MAX_M", DECODE_MAX_M)
        _int_env("GLM_DENSE_FAST_PREFILL_MIN_M", 0)
    except Exception as exc:  # noqa: BLE001
        msg = f"glm-dense-fast: refusing to start: {exc!r}"
        print(msg, flush=True)
        sys.stderr.write(msg + "\n")
        os._exit(1)
    _log(f"armed: GLM_DENSE_FAST={dec!r} GLM_DENSE_FAST_PREFILL={pre!r} table {tbl['_hash']} "
         f"({len(entries(tbl))} entries)")
    for name, fn in HOOKS.items():
        if name in sys.modules:
            fn(sys.modules[name])

    class _Finder(importlib.abc.MetaPathFinder):
        _glm_dense_fast = True

        def find_spec(self, name, path, target=None):
            if name not in HOOKS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec, _fn=HOOKS[name]):
                _orig(module)
                _fn(module)
            spec.loader.exec_module = exec_module
            return spec

    if not any(getattr(f, "_glm_dense_fast", False) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


# ------------------------------------------------------------------------------------------------------------------
# table generation (offline): python3 overlay/glm_dense_fast.py --make-table BEST_CONFIGS RESULTS DENSE_PREFILL
# ------------------------------------------------------------------------------------------------------------------
SHAPES = {  # name -> (K, PN, mx, calls per decode step) ; from the rank-0 trace of 2026-09-28
    "kda_in_proj": (4096, 6464, True, 34),
    "kda_o": (2048, 4096, True, 34),
    "mla_qkv_a": (4096, 2048, False, 11),
    "mla_q_b": (1536, 4096, False, 11),
    "mla_o": (4096, 4096, False, 11),
    "shared_gate_up": (4096, 1024, False, 42),
    "shared_down": (512, 4096, False, 42),
}
PREFILL_ALIAS = {"mla_qkv_a": "mla_qa_kva"}  # the prefill study's name for the same projection
MIN_DECODE_SPEEDUP = 1.02    # every measured M of a bucket must beat stock by this (op vs op, cold weights)
MIN_PREFILL_SPEEDUP = 1.10   # min_m = smallest measured M from which every larger measured M beats stock by this


def make_table(best_configs: dict, results: list, prefill_rows: list, sources: dict) -> dict:
    shapes = {}
    for name, (K, PN, mx, calls) in SHAPES.items():
        e = {"K": K, "PN": PN, "fmt": "mx" if mx else "blk", "calls_per_step": calls, "decode": {}}
        skipped = {}
        for b, (lo, hi) in BUCKET_RANGE.items():
            cfg = best_configs.get("dense", {}).get(f"{K}x{PN}x{'mx' if mx else 'blk'}x{b}")
            if not cfg:
                continue
            rows = [r for r in results if r.get("part") == "dense" and r.get("stage") == 2 and r.get("shape") == name
                    and lo <= r.get("M", 0) <= hi and r.get("cfg") == cfg and "cand_us" in r and "rejected" not in r]
            if not rows:
                skipped[b] = "no stage-2 measurement of the bucket config"
                continue
            sp = {str(r["M"]): round(r["stock_op_us"] / r["cand_us"], 3) for r in rows}
            us = {str(r["M"]): [r["stock_op_us"], r["cand_us"]] for r in rows}
            if min(sp.values()) < MIN_DECODE_SPEEDUP:
                skipped[b] = f"speedup {sp} < {MIN_DECODE_SPEEDUP}"
                continue
            e["decode"][b] = {"cfg": cfg, "speedup": sp, "stock_cand_us": us}
        if skipped:
            e["decode_skipped"] = skipped
        pname = PREFILL_ALIAS.get(name, name)
        pr = sorted((r for r in prefill_rows if r.get("kind") == "dense" and r.get("shape") == pname),
                    key=lambda r: r["M"])
        if pr:
            sp = {str(r["M"]): r["speedup_cold"] for r in pr}
            min_m = None
            for i, r in enumerate(pr):
                if all(q["speedup_cold"] >= MIN_PREFILL_SPEEDUP for q in pr[i:]):
                    min_m = r["M"]
                    break
            if min_m is not None:
                e["prefill"] = {"min_m": min_m, "speedup": sp}
            else:
                e["prefill_skipped"] = f"speedup {sp}: no M from which every larger M >= {MIN_PREFILL_SPEEDUP}"
        shapes[name] = e
    return {"about": "GLM_DENSE_FAST dispatch table (overlay/glm_dense_fast.py). decode buckets: '1' M<=8, "
                     "'2' 9<=M<=16, '4' 17<=M<=32; speedup = stock Marlin op / candidate, cold weights, GB10.",
            "rules": {"min_decode_speedup": MIN_DECODE_SPEEDUP, "min_prefill_speedup": MIN_PREFILL_SPEEDUP},
            "sources": sources, "shapes": shapes}


def _main(argv):
    if len(argv) >= 4 and argv[0] == "--make-table":
        with open(argv[1]) as f:
            best = json.load(f)
        res = [json.loads(line) for line in open(argv[2]) if line.strip()]
        pre = [json.loads(line) for line in open(argv[3]) if line.strip()]
        srcs = {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] for p in argv[1:4]}
        print(json.dumps(make_table(best, res, pre, srcs), indent=1, sort_keys=True))
        return 0
    if argv and argv[0] == "--build":
        load_ext(verbose=True)
        return 0
    print("usage: glm_dense_fast.py --make-table BEST_CONFIGS.json RESULTS.jsonl PREFILL_DENSE.jsonl\n"
          "       glm_dense_fast.py --build      (JIT-build the extension into the build dir, e.g. before a boot)")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
