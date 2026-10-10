# SPDX-License-Identifier: Apache-2.0
"""GLM_FLASHKDA_PREFILL=1: the KDA chunked prefill runs FlashKDA (one fused CUDA kernel pair) instead of the FLA
Triton `chunk_kda_with_fused_gate`. Default OFF. Numerics-changing: judge it with the KLD gate, not a text hash.

Image glm53-roce:rel0928 (vLLM 0.1.dev20051+g487ecf187, tonyd2wild v11), SM121 (GB10), TP4: 16 KDA heads per rank,
head dim 128, 34 KDA layers, prefill chunks up to 6912 tokens, varlen through cu_seqlens. The image predates vLLM
#55737, so `vllm.models.glm5next.nvidia.kda` (the repo's overlay/glm5next_kda.py) calls the module-level name
`chunk_kda_with_fused_gate` from its prefill branch. Its `_forward` is hash-pinned by glm_exact_hooks, so it is not
edited: this module replaces two module-level names after the kda module executes:

  * `chunk_kda_with_fused_gate` -> dispatcher. The previous value (`prev`, saved at install; possibly another
    overlay's wrapper, e.g. GLM_KDA_CONV_SPLIT, which chains the same way) gets every call we do not take.
  * `_cast_sigmoid` -> the stock compiled function plus a tap: FlashKDA wants the raw bf16 beta logits (it applies
    the sigmoid in-kernel, as vLLM #55737 passes them), while the caller hands chunk_kda the fp32 sigmoid. The tap
    remembers (sigmoid output, raw input) so the dispatcher can pass the raw tensor when `beta` is that output.
    Without a matching tap the raw logits are recovered as bf16(logit(beta)) (exact for |x| < ~10; saturated
    values stay saturated) and counted.

Taken only when all hold (anything else -> prev, unchanged arguments):
  keyword call; use_qk_l2norm_in_kernel; safe_gate with a finite lower_bound in [-5, 0]; q/k/v/raw_g bf16 CUDA
  [B, T, H, 128] of one shape; beta [B, T, H] (fp32 sigmoid or bf16 raw); A_log numel H and g_bias numel H*128,
  fp32; initial_state None or [N, H, 128, 128] fp32/bf16; B == 1; cu_seqlens None or int32/int64 1-D; no extra
  kwargs; T >= GLM_FLASHKDA_MIN_T; not under CUDA-graph capture; the load-time self-test passed on every rank; and
  the runtime gate (GLM_FLASHKDA_PREFILL, or the glm_ab variant's value when the harness is armed) is on.
Contract kept: returns (o, final_state) with o bf16 [B, T, H, 128] (a fresh tensor, like stock's) and final_state
fp32 [N, H, 128, 128] in the image's [N, H, V, K] layout (None when output_final_state is False). A bf16
initial_state is widened to fp32 first (stock also carries the state in fp32 and returns it in fp32).

Build: torch.utils.cpp_extension.load of overlay/glm_flashkda_csrc (FlashKDA wip-fp32-state, see PROVENANCE there)
for TORCH_CUDA_ARCH_LIST=12.1a (then 12.1), with the CUTLASS/CuTe headers FlashInfer ships in the image, into
GLM_FLASHKDA_BUILD_DIR (default /cache/glm_flashkda when /cache is writable: start.sh mounts $OVERLAY_REMOTE/cache
there), keyed by a hash of the sources, flags, arch, torch and CUDA versions; each node builds once (~minutes).
Loaded with torch.ops.load_library (op namespace glm_flashkda).

Load time (BaseModelLoader.load_model wrapper, every rank, models that have KDA layers): build, then self-test on
the first KDA layer (its own A_log, dt_bias, lower_bound; synthetic q/k/v/g/beta, two varlen sequences, random
fp32 initial state): FlashKDA vs the stock chunk path (`prev`): finite, output rel L2 and final-state rel L2 <=
GLM_FLASHKDA_TOL (default 2e-2). The pass flag is MIN all-reduced over the TP CPU group, so it is on for every
rank or for none. Any failure turns it off everywhere and logs loudly (fail closed to stock; the boot continues).

Measured (diagnostics/glm-flashkda-20260928, GB10 Spark_04, 2026-09-28, 16 heads, layer-0 A_log/dt_bias, lb -5,
synthetic activations; per layer-call, caller layout incl. the q/k/v copies): T=2048 1.81 -> 0.72 ms (x2.5), T=6912
6.56 -> 2.44 ms (x2.7), varlen 2 seqs with state 6.57 -> 2.51 ms; ~140 ms saved per 6912-token chunk (34 layers).
Error vs an fp64 sequential reference: output 4.2e-3 (stock FLA 5.1e-3), final state 3.1e-3 (stock 4.0e-3); vs
stock 6.5e-3 / 4.8e-3; bitwise deterministic. JIT build 25 s per node, then cached.

Env:
  GLM_FLASHKDA_PREFILL=0|1       install gate (read once) and, with glm_ab armed, per-call variant switch (bool)
  GLM_FLASHKDA_MIN_T=N            take only calls with T >= N (default 1)
  GLM_FLASHKDA_TOL=x              self-test rel L2 bound (default 2e-2)
  GLM_FLASHKDA_BUILD_DIR=dir      JIT cache root
  GLM_FLASHKDA_ARCH=12.1a         first arch tried (then 12.1)
  GLM_FLASHKDA_CUTLASS=dir        CUTLASS include dir (default: the image's flashinfer/data/cutlass/include)

Credits: FlashKDA by MoonshotAI and the vLLM FlashKDA contributors (MIT); Matt Mastracci (mmastrac) for the
wip-fp32-state kernel numerics, the fp32-state work behind FlashKDA #13 and the vllm_shim this call follows;
JaredforReal for vLLM #55737 (how GLM's KDA prefill calls FlashKDA: raw gate and beta logits, in-kernel l2norm,
bounded gate, state layout, varlen); simon-veitner-redhat for vLLM #58846 (FlashKDA pin bump) and the TMA
store-wait fix vendored with the kernel.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time

TARGET = "vllm.models.glm5next.nvidia.kda"
LOADER_MOD = "vllm.model_executor.model_loader.base_loader"
KEY = "GLM_FLASHKDA_PREFILL"
_OFF = ("", "0", "off", "false", "no")
HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "glm_flashkda_csrc")
CSRC_FILES = ("flash_kda.cpp", "torch_api.cpp", "smxx/fwd_launch.cu", "flash_kda.h", "fwd.h", "smxx/utils.cuh",
              "smxx/fwd_kernel1.cuh", "smxx/fwd_kernel2.cuh")
D = 128
REQUIRE_CUDA = True   # the CPU test clears it

ENABLED = os.environ.get(KEY, "0").strip().lower() not in _OFF
S = {"prev": None, "mod": None, "ready": False, "tested": False, "ext": None, "tap": None,
     "taken": 0, "passed": 0, "reasons": {}, "logit_beta": 0, "logged": 0, "detail": ""}
_LOCK = threading.Lock()


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-flashkda: {msg}\n")
    sys.stderr.flush()


def _float_env(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def call_on() -> bool:
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False) and KEY in ab.KNOWN:
        return ab.truthy(ab.env(KEY, "0"))
    return ENABLED


def _capturing() -> bool:
    import torch
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# ------------------------------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------------------------------
def build_root() -> str:
    root = os.environ.get("GLM_FLASHKDA_BUILD_DIR", "").strip()
    if root:
        return root
    if os.path.isdir("/cache") and os.access("/cache", os.W_OK):
        return "/cache/glm_flashkda"
    return os.path.join(os.path.expanduser("~"), ".cache", "glm_flashkda")


def cutlass_include() -> str:
    env = os.environ.get("GLM_FLASHKDA_CUTLASS", "").strip()
    cands = [env] if env else []
    try:
        import flashinfer
        cands.append(os.path.join(os.path.dirname(flashinfer.__file__), "data", "cutlass", "include"))
    except Exception:  # noqa: BLE001
        pass
    for p in cands:
        if p and os.path.isfile(os.path.join(p, "cute", "tensor.hpp")) \
                and os.path.isfile(os.path.join(p, "cutlass", "pipeline", "sm90_pipeline.hpp")):
            return p
    raise RuntimeError(f"no CUTLASS include dir with cute/tensor.hpp among {cands}")


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
    common = ["-O3", "-std=c++17", "-DTORCH_TARGET_VERSION=0x020a000000000000", "-DUSE_CUDA"]
    cflags = common + ["-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                       "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                       "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math", "-lineinfo",
                       "--ptxas-options=--register-usage-level=10"]
    # torch's ATen/cuda headers want cusparse.h etc., shipped here only in the pip nvidia wheels: search them last.
    for d in after_dirs:
        cflags += ["-Xcompiler", "-idirafter," + d]
    cxx = common + ["-Wno-psabi"]
    for d in after_dirs:
        cxx += ["-idirafter", d]
    return cflags, cxx


def build_tag(arch: str, cflags, cxx, incs) -> str:
    import torch
    h = hashlib.sha256()
    for f in CSRC_FILES:
        with open(os.path.join(CSRC, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read())
    h.update(json.dumps([arch, cflags, cxx, incs, torch.__version__, torch.version.cuda]).encode())
    return h.hexdigest()[:12]


def _clear_stale_lock(bdir: str, max_age_s: float = 1800.0) -> None:
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
    """Build (or reuse) and load the extension; returns torch.ops.glm_flashkda."""
    if S["ext"] is not None:
        return S["ext"]
    with _LOCK:
        if S["ext"] is not None:
            return S["ext"]
        import torch
        from torch.utils.cpp_extension import load
        srcs = [os.path.join(CSRC, f) for f in CSRC_FILES if f.endswith((".cpp", ".cu"))]
        incs = [CSRC, cutlass_include()]
        cflags, cxx = build_flags(_nv_include_dirs())
        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        last = None
        t0 = time.time()
        bdir = ""
        try:
            for arch in dict.fromkeys((os.environ.get("GLM_FLASHKDA_ARCH", "12.1a"), "12.1")):
                bdir = os.path.join(build_root(), f"glm_flashkda_ext-{build_tag(arch, cflags, cxx, incs)}")
                os.makedirs(bdir, exist_ok=True)
                _clear_stale_lock(bdir)
                os.environ["TORCH_CUDA_ARCH_LIST"] = arch
                try:
                    load(name="glm_flashkda_ext", sources=srcs, extra_include_paths=incs, extra_cuda_cflags=cflags,
                         extra_cflags=cxx, build_directory=bdir, verbose=verbose, is_python_module=False)
                    break
                except (ValueError, RuntimeError) as exc:
                    last = exc
                    _log(f"build for {arch} failed: {str(exc)[-400:]}")
            else:
                raise last
        finally:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
        ops = torch.ops.glm_flashkda
        ops.fwd  # noqa: B018  (raises if the library did not register)
        _log(f"extension ready in {time.time() - t0:.1f} s ({bdir})")
        S["ext"] = ops
        return ops


# ------------------------------------------------------------------------------------------------------------------
# the call
# ------------------------------------------------------------------------------------------------------------------
_TAKEN_KW = {"q", "k", "v", "raw_g", "beta", "A_log", "g_bias", "scale", "initial_state", "output_final_state",
             "use_qk_l2norm_in_kernel", "cu_seqlens", "safe_gate", "lower_bound"}


def why_not(args, kw, min_t: int = 1) -> str:
    """'' when FlashKDA can serve this call, else a short reason (the call then goes to prev unchanged)."""
    import torch
    if args:
        return "positional"
    if set(kw) - _TAKEN_KW:
        return "extra_kwargs"
    q, k, v, g, beta = (kw.get(n) for n in ("q", "k", "v", "raw_g", "beta"))
    if any(t is None for t in (q, k, v, g, beta)):
        return "missing"
    if not kw.get("use_qk_l2norm_in_kernel", False):
        return "no_l2norm"
    lb = kw.get("lower_bound", -5.0)
    if not kw.get("safe_gate", False) or lb is None or not (-5.0 <= float(lb) <= 0.0):
        return "gate"
    if q.dim() != 4 or q.shape[-1] != D or (REQUIRE_CUDA and not q.is_cuda):
        return "q_shape"
    B, T, H, _ = q.shape
    for t in (q, k, v, g):
        if t.dtype != torch.bfloat16 or tuple(t.shape) != (B, T, H, D) or t.device != q.device:
            return "qkvg"
    if tuple(beta.shape) != (B, T, H) or beta.dtype not in (torch.float32, torch.bfloat16):
        return "beta"
    a, gb = kw.get("A_log"), kw.get("g_bias")
    if a is None or gb is None or a.numel() != H or gb.numel() != H * D \
            or a.dtype != torch.float32 or gb.dtype != torch.float32:
        return "a_log_bias"
    if B != 1:
        return "batch"
    cu = kw.get("cu_seqlens")
    if cu is not None and (cu.dim() != 1 or cu.numel() < 2 or cu.dtype not in (torch.int32, torch.int64)):
        return "cu_seqlens"
    N = 1 if cu is None else cu.numel() - 1
    h0 = kw.get("initial_state")
    if h0 is not None and (tuple(h0.shape) != (N, H, D, D) or h0.dtype not in (torch.float32, torch.bfloat16)):
        return "initial_state"
    if T < min_t:
        return "short"
    if _capturing():
        return "capturing"
    return ""


def raw_beta(beta):
    """The bf16 beta logits for FlashKDA: the tapped raw input when `beta` is the tapped sigmoid, else a logit."""
    import torch
    tap, S["tap"] = S["tap"], None
    if beta.dtype == torch.bfloat16:
        return beta
    if tap is not None:
        y, x = tap
        if (beta.data_ptr() == y.data_ptr() and tuple(beta.shape[1:]) == tuple(y.shape)
                and tuple(x.shape) == tuple(y.shape) and x.dtype == torch.bfloat16):
            return x.unsqueeze(0)
    S["logit_beta"] += 1
    if S["logit_beta"] <= 2:
        _log("beta arrived without a matching raw tap: using bf16(logit(beta))")
    return torch.logit(beta.float()).to(torch.bfloat16)


def flashkda_call(q, k, v, raw_g, beta_raw, A_log, g_bias, scale=None, initial_state=None,
                  output_final_state=True, cu_seqlens=None, lower_bound=-5.0):
    """FlashKDA forward with the stock return contract: (o bf16 [B,T,H,D], final fp32 [N,H,D,D] or None)."""
    import torch
    ops = load_ext()
    B, T, H, _ = q.shape          # B == 1 (why_not); the build only has the varlen, fp32-state-out instances
    if cu_seqlens is None:
        cu = torch.tensor([0, T], dtype=torch.int32, device=q.device)
    else:
        cu = cu_seqlens.contiguous()
    N = cu.numel() - 1
    if scale is None:
        scale = D ** -0.5
    ws = torch.empty(int(ops.get_workspace_size(B * T, H, N)), dtype=torch.uint8, device=q.device)
    out = torch.empty((B, T, H, D), dtype=torch.bfloat16, device=q.device)
    h0 = None
    if initial_state is not None:
        h0 = initial_state.to(torch.float32).contiguous()
    ht = torch.empty((N, H, D, D), dtype=torch.float32, device=q.device)
    ops.fwd(q.contiguous(), k.contiguous(), v.contiguous(), raw_g.contiguous(), beta_raw, float(scale), out, ws,
            A_log.reshape(-1).contiguous(), g_bias.reshape(H, D).contiguous(), float(lower_bound), h0, ht, cu,
            None, None)
    return out, (ht if output_final_state else None)


def dispatch(*args, **kw):
    """Replacement for the kda module's `chunk_kda_with_fused_gate`."""
    cs = sys.modules.get("glm_kda_conv_split")
    take = getattr(cs, "take", None) if cs is not None else None
    if take is not None and not args and all(kw.get(n) is not None for n in ("q", "k", "v")):
        kw = dict(kw)
        kw["q"], kw["k"], kw["v"] = take(kw["q"], kw["k"], kw["v"])
    prev = S["prev"]
    if not call_on():
        S["tap"] = None
        return prev(*args, **kw)
    reason = "not_ready" if not S["ready"] else why_not(args, kw, int(_float_env("GLM_FLASHKDA_MIN_T", 1)))
    if reason:
        S["tap"] = None
        S["passed"] += 1
        S["reasons"][reason] = S["reasons"].get(reason, 0) + 1
        if S["reasons"][reason] == 1:
            _log(f"call left on stock: {reason}" + (" (the load-time self-test did not pass or did not run)"
                                                    if reason == "not_ready" else ""))
        return prev(*args, **kw)
    o, ht = flashkda_call(kw["q"], kw["k"], kw["v"], kw["raw_g"], raw_beta(kw["beta"]), kw["A_log"], kw["g_bias"],
                          kw.get("scale"), kw.get("initial_state"), kw.get("output_final_state", False),
                          kw.get("cu_seqlens"), kw.get("lower_bound", -5.0))
    S["taken"] += 1
    if S["logged"] < 2:
        S["logged"] += 1
        _log(f"FlashKDA prefill taken: q {tuple(kw['q'].shape)} N="
             f"{1 if kw.get('cu_seqlens') is None else kw['cu_seqlens'].numel() - 1}")
    return o, ht


dispatch.__glm_conv_split_aware__ = True


# ------------------------------------------------------------------------------------------------------------------
# self-test (load time)
# ------------------------------------------------------------------------------------------------------------------
def rel_l2(a, b) -> float:
    a = a.double()
    b = b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def make_inputs(H, T, lens, device, seed=1234, state_scale=0.05, dtype_state=None):
    """Synthetic KDA prefill inputs shaped like the GLM call: q/k/v/raw_g bf16 [1,T,H,D], raw beta bf16 [1,T,H],
    varlen cu_seqlens int32 over `lens`, fp32 initial state [N,H,D,D] (None if state_scale is 0)."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape, s=1.0):
        return (torch.randn(*shape, generator=gen) * s).to(device=device, dtype=torch.bfloat16)

    assert sum(lens) == T
    q, k, v = rn(1, T, H, D), rn(1, T, H, D), rn(1, T, H, D)
    g = rn(1, T, H, D)
    beta = rn(1, T, H)
    cu = torch.tensor([0] + list(__import__("itertools").accumulate(lens)), dtype=torch.int32, device=device)
    h0 = None
    if state_scale:
        h0 = (torch.randn(len(lens), H, D, D, generator=gen) * state_scale).to(device)
        if dtype_state is not None:
            h0 = h0.to(dtype_state)
    return q, k, v, g, beta, cu, h0


def self_test(layer, prev, T=1024, lens=(300, 724), tol=None):
    """(ok, detail) for one KDA layer: FlashKDA vs the stock chunk path on synthetic inputs."""
    import torch
    tol = _float_env("GLM_FLASHKDA_TOL", 2e-2) if tol is None else tol
    if not getattr(layer, "kda_safe_gate", False):
        return False, "layer has no safe_gate"
    A_log, dt_bias, lb = layer.A_log, layer.dt_bias, float(layer.kda_lower_bound)
    H = A_log.numel()
    dev = A_log.device
    q, k, v, g, beta, cu, h0 = make_inputs(H, T, list(lens), dev)
    kw = dict(q=q, k=k, v=v, raw_g=g, beta=beta.float().sigmoid(), A_log=A_log, g_bias=dt_bias,
              initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
              safe_gate=True, lower_bound=lb)
    reason = why_not((), kw)
    if reason:
        return False, f"layer call not eligible: {reason}"
    # Clones: with a contiguous v the FLA path writes its output into v in place (chunk_gla_fwd_o_gk(o=v)).
    o_ref, h_ref = prev(**{**kw, "q": q.clone(), "k": k.clone(), "v": v.clone(), "initial_state": h0.clone()})
    o, h = flashkda_call(q, k, v, g, beta, A_log, dt_bias, None, h0.clone(), True, cu, lb)
    o2, h2 = flashkda_call(q, k, v, g, beta, A_log, dt_bias, None, h0.clone(), True, cu, lb)
    torch.cuda.synchronize()
    fin = bool(torch.isfinite(o).all()) and bool(torch.isfinite(h).all())
    ro, rh = rel_l2(o, o_ref), rel_l2(h, h_ref.float())
    det = bool(torch.equal(o, o2)) and bool(torch.equal(h, h2))
    ok = fin and ro <= tol and rh <= tol and o.dtype == o_ref.dtype and h.dtype == h_ref.dtype \
        and tuple(o.shape) == tuple(o_ref.shape) and tuple(h.shape) == tuple(h_ref.shape)
    return ok, (f"H={H} T={T} lens={list(lens)} lb={lb} finite={fin} rel_o={ro:.2e} rel_state={rh:.2e} "
                f"deterministic={det} tol={tol:g}")


def _tp():
    try:
        from vllm.distributed.parallel_state import get_tp_group
        g = get_tp_group()
        return g.cpu_group, g.world_size, g.rank_in_group
    except Exception:  # noqa: BLE001
        return None, 1, 0


def agree(ok: bool, group=None, world: int = 1) -> bool:
    if group is None or world <= 1:
        return bool(ok)
    import torch
    import torch.distributed as dist
    t = torch.tensor([int(bool(ok))], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(int(t[0]))


def kda_layers(model) -> list:
    return [m for m in model.modules() if type(m).__name__ == "Glm5NextLinearAttention"]


def prepare(model, *, group=None, world: int = 1, rank: int = 0, prev=None) -> bool:
    """Build + self-test + TP agreement. Collective when any KDA layer exists (same on every rank)."""
    layers = kda_layers(model)
    if not layers:
        return S["ready"]
    ok, detail = False, ""
    t0 = time.time()
    try:
        prev = prev or S["prev"]
        if prev is None:
            raise RuntimeError("chunk_kda_with_fused_gate was never wrapped (kda module not hooked)")
        load_ext()
        ok, detail = self_test(layers[0], prev)
    except Exception as exc:  # noqa: BLE001  (a raise here would leave peers in the all-reduce)
        ok, detail = False, f"{type(exc).__name__}: {str(exc)[-600:]}"
    all_ok = agree(ok, group, world)
    S["ready"], S["tested"], S["detail"] = all_ok, True, detail
    if all_ok:
        _log(f"rank {rank}: self-test PASS on {len(layers)} KDA layers ({time.time() - t0:.1f} s): {detail}")
    else:
        _log(f"rank {rank}: *** FlashKDA prefill OFF on every rank (fail closed to stock) *** local "
             f"{'pass' if ok else 'FAIL'}: {detail}")
    return all_ok


# ------------------------------------------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------------------------------------------
def install(mod) -> None:
    if getattr(mod, "__glm_flashkda__", False):
        return
    S["prev"], S["mod"] = mod.chunk_kda_with_fused_gate, mod
    mod.chunk_kda_with_fused_gate = dispatch
    orig_sig = getattr(mod, "_cast_sigmoid", None)
    if orig_sig is not None:
        def _cast_sigmoid(x):
            y = orig_sig(x)
            S["tap"] = (y, x) if call_on() else None
            return y
        _cast_sigmoid.__wrapped__ = orig_sig
        mod._cast_sigmoid = _cast_sigmoid
    mod.__glm_flashkda__ = True
    _log("installed (chunk_kda_with_fused_gate dispatcher + _cast_sigmoid raw-beta tap)")


def install_loader(mod) -> None:
    cls = mod.BaseModelLoader
    if getattr(cls, "_glm_flashkda", False):
        return
    orig = cls.load_model

    def load_model(self, vllm_config, model_config, prefix: str = ""):
        model = orig(self, vllm_config, model_config, prefix)
        group, world, rank = _tp()
        prepare(model, group=group, world=world, rank=rank)
        return model

    load_model.__wrapped__ = orig
    cls.load_model = load_model
    cls._glm_flashkda = True


def register() -> None:
    """From sitecustomize (install gate already checked there; also safe to call when off)."""
    if not ENABLED:
        return
    import glm_prefill_hooks
    glm_prefill_hooks.after_import(TARGET, install)
    glm_prefill_hooks.after_import(LOADER_MOD, install_loader)
    _log(f"armed: {KEY}={os.environ.get(KEY)!r}")
