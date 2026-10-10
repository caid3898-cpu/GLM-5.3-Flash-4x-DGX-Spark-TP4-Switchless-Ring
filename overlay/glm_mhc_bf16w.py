# SPDX-License-Identifier: Apache-2.0
"""GLM_MHC_BF16W=1: the decode mHC fused post+pre kernel reads hc_attn_fn / hc_ffn_fn as BF16 instead of FP32.
Bit-exact by construction and verified bitwise at install (fail-closed).

What. The 90 hc_*_fn matrices [24, 16384] are BF16 in the checkpoint and upcast to FP32 parameters at load
(glm5next_model.py: nn.Parameter(torch.empty(mix_hc, d_model, dtype=torch.float32))). The decode path
(num_tokens <= 16) runs vLLM's TileLang `mhc_fused_tilelang` (vllm/model_executor/kernels/mhc/tilelang_kernels.py),
whose grid is (tokens, n_tiles, split_k): every token's CTAs read the whole 1.5 MiB FP32 matrix. Per step that is
135 MiB of cold DRAM reads at c1 and, because each token re-reads it, ~24 MiB of L2 reads per call at M=16
(2026-09-28 traces: the warm kernel grows 0.85 us per token, 6.7 us at M=4 -> 16.9 us at M=16).

How. Two text substitutions in the stock kernel source, nothing else:
    weight_t: T.Tensor((n_out, hc, h), T.float32)       ->  ... T.bfloat16)
    acc[n] += weight_t[...] * new_r[j]                  ->  acc[n] += T.float32(weight_t[...]) * new_r[j]
The patched function is compiled by TileLang from the same schedule. Converting a BF16 value to FP32 is exact, and
install checks, per matrix, that the FP32 parameter equals its BF16 twin widened (so every FMA sees the same operand
bits, in the same order, with the same warp/shared-memory reduction). A matrix that is not exactly BF16-representable
keeps the stock path. Before any capture, install runs the stock and patched kernels through the real
mhc_fused_post_pre_tilelang op on the first layer's matrix for M = 1..16 and compares every output bitwise; on any
difference (or a compile error) the patch disables itself and the stock kernel stays. Prior attempt (overlay/mhc_fused.py, a Triton
re-implementation, 2026-09-27) failed that bitwise gate at M=1; this one keeps TileLang's own codegen.

Scope. Only the small-FMA decode path (num_tokens <= 16). Larger batches (deep_gemm TF32 prenorm GEMM), layer 0's
standalone pre and hc_head keep the FP32 parameters, which stay allocated (+71 MiB of BF16 twins per rank).
Graph safety: twins are allocated and the patched kernel is compiled for both tile shapes in eager install, before
capture; the dispatcher is a Python-level pointer lookup at capture time.
Switch: GLM_MHC_BF16W=0|1 (install gate at start; per call through glm_ab when armed).

Credits: vLLM TileLang mHC kernels (vLLM contributors; tilelang by the TileLang authors); the BF16-origin
observation and the weight-read inventory are from our 2026-09-26 weight audit.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys

TARGET_MODEL = "vllm.models.glm5next.nvidia.model"
TARGET_TK = "vllm.model_executor.kernels.mhc.tilelang_kernels"
_OFF = ("", "0", "off", "false", "no")

DECL_OLD = "    weight_t: T.Tensor((n_out, hc, h), T.float32)"
DECL_NEW = "    weight_t: T.Tensor((n_out, hc, h), T.bfloat16)"
FMA_OLD = "acc[n] += weight_t[i_nt * tile_n + n, j, h_idx] * new_r[j]"
CASTS = ("T.float32({})", 'T.Cast("float32", {})')
FN_NAME = "mhc_fused_tilelang"
NEW_NAME = "mhc_fused_bf16w_tilelang"

S = {"ok": False, "twins": {}, "kernel": None, "stock": None, "cast": None, "calls_bf16": 0, "calls_stock": 0,
     "skipped": 0, "done": False}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-mhc-bf16w: {msg}\n")


def env(name, default=None):
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def _on() -> bool:
    return str(env("GLM_MHC_BF16W", "0")).strip().lower() not in _OFF


def installed() -> bool:
    return os.environ.get("GLM_MHC_BF16W", "0").strip().lower() not in _OFF


# -- source patch (pure text; CPU-testable) --------------------------------------------------------------

def extract_kernel_source(module_source: str, name: str = FN_NAME) -> str:
    """The decorator + def of `name` up to the next top-level decorator/def."""
    start_def = module_source.index(f"\ndef {name}(")
    start = module_source.rindex("\n@tilelang.jit(", 0, start_def) + 1
    nxt = [i for i in (module_source.find("\n@tilelang.jit(", start_def + 1), module_source.find("\ndef ", start_def + 1))
           if i > 0]
    end = min(nxt) + 1 if nxt else len(module_source)
    return module_source[start:end]


def patch_source(src: str, cast: str = CASTS[0]) -> str:
    """Apply the two substitutions (each must match exactly once) and rename the function."""
    for old in (DECL_OLD, FMA_OLD, f"def {FN_NAME}("):
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"glm-mhc-bf16w: expected exactly one {old!r} in the stock kernel, found {n}")
    w = "weight_t[i_nt * tile_n + n, j, h_idx]"
    fma_new = f"acc[n] += {cast.format(w)} * new_r[j]"
    out = src.replace(DECL_OLD, DECL_NEW).replace(FMA_OLD, fma_new).replace(f"def {FN_NAME}(", f"def {NEW_NAME}(")
    if "T.float32)" in out.split("weight_t: T.Tensor", 1)[1].split("\n", 1)[0]:
        raise RuntimeError("glm-mhc-bf16w: weight_t declaration still FP32 after patch")
    return out


KERNEL_HEADER = """# Generated by glm_mhc_bf16w.py from vLLM's tilelang_kernels.py (Apache-2.0, vLLM contributors).
import math  # noqa: F401
from vllm.model_executor.kernels.mhc.tilelang_kernels import ENABLE_PDL, T, pass_configs, tilelang  # noqa: F401

"""


def kernel_module_text(tk_source: str, cast: str) -> tuple[str, str]:
    """(module text, sha16 of the stock kernel source). The patched kernel lives in a real file because TileLang's
    TVMScript parser reads the function source through inspect."""
    src = extract_kernel_source(tk_source, FN_NAME)
    return KERNEL_HEADER + patch_source(src, cast), hashlib.sha256(src.encode()).hexdigest()[:16]


def build_kernel(tk, cast: str):
    import importlib.util
    text, h = kernel_module_text(inspect.getsource(tk), cast)
    d = os.environ.get("GLM_MHC_BF16W_CACHE", "/cache/glm_mhc_bf16w")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = "/tmp/glm_mhc_bf16w"
        os.makedirs(d, exist_ok=True)
    tag = hashlib.sha256(text.encode()).hexdigest()[:12]
    path = os.path.join(d, f"glm_mhc_bf16w_kernel_{tag}.py")
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    spec = importlib.util.spec_from_file_location(f"glm_mhc_bf16w_kernel_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, NEW_NAME), h


# -- twins ---------------------------------------------------------------------------------------------

def make_twins(model) -> dict:
    """{fp32 data_ptr: bf16 twin} for every hc_*_fn that is exactly BF16-representable."""
    import torch
    twins, n_bad, nbytes = {}, 0, 0
    for layer in getattr(model, "layers", []):
        for name in ("hc_attn_fn", "hc_ffn_fn"):
            fn = getattr(layer, name, None)
            if fn is None or not fn.is_cuda or fn.dtype != torch.float32 or not fn.is_contiguous():
                continue
            tw = fn.detach().to(torch.bfloat16).contiguous()
            if torch.equal(tw.float(), fn.detach()):
                twins[fn.data_ptr()] = tw
                nbytes += tw.numel() * 2
            else:
                n_bad += 1
    _log(f"{len(twins)} hc fn matrices have exact BF16 twins ({nbytes / 2**20:.1f} MiB); {n_bad} not representable "
         f"(stock FP32 path)")
    return twins


# -- dispatcher ----------------------------------------------------------------------------------------

def dispatch(*args, **kwargs):
    """Replacement for tilelang_kernels.mhc_fused_tilelang (looked up at call time by the tilelang.py wrapper).
    Positional arg 4 is weight_t = fn.view(24, 4, hidden): same data_ptr as the FP32 parameter."""
    if S["ok"] and _on() and len(args) >= 5:
        w = args[4]
        tw = S["twins"].get(w.data_ptr()) if hasattr(w, "data_ptr") else None
        if tw is not None and str(w.dtype) == "torch.float32" and w.numel() == tw.numel() and w.is_contiguous():
            a = list(args)
            a[4] = tw.view(tuple(w.shape))
            S["calls_bf16"] += 1
            return S["kernel"](*a, **kwargs)
        S["skipped"] += 1
    S["calls_stock"] += 1
    return S["stock"](*args, **kwargs)


def _probe(model, tk) -> bool:
    """Bitwise: stock vs patched through the real op, M = 1..16, first layer with a twin, random inputs."""
    import torch
    layer = next((lay for lay in getattr(model, "layers", [])
                  if getattr(lay, "hc_ffn_fn", None) is not None and lay.hc_ffn_fn.data_ptr() in S["twins"]), None)
    if layer is None:
        return False
    fn = layer.hc_ffn_fn
    hidden = fn.shape[1] // 4
    op = torch.ops.vllm.mhc_fused_post_pre_tilelang
    g = torch.Generator(device=fn.device).manual_seed(20260928)
    nw = getattr(getattr(layer, "post_attention_layernorm", None), "weight", None)
    for m in (1, 2, 3, 4, 5, 7, 8, 9, 12, 16):
        x = torch.randn(m, hidden, device=fn.device, dtype=torch.bfloat16, generator=g)
        res = torch.randn(m, 4, hidden, device=fn.device, dtype=torch.bfloat16, generator=g)
        post = torch.rand(m, 4, 1, device=fn.device, dtype=torch.float32, generator=g) * 2
        comb = torch.rand(m, 4, 4, device=fn.device, dtype=torch.float32, generator=g)
        args = (x, res, post, comb, fn, layer.hc_ffn_scale, layer.hc_ffn_base, layer.rms_norm_eps, layer.hc_eps,
                layer.hc_eps, layer.mhc_post_mult_value, layer.mhc_sinkhorn_iterations, 1, 1,
                nw.data if nw is not None else None,
                getattr(layer.post_attention_layernorm, "variance_epsilon", 1e-5))
        outs = []
        for flag in (False, True):
            S["ok"] = flag
            outs.append([t.clone() for t in op(*args)])
        S["ok"] = False
        for a, b in zip(*outs):
            if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                _log(f"bitwise MISMATCH at M={m}: max |d| {(a.float() - b.float()).abs().max().item():.3e}")
                return False
    torch.cuda.synchronize()
    return True


def install(model) -> bool:
    """After weights are loaded, before capture. Returns True when the patched kernel is live."""
    import torch
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tk
    if S["done"]:
        return S["ok"]
    S["done"] = True
    if torch.cuda.is_current_stream_capturing():
        _log("install called during capture; stock kernel")
        return False
    S["stock"] = tk.mhc_fused_tilelang
    S["twins"] = make_twins(model)
    if not S["twins"]:
        return False
    tk.mhc_fused_tilelang = dispatch
    for cast in CASTS:
        try:
            S["kernel"], h = build_kernel(tk, cast)
            S["cast"] = cast
            if _probe(model, tk):
                S["ok"] = True
                _log(f"live: stock source sha {h}, cast {cast!r}, bitwise probe M=1..16 passed")
                return True
            _log(f"cast {cast!r}: probe failed")
        except Exception as exc:  # noqa: BLE001
            _log(f"cast {cast!r}: {exc!r}")
    S["ok"] = False
    S["twins"] = {}
    tk.mhc_fused_tilelang = S["stock"]
    _log("disabled: stock kernel restored")
    return False


def install_model(mod) -> None:
    cls = mod.Glm5NextModel
    if getattr(cls, "_glm_mhc_bf16w", False):
        return
    cls._glm_mhc_bf16w = True
    orig = cls.forward

    def forward(self, *args, **kwargs):
        if not S["done"]:
            import torch
            if not torch.cuda.is_current_stream_capturing():
                try:
                    install(self)
                except Exception as exc:  # noqa: BLE001
                    _log(f"install failed ({exc!r}); stock kernel")
        return orig(self, *args, **kwargs)

    cls.forward = forward


def register() -> None:
    if not installed():
        return
    import importlib.abc
    import importlib.util
    if TARGET_MODEL in sys.modules:
        install_model(sys.modules[TARGET_MODEL])
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET_MODEL:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                install_model(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
