# SPDX-License-Identifier: Apache-2.0
"""GLM_MHC_FUSED=1: install overlay/mhc_fused.py (bf16-load / fp32-compute mHC projection kernels, bit-exact to the
stock TileLang FMA path; arithmetic adapted from
vLLM's mhc/tilelang_kernels.py) into the serving engine, and make it switchable per call for the in-boot A/B.

mhc_fused.install(model) must run after the weights are loaded and before any CUDA graph capture; it probes its
kernels bitwise against the stock ones on every device and raises on any mismatch. This hook calls it on the first
eager Glm5NextModel.forward (the memory-profile run, before capture). If install raises, the engine keeps the stock
kernels (logged, fail-safe). After install the two patched tilelang_kernels entry points dispatch per call: the
fused kernels when GLM_MHC_FUSED is on for the current (in-boot) variant, the stock ones otherwise.
"""
from __future__ import annotations

import os
import sys

TARGET_MODEL = "vllm.models.glm5next.nvidia.model"
_OFF = ("", "0", "off", "false", "no")
S = {"done": False, "ok": False, "fused_calls": 0, "stock_calls": 0}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-mhc-hook: {msg}\n")


def env(name: str, default=None):
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def _on() -> bool:
    return str(env("GLM_MHC_FUSED", "0")).strip().lower() not in _OFF


def _install(model) -> None:
    S["done"] = True
    try:
        import mhc_fused as mf
        from vllm.model_executor.kernels.mhc import tilelang_kernels as tk
        os.environ["GLM_MHC_FUSED"] = "1"
        if not mf.install(model):
            _log("mhc_fused.install returned False; stock kernels")
            return
        new_pre, new_fused = tk.hc_prenorm_gemm_tilelang, tk.mhc_fused_tilelang
        old_pre, old_fused = mf._STATE["pre"], mf._STATE["fused"]

        def pre(*a, **k):
            if _on():
                S["fused_calls"] += 1
                return new_pre(*a, **k)
            S["stock_calls"] += 1
            return old_pre(*a, **k)

        def fused(*a, **k):
            return (new_fused if _on() else old_fused)(*a, **k)

        tk.hc_prenorm_gemm_tilelang, tk.mhc_fused_tilelang = pre, fused
        S["ok"] = True
        _log(f"installed: {len(mf._STATE['weights'])} hc_fn weights packed to bf16, bitwise probes passed")
    except Exception as exc:  # noqa: BLE001
        _log(f"NOT installed ({exc!r}); stock mHC kernels stay")


def install_model(mod) -> None:
    import torch
    cls = mod.Glm5NextModel
    if getattr(cls, "_glm_mhc_hook", False):
        return
    cls._glm_mhc_hook = True
    orig = cls.forward

    def forward(self, *args, **kwargs):
        if not S["done"] and not torch.cuda.is_current_stream_capturing():
            _install(self)
        return orig(self, *args, **kwargs)

    cls.forward = forward
    _log("mHC fused kernels install on the first eager target forward")


def register() -> None:
    import importlib.abc
    import importlib.util

    if os.environ.get("GLM_MHC_FUSED", "0").strip().lower() in _OFF:
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
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                install_model(module)
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
