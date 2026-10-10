# SPDX-License-Identifier: Apache-2.0
"""GLM_ROUTER_DEDUP=1|check: skip the router GEMM whose output the MoE runner discards. Exact.

Tony v11 image (vLLM 0.1.dev20051+g487ecf187). `Glm5NextMoE.forward` (overlay/glm5next_model.py:266) computes
`router_logits, _ = self.gate(hidden_states)` and passes them to `self.experts` (a MoERunner built with
`gate=self.gate`). `MoERunner._forward_impl` (fused_moe/runner/moe_runner.py:858-863) recomputes
`router_logits` from its gate before any use whenever the runner holds a gate, so the first GEMM (288 x 4096 bf16
per layer, 42 MoE layers, plus its fp32 cast) is dead work. Upstream fix: vllm#55736 (JaredforReal), flagged in
MiaAI-Lab issue #271; the same discard trap was found by our verify-cut work (diagnostics/glm-verify-cut).

This wraps `GateLinear.forward` and `Glm5NextMoE.forward`: for a layer whose runner holds exactly this gate
(`self.experts.gate is self.gate`), the model's own gate call returns the MoE input itself as a placeholder
instead of running the GEMM; the runner then computes the real logits as it always did. Layers whose runner has no
gate keep the original path. Nothing the runner reads changes, so outputs are bit-identical by construction.

check mode (TEST ONLY): every eager MoE call also verifies the data flow: the tensor that reaches
`MoERunner._maybe_dispatch` as router_logits must not be the placeholder (a different storage from the MoE input)
and must have the router's shape; violations are counted and logged, and a violation raises (fail closed).
"""
from __future__ import annotations

import os
import sys

TARGET_MODEL = "vllm.models.glm5next.nvidia.model"
TARGET_GATE = "vllm.model_executor.layers.fused_moe.router.gate_linear"
TARGET_RUNNER = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
_OFF = ("", "0", "off", "false", "no")
S = {"skipped": 0, "kept": 0, "checked": 0, "violations": 0, "logged": False}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-router-dedup: {msg}\n")


def env(name: str, default=None):
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def _mode() -> str:
    return str(env("GLM_ROUTER_DEDUP", "0")).strip().lower()


def _check() -> bool:
    return os.environ.get("GLM_ROUTER_DEDUP_CHECK", "0").strip().lower() not in _OFF or _mode() == "check"


def install_gate(mod) -> None:
    cls = mod.GateLinear
    if getattr(cls, "_glm_router_dedup", False):
        return
    cls._glm_router_dedup = True
    orig = cls.forward

    def forward(self, x, *args, **kwargs):
        if self.__dict__.get("_glm_skip_once"):
            self.__dict__["_glm_skip_once"] = False
            S["skipped"] += 1
            return x, None  # placeholder: the runner recomputes the logits from this gate
        return orig(self, x, *args, **kwargs)

    cls.forward = forward


def install_model(mod) -> None:
    cls = mod.Glm5NextMoE
    if getattr(cls, "_glm_router_dedup", False):
        return
    cls._glm_router_dedup = True
    orig = cls.forward

    def forward(self, hidden_states, *args, **kwargs):
        if _mode() in _OFF:
            return orig(self, hidden_states, *args, **kwargs)
        runner_gate = getattr(self.experts, "gate", None)
        if runner_gate is not None and runner_gate is self.gate:
            self.gate.__dict__["_glm_skip_once"] = True
            try:
                return orig(self, hidden_states, *args, **kwargs)
            finally:
                self.gate.__dict__["_glm_skip_once"] = False
        S["kept"] += 1
        if not S["logged"]:
            S["logged"] = True
            _log(f"a MoE layer's runner has no gate of its own ({type(self.experts).__name__}); kept the router GEMM")
        return orig(self, hidden_states, *args, **kwargs)

    cls.forward = forward
    _log("Glm5NextMoE skips its own router GEMM when the runner recomputes it (vllm#55736)")


def install_runner(mod) -> None:
    if not _check():
        return
    import torch
    cls = mod.MoERunner
    if getattr(cls, "_glm_router_dedup", False):
        return
    cls._glm_router_dedup = True
    orig_impl, orig_dispatch = cls._forward_impl, cls._maybe_dispatch

    def _forward_impl(self, hidden_states, router_logits, *args, **kwargs):
        self.__dict__["_glm_in_ptr"] = (hidden_states.data_ptr(), router_logits.data_ptr())
        return orig_impl(self, hidden_states, router_logits, *args, **kwargs)

    def _maybe_dispatch(self, hidden_states, router_logits, *args, **kwargs):
        if not torch.cuda.is_current_stream_capturing() and _mode() not in _OFF:
            in_h, in_r = self.__dict__.get("_glm_in_ptr", (None, None))
            S["checked"] += 1
            leaked = router_logits.data_ptr() == in_h  # the placeholder is the MoE input itself
            shape_ok = self.gate is None or router_logits.shape[-1] == self.gate.weight.shape[0]
            bad = leaked or not shape_ok
            if bad:
                S["violations"] += 1
                _log(f"VIOLATION: placeholder reached dispatch (checked {S['checked']})")
                raise RuntimeError("glm-router-dedup: the placeholder router_logits reached the MoE dispatch")
            if S["checked"] in (1, 1000) or S["checked"] % 20000 == 0:
                _log(f"check: {S['checked']} eager MoE calls, placeholder never reached dispatch "
                     f"(skipped {S['skipped']}, kept {S['kept']})")
        return orig_dispatch(self, hidden_states, router_logits, *args, **kwargs)

    cls._forward_impl, cls._maybe_dispatch = _forward_impl, _maybe_dispatch
    _log("check mode: router_logits data flow verified on every eager MoE call")


HOOKS = {TARGET_MODEL: install_model, TARGET_GATE: install_gate, TARGET_RUNNER: install_runner}


def register() -> None:
    import importlib.abc
    import importlib.util

    if os.environ.get("GLM_ROUTER_DEDUP", "0").strip().lower() in _OFF:
        return
    for name, fn in HOOKS.items():
        if name in sys.modules:
            fn(sys.modules[name])

    class _Finder(importlib.abc.MetaPathFinder):
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
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec, _fn=HOOKS[name]):
                _orig(module)
                _fn(module)
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
