# SPDX-License-Identifier: Apache-2.0
"""GLM_L2_PREFETCH_MOE=1: window C for overlay/glm_l2_prefetch.py. Exact (a prefetch is only a cache hint).

Windows A and B (deployed) cover the 34 KDA layers: A = KDA core -> o_proj head, B = post-attention all-reduce ->
hc_ffn_fn / router / shared gate_up. Nothing is prefetched during the other 45 all-reduces of a target step: the
MoE (42) and dense-MLP (3) output all-reduces. At c1 those take 6-31 us each (09-28 trace: post-MoE mean ~17 us,
floor 6 us); during them DRAM is idle (1-8 polling CTAs), and the next thing read from DRAM is the next layer's
hc_attn_fn (fp32, read by the fused hc post/pre) followed by its first attention projection
(KDA in_proj_qkvbfg_a, ~25 MB/rank, 118-139 us at 4 rows; MLA fused_qkv_a_proj, ~38 us).

Window C arms, at the start of every MoE / dense-MLP forward of a decode/verify-sized target step, a table of the
first GLM_L2_PREFETCH_MOE_MB (default 4) MiB of [next.hc_attn_fn, next attention's first projection (scales first,
then the packed weight)]. glm_l2_prefetch's RoCE hook consumes the armed table right before the layer's output
all-reduce (the only custom all-reduce inside MoE/MLP forward: the shared expert runs with reduce_results=False),
forking the l2pf kernel on its side stream; the model-forward join is unchanged. The last layer arms nothing.

Requires GLM_L2_PREFETCH=1 and GLM_L2_PREFETCH_AR=1 (it reuses their kernel, side stream, join and RoCE hook).
Knobs (read per call; switchable in-boot through overlay/glm_ab.py): GLM_L2_PREFETCH_MOE=0|1,
GLM_L2_PREFETCH_MOE_MB (4). The token cap is the base GLM_L2_PREFETCH_MAXTOK (32).

Credits: ours (ds41 adapter/l2_prefetch.py and overlay/glm_l2_prefetch.py; the ds41 AHEAD extension, which
prefetched the next layer during DS4.1's MoE phase, was flat there, which is why this window fires at the all-reduce
instead of during the expert GEMMs); PTX cp.async.bulk.prefetch.L2 (NVIDIA).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

TARGET_MODEL = "vllm.models.glm5next.nvidia.model"
FIRST_PROJ = ("in_proj_qkvbfg_a", "fused_qkv_a_proj", "q_a_proj", "kv_a_proj_with_mqa", "q_proj")
_OFF = ("", "0", "off", "false", "no")
_state = {"logged": False, "arms": 0}


def _base():
    import glm_l2_prefetch as base
    return base


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-l2-prefetch-c: {msg}\n")


def installed() -> bool:
    return (os.environ.get("GLM_L2_PREFETCH_MOE", "0").strip().lower() not in _OFF
            and os.environ.get("GLM_L2_PREFETCH", "0").strip().lower() not in _OFF)


def first_proj(attn):
    """The first weight GEMM of a GLM-5.3 attention module (KDA or MLA), or None."""
    if attn is None:
        return None
    for name in FIRST_PROJ:
        mod = getattr(attn, name, None)
        if mod is not None and any(True for _ in mod.named_parameters(recurse=False)):
            return mod
    return None


def queue_for_next(layer):
    """[(ptr, nbytes)] in the order the next layer reads them from DRAM."""
    base = _base()
    queue = []
    if layer is None:
        return queue
    fn = getattr(layer, "hc_attn_fn", None)
    if fn is not None and getattr(fn, "is_cuda", False) and fn.is_contiguous():
        queue.append((fn.data_ptr(), fn.numel() * fn.element_size()))
    proj = first_proj(getattr(layer, "self_attn", None))
    if proj is not None:
        queue += base.module_tensors(proj)
    return queue


def _budget() -> int:
    return _base()._mib("GLM_L2_PREFETCH_MOE_MB", "4")


def plan_c(mlp):
    """The window-C table of one MoE/MLP module for the current budget (cached per budget, so in-boot A/B variants
    with different GLM_L2_PREFETCH_MOE_MB capture their own tables). False when there is nothing to prefetch."""
    cache = mlp.__dict__.setdefault("_glm_l2_c", {})
    budget = _budget()
    p = cache.get(budget)
    if p is None:
        base = _base()
        nxt = mlp.__dict__.get("_glm_l2_next")
        segs = base.take(queue_for_next(nxt), budget) if nxt is not None else []
        p = base._table(segs) if segs else False
        cache[budget] = p
        if p and not _state["logged"]:
            _state["logged"] = True
            proj = first_proj(getattr(nxt, "self_attn", None))
            _log(f"window C (MoE/MLP output all-reduce -> next hc_attn_fn + {type(proj).__name__}) "
                 f"{p[2] / 2**20:.2f} MiB")
    return p


def link_layers(model) -> None:
    layers = list(getattr(model, "layers", []))
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            mlp.__dict__["_glm_l2_next"] = layers[i + 1] if i + 1 < len(layers) else None


def _capturing() -> bool:
    import torch
    return bool(torch.cuda.is_current_stream_capturing())


def _wrap_mlp_forward(cls) -> None:
    if getattr(cls, "_glm_l2_c", False):
        return
    cls._glm_l2_c = True
    orig = cls.forward

    def forward(self, hidden_states, *args, **kwargs):
        base = _base()
        armed = None
        n = int(hidden_states.size(0))
        if (0 < n <= int(base.env("GLM_L2_PREFETCH_MAXTOK", "32")) and base._state["depth"] > 0
                and base._on("GLM_L2_PREFETCH") and base._on("GLM_L2_PREFETCH_MOE")):
            p = self.__dict__.get("_glm_l2_c", {}).get(_budget())
            if p is None and not _capturing():
                p = plan_c(self)
            if p:
                armed = p
                base._state["armed"] = p
                _state["arms"] += 1
        try:
            return orig(self, hidden_states, *args, **kwargs)
        finally:
            if armed is not None and base._state["armed"] is armed:
                base._state["armed"] = None  # not consumed (no custom all-reduce ran): never leak into the next layer

    cls.forward = forward


def install_model(mod) -> None:
    _wrap_mlp_forward(mod.Glm5NextMoE)
    _wrap_mlp_forward(mod.Glm5NextMLP)
    cls = mod.Glm5NextModel
    if not getattr(cls, "_glm_l2_c_link", False):
        cls._glm_l2_c_link = True
        orig = cls.forward

        def forward(self, *args, **kwargs):
            if not self.__dict__.get("_glm_l2_c_linked"):
                link_layers(self)
                self.__dict__["_glm_l2_c_linked"] = True
            return orig(self, *args, **kwargs)

        cls.forward = forward
    try:
        _base().install_roce()  # idempotent; window B normally installed it already
    except Exception as exc:  # noqa: BLE001
        _log(f"RoCE hook unavailable ({exc!r}); window C arms but nothing consumes it")
    _log("MoE / dense-MLP forwards arm window C")


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


def register() -> None:
    """Idempotent. No-op unless GLM_L2_PREFETCH_MOE and GLM_L2_PREFETCH are set (default off)."""
    if not installed():
        return
    if TARGET_MODEL in sys.modules:
        install_model(sys.modules[TARGET_MODEL])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
