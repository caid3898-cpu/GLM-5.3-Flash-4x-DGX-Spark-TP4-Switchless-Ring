# SPDX-License-Identifier: Apache-2.0
"""GLM_L2_PREFETCH_MLA / _MLA_AR / _DRAFT: windows M, B-MLA and D for overlay/glm_l2_prefetch.py. Exact (a prefetch
is only a cache hint; no tensor, kernel or reduction order changes).

Why. The deployed windows A and B live in `Glm5NextLinearAttention`, so they cover the 34 KDA layers only. The 2026-09-28
production traces are therefore a natural experiment: the same MoE-side kernels run L2-warm after a KDA attention and
cold after an MLA attention (diagnostics/glm-bytes-20260928/data/windows_evidence.txt, medians over 25 c1 steps):

    kernel (c1, M=4)                          after KDA (warm)   after MLA (cold)
    post-attention mhc_fused (hc_ffn_fn)           6.7 us            10.8 us
    router GEMM + splitK reduce                   17.4 us            27.6 us
    shared gate_up (aux stream)                   25.3 us            33.5 us

and inside the 11 MLA attentions every weight is read cold, several of them by latency-bound kernels that sit behind
~80-150 us of DRAM-idle indexer work (wk 10.3 us for 1.3 MB, head-gate gemmSN 66.7 us for 0.5 MB, W_UK bmm 24.4 us
and W_UV bmm 21.1 us for 4 MiB each, o_proj 78 us for 16.8 MB).

Window M (GLM_L2_PREFETCH_MLA=1): at `Indexer.forward` entry (after q_b; the main stream then runs wq_b, the wq_b
all-gather, wk, the head gate, the index-logit/top-k chain and the sparse attention: DRAM mostly idle), a side stream
prefetches GLM_L2_PREFETCH_MLA_MB (default 16) MiB in consumption order:
    indexer wk_weights_proj (1.25 MiB) -> head-gate fp32 copy _wp_fp32 (0.5 MiB, stock gate path only)
    -> index_kpool_compress_gate (1 MiB) -> W_UK_T (4 MiB) -> W_UV (4 MiB) -> o_proj (scales, then weight)
Window B-MLA (GLM_L2_PREFETCH_MLA_AR=1): window B for MLA layers. Right before the MLA o_proj all-reduce the base
module's RoCE hook forks the layer's window-B table (hc_ffn_fn, router, shared gate_up; GLM_L2_PREFETCH_AR_MB).
Window D (GLM_L2_PREFETCH_DRAFT=1): the DFlash2 drafter reads a replicated BF16 kernel_projection (8 MiB, 45 us,
~186 GB/s) right after each of its 10 all-reduces. At each drafter o_proj / down_proj all-reduce, prefetch the next
kernel_projection weight (GLM_L2_PREFETCH_DRAFT_MB, default 8). Joined at the end of the drafter model forward.

Joins, graphs, gates: as in glm_l2_prefetch.py. Tables are built on an eager decode-sized forward, never during
capture (a graph captured before that prefetches nothing); a decision is per captured shape. Target windows need the
base module's depth > 0 (inside Glm5NextModel.forward) and <= GLM_L2_PREFETCH_MAXTOK tokens; window D uses its own
depth and GLM_L2_PREFETCH_DRAFT_MAXTOK (default 64). Knobs are read per call through glm_ab (in-boot A/B; window D
needs GLM_AB_DRAFT_SETS=1 there, since it lives in the drafter graphs).

Requires GLM_L2_PREFETCH=1 (kernel, side stream, join) and GLM_L2_PREFETCH_AR=1 (RoCE hook) for B-MLA and D.

Credits: ours (ds41 adapter/l2_prefetch.py -> overlay/glm_l2_prefetch.py windows A/B, glm-steplevel window C);
DFlash2 drafter by Inco AI; vLLM MLA absorbed-weight decode (W_UK_T / W_UV); PTX cp.async.bulk.prefetch.L2 (NVIDIA).
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys

TARGET_ATTN = "vllm.models.glm5next.nvidia.attention"
TARGET_MODEL = "vllm.models.glm5next.nvidia.model"
TARGET_DRAFT = "vllm.model_executor.models.qwen3_dflash"
_OFF = ("", "0", "off", "false", "no")
_state = {"logged": set(), "forks_m": 0, "arms_b": 0, "arms_d": 0, "draft_depth": 0}


def _base():
    import glm_l2_prefetch as base
    return base


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-l2-prefetch-mla: {msg}\n")


def _once(key: str, msg: str) -> None:
    if key not in _state["logged"]:
        _state["logged"].add(key)
        _log(msg)


def _raw_on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in _OFF


def installed() -> bool:
    return _raw_on("GLM_L2_PREFETCH") and any(
        _raw_on(k) for k in ("GLM_L2_PREFETCH_MLA", "GLM_L2_PREFETCH_MLA_AR", "GLM_L2_PREFETCH_DRAFT"))


def _capturing() -> bool:
    import torch
    return bool(torch.cuda.is_current_stream_capturing())


def _rows(x) -> int:
    try:
        return int(x.size(0))
    except Exception:  # noqa: BLE001
        return 0


# -- tensors -------------------------------------------------------------------------------------------

def tensor_span(t):
    """(ptr, nbytes) of the memory a tensor's elements occupy (views included), or None. A strided view covers its
    whole span; a span more than 1/8 larger than the element bytes is refused (it would prefetch foreign data)."""
    if t is None or not getattr(t, "is_cuda", False):
        return None
    try:
        n = t.numel()
        if n == 0:
            return None
        es = t.element_size()
        hi = sum((s - 1) * st for s, st in zip(t.shape, t.stride()) if s > 0)
        lo = sum((s - 1) * st for s, st in zip(t.shape, t.stride()) if s > 0 and st < 0)
        span = (hi - lo + 1) * es
        if span > n * es * 9 // 8:
            return None
        return (t.data_ptr() + lo * es, span)
    except Exception:  # noqa: BLE001
        return None


def find_attr_tensor(module, name):
    """The first CUDA tensor attribute `name` on `module` or any submodule (W_UK_T / W_UV live on the inner
    MLAAttention layer of vLLM's MultiHeadLatentAttentionWrapper)."""
    mods = module.modules() if hasattr(module, "modules") else [module]
    for m in mods:
        t = getattr(m, name, None)
        if t is not None and getattr(t, "is_cuda", False):
            return t
    return None


def queue_m(attn, gate_stock: bool):
    """[(ptr, nbytes)] of window M in the order the MLA core reads them after Indexer.forward starts."""
    base = _base()
    q = []
    idx = getattr(attn, "indexer", None)
    if idx is not None:
        wkw = getattr(idx, "wk_weights_proj", None)
        if wkw is not None:
            q += base.module_tensors(wkw)
        if gate_stock:
            s = tensor_span(idx.__dict__.get("_wp_fp32"))
            if s:
                q.append(s)
        s = tensor_span(getattr(idx, "index_kpool_compress_gate", None))
        if s:
            q.append(s)
    for name in ("W_UK_T", "W_UV"):
        s = tensor_span(find_attr_tensor(attn, name))
        if s:
            q.append(s)
    o = getattr(attn, "o_proj", None)
    if o is not None:
        q += base.module_tensors(o)
    return q


def plan_m(attn):
    """Window-M table of one MLA attention, cached per (budget, gate path). False = nothing to prefetch."""
    base = _base()
    budget = base._mib("GLM_L2_PREFETCH_MLA_MB", "16")
    gate_stock = str(base.env("GLM_GATE_GEMV", "0")).strip().lower() in _OFF
    cache = attn.__dict__.setdefault("_glm_l2_m", {})
    key = (budget, gate_stock)
    p = cache.get(key)
    if p is None:
        segs = base.take(queue_m(attn, gate_stock), budget)
        p = base._table(segs) if segs else False
        cache[key] = p
        _once("m", f"window M (MLA indexer -> wk/gate/W_UK/W_UV/o_proj) {p[2] / 2**20 if p else 0:.2f} MiB "
                   f"in {p[1] if p else 0} segments; budget {budget / 2**20:.2f} MiB")
    return p


def _on_m(n: int) -> bool:
    base = _base()
    return (0 < n <= int(base.env("GLM_L2_PREFETCH_MAXTOK", "32")) and base._state["depth"] > 0
            and base._on("GLM_L2_PREFETCH") and base._on("GLM_L2_PREFETCH_MLA"))


# -- target hooks --------------------------------------------------------------------------------------

def install_attn(mod) -> None:
    cls = getattr(mod, "Indexer", None)
    if cls is None or getattr(cls, "_glm_l2_m", False):
        return
    cls._glm_l2_m = True
    orig = cls.forward

    def forward(self, hidden_states, *args, **kwargs):
        attn = self.__dict__.get("_glm_l2m_attn")
        n = _rows(hidden_states)
        build_after = False
        if attn is not None and _on_m(n):
            key = (_base()._mib("GLM_L2_PREFETCH_MLA_MB", "16"),
                   str(_base().env("GLM_GATE_GEMV", "0")).strip().lower() in _OFF)
            p = attn.__dict__.get("_glm_l2_m", {}).get(key)
            if p is None:
                build_after = not _capturing()  # _wp_fp32 is created lazily by the first forward
            elif p:
                _base()._fork(p)
                _state["forks_m"] += 1
                if _state["forks_m"] == 1:
                    _log(f"first window-M fork: {n} tokens, capturing={_capturing()}")
        out = orig(self, hidden_states, *args, **kwargs)
        if build_after:
            plan_m(attn)
        return out

    cls.forward = forward
    _log("Indexer.forward forks window M (MLA layers)")


def _wrap_o_proj_b(attn) -> None:
    """Window B for one MLA attention: arm the base window-B table right before its o_proj all-reduce."""
    o = getattr(attn, "o_proj", None)
    if o is None or o.__dict__.get("_glm_l2_bm"):
        return
    orig = o.forward

    def forward(x, *args, **kwargs):
        base = _base()
        armed = None
        n = _rows(x)
        if (0 < n <= int(base.env("GLM_L2_PREFETCH_MAXTOK", "32")) and base._state["depth"] > 0
                and base._on("GLM_L2_PREFETCH") and base._on("GLM_L2_PREFETCH_AR")
                and base._on("GLM_L2_PREFETCH_MLA_AR")):
            pb = attn.__dict__.get("_glm_l2_b")
            if pb is None and not _capturing():
                pb = base._plan_b(attn)
            if pb:
                armed = pb
                base._state["armed"] = pb
                _state["arms_b"] += 1
        try:
            return orig(x, *args, **kwargs)
        finally:
            if armed is not None and base._state["armed"] is armed:
                base._state["armed"] = None  # not consumed: never leak into the MoE output all-reduce

    o.forward = forward
    o.__dict__["_glm_l2_bm"] = True


def is_mla(attn) -> bool:
    return attn is not None and hasattr(attn, "indexer") and hasattr(attn, "o_proj") and not hasattr(
        attn, "in_proj_qkvbfg_a")


def link_model(model) -> int:
    """Link every MLA Indexer to its attention module and wrap the MLA o_proj forwards. Returns #MLA layers."""
    base = _base()
    base._link_layers(model)  # attn -> decoder layer (window-B tables read hc_ffn_fn / router / shared)
    n = 0
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if not is_mla(attn):
            continue
        n += 1
        idx = getattr(attn, "indexer", None)
        if idx is not None:
            idx.__dict__["_glm_l2m_attn"] = attn
        if _raw_on("GLM_L2_PREFETCH_MLA_AR"):
            _wrap_o_proj_b(attn)
    return n


def install_model(mod) -> None:
    cls = mod.Glm5NextModel
    if getattr(cls, "_glm_l2_mla_link", False):
        return
    cls._glm_l2_mla_link = True
    orig = cls.forward

    def forward(self, *args, **kwargs):
        if not self.__dict__.get("_glm_l2_mla_linked"):
            n = link_model(self)
            self.__dict__["_glm_l2_mla_linked"] = True
            _log(f"linked {n} MLA layers (M={_raw_on('GLM_L2_PREFETCH_MLA')}, "
                 f"B-MLA={_raw_on('GLM_L2_PREFETCH_MLA_AR')})")
        return orig(self, *args, **kwargs)

    cls.forward = forward
    if _raw_on("GLM_L2_PREFETCH_MLA_AR") or _raw_on("GLM_L2_PREFETCH_DRAFT"):
        try:
            _base().install_roce()  # idempotent
        except Exception as exc:  # noqa: BLE001
            _log(f"RoCE hook unavailable ({exc!r}); B-MLA / D arm but nothing consumes them")


# -- drafter (window D) ----------------------------------------------------------------------------------

def _conv_weight(conv):
    kp = getattr(conv, "kernel_projection", None) if conv is not None else None
    return getattr(kp, "weight", None) if kp is not None else None


def plan_d(owner, tensor):
    """Window-D table for one conv kernel_projection weight, cached on `owner` per budget."""
    base = _base()
    budget = base._mib("GLM_L2_PREFETCH_DRAFT_MB", "8")
    cache = owner.__dict__.setdefault("_glm_l2_d", {})
    p = cache.get(budget)
    if p is None:
        s = tensor_span(tensor)
        segs = base.take([s], budget) if s else []
        p = base._table(segs) if segs else False
        cache[budget] = p
        _once("d", f"window D (drafter all-reduce -> next kernel_projection) {p[2] / 2**20 if p else 0:.2f} MiB")
    return p


def _on_d(n: int) -> bool:
    base = _base()
    return (0 < n <= int(base.env("GLM_L2_PREFETCH_DRAFT_MAXTOK", "64")) and _state["draft_depth"] > 0
            and base._on("GLM_L2_PREFETCH") and base._on("GLM_L2_PREFETCH_AR") and base._on("GLM_L2_PREFETCH_DRAFT"))


def _wrap_arming(linear, owner, tensor) -> None:
    """Arm the table of `tensor` right before the all-reduce inside `linear` (a RowParallelLinear)."""
    if linear is None or tensor is None or linear.__dict__.get("_glm_l2_dw"):
        return
    orig = linear.forward

    def forward(x, *args, **kwargs):
        base = _base()
        armed = None
        if _on_d(_rows(x)):
            p = owner.__dict__.get("_glm_l2_d", {}).get(base._mib("GLM_L2_PREFETCH_DRAFT_MB", "8"))
            if p is None and not _capturing():
                p = plan_d(owner, tensor)
            if p:
                armed = p
                base._state["armed"] = p
                _state["arms_d"] += 1
        try:
            return orig(x, *args, **kwargs)
        finally:
            if armed is not None and base._state["armed"] is armed:
                base._state["armed"] = None

    linear.forward = forward
    linear.__dict__["_glm_l2_dw"] = True


def link_draft(model) -> int:
    layers = list(getattr(model, "layers", []))
    n = 0
    for i, layer in enumerate(layers):
        mlp_conv, attn = getattr(layer, "mlp_conv", None), getattr(layer, "self_attn", None)
        if mlp_conv is None or attn is None:
            continue  # not a DFlash2 layer
        n += 1
        _wrap_arming(getattr(attn, "o_proj", None), mlp_conv, _conv_weight(mlp_conv))
        nxt = layers[i + 1] if i + 1 < len(layers) else None
        nconv = getattr(nxt, "attention_conv", None) if nxt is not None else None
        if nconv is not None:
            _wrap_arming(getattr(getattr(layer, "mlp", None), "down_proj", None), nconv, _conv_weight(nconv))
    return n


def install_draft(mod) -> None:
    cls = getattr(mod, "DFlashQwen3Model", None)
    if cls is None or getattr(cls, "_glm_l2_d", False):
        return
    cls._glm_l2_d = True
    orig = cls.forward

    def forward(self, *args, **kwargs):
        base = _base()
        if not self.__dict__.get("_glm_l2_d_linked"):
            n = link_draft(self)
            self.__dict__["_glm_l2_d_linked"] = True
            _log(f"drafter: {n} DFlash2 layers arm window D")
        _state["draft_depth"] += 1
        try:
            return orig(self, *args, **kwargs)
        finally:
            _state["draft_depth"] -= 1
            base._state["armed"] = None
            if _state["draft_depth"] == 0 and base._state["depth"] == 0:
                base.join_all()

    cls.forward = forward
    try:
        _base().install_roce()
    except Exception as exc:  # noqa: BLE001
        _log(f"RoCE hook unavailable ({exc!r}); window D arms but nothing consumes it")


HOOKS = {TARGET_ATTN: install_attn, TARGET_MODEL: install_model, TARGET_DRAFT: install_draft}


def active_hooks() -> dict:
    """The module hooks this process installs (window D only when GLM_L2_PREFETCH_DRAFT is set at start)."""
    hooks = dict(HOOKS)
    if not _raw_on("GLM_L2_PREFETCH_DRAFT"):
        hooks.pop(TARGET_DRAFT)
    return hooks


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        hooks = active_hooks()
        if name not in hooks:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module
        fn = hooks[name]

        def exec_module(module, _orig=orig_exec, _fn=fn):
            _orig(module)
            _fn(module)

        spec.loader.exec_module = exec_module
        return spec


def register() -> None:
    """Idempotent. No-op unless GLM_L2_PREFETCH and one of the three window switches are set (default off)."""
    if not installed():
        return
    pending = False
    for name, fn in active_hooks().items():
        if name in sys.modules:
            fn(sys.modules[name])
        else:
            pending = True
    if pending and not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
