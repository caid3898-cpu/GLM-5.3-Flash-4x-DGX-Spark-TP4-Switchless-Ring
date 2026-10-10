# SPDX-License-Identifier: Apache-2.0
"""GLM_L2PF_HC / GLM_L2PF_ROUTER: corrected and extended all-reduce L2 prefetch tables. Exact (a prefetch is only a
cache hint; no tensor, kernel or reduction order changes). Default off. Needs GLM_L2_PREFETCH=1 + GLM_L2_PREFETCH_AR=1
(glm_l2_prefetch: kernel, side stream, RoCE hook, join); window C additionally GLM_L2_PREFETCH_MOE=1.

Why (diagnostics/glm-tinykernels-20260928, profile glm-prof2-20260928 p21905, all-on base, c1):
  * window B (post-attention all-reduce, KDA via glm_l2_prefetch, MLA via glm_l2_prefetch_mla) queues the layer's
    hc_ffn_fn FP32 parameter (1.5 MiB) first. With GLM_MHC_BF16W on, the fused mHC kernel reads the BF16 twin
    (0.75 MiB) instead, so the first 1.5 MiB of the 4 MiB budget warm a tensor nobody reads and the twin stays cold.
    Window C (MoE/MLP output all-reduce, glm_l2_prefetch_c) has the same problem with the next layer's hc_attn_fn.
  * the MoE router region (router GEMM + reduce + cast + top-k + align + sort, 37 us/layer) overlaps the shared
    expert on the aux stream (39.5 us busy): together they read router 2.36 MB + shared 6.3 MB at ~217 GB/s, i.e. the
    region is DRAM-bound in aggregate, which is why a faster router GEMV (GLM_ROUTER_GEMV) did not move the step. Only
    fewer DRAM bytes on the critical path help: warm them in L2 while the all-reduce waits (DRAM idle).

GLM_L2PF_HC=1      window B: the hc_ffn_fn entry becomes the tensor the mHC kernel actually reads (its BF16 twin when
                   glm_mhc_bf16w is live for the call, else the FP32 parameter); window C: same for the next layer's
                   hc_attn_fn. Budgets unchanged (GLM_L2_PREFETCH_AR_MB, GLM_L2_PREFETCH_MOE_MB).
GLM_L2PF_ROUTER=1  window B: queue hc_ffn (as above if GLM_L2PF_HC, else stock) -> router -> shared gate_up ->
                   shared down (dense layers: gate_up -> down) with budget GLM_L2PF_ROUTER_MB (default 10).
Mechanics: the base tables stay as they are (built by glm_l2_prefetch._plan_b / glm_l2_prefetch_c.plan_c on an eager
forward); this module wraps those two builders to register each base table and build its alternatives right away
(eager, never during capture), and wraps glm_l2_prefetch._fork (the call the RoCE hook uses) to substitute the
alternative for the current flags. Switches are read per call through overlay/glm_ab.py (per captured variant).
"""
from __future__ import annotations

import os
import sys

_OFF = ("", "0", "off", "false", "no")
KEYS = ("GLM_L2PF_HC", "GLM_L2PF_ROUTER")
_REG: dict = {}          # id(base table) -> {"kind": "B"|"C", "alts": {(hc, router, bf16w): table|False}}
S = {"subst": 0, "built": 0, "logged": set()}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-l2pf-v2: {msg}\n")


def env(name, default=None):
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def on(name) -> bool:
    return str(env(name, "0")).strip().lower() not in _OFF


def _bf16w_live() -> bool:
    m = sys.modules.get("glm_mhc_bf16w")
    return bool(m is not None and m.S.get("ok") and m._on())


def _hc_entry(fn, use_twin: bool):
    """(ptr, nbytes) of what the mHC kernel reads for this hc matrix."""
    if fn is None or not getattr(fn, "is_cuda", False) or not fn.is_contiguous():
        return []
    if use_twin:
        m = sys.modules.get("glm_mhc_bf16w")
        tw = m.S["twins"].get(fn.data_ptr()) if m is not None else None
        if tw is not None:
            return [(tw.data_ptr(), tw.numel() * tw.element_size())]
    return [(fn.data_ptr(), fn.numel() * fn.element_size())]


def _mib(name, default):
    return int(float(env(name, default)) * (1 << 20))


def queue_b(layer, hc_fix: bool, bf16w: bool, router: bool):
    import glm_l2_prefetch as base
    q = _hc_entry(getattr(layer, "hc_ffn_fn", None), hc_fix and bf16w)
    mlp = getattr(layer, "mlp", None)
    gate = getattr(mlp, "gate", None)
    if gate is not None:
        q += base.module_tensors(gate)
    shared = getattr(mlp, "shared_experts", None)
    owner = shared if shared is not None else mlp
    for name in (("gate_up_proj", "down_proj") if router else ("gate_up_proj",)):
        mod = getattr(owner, name, None)
        if mod is not None:
            q += base.module_tensors(mod)
    budget = _mib("GLM_L2PF_ROUTER_MB", "10") if router else base._mib("GLM_L2_PREFETCH_AR_MB", "4")
    return base.take(q, budget)


def queue_c(nxt, bf16w: bool):
    import glm_l2_prefetch as base
    import glm_l2_prefetch_c as wc
    if nxt is None:
        return []
    q = _hc_entry(getattr(nxt, "hc_attn_fn", None), bf16w)
    proj = wc.first_proj(getattr(nxt, "self_attn", None))
    if proj is not None:
        q += base.module_tensors(proj)
    return base.take(q, wc._budget())


def _build(segs):
    import glm_l2_prefetch as base
    S["built"] += 1
    return base._table(segs) if segs else False


def _register_b(table, attn):
    layer = attn.__dict__.get("_glm_l2_layer")
    if not table or layer is None:
        return
    alts = {}
    for hc in (0, 1):
        for router in (0, 1):
            for bw in (0, 1):
                if hc or router:
                    alts[(hc, router, bw)] = _build(queue_b(layer, bool(hc), bool(bw), bool(router)))
    _REG[id(table)] = {"kind": "B", "alts": alts, "base": table}
    if "b" not in S["logged"]:
        S["logged"].add("b")
        _log("window B alternatives (MiB): " + ", ".join(
            f"hc{k[0]}/router{k[1]}/bf16w{k[2]}={(v[2] / 2**20 if v else 0):.2f}" for k, v in sorted(alts.items()))
             + f"; base {table[2] / 2**20:.2f}")


def _register_c(table, mlp):
    if not table:
        return
    nxt = mlp.__dict__.get("_glm_l2_next")
    alts = {(1, 0, bw): _build(queue_c(nxt, bool(bw))) for bw in (0, 1)}
    _REG[id(table)] = {"kind": "C", "alts": alts, "base": table}
    if "c" not in S["logged"]:
        S["logged"].add("c")
        _log("window C alternatives (MiB): " + ", ".join(
            f"bf16w{k[2]}={(v[2] / 2**20 if v else 0):.2f}" for k, v in sorted(alts.items()))
             + f"; base {table[2] / 2**20:.2f}")


def substitute(item):
    """The table to fork for the current flags (the base table when no lever applies)."""
    r = _REG.get(id(item)) if item else None
    if r is None:
        return item
    hc = on("GLM_L2PF_HC")
    router = on("GLM_L2PF_ROUTER") and r["kind"] == "B"
    if not (hc or router):
        return item
    alt = r["alts"].get((int(hc), int(router), int(_bf16w_live())))
    if alt is None:
        return item
    S["subst"] += 1
    return alt if alt else item


def install() -> None:
    import glm_l2_prefetch as base
    if getattr(base, "_glm_l2pf_v2", False):
        return
    base._glm_l2pf_v2 = True
    orig_plan_b, orig_fork = base._plan_b, base._fork

    def _plan_b(attn):
        p = orig_plan_b(attn)
        if p and id(p) not in _REG:
            _register_b(p, attn)
        return p

    def _fork(item):
        return orig_fork(substitute(item))

    base._plan_b, base._fork = _plan_b, _fork
    wc = sys.modules.get("glm_l2_prefetch_c")
    if wc is None and os.environ.get("GLM_L2_PREFETCH_MOE", "0").strip().lower() not in _OFF:
        import glm_l2_prefetch_c as wc  # noqa: F811
    if wc is not None and not getattr(wc, "_glm_l2pf_v2", False):
        wc._glm_l2pf_v2 = True
        orig_plan_c = wc.plan_c

        def plan_c(mlp):
            p = orig_plan_c(mlp)
            if p and id(p) not in _REG:
                _register_c(p, mlp)
            return p

        wc.plan_c = plan_c
    _log("installed: window B/C tables substituted per call (" +
         " ".join(f"{k}={os.environ.get(k, '0')}" for k in KEYS) + ")")


def register() -> None:
    if not any(os.environ.get(k, "0").strip().lower() not in _OFF for k in KEYS):
        return
    if os.environ.get("GLM_L2_PREFETCH", "0").strip().lower() in _OFF:
        _log("GLM_L2_PREFETCH is off: nothing to extend")
        return
    install()
