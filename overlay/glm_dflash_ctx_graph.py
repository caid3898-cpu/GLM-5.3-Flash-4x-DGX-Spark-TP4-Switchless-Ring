# SPDX-License-Identifier: Apache-2.0
"""GLM_DFLASH_CTX_GRAPH=1 (+ GLM_DFLASH_CTX_GRAPH_CHECK=1): replay the DFlash context-KV precompute from exact-shape CUDA graphs. Exact.

Tony v11 image (vLLM 0.1.dev20051+g487ecf187), V2 model runner, `vllm.v1.worker.gpu.spec_decode.dflash.speculator`
(DFlash2Speculator inherits propose/capture from DFlashSpeculator).

Every propose() runs `self.model.precompute_and_store_context_kv(self.hidden_states[:N], self.context_positions[:N],
slots)` eagerly (N = the target step's token count): fused RMSNorm, one K/V GEMM for the 5 draft layers, a permute
copy, grouped K RMSNorm, RoPE and 5 KV-cache inserts, ~13 launches between the target and the draft graphs. Its
inputs always live in stable buffers (speculator.hidden_states, .context_positions, ._context_slot_mappings rows,
all written by prepare_dflash_inputs / propose before the call), so a graph captured per exact N replays it.

  * capture: right after DFlashSpeculator.capture(), one FULL graph per N in 1..GLM_DFLASH_CTX_GRAPH_MAX (default
    256 = the target's max capture size), into the global graph pool, inside vLLM's graph_capture context; the slot
    rows are filled with PAD_SLOT_ID during capture so the warm-up/capture calls write no KV;
  * replay: only when N has a graph and the call's tensors are exactly the captured buffer slices (pointer, length);
    anything else (dummy runs, N > max, a different buffer) runs the original eager method. No padding, no bucket:
    the same kernels on the same shapes, so the cache bytes are identical to eager.
  * check mode (GLM_DFLASH_CTX_GRAPH_CHECK=1, TEST ONLY): the captured graphs also copy the K/V fed to each cache insert
    into static buffers; after every replay the eager method runs again (its cache writes are the reference) and the
    K/V are compared bitwise per layer. Counts are logged every GLM_DFLASH_CTX_GRAPH_LOG_EVERY (500) calls.

Credit: the design is vLLM's DFlash context-KV CUDA graph work in the entrpi live checkout
(`DFlashContextCudaGraphManager` / `_capture_context_kv` in vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py,
"[GG] ... capture DSpark context KV (#251)"), here with exact row counts instead of capture buckets so no padded row
reaches a GEMM; the DS4.1 eager-glue work (ds41 adapter/eager_glue.py, ours) motivated it.
"""
from __future__ import annotations

import os
import sys

TARGET = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
_OFF = ("", "0", "off", "false", "no")
S = {"graphs": {}, "replays": 0, "eager": 0, "checked": 0, "mismatch": 0, "rec": None, "dbg": None,
     "capturing": False, "calls": 0, "disabled": False}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-dflash-ctx-graph: {msg}\n")


def env(name: str, default=None):
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def _mode() -> str:
    return str(env("GLM_DFLASH_CTX_GRAPH", "0")).strip().lower()


def _check_on() -> bool:
    """GLM_DFLASH_CTX_GRAPH_CHECK=1 (TEST ONLY; a separate variable so the in-boot harness's union of install gates,
    which rewrites GLM_DFLASH_CTX_GRAPH to 1, cannot drop it)."""
    return os.environ.get("GLM_DFLASH_CTX_GRAPH_CHECK", "0").strip().lower() not in _OFF \
        or os.environ.get("GLM_DFLASH_CTX_GRAPH", "").strip().lower() == "check"


def _ptrs(slots):
    if slots is None:
        return None
    if isinstance(slots, (list, tuple)):
        return tuple((s.data_ptr(), s.shape[0]) if s is not None else None for s in slots)
    return ((slots.data_ptr(), slots.shape[0]),)


def _same(a, b) -> bool:
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.element_size() == 1:
        a, b = a.view(torch.uint8), b.view(torch.uint8)
    return bool(torch.equal(a, b))


def _args(spec, n):
    if spec._layer_group_idx is not None:
        slots = [spec._context_slot_mappings[g][:n] for g in spec._layer_group_idx]
    else:
        slots = spec._context_slot_mappings[0][:n]
    return spec.hidden_states[:n], spec.context_positions[:n], slots


def _key(states, positions, slots):
    return (states.data_ptr(), states.shape[0], tuple(states.stride()), positions.data_ptr(), positions.shape[0],
            _ptrs(slots))


def _install_recorders(model) -> None:
    """check mode: wrap each draft layer's cache insert so it can record the K/V it writes."""
    layers = getattr(getattr(model, "model", model), "_attn_layers", None)
    if not layers:
        raise RuntimeError("draft model has no _attn_layers")
    for i, attn in enumerate(layers):
        impl = attn.impl
        if getattr(impl, "_glm_ctx_rec", False):
            continue
        orig = impl.do_kv_cache_update

        def rec(layer, key, value, kv_cache, slot_mapping, *a, _orig=orig, _i=i, **kw):
            r = S["rec"]
            if r is not None:
                r(_i, key, value)
            return _orig(layer, key, value, kv_cache, slot_mapping, *a, **kw)
        impl.do_kv_cache_update = rec
        impl._glm_ctx_rec = True


def _capture(spec) -> None:
    import torch
    from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
    from vllm.distributed.parallel_state import graph_capture
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    model = spec.model
    orig = model.precompute_and_store_context_kv
    check = _check_on()
    cc = spec.vllm_config.compilation_config
    maxn = min(int(os.environ.get("GLM_DFLASH_CTX_GRAPH_MAX", "256")), int(spec.max_num_tokens),
               int(cc.max_cudagraph_capture_size or 256))
    if check:
        _install_recorders(model)
    saved_slots = spec._context_slot_mappings.clone()
    saved_pos = spec.context_positions[:maxn].clone()
    spec._context_slot_mappings.fill_(PAD_SLOT_ID)
    spec.context_positions[:maxn].zero_()
    pool = current_platform.get_global_graph_pool()
    graphs = {}
    try:
        with graph_capture(device=spec.device):
            for n in range(maxn, 0, -1):  # largest first, like vLLM, so smaller graphs reuse pool blocks
                st, pos, slots = _args(spec, n)
                if check:
                    def alloc(i, k, v):
                        if S["dbg"] is None:
                            S["dbg"] = {}
                        if i not in S["dbg"]:
                            S["dbg"][i] = (torch.empty((maxn,) + tuple(k.shape[1:]), dtype=k.dtype, device=k.device),
                                           torch.empty((maxn,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device))
                    S["rec"] = alloc
                orig(st, pos, slots)  # warm-up (writes nothing: PAD slots)
                S["rec"] = None
                g = torch.cuda.CUDAGraph()
                set_graph_pool_id(pool)
                if check:
                    def into(i, k, v, _n=n):
                        S["dbg"][i][0][:_n].copy_(k)
                        S["dbg"][i][1][:_n].copy_(v)
                    S["rec"] = into
                S["capturing"] = True
                try:
                    with torch.cuda.graph(g, pool):
                        orig(st, pos, slots)
                finally:
                    S["capturing"] = False
                    S["rec"] = None
                graphs[n] = (g, _key(st, pos, slots))
    except Exception as exc:  # noqa: BLE001
        _log(f"capture failed at N={n}: {exc!r}; the precompute stays eager")
        graphs = {}
    finally:
        spec._context_slot_mappings.copy_(saved_slots)
        spec.context_positions[:maxn].copy_(saved_pos)
    S["graphs"] = graphs
    if graphs:
        _log(f"captured {len(graphs)} context-KV graphs (N = 1..{maxn}){' with K/V check buffers' if check else ''}")

    def precompute_and_store_context_kv(context_states, context_positions, context_slot_mapping=None):
        S["calls"] += 1
        mode = _mode()
        ent = S["graphs"].get(int(context_states.shape[0])) if mode not in _OFF else None
        if (ent is None or context_slot_mapping is None or S["capturing"]
                or torch.cuda.is_current_stream_capturing()
                or _key(context_states, context_positions, context_slot_mapping) != ent[1]):
            S["eager"] += 1
            return orig(context_states, context_positions, context_slot_mapping)
        ent[0].replay()
        S["replays"] += 1
        if S["dbg"] is not None and _check_on():
            n = int(context_states.shape[0])
            got = {i: (kv[0][:n].clone(), kv[1][:n].clone()) for i, kv in S["dbg"].items()}
            ref = {}
            S["rec"] = lambda i, k, v: ref.__setitem__(i, (k.clone(), v.clone()))
            try:
                orig(context_states, context_positions, context_slot_mapping)  # eager rewrite = reference bytes
            finally:
                S["rec"] = None
            S["checked"] += 1
            bad = 0
            for i, (k, v) in ref.items():
                gk, gv = got[i]
                if not (_same(gk, k) and _same(gv, v)):
                    bad += 1
            if bad or len(ref) != len(got):
                S["mismatch"] += 1
        every = int(os.environ.get("GLM_DFLASH_CTX_GRAPH_LOG_EVERY", "500") or 0)
        if every and S["calls"] % every == 0:
            _log(f"calls={S['calls']} replays={S['replays']} eager={S['eager']} checked={S['checked']} "
                 f"mismatch={S['mismatch']}")

    model.precompute_and_store_context_kv = precompute_and_store_context_kv


def install(mod) -> None:
    cls = mod.DFlashSpeculator
    if getattr(cls, "_glm_ctx_graph", False):
        return
    cls._glm_ctx_graph = True
    orig_capture = cls.capture

    def capture(self, *args, **kwargs):
        out = orig_capture(self, *args, **kwargs)
        try:
            _capture(self)
        except Exception as exc:  # noqa: BLE001
            _log(f"not installed: {exc!r}")
        return out

    cls.capture = capture
    _log("DFlash context-KV precompute will be captured per exact row count after the draft graphs")


def register() -> None:
    import importlib.abc
    import importlib.util

    if os.environ.get("GLM_DFLASH_CTX_GRAPH", "0").strip().lower() in _OFF:
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET:
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
                install(module)
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
