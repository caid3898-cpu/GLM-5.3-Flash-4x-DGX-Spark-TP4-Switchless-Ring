# SPDX-License-Identifier: Apache-2.0
"""GLM_TRITON_MLA_PREFILL=0|1: Triton sparse-MLA attention for the eager (prefill / mixed) steps.

Idea and GB10 measurements: Matt Mastracci (mmastrac), GLM-5.3-Flash 4x GX10 recipe PR #4; independent
reproduction by chuck-ads. Independent reimplementation, no code copied (kernel: glm_sparse_mla_kernel.py).

Where it sits. With FLASHINFER_MLA_SPARSE_SM90 and no dense-MHA prefill backend, every row of the 11 DSA
layers (decode and prefill alike) goes MLAAttention.forward_impl -> W_UK bmm -> impl.forward_mqa((ql_nope,
q_pe), kv_cache, attn_metadata, layer) -> _v_up_proj (W_UV bmm). The stock forward_mqa converts the top-k
request-local ids to global slots (triton_convert_req_index_to_global_index, compacted), clamps the -1 tail,
copies them into the wrapper's reserved kv_indices and runs BatchMLAPagedAttention (FA2 on SM121) with the
per-row lengths planned host-side by the metadata builder. This adapter wraps forward_mqa and, for an eager step
that carries prefill rows, computes ALL of that step's rows (the plan covers the whole batch, so the decode rows
of a mixed step go the same way) with one Triton launch that gathers the fp8 latent rows itself (block_table +
req_id_per_token; no convert kernel, no clamp, no copies, no host lengths). It returns the same [T, H, 512] bf16
tensor forward_mqa returns; W_UV runs unchanged after it.

FlashInfer stays for decode: CUDA-graph FULL_DECODE_ONLY steps are captured (this wrapper declines under
capture, so the graphs bake the stock kernel) and replayed without Python; eager decode-only steps also stay stock.
The builder still plans every step (unchanged; the plan is simply unused on the steps taken here).

Per call (rank-invariant: every input of the decision is batch metadata identical on every TP rank):
  * the switch: GLM_TRITON_MLA_PREFILL, read through glm_ab.env when the in-boot A/B harness is armed and knows
    the key (so a variant can turn it on or off), else the install gate;
  * never under CUDA-graph capture;
  * the step carries prefill rows (attn_metadata.num_prefills > 0) and >= GLM_TRITON_MLA_MIN_ROWS rows (1);
  * layouts are exactly the qualified ones (NoPE, D 512, fp8_e4m3 or bf16 cache, int32 top-k/block table/
    req ids, unit inner strides, no DCP); anything else, or any exception from the Triton path, goes stock.
    An exception also disables the Triton path for the rest of the process (logged once).

Measured (diagnostics/glm-tmla-20260928, one GB10, T 6912 rows, 16 heads, top-k 2048, fp8 cache): stock forward_mqa
49.1 / 50.4 / 42.5 ms (32k / 128k context / causal chunk from 0) vs Triton 13.8 / 14.6 / 13.0 ms (3.3-3.6x, 31-34
TFLOPS); error vs an fp32 reference equal to stock's (rel L2 1.77e-3 vs 1.76e-3), Triton vs stock rel L2 1.6e-3.

Knobs: GLM_TRITON_MLA_CFG="BN,warps,stages[,split[,skip]]" (tile width, num_warps, num_stages, latent split, skip
empty tiles; default in the kernel module),
GLM_TRITON_MLA_MIN_ROWS (1), GLM_TRITON_MLA_CHECK=N (the first N Triton calls also run stock and log the max abs /
rel L2 difference; debug only, doubles the MLA time of those calls).
"""
from __future__ import annotations

import os
import sys

KEY = "GLM_TRITON_MLA_PREFILL"
TARGET = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"
# sha256[:16] of glm_prefill_hooks.func_source() in the qualified image (vLLM 487ecf187, tonyd2wild v11,
# extracted 2026-09-18). The Triton path replaces forward_mqa's semantics, so a drift refuses to install.
EXPECTED = {
    f"{TARGET}:FlashInferMLASparseSM90Impl.forward_mqa": "ab7dc07a2c8fa5b8",
    "vllm.model_executor.layers.attention.mla_attention:MLAAttention._v_up_proj": "57a84e6f2b92512d",
    "vllm.v1.attention.backends.mla.sparse_utils:triton_convert_req_index_to_global_index": "9753ba703712a78d",
}
_OFF = ("", "0", "off", "false", "no")

_state = {"disabled": False, "calls": 0, "stock": 0, "checks": 0, "reasons": {}, "installed": False}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-tmla: {msg}\n")


def _on(value) -> bool:
    return value is not None and str(value).strip().lower() not in _OFF


def wanted(env=None) -> bool:
    env = os.environ if env is None else env
    return _on(env.get(KEY, "0"))


def _cfg(env=None):
    env = os.environ if env is None else env
    raw = env.get("GLM_TRITON_MLA_CFG", "").strip()
    if not raw:
        return None
    parts = [int(x) for x in raw.split(",")]
    if len(parts) == 3:
        parts.append(1)
    if len(parts) == 4:
        parts.append(0)
    if len(parts) != 5:
        raise RuntimeError(f"GLM_TRITON_MLA_CFG={raw!r}: expected BN,warps,stages[,split[,skip]]")
    bn, warps, stages, split, skip = parts
    if bn not in (16, 32, 64, 128) or warps not in (1, 2, 4, 8, 16) or not 1 <= stages <= 6 \
            or split not in (1, 2, 4, 8, 16) or skip not in (0, 1):
        raise RuntimeError(f"GLM_TRITON_MLA_CFG={raw!r}: BN in 16|32|64|128, warps 1..16, stages 1..6, "
                           "split in 1|2|4|8|16, skip 0|1")
    return bn, warps, stages, split, skip


def _ab():
    ab = sys.modules.get("glm_ab")
    return ab if ab is not None and getattr(ab, "ACTIVE", False) else None


def call_enabled() -> bool:
    """The switch for this forward: the install gate, or the runtime A/B variant's value when glm_ab is armed and
    knows the key (the in-boot switch is a collective at the same step on every rank)."""
    if _state["disabled"]:
        return False
    ab = _ab()
    if ab is None or KEY not in ab.KNOWN:
        return True
    return _on(ab.env(KEY, "0"))


def _capturing(torch) -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001  (no CUDA: tests)
        return False


def decline_reason(impl, q, kv_cache, md, torch, min_rows: int = 1):
    """None when the Triton path takes this call, else a short reason (the call then goes stock)."""
    if not call_enabled():
        return "off"
    if _capturing(torch):
        return "capture"
    if md is None:
        return "no-metadata"
    if int(getattr(md, "num_prefills", 0) or 0) <= 0:
        return "no-prefill"
    if not isinstance(q, tuple) or len(q) != 2:
        return "q-not-split"
    q_nope, q_rope = q
    T = q_nope.shape[0] if q_nope.dim() == 3 else -1
    if T < max(int(min_rows), 1):
        return "rows"
    if int(getattr(impl, "qk_rope_head_dim", -1)) != 0 or q_rope.shape[-1] != 0:
        return "rope"
    H = int(getattr(impl, "num_heads", 0))
    D = int(getattr(impl, "kv_lora_rank", 0))
    if D != 512 or H not in (16, 32, 64) or tuple(q_nope.shape[1:]) != (H, D):
        return "shape"
    if q_nope.dtype != torch.bfloat16 or q_nope.stride(-1) != 1:
        return "q-layout"
    if int(getattr(impl, "dcp_world_size", 1) or 1) != 1:
        return "dcp"
    kvd = getattr(impl, "kv_cache_dtype", "")
    if kvd not in ("fp8", "fp8_e4m3", "auto", "bfloat16"):
        return "kv-dtype"
    if kv_cache.dim() != 3 or kv_cache.shape[-1] < D or kv_cache.stride(-1) != 1:
        return "kv-layout"
    fp8 = kvd in ("fp8", "fp8_e4m3")
    if fp8 and kv_cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        return "kv-dtype"
    if not fp8 and kv_cache.dtype != torch.bfloat16:
        return "kv-dtype"
    if int(getattr(md, "block_size", -1)) != kv_cache.shape[1]:
        return "block-size"
    buf = getattr(impl, "topk_indices_buffer", None)
    bt = getattr(md, "block_table", None)
    rid = getattr(md, "req_id_per_token", None)
    if buf is None or bt is None or rid is None:
        return "no-topk"
    if buf.dtype != torch.int32 or buf.dim() != 2 or buf.shape[0] < T or buf.stride(-1) != 1:
        return "topk-layout"
    if bt.dtype != torch.int32 or bt.dim() != 2 or rid.dtype != torch.int32 or rid.shape[0] < T:
        return "bt-layout"
    return None


def make_forward_mqa(prev, kernel_fn=None, env=None):
    """Wrapper for FlashInferMLASparseSM90Impl.forward_mqa chaining to `prev` for every call it does not take.
    kernel_fn(q, kv_cache, topk, req_id, block_table, sm_scale, kv_scale, cfg) -> out [T, H, D] (tests inject)."""
    env = os.environ if env is None else env
    min_rows = int(env.get("GLM_TRITON_MLA_MIN_ROWS", "1"))
    cfg = _cfg(env)
    check_n = int(env.get("GLM_TRITON_MLA_CHECK", "0") or 0)

    def _kernel():
        nonlocal kernel_fn
        if kernel_fn is None:
            from glm_sparse_mla_kernel import sparse_mla_fwd

            def kernel_fn(q, kv, topk, rid, bt, sm_scale, kv_scale, cfg):  # noqa: E306
                return sparse_mla_fwd(q, kv, topk, rid, bt, sm_scale, kv_scale, cfg=cfg)
        return kernel_fn

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        import torch

        why = decline_reason(self, q, kv_c_and_k_pe_cache, attn_metadata, torch, min_rows)
        if why is not None:
            _state["stock"] += 1
            _state["reasons"][why] = _state["reasons"].get(why, 0) + 1
            return prev(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
        try:
            q_nope = q[0]
            T = q_nope.shape[0]
            kv = kv_c_and_k_pe_cache
            fp8 = self.kv_cache_dtype in ("fp8", "fp8_e4m3")
            if fp8 and kv.dtype != torch.float8_e4m3fn:
                kv = kv.view(torch.float8_e4m3fn)
            kv_scale = float(getattr(layer, "_k_scale_float", 1.0) or 1.0) if fp8 else 1.0
            out = _kernel()(q_nope, kv, self.topk_indices_buffer[:T],
                            attn_metadata.req_id_per_token[:T], attn_metadata.block_table,
                            float(self.scale), kv_scale, cfg)
        except Exception as exc:  # noqa: BLE001
            _state["disabled"] = True
            _log(f"Triton path failed, stock for the rest of this process: {exc!r}")
            return prev(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
        _state["calls"] += 1
        if _state["calls"] == 1:
            _log(f"first Triton call: rows {T}, prefills {attn_metadata.num_prefills}, cfg {cfg or 'default'}")
        if _state["checks"] < check_n:
            _state["checks"] += 1
            ref, _ = prev(self, q, kv_c_and_k_pe_cache, attn_metadata, layer)
            d = (out.float() - ref.float())
            rel = float(d.norm() / ref.float().norm().clamp_min(1e-30))
            _log(f"check {_state['checks']}/{check_n}: rows {T} max_abs {float(d.abs().max()):.3e} rel_l2 {rel:.3e}")
        return out, None

    forward_mqa._glm_tmla_prev = prev
    return forward_mqa


def install(mod, env=None) -> None:
    """After-import hook for TARGET: drift guard, then wrap forward_mqa (chaining to the current binding)."""
    env = os.environ if env is None else env
    if env.get("GLM_TRITON_MLA_ALLOW_DRIFT", "0") != "1":
        import glm_prefill_hooks
        glm_prefill_hooks.check_sources(EXPECTED, "glm-tmla")
    cls = mod.FlashInferMLASparseSM90Impl
    prev = cls.forward_mqa
    if getattr(prev, "_glm_tmla_prev", None) is not None:
        return
    cls.forward_mqa = make_forward_mqa(prev, env=env)
    _state["installed"] = True
    _log(f"installed on {cls.__name__}.forward_mqa (cfg {env.get('GLM_TRITON_MLA_CFG', '') or 'default'}, "
         f"min rows {env.get('GLM_TRITON_MLA_MIN_ROWS', '1')}, glm_ab {'armed' if _ab() else 'off'})")


def status() -> dict:
    return {"installed": _state["installed"], "disabled": _state["disabled"], "triton_calls": _state["calls"],
            "stock_calls": _state["stock"], "reasons": dict(_state["reasons"])}


def register(env=None) -> bool:
    """Entry point from sitecustomize. Inert (imports nothing) unless GLM_TRITON_MLA_PREFILL is on."""
    env = os.environ if env is None else env
    if not wanted(env):
        return False
    _cfg(env)  # a malformed config is fatal at startup, not at the first prefill
    import glm_prefill_hooks
    glm_prefill_hooks.after_import(TARGET, lambda mod: install(mod, env))
    return True
