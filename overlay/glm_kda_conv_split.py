# SPDX-License-Identifier: Apache-2.0
"""GLM_KDA_CONV_SPLIT=1: the KDA prefill short conv produces q, k and v as three dense tensors. Bit-exact intent.

Tony v11 image (vLLM 0.1.dev20051+g487ecf187), `vllm.models.glm5next.nvidia.kda` (the repo's glm5next_kda.py).

Stock prefill (Glm5NextLinearAttention._forward): one merged `causal_conv1d_fn` over q|k|v (3 x 2048 channels per
rank), output [3P, T] channel-last, i.e. [T, 3P] row-major after the caller's transpose; the caller splits it into
q / k / v views with a 3P token stride, and `chunk_kda_with_fused_gate` (fla) calls `.contiguous()` on each: three
[T, 2048] bf16 copies per KDA layer and prefill chunk (34 layers; 47 ms of a 2554 ms 5760-row chunk, 1.8 %, in the
2026-09-27 profile, diagnostics/glm-prefill2-20260928/budget_p32k_5760.txt).

Here, without touching `_forward` (its source is hash-pinned by glm_exact_hooks):
  * the kda module's `causal_conv1d_fn` is wrapped: an eligible merged call runs the stock conv three times, on the
    q, k and v channel slices of x / weight / bias / conv_states (the conv is independent per channel, so every
    output element and every conv-state write is the stock one; the image's own comment in _forward says the merged
    call is bit-identical to three calls). Each output is dense [T, P]. The wrapper returns a correctly shaped,
    uninitialised carrier with the stock output's strides, and remembers the three dense tensors;
  * the kda module's `chunk_kda_with_fused_gate` is wrapped: when its q / k / v are exactly the caller's split views
    of the carrier, the dense tensors are passed instead (same values, contiguous, so fla's `.contiguous()` is a no-op). If anything
    else arrives while a carrier is pending, the carrier is first filled with the real values (one copy), so a
    missed substitution can cost time but never correctness.
An overlay that wraps the chunk name after this one must call take() first and mark its wrapper
`__glm_conv_split_aware__` (glm_flashkda does); a non-aware outer wrapper just disables the split (stock conv).
Only eager calls (never under CUDA-graph capture); decode uses `causal_conv1d_update`, untouched.

In-boot A/B: GLM_KDA_CONV_SPLIT is a glm_ab KNOWN key (bool, registered in sitecustomize): the install gate is the
union of the variants, each conv call reads the runtime variant.

Credits: the idea of a conv that writes q, k and v as separate dense tensors is from Matt Mastracci (mmastrac,
GLM-5.3-Flash 4x GX10 recipe, PR #4, `causal_conv1d` out_group); independent implementation, no code copied.
"""
from __future__ import annotations

import os
import sys

TARGET = "vllm.models.glm5next.nvidia.kda"
_OFF = ("", "0", "off", "false", "no")
ENABLED = os.environ.get("GLM_KDA_CONV_SPLIT", "0").strip().lower() not in _OFF
S = {"pending": None, "prev_conv": None, "prev_chunk": None, "chunk_fn": None, "mod": None,
     "taken": 0, "substituted": 0, "filled": 0, "logged": 0}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-kda-conv-split: {msg}\n")
    sys.stderr.flush()


def call_on() -> bool:
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False) and "GLM_KDA_CONV_SPLIT" in ab.KNOWN:
        return ab.truthy(ab.env("GLM_KDA_CONV_SPLIT", "0"))
    return ENABLED


def _capturing() -> bool:
    import torch
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def eligible(x, weight, bias, conv_states) -> int:
    """Channels per slice P, or 0 when the merged call is not the KDA q|k|v prefill conv we expect."""
    if x is None or x.dim() != 2 or weight is None or weight.dim() != 2 or conv_states is None:
        return 0
    dim = x.shape[0]
    if dim % 3 or weight.shape[0] != dim or conv_states.dim() != 3 or conv_states.shape[1] != dim:
        return 0
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != dim):
        return 0
    if x.stride(0) != 1:          # the stock conv only runs channel-last; anything else stays stock
        return 0
    return dim // 3


def _is_split_view(t, carrier, off, P) -> bool:
    try:
        return (t.untyped_storage().data_ptr() == carrier.untyped_storage().data_ptr()
                and t.storage_offset() == carrier.storage_offset() + off
                and t.numel() == carrier.shape[0] * P and t.dtype == carrier.dtype)
    except (AttributeError, RuntimeError):
        return False


def _fill(pend) -> None:
    import torch
    carrier, P, dense = pend
    torch.cat(dense, dim=1, out=carrier)
    S["filled"] += 1
    if S["filled"] <= 3:
        _log(f"carrier filled instead of substituted ({S['filled']}x): an unexpected consumer")


def take(q, k, v):
    """(q, k, v) to hand to a KDA prefill kernel: the dense tensors when these are the pending carrier's split views,
    else unchanged (a pending carrier is then filled first). Idempotent. Any other overlay that wraps
    chunk_kda_with_fused_gate after this one and reads q / k / v itself must call take() first and set
    `__glm_conv_split_aware__ = True` on its wrapper; otherwise the split is simply not used."""
    pend = S["pending"]
    S["pending"] = None
    if pend is None:
        return q, k, v
    carrier, P, dense = pend
    if (q is not None and k is not None and v is not None and _is_split_view(q, carrier, 0, P)
            and _is_split_view(k, carrier, P, P) and _is_split_view(v, carrier, 2 * P, P)):
        S["substituted"] += 1
        return tuple(d.view(t.shape) for d, t in zip(dense, (q, k, v)))
    _fill(pend)
    return q, k, v


def _chunk_wrapper(*args, **kw):
    if S["pending"] is not None:
        if args:
            _fill(S["pending"])
            S["pending"] = None
        else:
            kw = dict(kw)
            kw["q"], kw["k"], kw["v"] = take(kw.get("q"), kw.get("k"), kw.get("v"))
    return S["prev_chunk"](*args, **kw)


_chunk_wrapper.__glm_conv_split_aware__ = True


def _chain_ok(mod) -> bool:
    """True when the chunk_kda name the caller will use substitutes before anything reads q / k / v."""
    fn = mod.chunk_kda_with_fused_gate
    return fn is S["chunk_fn"] or bool(getattr(fn, "__glm_conv_split_aware__", False))


def install(mod) -> None:
    if getattr(mod, "__glm_kda_conv_split__", False):
        return
    import torch

    prev_conv = mod.causal_conv1d_fn
    S["prev_conv"], S["mod"] = prev_conv, mod
    S["prev_chunk"] = mod.chunk_kda_with_fused_gate
    S["chunk_fn"] = _chunk_wrapper
    mod.chunk_kda_with_fused_gate = _chunk_wrapper

    def conv(x, weight, bias=None, *args, **kw):
        pend = S["pending"]
        if pend is not None:          # a carrier nobody consumed: make it real before it can be read
            S["pending"] = None
            _fill(pend)
        conv_states = kw.get("conv_states", args[0] if args else None)
        P = eligible(x, weight, bias, conv_states) if not args else 0
        if not P or not call_on() or _capturing() or not _chain_ok(mod):
            if P and call_on() and not _chain_ok(mod) and S["logged"] < 3:
                S["logged"] += 1
                _log("chunk_kda_with_fused_gate was wrapped by a non-aware overlay: split not used")
            return prev_conv(x, weight, bias, *args, **kw)
        outs = []
        for j in range(3):
            sl = slice(j * P, (j + 1) * P)
            kwj = dict(kw)
            kwj["conv_states"] = conv_states[:, sl]
            outs.append(prev_conv(x[sl], weight[sl], None if bias is None else bias[sl], **kwj))
        n = x.shape[1]
        dense = []
        for o in outs:                 # [P, T] with strides (1, P) -> [T, P] dense
            d = o.transpose(0, 1)
            if not d.is_contiguous():
                d = d.contiguous()     # a conv build that allocates channel-first: still correct
            dense.append(d)
        carrier = torch.empty((n, 3 * P), dtype=dense[0].dtype, device=x.device)
        S["pending"] = (carrier, P, dense)
        S["taken"] += 1
        if S["logged"] < 2:
            S["logged"] += 1
            _log(f"split conv taken: T={n} P={P} dense={[tuple(d.stride()) for d in dense]}")
        return carrier.transpose(0, 1)   # [3P, T], strides (1, 3P): the stock output's layout

    conv.__wrapped__ = prev_conv
    mod.causal_conv1d_fn = conv
    mod.__glm_kda_conv_split__ = True
    _log("installed (causal_conv1d_fn + chunk_kda_with_fused_gate wrappers)")


def register() -> None:
    """From sitecustomize: install right after the kda module executes (chains with the other finders)."""
    import glm_prefill_hooks
    glm_prefill_hooks.after_import(TARGET, install)
