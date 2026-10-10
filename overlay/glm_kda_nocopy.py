# SPDX-License-Identifier: Apache-2.0
# Kernel body: copied from vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
# (fused_recurrent_gated_delta_rule_fwd_kernel), which is from flash-linear-attention
# (MIT, Copyright (c) 2023-2025, Songlin Yang, Yu Zhang). Only the q/k/v/g/beta addressing changed.
"""GLM_KDA_NOCOPY=1: the KDA recurrent kernel reads q/k/v/g/beta through their token stride. Exact.

Tony v11 image, vLLM 0.1.dev20051+g487ecf187, `vllm.models.glm5next.nvidia.kda`.

In `Glm5NextLinearAttention` the merged in-projection output is [T, 6416] (q|k|v 3 x 2048, beta 16,
f_a 128, g_a 128), the short conv writes q|k|v back in place, and `_forward` hands the kernel views:
q/k/v = [1, T, 16, 128] with token stride 6416 (6144 in mixed steps, after index_select), and
beta = [1, T, 16] with token stride 6416. `fla.ops.kda.fused_recurrent_kda` calls `.contiguous()` on all
five, i.e. four gather copies per KDA layer and step (34 layers; audit: 5.7-7.2 us per layer at T > 1,
0.19-0.25 ms per step; zero at T == 1 where the views are already contiguous).

This module swaps `fused_recurrent_kda` inside the kda module for a dispatcher that, when every input has a
dense head/channel layout and B == 1, launches a copy of the stock kernel whose only change is
    p = base + bos * stride_token + head * D + o          (stock: base + (bos * H + head) * D + o)
    p += stride_token                                      (stock: p += H * D)
for q, k, v, g and beta. The loaded values are the same elements the contiguous copy would hold, and the
arithmetic, tile shapes, num_warps and num_stages are unchanged, so outputs and states are bit-identical
(verified on GPU by gpu/gpu_exact.py kda). All other calls (prefill chunk path, non-dense layouts, already
contiguous inputs) go to the previous function unchanged. The Kimi-K3 recurrent kernel in the same image
already takes row strides, but its gate is written as lower_bound * sigmoid(...) and its BV tiling is
different, so it is not a bit-exact substitute; that is why the GLM kernel is copied here instead.

Composes with GLM_KDA_STASH: if the stash dispatcher is already installed when this one installs,
spec-verify calls (num_accepted_tokens given) are left to it.
"""
from __future__ import annotations

import os
import sys

TARGET = "vllm.models.glm5next.nvidia.kda"
ENABLED = os.environ.get("GLM_KDA_NOCOPY", "0").strip().lower() in ("1", "on", "true")

_state = {"kernel": None, "calls": 0, "strided": 0}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-kda-nocopy: {msg}\n")


# ---------------------------------------------------------------------------------------------------
# eligibility and strides (pure; CPU-testable)
# ---------------------------------------------------------------------------------------------------
def token_strides(q, k, v, g, beta):
    """Token strides (s_q, s_k, s_v, s_g, s_beta) if the kernel can read the tensors in place, else None.

    Requires B == 1 (the stock kernel's non-varlen indexing treats B*T as one token axis; GLM always
    passes B == 1 with cu_seqlens) and a dense layout inside a token: [H, K] with strides (K, 1) for q/k,
    [HV, V] with (V, 1) for v, [HV, K] with (K, 1) for g (KDA), [HV] with stride 1 for scalar beta."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or g is None or beta is None:
        return None
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1 or g.shape[0] != 1 or beta.shape[0] != 1:
        return None
    H, K = k.shape[2], k.shape[3]
    HV, V = v.shape[2], v.shape[3]
    T = k.shape[1]
    if q.shape != k.shape or v.shape[1] != T or g.shape[1] != T or beta.shape[1] != T:
        return None
    if q.stride(3) != 1 or q.stride(2) != K or k.stride(3) != 1 or k.stride(2) != K:
        return None
    if v.stride(3) != 1 or v.stride(2) != V:
        return None
    if g.dim() == 4:  # KDA gate [1, T, HV, K]
        if tuple(g.shape[2:]) != (HV, K) or g.stride(3) != 1 or g.stride(2) != K:
            return None
    elif g.dim() == 3:  # GDN scalar gate [1, T, HV]
        if g.shape[2] != HV or g.stride(2) != 1:
            return None
    else:
        return None
    if beta.dim() == v.dim():  # headwise beta [1, T, HV, V]
        if tuple(beta.shape[2:]) != (HV, V) or beta.stride(3) != 1 or beta.stride(2) != V:
            return None
    elif beta.dim() == 3:
        if beta.shape[2] != HV or beta.stride(2) != 1:
            return None
    else:
        return None
    return (q.stride(1), k.stride(1), v.stride(1), g.stride(1), beta.stride(1))


def wants_strided(kw, prev_is_stock: bool):
    """Strides to use, or None to call the previous function."""
    q, k, v, g, beta = (kw.get(n) for n in ("q", "k", "v", "g", "beta"))
    if q is None or k is None or v is None or g is None or beta is None:
        return None
    if not prev_is_stock and kw.get("num_accepted_tokens") is not None:
        return None  # GLM_KDA_STASH owns spec-verify
    if kw.get("cu_seqlens") is not None and q.shape[0] != 1:
        return None  # stock raises; let it
    if all(t.is_contiguous() for t in (q, k, v, g, beta)):
        return None  # nothing to save
    return token_strides(q, k, v, g, beta)


# ---------------------------------------------------------------------------------------------------
# kernel (built lazily)
# ---------------------------------------------------------------------------------------------------
def _build_kernel():
    from vllm.third_party.flash_linear_attention.ops.op import exp
    from vllm.triton_utils import tl, triton

    @triton.heuristics(
        {
            "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
            "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
            "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        }
    )
    @triton.jit(do_not_specialize=["N", "T"])
    def fused_recurrent_gated_delta_rule_fwd_kernel_strided(
        q,
        k,
        v,
        g,
        beta,
        o,
        h0,
        ht,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        a_log,
        g_bias,
        scale,
        stride_q_tok,
        stride_k_tok,
        stride_v_tok,
        stride_g_tok,
        stride_beta_tok,
        N: tl.int64,  # num of sequences
        T: tl.int64,  # num of tokens
        B: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        stride_init_state_token: tl.constexpr,
        stride_final_state_token: tl.constexpr,
        stride_indices_seq: tl.constexpr,
        stride_indices_tok: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
        INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
        IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
        USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        IS_CONTINUOUS_BATCHING: tl.constexpr,
        IS_SPEC_DECODING: tl.constexpr,
        IS_KDA: tl.constexpr,
        SIGMOID_BETA: tl.constexpr,  # beta holds raw logits; sigmoid at fp32 load
        COMPUTE_GATE: tl.constexpr,  # g holds raw logits; KDA gate computed in-kernel
        SAFE_GATE: tl.constexpr,  # bounded gate variant (only branch implemented)
        LOWER_BOUND: tl.constexpr,
    ):
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)
        if IS_VARLEN:
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int64),
                tl.load(cu_seqlens + i_n + 1).to(tl.int64),
            )
            all = T
            T = eos - bos
        else:
            bos, eos = i_n * T, i_n * T + T
            all = B * T

        if T == 0:
            # no tokens to process for this sequence
            return

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)

        # CHANGED: token stride instead of the contiguous (bos * H + i_h) * K
        p_q = q + bos * stride_q_tok + i_h * K + o_k
        p_k = k + bos * stride_k_tok + i_h * K + o_k
        p_v = v + bos * stride_v_tok + i_hv * V + o_v
        if IS_BETA_HEADWISE:
            p_beta = beta + bos * stride_beta_tok + i_hv * V + o_v
        else:
            p_beta = beta + bos * stride_beta_tok + i_hv

        if not IS_KDA:
            p_g = g + bos * stride_g_tok + i_hv
        else:
            p_gk = g + bos * stride_g_tok + i_hv * K + o_k

        # Per-head gate amplitude, hoisted out of the token loop (COMPUTE_GATE).
        if COMPUTE_GATE:
            b_a_log = tl.exp(tl.load(a_log + i_h).to(tl.float32))

        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        b_h = tl.zeros([BV, BK], dtype=tl.float32)
        if USE_INITIAL_STATE:
            if IS_CONTINUOUS_BATCHING:
                if IS_SPEC_DECODING:
                    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                else:
                    i_t = 0
                # Load state index and check for invalid entries
                state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(
                    tl.int64
                )
                # Skip if state index is invalid (NULL_BLOCK_ID=0)
                if state_idx <= 0:
                    return
                p_h0 = h0 + state_idx * stride_init_state_token
            else:
                p_h0 = h0 + bos * HV * V * K
            p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        for i_t in range(0, T):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

            if USE_QK_L2NORM_IN_KERNEL:
                b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * scale
            # [BV, BK]
            if not IS_KDA:
                b_g = tl.load(p_g).to(tl.float32)
                b_h *= exp(b_g)
            else:
                b_gk = tl.load(p_gk).to(tl.float32)
                if COMPUTE_GATE:
                    b_gk += tl.load(
                        g_bias + i_h * K + o_k, mask=mask_k, other=0.0
                    ).to(tl.float32)
                    b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))
                b_h *= exp(b_gk[None, :])
            # [BV]
            b_v -= tl.sum(b_h * b_k[None, :], 1)
            if IS_BETA_HEADWISE:
                b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
            else:
                b_beta = tl.load(p_beta).to(tl.float32)
            if SIGMOID_BETA:
                b_beta = tl.sigmoid(b_beta)
            b_v *= b_beta
            # [BV, BK]
            b_h += b_v[:, None] * b_k[None, :]
            # [BV]
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            # keep the states for multi-query tokens
            if INPLACE_FINAL_STATE:
                # Load state index and check for invalid entries
                final_state_idx = tl.load(
                    ssm_state_indices + i_n * stride_indices_seq + i_t
                ).to(tl.int64)
                # Only store if state index is valid (not NULL_BLOCK_ID=0)
                if final_state_idx > 0:
                    p_ht = ht + final_state_idx * stride_final_state_token
                    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
            else:
                p_ht = ht + (bos + i_t) * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

            # CHANGED: advance by the token stride
            p_q += stride_q_tok
            p_k += stride_k_tok
            p_o += HV * V
            p_v += stride_v_tok
            if not IS_KDA:
                p_g += stride_g_tok
            else:
                p_gk += stride_g_tok
            p_beta += stride_beta_tok

    return fused_recurrent_gated_delta_rule_fwd_kernel_strided


def fused_recurrent_kda_strided(
    q, k, v, g, beta=None, scale=None, initial_state=None, inplace_final_state=True,
    use_qk_l2norm_in_kernel=True, cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
    out=None, sigmoid_beta=False, a_log=None, g_bias=None, compute_gate=False, lower_bound=-5.0,
    _strides=None, **kwargs,
):
    """fla.ops.kda.fused_recurrent_kda + fused_recurrent_kda_fwd without the five .contiguous() calls."""
    import torch
    from vllm.utils.math_utils import cdiv, next_power_of_2

    if _state["kernel"] is None:
        _state["kernel"] = _build_kernel()
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            f"Please flatten variable-length inputs before processing."
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    s_q, s_k, s_v, s_g, s_b = _strides

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 8)
    NK, NV = cdiv(K, BK), cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if compute_gate:
        assert a_log is not None and g_bias is not None, "compute_gate requires a_log and g_bias"
        assert lower_bound is not None, "compute_gate implements the bounded (safe_gate) branch only"
        a_log = a_log.reshape(-1).contiguous()
        g_bias = g_bias.reshape(-1).contiguous()

    if out is None:
        # stock: torch.empty_like(k) of the contiguous copy == a contiguous tensor of k's shape
        o = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    else:
        assert out.shape == k.shape and out.dtype == k.dtype
        assert out.is_contiguous()
        o = out
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * HV)
    _state["kernel"][grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        stride_q_tok=s_q,
        stride_k_tok=s_k,
        stride_v_tok=s_v,
        stride_g_tok=s_g,
        stride_beta_tok=s_b,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_KDA=True,
        SIGMOID_BETA=sigmoid_beta,
        a_log=a_log,
        g_bias=g_bias,
        COMPUTE_GATE=compute_gate,
        SAFE_GATE=True,
        LOWER_BOUND=lower_bound if lower_bound is not None else -5.0,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, final_state


def _ab_flag(name: str) -> bool:
    """Per-call gate under the in-boot A/B harness (overlay/glm_ab.py, TEST ONLY); True when it is off."""
    ab = sys.modules.get("glm_ab")
    return True if ab is None else ab.flag(name)


def make_dispatch(prev, pristine):
    prev_is_stock = prev is pristine

    def fused_recurrent_kda(*args, **kw):
        if not _ab_flag("GLM_KDA_NOCOPY"):
            return prev(*args, **kw)
        _state["calls"] += 1
        # the stash dispatcher (installed before this one) owns spec-verify only while it is on
        stash_owns = not prev_is_stock and _ab_flag("GLM_KDA_STASH")
        strides = None if args else wants_strided(kw, not stash_owns)
        if strides is None:
            return prev(*args, **kw)
        _state["strided"] += 1
        return fused_recurrent_kda_strided(**kw, _strides=strides)

    fused_recurrent_kda.__wrapped__ = prev
    return fused_recurrent_kda


def install(mod) -> None:
    from vllm.third_party.flash_linear_attention.ops import kda as fla_kda

    prev = mod.fused_recurrent_kda
    mod.fused_recurrent_kda = make_dispatch(prev, fla_kda.fused_recurrent_kda)
    _log("KDA recurrent kernel reads q/k/v/g/beta in place (GLM_KDA_NOCOPY=1)"
         + ("" if prev is fla_kda.fused_recurrent_kda else "; spec-verify left to the already-installed "
            "dispatcher (GLM_KDA_STASH)"))
