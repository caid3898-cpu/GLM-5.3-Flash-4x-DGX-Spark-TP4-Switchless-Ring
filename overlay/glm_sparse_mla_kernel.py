# SPDX-License-Identifier: Apache-2.0
"""Triton sparse-MLA attention for GLM-5.3-Flash rows on GB10 (SM121): top-k latent rows, fp8 or bf16 cache.

Idea and GB10 measurements: Matt Mastracci (mmastrac), GLM-5.3-Flash 4x GX10 recipe PR #4 ("gb10 sparse MLA":
one program per query token, every head in registers, the top-k latent rows gathered in blocks with an online
softmax, no host plan and no host KV lengths). Independent reproduction and ablations: chuck-ads.
This file is an independent reimplementation from that description; no code was copied.

What one program computes (query row t, H heads, latent width D = kv_lora_rank = 512, NoPE so no rope part):

    idx   = topk_indices[t, :]                  request-local token ids, -1 = unused (any column)
    slot  = block_table[req_id[t], idx // PAGE] * PAGE + idx % PAGE     (as the stock convert kernel)
    K = V = dequant(kv_cache[slot, :D])         fp8 e4m3 -> bf16 (exact), times the per-tensor scale in bf16
    out[t] = softmax(q[t] @ K^T * sm_scale) @ V  bf16 operands, fp32 accumulation, P rounded to bf16 for P@V

Numerics follow FlashInfer's FA2 MLA path (mla.cuh): bf16 q (no fp16 cast), fp8 -> bf16 repack with the scale
multiplied in bf16, fp32 QK, exp2(s * sm_scale * log2e - m_scaled) with the fully-masked clamp, P cast to bf16,
fp32 P@V, a final divide. Only the summation order (tile width, key order) differs.

Invalid entries (-1, or a block id past the block table) are masked; the key loop stops after the last valid
column of the row, so the -1 tail of short (causal) rows and of the rounded-up buffer width costs no blocks.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634

# GB10 defaults (48 SMs, ~101 KB smem/SM): (BN keys per tile, num_warps, num_stages, SPLIT). SPLIT > 1 cuts the
# latent into SPLIT chunks computed as one batched dot (each warp group holds a D/SPLIT slice of q and of the
# accumulator instead of every warp holding all of q; the QK partials are summed across the chunks).
# An optional 5th value SKIP=1 jumps over key tiles with no valid entry (the -1 middle of short causal rows); measured
# slower (the branch defeats the pipelining: 24.8 vs 13.8 ms), so off.
# Overridden by GLM_TRITON_MLA_CFG="BN,warps,stages[,split[,skip]]" or the bench sweep.
DEFAULT_CFG = (64, 4, 1, 1)


@triton.jit
def _sparse_mla_fwd_kernel(
    q_ptr, q_s_t, q_s_h,                 # q [T, H, D] (last stride 1)
    kv_ptr, kv_s_blk, kv_s_tok,          # cache [num_blocks, PAGE, >=D] (last stride 1), fp8 or bf16
    idx_ptr, idx_s_t,                    # topk [T, W] int32, request-local token ids, -1 = unused
    req_ptr,                             # req_id_per_token [T] int32
    bt_ptr, bt_s_r,                      # block_table [R, MB] int32 (last stride 1)
    out_ptr, o_s_t, o_s_h,               # out [T, H, D] bf16 (last stride 1)
    max_blocks,                          # MB
    qk_scale_log2,                       # sm_scale * log2(e)
    kv_scale,                            # per-tensor dequant scale (HAS_SCALE only)
    W: tl.constexpr, W_POW2: tl.constexpr, PAGE: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr, SKIP: tl.constexpr,
    KV_FP8: tl.constexpr, HAS_SCALE: tl.constexpr,
):
    DC: tl.constexpr = D // SPLIT
    t = tl.program_id(0).to(tl.int64)
    req = tl.load(req_ptr + t)

    # Last valid column of this row: the key loop stops there.
    cols = tl.arange(0, W_POW2)
    row = tl.load(idx_ptr + t * idx_s_t + cols, mask=cols < W, other=-1)
    n_cols = tl.max(tl.where(row >= 0, cols, -1), axis=0) + 1

    offs_h = tl.arange(0, H)
    offs_c = tl.arange(0, SPLIT)
    offs_j = tl.arange(0, DC)
    # [SPLIT, H, DC]: element (c, h, j) is q[t, h, c * DC + j]
    q_off = offs_c[:, None, None] * DC + offs_h[None, :, None] * q_s_h + offs_j[None, None, :]
    q = tl.load(q_ptr + t * q_s_t + q_off)

    m_i = tl.full([H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([SPLIT, H, DC], dtype=tl.float32)
    bt_row = bt_ptr + req.to(tl.int64) * bt_s_r

    for start in range(0, n_cols, BN):
        offs_n = start + tl.arange(0, BN)
        tok = tl.load(idx_ptr + t * idx_s_t + offs_n, mask=offs_n < n_cols, other=-1)
        if SKIP:
            live = tl.max(tok, axis=0) >= 0
        else:
            live = True
        if live:
            blk = tok // PAGE
            valid = (tok >= 0) & (blk < max_blocks)
            page = tl.load(bt_row + blk, mask=valid, other=0)
            roff = page.to(tl.int64) * kv_s_blk + (tok - blk * PAGE).to(tl.int64) * kv_s_tok
            # [SPLIT, BN, DC]: element (c, n, j) is latent row n, channel c * DC + j
            k_off = roff[None, :, None] + (offs_c[:, None, None] * DC + offs_j[None, None, :])
            k = tl.load(kv_ptr + k_off, mask=valid[None, :, None], other=0.0)
            if KV_FP8:
                k = k.to(tl.bfloat16)
            if HAS_SCALE:
                # bf16 x bf16 product is exact in fp32; one rounding == FlashInfer's __hmul2.
                k = (k.to(tl.float32) * kv_scale).to(tl.bfloat16)
            if SPLIT == 1:
                s = tl.dot(tl.reshape(q, (H, DC)), tl.trans(tl.reshape(k, (BN, DC))))       # [H, BN] fp32
            else:
                s = tl.sum(tl.dot(q, tl.permute(k, (0, 2, 1))), axis=0)                     # [H, BN] fp32
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_scaled = tl.maximum(m_new * qk_scale_log2, -3.4028234663852886e38)
            alpha = tl.math.exp2(m_i * qk_scale_log2 - m_scaled)
            p = tl.math.exp2(s * qk_scale_log2 - m_scaled[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            pb = p.to(tl.bfloat16)
            if SPLIT == 1:
                pv = tl.reshape(tl.dot(pb, tl.reshape(k, (BN, DC))), (SPLIT, H, DC))
            else:
                pv = tl.dot(tl.broadcast_to(pb[None, :, :], (SPLIT, H, BN)), k)              # [SPLIT, H, DC]
            acc = acc * alpha[None, :, None] + pv
            m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = acc / l_safe[None, :, None]
    tl.store(out_ptr + t * o_s_t + (offs_c[:, None, None] * DC + offs_h[None, :, None] * o_s_h
                                    + offs_j[None, None, :]), o.to(tl.bfloat16))


def sparse_mla_fwd(q: torch.Tensor, kv_cache: torch.Tensor, topk: torch.Tensor, req_id: torch.Tensor,
                   block_table: torch.Tensor, sm_scale: float, kv_scale: float = 1.0,
                   out: torch.Tensor | None = None, cfg: tuple | None = None) -> torch.Tensor:
    """q [T, H, D] bf16 (any row/head strides, unit last stride); kv_cache [num_blocks, PAGE, >=D] fp8_e4m3fn or
    bf16 (uint8 storage must be viewed as fp8 by the caller); topk [>=T, W] int32; req_id [>=T] int32;
    block_table [R, MB] int32. Returns out [T, H, D] bf16 (contiguous unless `out` is given)."""
    T, H, D = q.shape
    cfg = tuple(cfg or DEFAULT_CFG)
    bn, warps, stages = cfg[:3]
    split = cfg[3] if len(cfg) > 3 else 1
    skip = bool(cfg[4]) if len(cfg) > 4 else False
    if out is None:
        out = torch.empty((T, H, D), dtype=torch.bfloat16, device=q.device)
    if T == 0:
        return out
    W = topk.shape[1]
    kv_fp8 = kv_cache.dtype == torch.float8_e4m3fn
    has_scale = kv_fp8 and float(kv_scale) != 1.0
    if block_table.stride(1) != 1:
        block_table = block_table.contiguous()
    _sparse_mla_fwd_kernel[(T,)](
        q, q.stride(0), q.stride(1),
        kv_cache, kv_cache.stride(0), kv_cache.stride(1),
        topk, topk.stride(0),
        req_id,
        block_table, block_table.stride(0),
        out, out.stride(0), out.stride(1),
        block_table.shape[1],
        float(sm_scale) * LOG2E,
        float(kv_scale),
        W=W, W_POW2=triton.next_power_of_2(W), PAGE=kv_cache.shape[1],
        H=H, D=D, BN=bn, SPLIT=split, SKIP=skip, KV_FP8=kv_fp8, HAS_SCALE=has_scale,
        num_warps=warps, num_stages=stages,
    )
    return out


def reference_fp32(q: torch.Tensor, kv_cache: torch.Tensor, topk: torch.Tensor, req_id: torch.Tensor,
                   block_table: torch.Tensor, sm_scale: float, kv_scale: float = 1.0) -> torch.Tensor:
    """fp32 reference (dequantized cache in fp32, exact softmax), row by row; for tests and the bench."""
    T, H, D = q.shape
    page = kv_cache.shape[1]
    fp8 = kv_cache.dtype == torch.float8_e4m3fn
    raw = kv_cache.view(torch.uint8) if fp8 else kv_cache       # gather on the byte view (fp8 indexing is patchy)
    out = torch.empty((T, H, D), dtype=torch.float32, device=q.device)
    for t in range(T):
        idx = topk[t]
        idx = idx[idx >= 0].long()
        blk = idx // page
        ok = blk < block_table.shape[1]
        idx, blk = idx[ok], blk[ok]
        pages = block_table[int(req_id[t])].long()[blk]
        k = raw[pages, idx % page, :D]
        k = (k.view(torch.float8_e4m3fn) if fp8 else k).float() * float(kv_scale)
        if k.shape[0] == 0:
            out[t] = 0.0
            continue
        s = (q[t].float() @ k.T) * sm_scale
        out[t] = torch.softmax(s, dim=-1) @ k
    return out


def flops(valid_counts: torch.Tensor, heads: int, dim: int) -> float:
    """QK + PV multiply-adds as FLOPs over the valid keys."""
    return 4.0 * heads * dim * float(valid_counts.sum())


__all__ = ["sparse_mla_fwd", "reference_fp32", "flops", "DEFAULT_CFG"]
