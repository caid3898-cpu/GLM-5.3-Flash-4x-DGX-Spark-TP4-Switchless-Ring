# SPDX-License-Identifier: Apache-2.0
"""Exact pure-spec GDN metadata preparation, opt-in GLM_GDN_METADATA_FAST=1.

Integration: gdn_metadata_fast.diff, image g487ecf187. Credit: vLLM's
GDNAttentionMetadataBuilder.build and mamba_get_block_table_tensor. This
fuses their integer/copy chain without caching any input pointer or value.
Four independent builder instances keep their own existing persistent outputs.
"""
from __future__ import annotations

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from vllm.triton_utils import triton, tl

    @triton.jit
    def prepare(BT, SL, QSL, ACC, OSTATE, OMASK, OTOKEN, OQSL, OACC,
                N: tl.constexpr, T: tl.constexpr, S: tl.constexpr,
                BT0: tl.constexpr, BT1: tl.constexpr, BTW: tl.constexpr, BS: tl.constexpr,
                ALIGN: tl.constexpr, B: tl.constexpr):
        x = tl.program_id(0) * B + tl.arange(0, B)
        row, col = x // S, x % S
        if ALIGN:
            seq = tl.load(SL + row, mask=row < N, other=0)
            start = tl.maximum(seq.to(tl.int64) - 1, 0) // BS
        else:
            start = tl.full((B,), 0, tl.int32)
        value = tl.load(BT + row * BT0 + (start + col) * BT1,
                        mask=(row < N) & (start + col < BTW), other=0)
        tl.store(OSTATE + x, value, mask=x < N * S)
        tl.store(OMASK + x, True, mask=x < N)
        tl.store(OTOKEN + x, x, mask=x < T)
        q = tl.load(QSL + x, mask=x <= N, other=0)
        tl.store(OQSL + x, q, mask=x <= N)
        a = tl.load(ACC + x, mask=x < N, other=1)
        tl.store(OACC + x, a, mask=x < N)

    _KERNEL = prepare
    return prepare


def try_build(builder, m, accepted, draft_cpu):
    """Return None for every unsupported case; no mutation before all guards pass.

    No graph captures the inputs here: this one eager kernel reads this step's
    actual arguments and writes the same addresses the target graph already
    consumes. All output allocation remains in the stock builder constructor.
    """
    import torch
    from overhead_common import disjoint_storage
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    if torch.cuda.is_current_stream_capturing():
        return None  # Eager producer, never capture transient metadata pointers.
    n = m.num_reqs
    if (not builder.use_spec_decode or not builder.use_full_cuda_graph
            or accepted is None or draft_cpu is None or n <= 0
            or n > builder.decode_cudagraph_max_bs):
        return None
    qcpu = m.query_start_loc_cpu
    if (draft_cpu.device.type != 'cpu' or qcpu.device.type != 'cpu'
            or draft_cpu.ndim != 1 or qcpu.ndim != 1
            or draft_cpu.dtype not in (torch.int32, torch.int64)
            or qcpu.dtype not in (torch.int32, torch.int64)
            or draft_cpu.numel() != n or qcpu.numel() != n + 1):
        return None
    # Host-only checks preserve stock branch semantics, including zero-length
    # padded requests, mixed prefills, and num_decode_draft_tokens == 0.
    drafts = draft_cpu.tolist()
    q = qcpu[:n + 1].tolist()
    s = builder.num_spec + 1
    if s <= 0:
        return None
    if (any(x < 0 for x in drafts) or sum(drafts) == 0 or q[0] != 0
            or any(not 0 < b-a <= s for a, b in zip(q, q[1:]))):
        return None
    t = q[-1]
    if t > builder.decode_cudagraph_max_bs or t != m.num_actual_tokens:
        return None
    mode = builder.vllm_config.cache_config.mamba_cache_mode
    if mode not in ('align', 'all', 'none'):
        return None
    bt, sl, qsl = m.block_table_tensor, m.seq_lens, m.query_start_loc
    outputs = (builder.spec_state_indices_tensor, builder.spec_sequence_masks,
               builder.spec_token_indx, builder.spec_query_start_loc,
               builder.num_accepted_tokens)
    required = ((outputs[0], torch.int32, 2, n*s),
                (outputs[1], torch.bool, 1, n),
                (outputs[2], torch.int32, 1, t),
                (outputs[3], torch.int32, 1, n+1),
                (outputs[4], torch.int32, 1, n))
    if any(x.dtype != dtype or x.ndim != ndim or x.numel() < count
           for x, dtype, ndim, count in required):
        return None
    if (not bt.is_cuda or bt.ndim != 2 or bt.shape[0] < n or bt.shape[1] < s
            or any(x.ndim != 1 for x in (accepted, sl, qsl))
            or accepted.numel() < n or sl.numel() < n or qsl.numel() < n+1
            or bt.dtype != torch.int32 or sl.dtype != torch.int32
            or qsl.dtype != torch.int32 or accepted.dtype != torch.int32
            or not all(x.device == bt.device and x.is_contiguous()
                       for x in (sl, qsl, accepted, *outputs))
            or outputs[0].shape[1] != s
            or outputs[2].numel() < t):
        return None
    bs = builder.kv_cache_spec.block_size
    if bs <= 0 or builder.kv_cache_spec.num_speculative_blocks != builder.num_spec:
        return None
    if mode == 'align':
        # CPU upper bound only: seq_lens_cpu can implicitly trigger D2H.
        upper = getattr(m, 'max_seq_len', None)
        if type(upper) is not int or not 0 <= upper <= 2147483647:
            return None
        if max(upper - 1, 0) // bs + s > bt.shape[1]:
            return None
    if not disjoint_storage([bt, sl, qsl, accepted, *outputs]):
        return None
    kernel = _kernel()
    extent = max(n*s, t, n+1)
    kernel[((extent+127)//128,)](
        bt, sl, qsl, accepted, *outputs, N=n, T=t, S=s,
        BT0=bt.stride(0), BT1=bt.stride(1), BTW=bt.shape[1], BS=bs,
        ALIGN=mode == 'align', B=128, num_warps=4)
    return GDNAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=0,
        num_decode_tokens=0, num_spec_decodes=n,
        num_spec_decode_tokens=t, num_actual_tokens=m.num_actual_tokens,
        spec_query_start_loc=outputs[3][:n+1],
        spec_state_indices_tensor=outputs[0][:n],
        spec_sequence_masks=outputs[1][:n],
        spec_token_indx=outputs[2][:t],
        non_spec_token_indx=builder.non_spec_token_indx[:0],
        num_accepted_tokens=outputs[4][:n])
