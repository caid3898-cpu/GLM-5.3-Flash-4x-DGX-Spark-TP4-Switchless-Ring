# SPDX-License-Identifier: Apache-2.0
# Arithmetic/scheduling adapted from vLLM contributors' mhc/tilelang_kernels.py.
"""Experimental GLM mHC BF16-load, FP32-compute kernels. See answer.md.

Call install(model) AFTER loading weights, BEFORE warmup/CUDA graph capture.
GLM_MHC_FUSED=1 and a matching full qualification report are required. No parameters are replaced. Unsupported calls
retain stock execution. Installation checks exact BF16 representability and
runs bitwise stock probes; it fails closed on any mismatch.
"""
import os
import functools
import torch
import triton
import triton.language as tl


@triton.jit
def _ordered_reduce(v, O: tl.constexpr, NT: tl.constexpr):
    # TileLang warp_reduce_sum: XOR 16,8,4,2,1; then serial warp 0..W-1.
    ids = tl.arange(0, NT)
    for shift in tl.static_range(5):
        idx = tl.broadcast_to((ids ^ (16 >> shift))[None, :], (O, NT))
        v = v + tl.gather(v, idx, axis=1)
    total = tl.full((O, 1), 0, tl.float32)
    for w in tl.static_range(NT // 32):
        idx = tl.full((O, 1), w * 32, tl.int32)
        total = total + tl.gather(v, idx, axis=1)
    return total


@triton.jit
def _projection(X, W, Y, Q, M, H: tl.constexpr,
                NT: tl.constexpr, TILE: tl.constexpr, O: tl.constexpr):
    t, tile = tl.program_id(0), tl.program_id(1)
    tid = tl.arange(0, NT)
    rows = tile * TILE + tl.arange(0, O)
    valid = (tl.arange(0, O) < TILE) & (rows < 24)
    acc = tl.full((O, NT), 0, tl.float32)
    sq = tl.full((1, NT), 0, tl.float32)
    for it in range(4 * H // NT):
        k = it * NT + tid
        x = tl.load(X + t * 4 * H + k).to(tl.float32)
        w = tl.load(W + rows[:, None] * 4 * H + k[None, :],
                    valid[:, None], other=0).to(tl.float32)
        acc = tl.fma(x[None, :], w, acc)
        sq = tl.fma(x[None, :], x[None, :], sq)
    total = _ordered_reduce(acc, O, NT)
    tl.store(Y + t * 24 + rows[:, None], total, valid[:, None])
    if tile == 0:
        tl.store(Q + t, tl.sum(_ordered_reduce(sq, 1, NT)))


@triton.jit
def _post_projection(C, R, P, X, W, Y, Q, ROUT,
                     M, H: tl.constexpr, S: tl.constexpr,
                     TILE: tl.constexpr, O: tl.constexpr):
    t, tile, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tid = tl.arange(0, 256)
    rows = tile * TILE + tl.arange(0, O)
    valid = (tl.arange(0, O) < TILE) & (rows < 24)
    acc = tl.full((O, 256), 0, tl.float32)
    sq = tl.full((1, 256), 0, tl.float32)
    for it in range(H // S // 256):
        h = split * (H // S) + it * 256 + tid
        x = tl.load(X + t * H + h).to(tl.float32)
        for j in tl.static_range(4):
            p = tl.load(P + t * 4 + j)
            # Separate rounded multiply, followed by four ordered FMAs.
            new = p * x
            for k in tl.static_range(4):
                c = tl.load(C + t * 16 + k * 4 + j)
                r = tl.load(R + t * 4 * H + k * H + h).to(tl.float32)
                new = tl.fma(c, r, new)
            if tile == 0:
                tl.store(ROUT + t * 4 * H + j * H + h, new)
                sq = tl.fma(new[None, :], new[None, :], sq)
            w = tl.load(W + rows[:, None] * 4 * H + j * H + h[None, :],
                        valid[:, None], other=0).to(tl.float32)
            # Deliberately use unrounded new, NOT ROUT's BF16 values.
            acc = tl.fma(w, new[None, :], acc)
    tl.store(Y + (split * M + t) * 24 + rows[:, None],
             _ordered_reduce(acc, O, 256), valid[:, None])
    if tile == 0:
        tl.store(Q + split * M + t, tl.sum(_ordered_reduce(sq, 1, 256)))


def _require_dense(name, tensor, shape, dtype, device):
    if (tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype
            or not tensor.is_cuda or tensor.device != device
            or not tensor.is_contiguous()):
        raise ValueError(f"{name}: expected contiguous CUDA {dtype} {shape} on {device}")


def _require_weight(weight):
    if (not weight.is_cuda or not weight.is_contiguous()
            or weight.dtype not in (torch.float32, torch.bfloat16)
            or tuple(weight.shape) not in ((24, 16384), (24, 4, 4096))):
        raise ValueError('Unsupported mHC weight layout')


def _require_disjoint(inputs, outputs):
    # Dense tensors on one device only. Metadata reads neither allocate nor sync.
    def span(t):
        start = t.data_ptr()
        return start, start + t.numel() * t.element_size()
    for i, dst in enumerate(outputs):
        lo, hi = span(dst)
        for other in (*inputs, *outputs[:i]):
            olo, ohi = span(other)
            if lo < ohi and olo < hi:
                raise ValueError('mHC destinations must not overlap')


def project(x, weight, out, sqrsum, n_thr, tile_n):
    """Preallocated CUDA API: no tensor allocation, copies, or synchronization.

    Only stock default configurations for physical M=1..256 are qualified.
    Outputs must be dense, disjoint from all inputs and each other. Element
    offsets are allowed. Warm each specialization outside graph capture.
    """
    if x.ndim != 2 or not 1 <= x.shape[0] <= 256:
        raise ValueError('Expected x[M,16384], M=1..256')
    m = x.shape[0]
    if (n_thr, tile_n) != ((1024, 4) if m < 128 else (512, 12)):
        raise ValueError('Unqualified projection configuration for physical M')
    _require_weight(weight)
    device = weight.device
    _require_dense('x', x, (m, 16384), torch.bfloat16, device)
    _require_dense('out', out, (1, m, 24), torch.float32, device)
    _require_dense('sqrsum', sqrsum, (1, m), torch.float32, device)
    _require_disjoint((x, weight), (out, sqrsum))
    return _projection[(m, triton.cdiv(24, tile_n))](
        x, weight, out, sqrsum, m, 4096, n_thr, tile_n,
        triton.next_power_of_2(tile_n), num_warps=8,
        enable_fp_fusion=False)


def post_project(comb, residual, post, x, weight, out, sqrsum, residual_out,
                 tile_n, splits):
    """Preallocated small-M API; same storage contract as project()."""
    if residual.ndim != 3 or not 1 <= residual.shape[0] <= 16:
        raise ValueError('Expected residual[M,4,4096], M=1..16')
    m = residual.shape[0]
    if (splits, tile_n) != ((8, 2) if m < 8 else (4, 3)):
        raise ValueError('Unqualified fused configuration for physical M')
    _require_weight(weight)
    device = weight.device
    _require_dense('residual', residual, (m, 4, 4096), torch.bfloat16, device)
    _require_dense('x', x, (m, 4096), torch.bfloat16, device)
    _require_dense('comb', comb, (m, 4, 4), torch.float32, device)
    if tuple(post.shape) not in ((m, 4), (m, 4, 1)):
        raise ValueError('Expected post[M,4] or post[M,4,1]')
    _require_dense('post', post, post.shape, torch.float32, device)
    _require_dense('out', out, (splits, m, 24), torch.float32, device)
    _require_dense('sqrsum', sqrsum, (splits, m), torch.float32, device)
    _require_dense('residual_out', residual_out, (m, 4, 4096), torch.bfloat16, device)
    _require_disjoint((comb, residual, post, x, weight), (out, sqrsum, residual_out))
    return _post_projection[(m, 24 // tile_n, splits)](
        comb, residual, post, x, weight, out, sqrsum, residual_out,
        m, 4096, splits, tile_n, triton.next_power_of_2(tile_n),
        num_warps=8, enable_fp_fusion=False)


def bit_equal(a, b):
    """Compare raw bytes, including signed zeros (torch.equal(float) does not)."""
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.contiguous().view(torch.uint8),
                        b.contiguous().view(torch.uint8)))


def pack_weight(weight):
    if (weight.shape != (24, 16384) or weight.dtype != torch.float32
            or not weight.is_cuda or not weight.is_contiguous()):
        raise ValueError('Expected loaded FP32 hc_fn [24,16384]')
    packed = weight.detach().to(torch.bfloat16).contiguous()
    if not bit_equal(weight, packed.float()):
        raise ValueError('hc_fn is not bit-exactly representable as BF16')
    return packed


def check_kernels(stock_pre, stock_fused, packed, tokens=(1, 7, 8, 16, 17, 127, 128, 256), seed=51):
    """No tolerance: compare intermediates against the actual installed stock JIT.

    Not a universal proof. Full token/seed/view/end-to-end coverage is in tests.
    Runs outside capture only; also compiles both pointer dtypes.
    """
    gen = torch.Generator(device=packed.device).manual_seed(seed)
    fp = packed.float()
    for m in tokens:
        r = torch.randn((m, 4, 4096), generator=gen, device=packed.device,
                        dtype=torch.bfloat16)
        nt, tile = (1024, 4) if m < 128 else (512, 12)
        ref = torch.empty((1, m, 24), device=packed.device)
        rs = torch.empty((1, m), device=packed.device)
        stock_pre(r.view(m, -1), fp, ref, rs, 4096, 4, 24, nt, tile, 1)
        for w in (fp, packed):
            out, sq = torch.empty_like(ref), torch.empty_like(rs)
            project(r.view(m, -1), w, out, sq, nt, tile)
            if not bit_equal(ref, out) or not bit_equal(rs, sq):
                raise RuntimeError(f'prenorm bitwise mismatch M={m}, dtype={w.dtype}')
        if m > 16:
            continue
        x = torch.randn((m, 4096), generator=gen, device=packed.device, dtype=torch.bfloat16)
        c = torch.randn((m, 4, 4), generator=gen, device=packed.device)
        p = torch.randn((m, 4), generator=gen, device=packed.device)
        splits, tile = (8, 2) if m < 8 else (4, 3)
        ref = torch.empty((splits, m, 24), device=packed.device)
        rs = torch.empty((splits, m), device=packed.device)
        rr = torch.empty_like(r)
        stock_fused(c, r, p, x, fp.view(24, 4, 4096), ref, rs, rr,
                    4, 4096, 24, 256, 256, tile, splits)
        for w in (fp, packed):
            out, sq, rout = torch.empty_like(ref), torch.empty_like(rs), torch.empty_like(rr)
            post_project(c, r, p, x, w.view(24, 4, 4096), out, sq, rout, tile, splits)
            if not all(bit_equal(a, b) for a, b in ((ref, out), (rs, sq), (rr, rout))):
                raise RuntimeError(f'post/pre bitwise mismatch M={m}, dtype={w.dtype}')


_STATE = None


def _lookup_weight(weights, w):
    if (not w.is_cuda or w.dtype != torch.float32
            or tuple(w.shape) not in ((24, 16384), (24, 4, 4096))
            or not w.is_contiguous()):
        return None
    item = weights.get((w.device, w.data_ptr()))
    if item is None:
        return None
    original, packed, version = item
    if (original.device != w.device or original.data_ptr() != w.data_ptr()
            or original.dtype != torch.float32
            or tuple(original.shape) != (24, 16384) or not original.is_contiguous()):
        raise RuntimeError('mHC weight storage/layout changed; reinstall required')
    if original._version != version:
        raise RuntimeError('mHC weights changed after packing; recapture required')
    return packed


def install(model):
    """Default-off production entry: requires a full passing qualification report.

    Set GLM_MHC_FUSED=1 and GLM_MHC_QUALIFICATION=/path/to/qualification.json.
    The report must match these files, installed compiler sources/binaries,
    wrappers, CUDA driver and GPU. No report is shipped as qualified.
    Weights must remain immutable, including during graph replay. Re-loading
    weights requires a fresh process and fresh captures; .data bypasses versions.
    """
    if os.environ.get('GLM_MHC_FUSED') != '1':
        return False
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Install before CUDA graph capture')
    from mhc_validation import require_qualification
    require_qualification(os.environ.get('GLM_MHC_QUALIFICATION'))
    return _install_for_qualification(model)


def _install_for_qualification(model):
    """Internal test bootstrap; production callers MUST use install(model).

    Shares the real installed lookup/dispatch, without needing the report that
    this process is about to create. Still requires the explicit enable flag.
    """
    global _STATE
    if os.environ.get('GLM_MHC_FUSED') != '1':
        return False
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Install before CUDA graph capture')
    if _STATE is not None:
        if _STATE['model'] is model:
            for original, _, _ in _STATE['weights'].values():
                if _lookup_weight(_STATE['weights'], original) is None:
                    raise RuntimeError('mHC weight storage changed; restart required')
            return True
        raise RuntimeError('Already installed for another model')
    from vllm.model_executor.kernels.mhc import tilelang_kernels as tk
    import hashlib
    from pathlib import Path
    expected = '03aeb3f7c0eabfe8391006ff80f8d4cfedba7cf3d0a168e0dee2d678d3e3bd24'
    if hashlib.sha256(Path(tk.__file__).read_bytes()).hexdigest() != expected:
        raise RuntimeError('Unqualified stock TileLang source revision; review '
                           'the new source and rerun qualification before updating hash')
    if tk.ENABLE_PDL:
        raise RuntimeError('Unqualified PDL launch protocol; candidate requires PDL disabled')
    weights = {}
    for layer in model.modules():
        for name in ('hc_attn_fn', 'hc_ffn_fn'):
            w = getattr(layer, name, None)
            if w is not None:
                weights[(w.device, w.data_ptr())] = (w, pack_weight(w), w._version)
    if not weights:
        raise ValueError('No GLM hc_attn_fn/hc_ffn_fn found')
    old_pre, old_fused = tk.hc_prenorm_gemm_tilelang, tk.mhc_fused_tilelang
    # Every parameter is round-trip checked; one arithmetic probe per device.
    checked_devices = set()
    for w, packed, _ in weights.values():
        if w.device not in checked_devices:
            with torch.cuda.device(w.device):
                check_kernels(old_pre, old_fused, packed)
            checked_devices.add(w.device)

    warmed = set()

    def launch(kind, tensors, config, call):
        # Include alignment classes and runtime integer specialization (M).
        # Fresh addresses with the same layout do not require new compilation.
        key = (kind, config, tuple((t.device, t.dtype, tuple(t.shape),
                                   tuple(t.stride()), t.data_ptr() % 16)
                                  for t in tensors))
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and key not in warmed:
            raise RuntimeError('Warm this mHC specialization before graph capture')
        result = call()
        if not capturing:
            warmed.add(key)
        return result

    @functools.wraps(old_pre)
    def pre(x, fn, out, sqrsum, hidden_size, hc_mult=4, n_out=24,
            n_thr=512, tile_n=12, n_splits=1):
        w = _lookup_weight(weights, fn)
        m = x.shape[0] if x.ndim == 2 else 0
        if (w is not None and 1 <= m <= 256 and hidden_size == 4096
                and hc_mult == 4 and n_out == 24 and n_splits == 1
                and (n_thr, tile_n) == ((1024, 4) if m < 128 else (512, 12))):
            return launch('pre', (x, w, out, sqrsum), (n_thr, tile_n),
                          lambda: project(x, w, out, sqrsum, n_thr, tile_n))
        return old_pre(x, fn, out, sqrsum, hidden_size, hc_mult,
                       n_out, n_thr, tile_n, n_splits)

    @functools.wraps(old_fused)
    def fused(comb_mix, residual_in, post_mix, x_in, weight_t, yp_out,
              rp_out, residual_out, hc, hidden, n_out, n_thr=256,
              h_blk=256, tile_n=1, split_k=None, **kwargs):
        alias = kwargs.pop('n_splits', None)
        if kwargs or (alias is not None and split_k is not None and alias != split_k):
            raise TypeError('Unexpected or conflicting mHC split arguments')
        splits = alias if alias is not None else (1 if split_k is None else split_k)
        w = _lookup_weight(weights, weight_t)
        m = residual_in.shape[0] if residual_in.ndim == 3 else 0
        if (w is not None and 1 <= m <= 16 and hc == 4
                and hidden == 4096 and n_out == 24 and n_thr == 256 and h_blk == 256
                and (splits, tile_n) == ((8, 2) if m < 8 else (4, 3))):
            return launch('fused', (comb_mix, residual_in, post_mix, x_in, w,
                                   yp_out, rp_out, residual_out), (tile_n, splits),
                          lambda: post_project(comb_mix, residual_in, post_mix, x_in,
                                               w, yp_out, rp_out, residual_out,
                                               tile_n, splits))
        return old_fused(comb_mix, residual_in, post_mix, x_in, weight_t,
                         yp_out, rp_out, residual_out, hc, hidden, n_out,
                         n_thr, h_blk, tile_n, splits)

    tk.hc_prenorm_gemm_tilelang = pre
    tk.mhc_fused_tilelang = fused
    _STATE = dict(model=model, weights=weights, pre=old_pre, fused=old_fused,
                  installed_pre=pre, installed_fused=fused, warmed=warmed)
    return True
