#!/usr/bin/env python3
"""CPU tests for overlay/glm_draft_conv_fused.py (torch CPU only; no CUDA, no Triton, no vLLM).

  python3 tests/test_glm_draft_conv_fused.py        (exit 0 = all pass)

1. The explicit-rounding reference equals the VERBATIM eager `_grouped_conv` of the image, bit for bit, over real
   and small shapes, taps 1-4, power-of-two and other block sizes, strided delta views like the real call, and
   inputs spanning many binades plus +-0, subnormals, overflow, inf and NaN.
2. A line-by-line emulation of the Triton program (grid = rows x H-blocks, masked loads, the same index arithmetic)
   equals the reference: the kernel's addressing is right before a GPU ever runs it.
3. Dispatch: eligibility reasons, qualification pass/fail bookkeeping, capture fail-closed, qualification off,
   check mode (returns the original bytes, counts differences in device-style counters), mode off.
4. Install hooks: the module global is replaced once, the drafter-graph replay hook counts replays, register() is a
   no-op without the switch.
"""
from __future__ import annotations

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import glm_draft_conv_fused as M  # noqa: E402

BF = torch.bfloat16


# ---- verbatim from the image: vllm/model_executor/models/qwen3_dflash2.py (g487ecf187), lines 23-43 --------------
def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)
# -------------------------------------------------------------------------------------------------------------------


def bits(t):
    return t.contiguous().view(torch.int16)


def same_bits(a, b):
    return a.shape == b.shape and bool(torch.equal(bits(a), bits(b)))


def make_inputs(m, taps, groups, gsize, gen, special=True, side=0, scale_lo=-20, scale_hi=20):
    """hidden [m, H]; delta = coefficients[:, side] of a [m, 2, taps, G] reshape (the real call's strided view);
    base = base_kernel[side] of a [2, taps, H] parameter."""
    h = groups * gsize

    def vals(n):
        v = torch.randn(n, generator=gen) * torch.pow(2.0, torch.randint(scale_lo, scale_hi, (n,), generator=gen).float())
        if special and n >= 16:
            idx = torch.randperm(n, generator=gen)[: max(4, n // 40)]
            sp = torch.tensor([0.0, -0.0, 1e-40, -1e-40, 3.0e38, -3.0e38, float("inf"), float("-inf"),
                               float("nan"), 1.0, -1.0, 2.0 ** -126])
            v[idx] = sp[torch.arange(idx.numel()) % sp.numel()]
        return v

    hidden = vals(m * h).view(m, h).to(BF)
    coef = vals(m * 2 * taps * groups).view(m, 2 * taps * groups).to(BF).reshape(m, 2, taps, groups)
    delta = coef[:, side]
    base_param = vals(2 * taps * h).view(2, taps, h).to(BF)
    base = base_param[side]
    return hidden, delta, base


# =====================================================================================================
# 1. reference == verbatim eager
# =====================================================================================================
def test_reference_matches_stock():
    gen = torch.Generator().manual_seed(1)
    cases = 0
    configs = [
        # (m, taps, groups, gsize, block_size)
        (8, 2, 256, 16, 8),      # production c1 (k=7 block) on the real H=4096
        (16, 2, 256, 16, 8), (32, 2, 256, 16, 8), (24, 2, 256, 16, 8),
        (1, 2, 8, 4, 8), (3, 2, 8, 4, 8), (8, 1, 8, 4, 8), (8, 3, 8, 4, 8), (16, 4, 16, 8, 8),
        (12, 2, 8, 4, 4), (12, 3, 8, 4, 6), (18, 2, 4, 16, 6), (40, 4, 32, 2, 5), (7, 2, 8, 4, 1),
    ]
    for (m, taps, groups, gsize, bs) in configs:
        for side in (0, 1):
            for special in (False, True):
                for lo, hi in ((-3, 3), (-20, 20), (-60, 60)):
                    x, d, b = make_inputs(m, taps, groups, gsize, gen, special=special, side=side,
                                          scale_lo=lo, scale_hi=hi)
                    want = _grouped_conv(x, d, b, bs, groups, gsize, taps)
                    got = M.grouped_conv_reference(x, d, b, bs, groups, gsize, taps)
                    assert same_bits(want, got), (m, taps, groups, gsize, bs, side, special, lo,
                                                  int((bits(want) != bits(got)).sum()))
                    assert got.is_contiguous() and got.dtype == BF
                    cases += 1
    return cases


def test_reference_is_order_sensitive():
    """The check can see the difference an unrounded (fused fp32) evaluation would make: the reference with the
    intermediate roundings removed must differ from eager on ordinary data."""
    gen = torch.Generator().manual_seed(2)
    x, d, b = make_inputs(32, 2, 256, 16, gen, special=False, scale_lo=-4, scale_hi=4)
    want = _grouped_conv(x, d, b, 8, 256, 16, 2)
    f32 = torch.float32
    xs = x.float().view(32, 256, 16)
    c = b.float().view(2, 256, 16).unsqueeze(0) + d.float().unsqueeze(-1)   # no rounding anywhere
    out = c[:, 0] * xs
    sh = torch.zeros_like(xs)
    sh[1:] = xs[:-1]
    out = out + c[:, 1] * sh * ((torch.arange(32) % 8) >= 1).to(f32).view(-1, 1, 1)
    loose = out.to(BF).view(32, 4096)
    ndiff = int((bits(want) != bits(loose)).sum())
    assert ndiff > 0, "an unrounded evaluation should differ somewhere; the test would be blind"
    return ndiff


# =====================================================================================================
# 2. Triton program emulation (same index arithmetic, masked loads with other=0.0, per-op rounding)
# =====================================================================================================
def emulate_kernel(x, d, b, block_size, groups, gsize, taps, BLOCK=1024):
    m_rows, h = x.shape
    f32 = torch.float32
    out = torch.empty((m_rows, h), dtype=BF)
    xf, df, bfv = x.reshape(-1), d, b.reshape(-1)
    sxm = x.stride(0)
    sdm, sdt, sdg = d.stride()
    sbt = b.stride(0)
    dflat = torch.as_strided(d, (d.untyped_storage().nbytes() // 2 - d.storage_offset(),), (1,), d.storage_offset())
    xflat = torch.as_strided(x, (x.untyped_storage().nbytes() // 2 - x.storage_offset(),), (1,), x.storage_offset())
    bflat = torch.as_strided(b, (b.untyped_storage().nbytes() // 2 - b.storage_offset(),), (1,), b.storage_offset())

    def rnd(t):
        return t.to(BF).to(f32)

    def load(flat, offs, mask):
        v = torch.zeros(offs.shape, dtype=f32)
        if bool(mask.any()):
            v[mask] = flat[offs[mask]].to(f32)
        return v

    nblk = (h + BLOCK - 1) // BLOCK
    for m in range(m_rows):
        for pb in range(nblk):
            hh = pb * BLOCK + torch.arange(BLOCK)
            hm = hh < h
            g = hh // gsize
            xv = load(xflat, m * sxm + hh, hm)
            b0 = load(bflat, hh, hm)
            d0 = load(dflat, m * sdm + g * sdg, hm)
            c0 = rnd(b0 + d0)
            o = rnd(c0 * xv)
            pos = m % block_size
            for tap in range(1, taps):
                bt = load(bflat, tap * sbt + hh, hm)
                dt = load(dflat, m * sdm + tap * sdt + g * sdg, hm)
                ct = rnd(bt + dt)
                src_ok = m >= tap
                sh = load(xflat, (m - tap) * sxm + hh, hm & torch.tensor(src_ok))
                t1 = rnd(ct * sh)
                mk = 1.0 if pos >= tap else 0.0
                t2 = rnd(t1 * mk)
                o = rnd(o + t2)
            out.view(-1)[(m * h + hh)[hm]] = o[hm].to(BF)
    del xf, df, bfv
    return out


def test_kernel_emulation():
    gen = torch.Generator().manual_seed(3)
    n = 0
    for (m, taps, groups, gsize, bs, block) in [(8, 2, 256, 16, 8, 1024), (5, 3, 12, 4, 4, 32), (9, 2, 10, 3, 8, 16),
                                                  (16, 4, 8, 8, 6, 64), (3, 2, 5, 5, 8, 128)]:
        for side in (0, 1):
            x, d, b = make_inputs(m, taps, groups, gsize, gen, special=True, side=side)
            want = M.grouped_conv_reference(x, d, b, bs, groups, gsize, taps)
            got = emulate_kernel(x, d, b, bs, groups, gsize, taps, BLOCK=block)
            assert same_bits(want, got), (m, taps, groups, gsize, bs, block, side)
            n += 1
    return n


# =====================================================================================================
# 3. dispatch
# =====================================================================================================
def _reset():
    M.S["ok"].clear()
    M.S["bad"].clear()
    M.S["counts"].clear()
    M.S["dev_counters"] = None
    M.S["warned"].clear()


def test_eligible_reasons():
    gen = torch.Generator().manual_seed(4)
    x, d, b = make_inputs(8, 2, 8, 4, gen, special=False)
    assert M.eligible(x, d, b, 8, 8, 4, 2) == "device"                                  # CPU tensors
    assert M.eligible(x, d, b, 8, 8, 4, 2, require_cuda=False) is None
    assert M.eligible(x.float(), d, b, 8, 8, 4, 2, require_cuda=False) == "dtype"
    assert M.eligible(x, d, b, 8, 4, 8, 2, require_cuda=False) == "shape"               # delta groups mismatch
    assert M.eligible(x.t().contiguous().t(), d, b, 8, 8, 4, 2, require_cuda=False) == "stride"
    assert M.eligible(x[:, None], d, b, 8, 8, 4, 2, require_cuda=False) == "rank"
    assert M.eligible(x, d, b, 0, 8, 4, 2, require_cuda=False) == "size"
    return 7


def _dispatch(impl, capturing=False, mode="on", qual=True):
    return M.make_dispatch(_grouped_conv, impl=impl, is_capturing=lambda: capturing, get_mode=lambda: mode,
                           qual_on=lambda: qual)


def test_dispatch_paths():
    gen = torch.Generator().manual_seed(5)
    orig_eligible = M.eligible
    M.eligible = lambda *a, **k: None          # CPU stand-in for "CUDA bf16 and well-formed"
    try:
        calls = {"impl": 0}

        def good(*a):
            calls["impl"] += 1
            return M.grouped_conv_reference(*a)

        def bad(*a):
            out = M.grouped_conv_reference(*a).clone()
            out.view(-1)[7] = out.view(-1)[7] + 1 if out.view(-1)[7].isfinite() else 0.0
            out.view(torch.int16).view(-1)[3] ^= 1
            return out

        x, d, b = make_inputs(8, 2, 8, 4, gen, special=True)
        want = _grouped_conv(x, d, b, 8, 8, 4, 2)

        # qualified -> fused
        _reset()
        f = _dispatch(good)
        out = f(x, d, b, 8, 8, 4, 2)
        assert same_bits(out, want) and M.S["counts"]["qualified"] == 1 and M.S["counts"]["fused_calls"] == 1
        n_after_qual = calls["impl"]
        f(x, d, b, 8, 8, 4, 2)
        assert calls["impl"] == n_after_qual + 1, "qualification must run once per key"

        # a wrong kernel is caught by qualification and never used
        _reset()
        f = _dispatch(bad)
        out = f(x, d, b, 8, 8, 4, 2)
        assert same_bits(out, want) and len(M.S["bad"]) == 1 and M.S["counts"]["fused_calls"] == 0
        out = f(x, d, b, 8, 8, 4, 2)
        assert same_bits(out, want) and M.S["counts"]["fallback_bad"] == 1

        # unknown key inside a capture: fail-closed to the original, not qualified
        _reset()
        f = _dispatch(good, capturing=True)
        out = f(x, d, b, 8, 8, 4, 2)
        assert same_bits(out, want) and not M.S["ok"] and M.S["counts"]["fallback_unqualified_in_capture"] == 1

        # qualified outside, then used inside a capture
        _reset()
        _dispatch(good)(x, d, b, 8, 8, 4, 2)
        f = _dispatch(good, capturing=True)
        out = f(x, d, b, 8, 8, 4, 2)
        assert same_bits(out, want) and M.S["counts"]["fused_calls"] == 2

        # qualification off: straight to the kernel (even a bad one: the switch is the operator's)
        _reset()
        f = _dispatch(bad, qual=False)
        out = f(x, d, b, 8, 8, 4, 2)
        assert not same_bits(out, want) and M.S["counts"]["fused_calls"] == 1

        # check mode: original bytes out, device-style counters see the difference
        _reset()
        f = _dispatch(bad, mode="check", qual=False)
        for _ in range(3):
            out = f(x, d, b, 8, 8, 4, 2)
            assert same_bits(out, want)
        calls_n, diff_n = M.counters()
        assert calls_n == 3 and diff_n >= 3, (calls_n, diff_n)
        _reset()
        f = _dispatch(good, mode="check", qual=False)
        f(x, d, b, 8, 8, 4, 2)
        assert M.counters() == (1, 0)

        # mode off / stock (the timing control arm): original, nothing counted, nothing qualified
        for md in ("off", "stock"):
            _reset()
            f = _dispatch(bad, mode=md)
            assert same_bits(f(x, d, b, 8, 8, 4, 2), want) and not M.S["counts"] and not M.S["ok"]
    finally:
        M.eligible = orig_eligible
        _reset()
    return 10


def test_ineligible_goes_to_original():
    _reset()
    gen = torch.Generator().manual_seed(6)
    x, d, b = make_inputs(8, 2, 8, 4, gen, special=False)

    def boom(*a):
        raise AssertionError("must not be called for CPU tensors")

    f = _dispatch(boom)
    out = f(x, d, b, 8, 8, 4, 2)
    assert same_bits(out, _grouped_conv(x, d, b, 8, 8, 4, 2)) and M.S["counts"]["fallback_device"] == 1
    _reset()
    return 1


def test_synthetic_and_bits():
    gen = torch.Generator().manual_seed(7)
    x, d, b = make_inputs(8, 2, 256, 16, gen, side=1)
    s = M.synthetic_like(d, gen)
    assert s.shape == d.shape and s.stride() == d.stride() and s.dtype == d.dtype
    sx = M.synthetic_like(x, gen)
    assert bool(torch.isnan(sx.float()).any()) and bool(torch.isinf(sx.float()).any())
    assert bool((sx.float() == 0).any())
    a = torch.tensor([1.0, float("nan"), -0.0], dtype=BF)
    same, n = M.bits_equal(a, a.clone())
    assert same and n == 0
    same, n = M.bits_equal(a, torch.tensor([1.0, float("nan"), 0.0], dtype=BF))
    assert not same and n == 1, "-0 vs +0 must count"
    return 4


def test_qualify_real_layout():
    """qualify() on CPU with the reference as the 'kernel' and eager as the original passes for the real layout
    (strided delta, base slice), i.e. the synthetic data exercises specials without tripping the reference."""
    gen = torch.Generator().manual_seed(8)
    x, d, b = make_inputs(8, 2, 256, 16, gen, side=1)
    key = M.conv_key(x, d, b, 8, 256, 16, 2)
    assert M.qualify(key, x, d, b, 8, 256, 16, 2, M.grouped_conv_reference, _grouped_conv)
    return 1


# =====================================================================================================
# 4. install hooks
# =====================================================================================================
def test_install_hooks():
    _reset()
    fake = types.ModuleType("fake_qwen3_dflash2")
    fake._grouped_conv = _grouped_conv
    M._install_model(fake)
    assert getattr(fake._grouped_conv, "_glm_conv_fused", False) and M.S["orig"] is _grouped_conv
    wrapped = fake._grouped_conv
    M._install_model(fake)
    assert fake._grouped_conv is wrapped, "idempotent"
    # default mode (env unset) -> original path
    os.environ.pop("GLM_DRAFT_CONV_FUSED", None)
    gen = torch.Generator().manual_seed(9)
    x, d, b = make_inputs(8, 2, 8, 4, gen)
    assert same_bits(fake._grouped_conv(x, d, b, 8, 8, 4, 2), _grouped_conv(x, d, b, 8, 8, 4, 2))

    class CudaGraphManager:
        def run_fullgraph(self, desc):
            return ("replayed", desc)

    class DFlashCudaGraphManager(CudaGraphManager):
        pass

    cg = types.ModuleType("fake_cg")
    cg.DFlashCudaGraphManager = DFlashCudaGraphManager
    M._install_cg(cg)
    M._install_cg(cg)
    r0 = M.S["replays"]
    os.environ["GLM_DRAFT_CONV_FUSED_LOG_EVERY"] = "0"
    try:
        assert DFlashCudaGraphManager().run_fullgraph("d") == ("replayed", "d")
        assert CudaGraphManager().run_fullgraph("t") == ("replayed", "t")
    finally:
        os.environ.pop("GLM_DRAFT_CONV_FUSED_LOG_EVERY", None)
    assert M.S["replays"] == r0 + 1, "only the drafter manager is hooked, once"

    # mode() parsing
    for raw, want_mode in (("", "off"), ("0", "off"), ("1", "on"), ("on", "on"), ("check", "check"),
                           ("stock", "stock"), ("STOCK", "stock")):
        os.environ["GLM_DRAFT_CONV_FUSED"] = raw
        assert M.mode() == want_mode, (raw, M.mode())
    os.environ.pop("GLM_DRAFT_CONV_FUSED", None)

    # register() without the switch: nothing
    before = list(sys.meta_path)
    M.register()
    assert sys.meta_path == before
    return 6


def main():
    tests = [test_reference_matches_stock, test_reference_is_order_sensitive, test_kernel_emulation,
             test_eligible_reasons, test_dispatch_paths, test_ineligible_goes_to_original, test_synthetic_and_bits,
             test_qualify_real_layout, test_install_hooks]
    failed = 0
    for t in tests:
        try:
            n = t()
            print(f"PASS {t.__name__} ({n})")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
