# SPDX-License-Identifier: Apache-2.0
"""GLM_DRAFT_CONV_FUSED=1|check: the DFlash2 drafter's grouped short convolution as ONE kernel. Bit-exact by design.

Target: Tony v11 image (vLLM 0.1.dev20051+g487ecf187), drafter vllm/model_executor/models/qwen3_dflash2.py
(incoai GLM-5.3-Flash-DFlash2-fp8blk, conv_kernel_size 2, conv_group_size 16, block size 1 + K_HI = 8).

Why. `_grouped_conv` runs as ten tiny eager PyTorch kernels (bf16 add for the coefficients, mul, arange, bitwise
and, F.pad fill + copy, mul, compare, mul by the mask, in-place add). DFlash2 calls it four times per draft layer
(attention_conv.prepare/finish, mlp_conv.prepare/finish), so the captured drafter graph carries 20 x 10 = 200 of its
313 nodes for it: 0.38 ms of the 3.2 ms draft step at c1 and c4 (glm-megagraph-20260928 REPORT.md, measured on the
09-28 traces). One kernel per call moves the same ~64-256 KB and removes 180 graph nodes: ~0.30-0.36 ms per step.

What is computed. Exactly the eager op sequence, per element, in the eager rounding order (every PyTorch bf16
elementwise op computes in fp32 and rounds once to bf16, round-to-nearest-even):
    c_t   = bf16(base[t, h] + delta[m, t, g(h)])                    t = 0 .. taps-1, g(h) = h // group_size
    out   = bf16(c_0 * x[m, h])
    for t in 1 .. taps-1:
        s   = x[m - t, h] if m >= t else +0.0                        (F.pad zeros)
        t1  = bf16(c_t * s)
        t2  = bf16(t1 * (1.0 if (m % block_size) >= t else 0.0))     (the bool mask promoted to bf16)
        out = bf16(out + t2)
The kernel keeps every intermediate rounding, multiplies by the mask instead of selecting (so -0 and NaN propagate
as in eager) and is launched with enable_fp_fusion=False (no mul+add contraction). `grouped_conv_reference` below is
the same computation in torch, used by the CPU tests to pin the semantics against the verbatim eager function.

Safety (fail-closed, output unchanged by construction):
  * dispatch: only CUDA bf16 inputs with a unit-stride last dimension and the documented shapes; anything else runs
    the original function.
  * qualification (default on): before a shape/stride key is used fused, the kernel and the original run once on
    synthetic data of that exact layout (random values over many binades plus +-0, subnormals, overflow, inf and
    NaN); any difference in raw bits (NaN positions included) marks the key eager for good and logs it. Keys first
    seen inside a CUDA-graph capture run eager (no qualification can run there); vLLM's warm-up pass precedes every
    capture, so in practice every captured key is qualified.
  * check mode (GLM_DRAFT_CONV_FUSED=check, TEST ONLY): every call computes BOTH, returns the ORIGINAL result, and
    adds the number of differing bf16 elements to a device counter with device ops, so the check also runs inside
    the captured drafter graphs on live traffic. The counters are read every GLM_DRAFT_CONV_FUSED_LOG_EVERY
    (default 2000) draft-graph replays (one tiny D2H) and logged on every rank.
  * timing (GLM_DRAFT_CONV_FUSED_TIMING=N): CUDA events around every drafter graph replay, median per (A/B variant,
    padded token count) logged every N replays; the drafter graph has fixed shapes, so a boot with =stock and one
    with =1 compare directly, and so do the two variants of an in-boot A/B (GLM_AB_DRAFT_SETS=1).

Knobs:
  GLM_DRAFT_CONV_FUSED=0|1|check|stock default 0: register() does nothing, torch is not imported; stock = hooks
                                       installed but the original conv runs (the control arm for timing)
  GLM_DRAFT_CONV_FUSED_QUAL=0|1        per-key qualification before first fused use (default 1)
  GLM_DRAFT_CONV_FUSED_BLOCK=1024      elements of one row per program (power of two)
  GLM_DRAFT_CONV_FUSED_LOG_EVERY=2000  counters / status cadence in drafter-graph replays (0 = never)
  GLM_DRAFT_CONV_FUSED_TIMING=0        N > 0: log the median drafter-graph replay time every N replays

Credits: DFlash2 drafter and `_grouped_conv` (vLLM qwen3_dflash2.py; incoai GLM-5.3-Flash-DFlash2); the fail-closed
qualify-then-use pattern and the "measured association" discipline follow our DS4.1 eager-glue work (vcapk) and the
09-27 mHC lesson (a fused kernel that was only close was rejected by a raw-byte check on GPU).
"""
from __future__ import annotations

import collections
import importlib.abc
import importlib.util
import os
import statistics
import sys
import zlib

_OFF = ("", "0", "off", "false", "no")
TARGET_MODEL = "vllm.model_executor.models.qwen3_dflash2"
TARGET_CG = "vllm.v1.worker.gpu.spec_decode.dflash.cudagraph"

S = {
    "orig": None,            # the original _grouped_conv
    "ok": set(),             # qualified keys
    "bad": set(),            # keys that failed qualification (eager for good)
    "counts": collections.Counter(),
    "dev_counters": None,    # check mode: int64 [2] = (calls, differing elements), device resident
    "replays": 0,
    "timing": collections.defaultdict(list),   # (A/B variant or None, padded tokens) -> ms per replay
    "pending_ev": collections.deque(),
    "installed": set(),
    "warned": set(),
}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-draft-conv-fused: {msg}\n")


def _env(name: str, default=None):
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def mode() -> str:
    """off | stock (hooks installed, original conv: the timing/control arm) | on | check."""
    v = str(_env("GLM_DRAFT_CONV_FUSED", "0")).strip().lower()
    if v in _OFF:
        return "off"
    if v in ("stock", "check"):
        return v
    return "on"


def _variant():
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        try:
            return int(ab.current())
        except Exception:  # noqa: BLE001
            return None
    return None


def _int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default)) or default))
    except (TypeError, ValueError):
        return default


# =====================================================================================================
# The semantics (torch only; CPU-testable). This is the specification the Triton kernel implements.
# =====================================================================================================

def grouped_conv_reference(hidden_states, delta, base, block_size: int, num_groups: int, group_size: int,
                           taps: int):
    """Per-element eager semantics of qwen3_dflash2._grouped_conv with every bf16 rounding made explicit.

    hidden_states [M, H] bf16, delta [M, taps, num_groups] bf16 (any strides), base [taps, H] bf16.
    Returns [M, H] bf16, contiguous."""
    import torch

    bf16, f32 = torch.bfloat16, torch.float32

    def rnd(t):
        return t.to(bf16).to(f32)

    m_rows = hidden_states.shape[0]
    x = hidden_states.to(f32).reshape(m_rows, num_groups, group_size)
    d = delta.to(f32)
    b = base.to(f32).reshape(taps, num_groups, group_size)
    out = rnd(rnd(b[0].unsqueeze(0) + d[:, 0, :, None]) * x)
    pos = torch.arange(m_rows) % int(block_size)
    for tap in range(1, taps):
        c = rnd(b[tap].unsqueeze(0) + d[:, tap, :, None])
        sh = torch.zeros_like(x)
        if m_rows > tap:
            sh[tap:] = x[:-tap]
        t1 = rnd(c * sh)
        msk = (pos >= tap).to(f32).view(-1, 1, 1)
        t2 = rnd(t1 * msk)
        out = rnd(out + t2)
    return out.to(bf16).reshape(m_rows, num_groups * group_size).contiguous()


def conv_key(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    """Everything the kernel's addressing and the qualification depend on (no data pointers)."""
    return (tuple(hidden_states.shape), tuple(hidden_states.stride()), tuple(delta.shape), tuple(delta.stride()),
            tuple(base.shape), tuple(base.stride()), int(block_size), int(num_groups), int(group_size), int(taps),
            str(hidden_states.device.type))


def eligible(hidden_states, delta, base, block_size, num_groups, group_size, taps,
             require_cuda: bool = True) -> str | None:
    """None when the fused kernel may run, else the reason (the caller then runs the original)."""
    import torch

    if delta.device != hidden_states.device or base.device != hidden_states.device:
        return "device"
    if require_cuda and hidden_states.device.type != "cuda":
        return "device"
    if not (hidden_states.dtype == delta.dtype == base.dtype == torch.bfloat16):
        return "dtype"
    if hidden_states.dim() != 2 or delta.dim() != 3 or base.dim() != 2:
        return "rank"
    m_rows, h = hidden_states.shape
    if m_rows < 1 or int(taps) < 1 or int(block_size) < 1:
        return "size"
    if int(num_groups) * int(group_size) != h or tuple(delta.shape) != (m_rows, int(taps), int(num_groups)):
        return "shape"
    if tuple(base.shape) != (int(taps), h):
        return "shape"
    if hidden_states.stride(1) != 1 or base.stride(1) != 1:
        return "stride"
    if m_rows * max(hidden_states.stride(0), h) >= 2 ** 31:
        return "size"
    return None


# =====================================================================================================
# Triton kernel (built lazily, CUDA only)
# =====================================================================================================
_K = {}


def _kernel():
    if "k" in _K:
        return _K["k"]
    import triton
    import triton.language as tl

    @triton.jit
    def _grouped_conv_kernel(x_ptr, d_ptr, b_ptr, o_ptr, H, sxm, sdm, sdt, sdg, sbt, BS,
                             GS: tl.constexpr, TAPS: tl.constexpr, BLOCK: tl.constexpr):
        m = tl.program_id(0)
        h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        hm = h < H
        g = h // GS
        x = tl.load(x_ptr + m * sxm + h, mask=hm, other=0.0).to(tl.float32)
        b0 = tl.load(b_ptr + h, mask=hm, other=0.0).to(tl.float32)
        d0 = tl.load(d_ptr + m * sdm + g * sdg, mask=hm, other=0.0).to(tl.float32)
        c0 = (b0 + d0).to(tl.bfloat16).to(tl.float32)
        out = (c0 * x).to(tl.bfloat16).to(tl.float32)
        pos = m % BS
        for tap in tl.static_range(1, TAPS):
            bt = tl.load(b_ptr + tap * sbt + h, mask=hm, other=0.0).to(tl.float32)
            dt = tl.load(d_ptr + m * sdm + tap * sdt + g * sdg, mask=hm, other=0.0).to(tl.float32)
            ct = (bt + dt).to(tl.bfloat16).to(tl.float32)
            src_ok = m >= tap
            sh = tl.load(x_ptr + (m - tap) * sxm + h, mask=hm & src_ok, other=0.0).to(tl.float32)
            t1 = (ct * sh).to(tl.bfloat16).to(tl.float32)
            mk = tl.where(pos >= tap, 1.0, 0.0)
            t2 = (t1 * mk).to(tl.bfloat16).to(tl.float32)
            out = (out + t2).to(tl.bfloat16).to(tl.float32)
        tl.store(o_ptr + m * H + h, out.to(tl.bfloat16), mask=hm)

    _K["k"] = _grouped_conv_kernel
    return _grouped_conv_kernel


def grouped_conv_fused(hidden_states, delta, base, block_size, num_groups, group_size, taps):
    """The fused kernel (no checks; see `eligible`)."""
    import torch
    import triton

    m_rows, h = hidden_states.shape
    out = torch.empty((m_rows, h), dtype=hidden_states.dtype, device=hidden_states.device)
    block = _int("GLM_DRAFT_CONV_FUSED_BLOCK", 1024)
    block = max(128, min(4096, triton.next_power_of_2(block)))
    grid = (m_rows, triton.cdiv(h, block))
    _kernel()[grid](hidden_states, delta, base, out, h, hidden_states.stride(0), delta.stride(0), delta.stride(1),
                    delta.stride(2), base.stride(0), int(block_size), GS=int(group_size), TAPS=int(taps),
                    BLOCK=block, num_warps=4, enable_fp_fusion=False)
    return out


# =====================================================================================================
# Qualification and comparison helpers
# =====================================================================================================

def bits_equal(a, b) -> tuple[bool, int]:
    """(identical raw bf16 bits, number of differing elements). NaN payloads count as differences."""
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, -1
    diff = int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum().item())
    return diff == 0, diff


def synthetic_like(t, gen, special: bool = True):
    """A tensor with t's shape, strides and dtype, filled with values over many binades plus specials."""
    import torch
    out = torch.empty_strided(tuple(t.shape), tuple(t.stride()), dtype=t.dtype, device=t.device)
    n = t.numel()
    vals = torch.randn(n, generator=gen, device="cpu", dtype=torch.float32)
    scale = torch.pow(2.0, torch.randint(-30, 30, (n,), generator=gen).to(torch.float32))
    vals = vals * scale
    if special and n >= 16:
        idx = torch.randperm(n, generator=gen)[: max(8, n // 50)]
        specials = torch.tensor([0.0, -0.0, 1e-40, -1e-40, 3.0e38, -3.0e38, float("inf"), float("-inf"),
                                 float("nan"), 1.0, -1.0, 2.0 ** -126], dtype=torch.float32)
        vals[idx] = specials[torch.arange(idx.numel()) % specials.numel()]
    out.copy_(vals.view(tuple(t.shape)).to(t.dtype))
    return out


def qualify(key, hidden_states, delta, base, block_size, num_groups, group_size, taps, impl, orig,
            rounds: int = 3) -> bool:
    """Run impl and orig on synthetic tensors of the exact layout; True when every round is bit-identical."""
    import torch
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260928 + zlib.crc32(repr(key).encode()))  # same data on every rank: same verdict
    for r in range(rounds):
        x = synthetic_like(hidden_states, gen, special=r > 0)
        d = synthetic_like(delta, gen, special=r > 0)
        b = synthetic_like(base, gen, special=r > 1)
        want = orig(x, d, b, block_size, num_groups, group_size, taps)
        got = impl(x, d, b, block_size, num_groups, group_size, taps)
        same, ndiff = bits_equal(got.reshape(want.shape), want)
        if not same:
            _log(f"qualification FAILED for {key}: round {r}, {ndiff} differing elements; this shape stays eager")
            return False
    return True


def _capturing() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _compiling() -> bool:
    """Under torch.compile tracing the original runs (Dynamo cannot trace the dispatch; Inductor fuses anyway)."""
    try:
        import torch
        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


# =====================================================================================================
# The replacement for qwen3_dflash2._grouped_conv
# =====================================================================================================

def make_dispatch(orig, impl=None, is_capturing=None, get_mode=None, qual_on=None):
    """Build the drop-in replacement. `impl`, `is_capturing`, `get_mode`, `qual_on` are injectable for tests."""
    impl = impl or grouped_conv_fused
    is_capturing = is_capturing or _capturing
    get_mode = get_mode or mode
    qual_on = qual_on or (lambda: str(_env("GLM_DRAFT_CONV_FUSED_QUAL", "1")).strip().lower() not in _OFF)

    def _grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps):
        m = get_mode()
        if m in ("off", "stock") or _compiling():
            return orig(hidden_states, delta, base, block_size, num_groups, group_size, taps)
        why = eligible(hidden_states, delta, base, block_size, num_groups, group_size, taps)
        if why is not None:
            S["counts"]["fallback_" + why] += 1
            return orig(hidden_states, delta, base, block_size, num_groups, group_size, taps)
        key = conv_key(hidden_states, delta, base, block_size, num_groups, group_size, taps)
        if key in S["bad"]:
            S["counts"]["fallback_bad"] += 1
            return orig(hidden_states, delta, base, block_size, num_groups, group_size, taps)
        if key not in S["ok"]:
            if not qual_on():
                S["ok"].add(key)
            elif is_capturing():
                S["counts"]["fallback_unqualified_in_capture"] += 1
                if "capture" not in S["warned"]:
                    S["warned"].add("capture")
                    _log(f"first sight of {key} inside a capture: eager for this capture")
                return orig(hidden_states, delta, base, block_size, num_groups, group_size, taps)
            elif qualify(key, hidden_states, delta, base, block_size, num_groups, group_size, taps, impl, orig):
                S["ok"].add(key)
                S["counts"]["qualified"] += 1
            else:
                S["bad"].add(key)
                return orig(hidden_states, delta, base, block_size, num_groups, group_size, taps)
        if m == "check":
            ref = orig(hidden_states, delta, base, block_size, num_groups, group_size, taps)
            got = impl(hidden_states, delta, base, block_size, num_groups, group_size, taps)
            _accumulate(ref, got)
            S["counts"]["checked_calls"] += 1
            return ref
        S["counts"]["fused_calls"] += 1
        return impl(hidden_states, delta, base, block_size, num_groups, group_size, taps)

    _grouped_conv._glm_conv_fused = True
    return _grouped_conv


def _accumulate(ref, got) -> None:
    """Device-side (graph-capturable) mismatch accounting: counters[0] += 1, counters[1] += #differing elements."""
    import torch
    c = S["dev_counters"]
    if c is None or c.device != ref.device:
        if _capturing():  # a counter born inside a capture would be re-zeroed by every replay
            S["counts"]["check_skipped_in_capture"] += 1
            return
        c = torch.zeros(2, dtype=torch.int64, device=ref.device)
        S["dev_counters"] = c
    ne = (ref.reshape(-1).view(torch.int16) != got.reshape(-1).view(torch.int16)).sum()
    c[0:1].add_(1)
    c[1:2].add_(ne)


def counters() -> tuple[int, int] | None:
    c = S["dev_counters"]
    if c is None:
        return None
    v = c.tolist()
    return int(v[0]), int(v[1])


def status() -> dict:
    med = {str(v): round(statistics.median(t[-4000:]), 4) for v, t in S["timing"].items() if t}
    return {"mode": mode(), "installed": sorted(S["installed"]), "qualified_keys": len(S["ok"]),
            "bad_keys": len(S["bad"]), "counts": dict(S["counts"]), "replays": S["replays"],
            "device_counters": counters() if mode() == "check" and not _capturing() else None,
            "draft_graph_ms_median_by_variant": med or None}


# =====================================================================================================
# Drafter-graph replay hook (counters and timing live here: the conv itself runs inside the graph)
# =====================================================================================================

def _after_replay() -> None:
    S["replays"] += 1
    every = _int("GLM_DRAFT_CONV_FUSED_LOG_EVERY", 2000)
    if every and S["replays"] % every == 0:
        st = status()
        _log(f"status {st}")
        c = st.get("device_counters")
        if c is not None and c[1] != 0:
            _log(f"CHECK MISMATCH: {c[1]} differing elements over {c[0]} conv calls")


def _install_cg(mod) -> None:
    cls = mod.DFlashCudaGraphManager
    if getattr(cls, "_glm_conv_fused", False):
        return
    cls._glm_conv_fused = True
    own = cls.__dict__.get("run_fullgraph")

    def base_run(self, desc):
        # resolved per call, so a wrapper installed on the base class later (overlay/glm_ab.py) is honoured
        return own(self, desc) if own is not None else super(cls, self).run_fullgraph(desc)

    def run_fullgraph(self, desc):
        every = _int("GLM_DRAFT_CONV_FUSED_TIMING", 0)
        if every and not _capturing():
            import torch
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            out = base_run(self, desc)
            b.record()
            pend = S["pending_ev"]
            pend.append((a, b, _variant(), desc.num_tokens))
            while pend and pend[0][1].query():
                x, y, v, n = pend.popleft()
                t = S["timing"][(v, n)]
                t.append(x.elapsed_time(y))
                if len(t) > 8000:
                    del t[:4000]
            if S["replays"] % every == 0:
                for (v, n), t in sorted(S["timing"].items(), key=lambda kv: str(kv[0])):
                    t = t[-every:]
                    if t:
                        _log(f"drafter graph replay ms: variant {v} tokens {n} median {statistics.median(t):.4f} "
                             f"p10 {sorted(t)[len(t) // 10]:.4f} n {len(t)}")
        else:
            out = base_run(self, desc)
        _after_replay()
        return out

    cls.run_fullgraph = run_fullgraph
    S["installed"].add("cudagraph")


def _install_model(mod) -> None:
    fn = getattr(mod, "_grouped_conv", None)
    if fn is None:
        _log("qwen3_dflash2 has no _grouped_conv: not installed")
        return
    if getattr(fn, "_glm_conv_fused", False):
        return
    S["orig"] = fn
    mod._grouped_conv = make_dispatch(fn)
    S["installed"].add("model")
    _log(f"qwen3_dflash2._grouped_conv wrapped (mode {mode()})")


HOOKS = {TARGET_MODEL: _install_model, TARGET_CG: _install_cg}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        fn = HOOKS.get(name)
        if fn is None:
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

        def exec_module(module, _orig=orig_exec, _fn=fn):
            _orig(module)
            _fn(module)

        loader.exec_module = exec_module
        return spec


def register() -> None:
    """Idempotent. No-op unless GLM_DRAFT_CONV_FUSED is set (default off)."""
    if str(os.environ.get("GLM_DRAFT_CONV_FUSED", "0")).strip().lower() in _OFF:
        return
    for name, fn in HOOKS.items():  # a module imported before us is patched in place
        if name in sys.modules:
            fn(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    _log("registered (default-off switch GLM_DRAFT_CONV_FUSED is on)")
