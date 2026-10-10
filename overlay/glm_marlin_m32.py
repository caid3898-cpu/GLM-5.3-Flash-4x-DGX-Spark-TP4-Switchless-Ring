"""Force the fused_marlin_moe M-tile (decode MoE) — default-off.

Env GLM_MARLIN_MOE_BLOCK_M=<16|32|48|64> arms the patch; anything else is a
no-op (cheap flag pre-check, no torch import at interpreter startup).

Why: vllm/model_executor/layers/fused_moe/experts/marlin_moe.py selects
block_size_m from the occupancy rule
    for bsm in [8, 16, 32, 48, 64]: if M * topk / E / bsm < 0.9: break
then floors to 16 ONLY when the input is 1-byte (fp8). Our decode MoE is
NVFP4 marlin — the packed fp4 input is 4-byte, so the floor does NOT apply
and the natural decode tile is 8 (observed live: "first forced align
8->32"). Prefill (M=6912) naturally selects 64 and passes through untouched.
The sweep leg forces 32 on the fused (standard-format) path ONLY — the
batched path uses batched_moe_align_block_size and must stay untouched.

Mechanics (identity handshake, consume-once):
  1. wrap module-global moe_align_block_size — the fused path is its ONLY
     caller in this module — and force the block arg when the natural
     selection is BELOW the target (prefill's 64 passes through unchanged);
     remember the returned sorted_token_ids object.
  2. wrap module-global _fused_marlin_moe; force its block_size_m kwarg only
     when the incoming sorted_token_ids IS the object from a forced align,
     then clear the handshake. Both paths pass these as kwargs
     (marlin_moe.py:340/363 fused, :513/536 batched), so align and kernel
     stay consistent by construction.

Numerics: M-tiling is zero-padding only — per-K-group FP4 dequant does not
depend on BLOCK_SIZE_M, so the change is exact by construction.
"""

import importlib.util
import os
import sys

MODNAME = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
_ALLOWED = (16, 32, 48, 64)


def _env_target():
    raw = os.environ.get("GLM_MARLIN_MOE_BLOCK_M", "").strip()
    try:
        target = int(raw)
    except ValueError:
        return None
    return target if target in _ALLOWED else None


def register():
    target = _env_target()
    if target is None or target <= 16:
        return  # 16 is the packed-input floor: nothing to force

    class _Finder:
        def find_spec(self, name, path, target_=None):
            if name != MODNAME:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            exec_module = loader.exec_module

            def patched_exec(module):
                exec_module(module)
                try:
                    patch_module(module, target)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write("glm-marlin-m32: patch failed: %r\n" % (exc,))
                    raise

            loader.exec_module = patched_exec
            return spec

    sys.meta_path.insert(0, _Finder())
    sys.stderr.write("glm-marlin-m32: armed target=%d\n" % target)


def patch_module(mm, target):
    """Wrap the two module globals in marlin_moe with the handshake."""
    state = {"last_sorted": None, "forced": False, "announced": False}
    orig_align = mm.moe_align_block_size
    orig_kernel = mm._fused_marlin_moe

    def align(topk_ids, block_size_m, *a, **k):
        state["forced"] = block_size_m < target
        if not state["forced"]:
            return orig_align(topk_ids, block_size_m, *a, **k)
        if not state["announced"]:
            state["announced"] = True
            sys.stderr.write("glm-marlin-m32: first forced align %d->%d\n"
                             % (block_size_m, target))
        out = orig_align(topk_ids, target, *a, **k)
        state["last_sorted"] = out[0]
        return out

    def kernel(*a, **k):
        last, forced = state["last_sorted"], state["forced"]
        state["last_sorted"] = None
        state["forced"] = False
        if forced and last is not None and k.get("sorted_token_ids") is last:
            k["block_size_m"] = target
        return orig_kernel(*a, **k)

    mm.moe_align_block_size = align
    mm._fused_marlin_moe = kernel
    sys.stderr.write("glm-marlin-m32: patch installed target=%d\n" % target)


def selection_outcome(m, topk, e):
    """Upstream marlin_moe.py:333 loop + packed-4-bit floor (oracle for tests)."""
    block_size_m = 8
    for bsm in (8, 16, 32, 48, 64):
        block_size_m = bsm
        if m * topk / e / bsm < 0.9:
            break
    return max(block_size_m, 16)  # packed-4-bit input floor
