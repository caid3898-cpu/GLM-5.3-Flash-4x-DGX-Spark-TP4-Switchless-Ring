#!/usr/bin/env python3
"""Clamp the long side of an image to ``GLM_MM_MAX_SIDE`` before the model's
``smart_resize`` picks a canvas (glm-mm-resize-20261008).

Why this exists
---------------
``--limit-mm-per-prompt`` bounds the image *count* (128) and
``max_image_tokens`` bounds the *token* cost of one image (4096). Neither
bounds the pixels the vision encoder actually decodes and holds: the crash
dump shows every image's ``pixel_values`` resident as a full bfloat16 tensor
(``slices=[[slice(0,16280)]]``), so 32 x 12 MP killed the worker with
``Worker proc VllmWorker-0 died unexpectedly (exit code: None)`` — a signal,
not a Python exception. Token-side compression does not shrink that tensor.

So the budget has to be applied to the *long side* before ``smart_resize``:
scaling to a max side of 1024 px shrinks both the token cost and, crucially,
the encoder's working set. Images already within the limit pass through
byte-identical — this is a ceiling, not a resampler.

``smart_resize`` is the single funnel for all three call sites in
``vllm/transformers_utils/processors/glm5next.py`` (preprocess, the
placeholder-count path and the per-image token accounting), so wrapping it
covers every path. We only pre-scale the requested ``(h, w)``; the alignment
rounding and the budget logic stay entirely upstream's.

Environment
-----------
GLM_MM_MAX_SIDE   long-side ceiling in pixels. 0/empty/unset = off.
                  Aspect ratio is preserved; the result is never enlarged.
GLM_MM_MAX_SIDE_DRY  set to 1 to log every decision without clamping.
"""

import os
import sys

# Alignment factors the upstream funnels round up to (patch_size * merge_size
# * patch_expand_factor = 14 * 2 * 1). A clamped side must stay a multiple of
# this so we do not change how upstream aligns the canvas.
ALIGN = 28

_STATE = {"hooked": False, "orig": None, "applied": 0, "seen": 0, "dry": False}


def _ceil_align(v: int) -> int:
    return ((v + ALIGN - 1) // ALIGN) * ALIGN


def _floor_align(v: int) -> int:
    return max(ALIGN, (v // ALIGN) * ALIGN)


def _clamp(h: int, w: int, max_side: int) -> tuple[int, int]:
    """Return (h, w) scaled so the long side is <= max_side, ratio preserved."""
    long_side = max(h, w)
    if long_side <= max_side:
        return h, w
    scale = max_side / long_side
    nh, nw = int(h * scale), int(w * scale)
    return _floor_align(nh), _floor_align(nw)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "off", "false", "no")


def register() -> bool:
    """Wrap the processor's smart_resize. Idempotent; never fatal."""
    if _STATE["hooked"]:
        return True
    try:
        import vllm.transformers_utils.processors.glm5next as P
    except Exception as e:  # noqa: BLE001
        print(f"glm-mm-resize: processor module unavailable, inert ({e})", file=sys.stderr)
        return False
    if not hasattr(P, "smart_resize"):
        print("glm-mm-resize: smart_resize absent, inert", file=sys.stderr)
        return False

    raw_max = os.environ.get("GLM_MM_MAX_SIDE", "").strip()
    try:
        max_side = int(raw_max)
    except ValueError:
        print(f"glm-mm-resize: bad GLM_MM_MAX_SIDE={raw_max!r}, inert", file=sys.stderr)
        return False
    if max_side <= 0:
        print("glm-mm-resize: max side <= 0, inert", file=sys.stderr)
        return False
    # Never let the ceiling fall below the alignment unit or a whole token.
    max_side = max(_ceil_align(max_side), ALIGN * 2)

    _STATE["dry"] = _env_on("GLM_MM_MAX_SIDE_DRY")
    orig = P.smart_resize

    def smart_resize(t: int, h: int, w: int, *args, **kwargs):  # noqa: ANN001
        _STATE["seen"] += 1
        ch, cw = _clamp(int(h), int(w), max_side)
        if (ch, cw) != (h, w) and not _STATE["dry"]:
            _STATE["applied"] += 1
        # Rewrite only the positionally-passed h/w (the two funnels that call
        # smart_resize pass everything else by keyword).
        if "h" in kwargs:
            kwargs["h"] = ch
        if "w" in kwargs:
            kwargs["w"] = cw
        if (ch, cw) != (h, w):
            return orig(t, ch, cw, *args, **kwargs)
        return orig(t, h, w, *args, **kwargs)

    smart_resize.__wrapped__ = orig
    P.smart_resize = smart_resize
    _STATE["orig"] = orig
    _STATE["hooked"] = True
    print(f"glm-mm-resize: armed max_side={max_side}px align={ALIGN} "
          f"dry={int(_STATE['dry'])} ({P.__name__})", file=sys.stderr)
    return True


def status() -> dict:
    return dict(_STATE, max_side=os.environ.get("GLM_MM_MAX_SIDE", ""))


if __name__ == "__main__":
    print(status())
