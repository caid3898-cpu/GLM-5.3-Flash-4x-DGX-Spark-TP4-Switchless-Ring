# SPDX-License-Identifier: Apache-2.0
"""GLM_MAMBA_ALIGN_FIX (default 1): prefill chunks end where KDA state checkpoints are valid.

The bug (GitHub issue #2, 2026-09-29)
  vLLM 487ecf187's ``Scheduler._mamba_block_aligned_split`` aligns prefill chunk ends to
  ``cache_config.block_size``. In this hybrid model that value is the smallest KV group block
  (1152, set by the DFlash2 drafter's group in ``EngineCore``), while a KDA/Mamba state slot
  covers ``cache_config.mamba_block_size`` = 2304 tokens (``--block-size``). The align-mode
  invariant is "slot i holds the state after exactly (i + 1) * 2304 tokens"; the KDA prefill
  writes only the final state of each chunk into slot ``(end - 1) // 2304``. A chunk that ends
  on an odd multiple of 1152 therefore leaves slot i holding S(end) while its prefix-cache hash
  claims (i + 1) * 2304 = end + 1152.
  With DFlash (``use_eagle``) the split also backs the last checkpoint off one 1152 unit, so for
  a 99,447-token prompt the chunks end ... 92160, 97920, 99447. Slot 42 keeps S(97920) under the
  hash of 43 * 2304 = 99072. Our coordinator repair (overlay/kv_cache_coordinator.py, which stops
  the drafter group from shrinking the hit) makes that block reachable. A warm request then
  resumes the recurrent (and short-conv) state 1152 tokens early: the KDA layers never see
  [97920, 99072), and the model reports phantom corruption in exactly that half block
  (measured: cold 0/8, warm 7/7 at T=0 and T=0.8; cold-vs-warm logprob drift 5-8x the noise).

The fix (two edits of the image's method, hash-pinned; everything else unchanged)
  1. align to ``cache_config.mamba_block_size`` (2304), the size a state slot actually covers;
  2. add the prompt's last full block boundary (``num_tokens // 2304 * 2304``) to the mandatory
     stops, so the block the repaired coordinator exposes is really materialized. The existing
     Eagle back-off checkpoint is kept.
  For the 99,447-token prompt the final chunks become 96768 -> 99072 -> 99447.
  Side effect: chunk ends align to 2304, so a 6905-token budget (6912 minus the 7 lookahead
  slots) gives 4608-token chunks instead of 5760.

The method is vLLM's (Apache-2.0); this is a two-line text edit of it. The same scheduler code is in upstream
vLLM 487ecf187, so any hybrid model whose smallest KV group block is below its mamba block size is exposed.

GLM_MAMBA_ALIGN_FIX=0 leaves the stock method (for A/B only; it is incorrect with prefix caching).
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys
import textwrap

EXPECTED_SHA = "712bf8266d6ce2a3"   # vllm 487ecf187 Scheduler._mamba_block_aligned_split, sha256 prefix
TAG = "# GLM_MAMBA_ALIGN_FIX"
EDITS = [
    ("        block_size = self.cache_config.block_size\n",
     "        block_size = self.cache_config.mamba_block_size or self.cache_config.block_size  " + TAG + "\n"),
    ("            # Never run past the last cacheable block boundary mid-chunk.\n"
     "            last_cache_position,\n",
     "            # Never run past the last cacheable block boundary mid-chunk.\n"
     "            last_cache_position,\n"
     "            # " + TAG + ": also materialize the prompt's last full block, which the\n"
     "            # repaired coordinator exposes to prefix hits (Eagle back-off kept above).\n"
     "            request.num_tokens // block_size * block_size,\n"),
]


def enabled() -> bool:
    v = os.environ.get("GLM_MAMBA_ALIGN_FIX", "1").strip()
    if v not in ("0", "1"):
        raise ValueError("GLM_MAMBA_ALIGN_FIX must be 0 or 1")
    return v == "1"


def patch_source(src: str) -> str:
    for old, new in EDITS:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"glm-mamba-align-fix: anchor found {n} times: {old.strip()[:70]!r}")
        src = src.replace(old, new)
    return src


def build(fn, module_globals: dict):
    """Return the patched function built from ``fn``'s source (checked against EXPECTED_SHA)."""
    src = inspect.getsource(fn)
    sha = hashlib.sha256(src.encode()).hexdigest()[:16]
    if sha != EXPECTED_SHA:
        raise RuntimeError(f"glm-mamba-align-fix: _mamba_block_aligned_split sha {sha} != {EXPECTED_SHA}; not installing")
    code = textwrap.dedent(patch_source(src))
    ns: dict = {}
    exec(compile(code, f"<glm_mamba_align_fix:{fn.__qualname__}>", "exec"), module_globals, ns)
    new = ns[fn.__name__]
    new.__qualname__ = fn.__qualname__
    new.__glm_mamba_align_fix__ = True
    return new


def install(mod) -> None:
    S = mod.Scheduler
    fn = S._mamba_block_aligned_split
    if getattr(fn, "__glm_mamba_align_fix__", False):
        return
    S._mamba_block_aligned_split = build(fn, vars(mod))
    sys.stderr.write("glm-mamba-align-fix: Scheduler._mamba_block_aligned_split aligned to mamba_block_size, "
                     "last full block stop added\n")


def register() -> None:
    if not enabled():
        sys.stderr.write("glm-mamba-align-fix: OFF (GLM_MAMBA_ALIGN_FIX=0)\n")
        return
    import importlib.abc
    import importlib.util

    name = "vllm.v1.core.sched.scheduler"
    if name in sys.modules:
        install(sys.modules[name])
        return

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != name:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            ex = spec.loader.exec_module

            def patched(module):
                ex(module)
                install(module)

            spec.loader.exec_module = patched
            return spec

    sys.meta_path.insert(0, _Hook())
