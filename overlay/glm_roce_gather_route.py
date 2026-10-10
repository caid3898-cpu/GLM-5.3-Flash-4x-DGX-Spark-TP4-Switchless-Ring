# SPDX-License-Identifier: Apache-2.0
"""GLM_ROCE_AG_DIM0_NCCL: large dim-0 (row) all-gathers go to NCCL instead of the RoCEnante gather. Default OFF. Exact.

Why (diagnostics/glm-prefill-overlap-20260928, SUMMARY_MORNING.md, window 0447): the mHC prefill shard
(overlay/glm_prefill_shard.py) gathers its owned rows with GroupCoordinator.all_gather(x, 0). The shard of a 5760-row
chunk is [1440, 4096] bf16 = 11.8 MB, under GLM_ROCE_GATHER_MAX_SIZE (16 MiB), so it rides the RoCE one-shot gather.
That cap was sized for the logits gather (last dim, at most 9.9 MB). Measured on the fleet, NCCL gathers these rows
19-23 % faster (2880 rows 1.15 -> 0.93 ms, 5760 rows 2.34 -> 1.86 ms, 6912 rows 2.81 -> 2.15 ms), with byte-identical
output and the same consumer-GEMM speed. Lowering GLM_ROCE_GATHER_MAX_SIZE instead would also move the logits gathers
(last dim) off RoCE, so this module adds a dim-0 condition and leaves the global limit alone.

Rule (wraps glm_roce.adapter.GlmRoceAllReduce.should_all_gather, which the image's CudaCommunicator.all_gather wrapper
asks before routing; returning False sends the call to the stock NCCL all-gather):
    NCCL  iff  switch on  and  inp.dim() >= 2  and  dim == 0 (after normalising a negative dim)
               and  inp.numel() * inp.element_size()  >  GLM_ROCE_AG_DIM0_NCCL_ABOVE   (per-rank shard bytes)
    otherwise the adapter's own decision (unchanged).
- Last-dim gathers (logits, the vocab-parallel argmax pairs, the draft-split gathers) are never touched: for a tensor
  of two or more dims, dim 0 is not the last dim. A 1-D tensor (dim 0 is also its last dim) is left alone too.
- Decode-size dim-0 gathers stay on RoCE: 128 verify rows are a 256 KiB shard, far below the 4 MiB default.
- Rank-invariant: the decision depends only on the switch, dim, ndim and byte count, which every TP rank shares for
  one collective (same env on every container; glm_ab switches variants on every rank together).
- Exact: an all-gather is a copy, and NCCL and RoCE write the same bytes (0447 byte audit).

Knobs:
  GLM_ROCE_AG_DIM0_NCCL=0|1              the switch (read through glm_ab when the in-boot A/B harness is armed).
  GLM_ROCE_AG_DIM0_NCCL_ABOVE=4MiB       per-rank shard bytes; strictly larger dim-0 shards go to NCCL. Accepts
                                         4194304, 4MiB, 4M, 512KiB ... A bad value raises at start-up (all ranks alike).
Logs (stderr): one "installed" line per process, one line for the first rerouted gather per process, and
`stats()` counters (routed / kept dim-0 calls) for tests and debugging.
"""
from __future__ import annotations

import os
import re
import sys

ENV = "GLM_ROCE_AG_DIM0_NCCL"
ENV_ABOVE = "GLM_ROCE_AG_DIM0_NCCL_ABOVE"
DEFAULT_ABOVE = "4MiB"
TARGET = "glm_roce.adapter"
_OFF = ("", "0", "off", "false", "no")
_S = {"above": None, "routed": 0, "kept": 0, "logged": False, "installed": False}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-roce-gather-route: {msg}\n")
    sys.stderr.flush()


def parse_bytes(value) -> int:
    """"4MiB", "4M", "4096KiB", "4194304" -> bytes (binary multiples). Raises ValueError on anything else."""
    m = re.fullmatch(r"\s*(\d+)\s*([kmg]?)(i?b?)\s*", str(value).lower())
    if m is None or (m.group(3) and not m.group(2) and m.group(3) != "b"):
        raise ValueError(f"{ENV_ABOVE}: invalid byte size {value!r}")
    n = int(m.group(1)) * {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[m.group(2)]
    if n <= 0:
        raise ValueError(f"{ENV_ABOVE}: must be positive, got {value!r}")
    return n


def _raw(name: str, default: str) -> str:
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return str(ab.env(name, default) or default)
    return os.environ.get(name, default)


def switch_on() -> bool:
    return _raw(ENV, "0").strip().lower() not in _OFF


def above() -> int:
    if _S["above"] is None:
        _S["above"] = parse_bytes(os.environ.get(ENV_ABOVE, DEFAULT_ABOVE))
    return _S["above"]


def to_nccl(inp, dim: int) -> bool:
    """True when this gather must bypass RoCE (the rule in the module docstring). Pure in (switch, dim, shape, dtype)."""
    nd = inp.dim()
    if nd < 2:
        return False
    d = dim + nd if dim < 0 else dim
    if d != 0:
        return False
    if not switch_on():
        return False
    return inp.numel() * inp.element_size() > above()


def stats() -> dict:
    return {"routed": _S["routed"], "kept": _S["kept"], "above": _S["above"], "installed": _S["installed"]}


def _install(mod) -> None:
    cls = mod.GlmRoceAllReduce
    if getattr(cls.should_all_gather, "_glm_gather_route", False):
        return
    orig = cls.should_all_gather

    def should_all_gather(self, inp, dim):
        if to_nccl(inp, dim):
            _S["routed"] += 1
            if not _S["logged"]:
                _S["logged"] = True
                _log(f"rank {getattr(self, 'rank', '?')}: first dim-0 gather routed to NCCL: shard "
                     f"{tuple(inp.shape)} {str(inp.dtype).replace('torch.', '')} "
                     f"{inp.numel() * inp.element_size()} B > {above()} B")
            return False
        if inp.dim() >= 2 and (dim + inp.dim() if dim < 0 else dim) == 0:
            _S["kept"] += 1
        return orig(self, inp, dim)

    should_all_gather._glm_gather_route = True
    should_all_gather.__wrapped__ = orig
    cls.should_all_gather = should_all_gather
    _S["installed"] = True
    _log(f"installed on GlmRoceAllReduce.should_all_gather: dim-0 gathers above {above()} B per rank -> NCCL; "
         f"last-dim gathers and GLM_ROCE_GATHER_MAX_SIZE unchanged")


def register() -> None:
    """Install the wrapper when the switch is on (the glm_ab union puts '1' into os.environ for an A/B key)."""
    import importlib.abc
    import importlib.util

    if os.environ.get(ENV, "0").strip().lower() in _OFF:
        return
    above()   # validate the threshold now: a bad value fails the start-up on every rank alike
    if TARGET in sys.modules:
        _install(sys.modules[TARGET])
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                _install(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
