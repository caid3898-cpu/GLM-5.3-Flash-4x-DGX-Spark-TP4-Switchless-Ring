# SPDX-License-Identifier: Apache-2.0
"""GLM_SHM_BUSY_LOOP_S=<seconds>: shorter busy-wait for vLLM's shared-memory message queue readers.

vLLM's `SpinCondition` (distributed/device_communicators/shm_broadcast.py) keeps a reader spinning on
`sched_yield()` for `busy_loop_s` = 1 s after every message before it falls back to a zmq poll. At
decode rates every worker therefore spins a CPU core all the time. On GB10 the CPU and GPU share one
power budget, so the spinning is not free. nacyot measured the effect on GB10
(https://artifacts.nacyot.com/vllm-spin-wait-gb10-en/) and mmastrac carries the same change in his
Spark stack; this module is an independent implementation of that idea.

Only the reader-side timeout changes; message contents and ordering are untouched, so model output is
exactly the same. Unset (the default) keeps vLLM's 1 s.
"""
from __future__ import annotations

import os
import sys

TARGET = "vllm.distributed.device_communicators.shm_broadcast"
ENV = "GLM_SHM_BUSY_LOOP_S"


def _value() -> float | None:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v >= 0 else None


def _patch(mod) -> None:
    value = _value()
    cls = getattr(mod, "SpinCondition", None)
    if value is None or cls is None or getattr(cls, "_glm_spin_patched", False):
        return
    orig_init = cls.__init__

    def __init__(self, is_reader, context, notify_address, busy_loop_s=1, **kw):
        orig_init(self, is_reader, context, notify_address, busy_loop_s=value if is_reader else busy_loop_s, **kw)

    cls.__init__ = __init__
    cls._glm_spin_patched = True
    sys.stderr.write(f"glm-shm-spin: SpinCondition reader busy_loop_s = {value}\n")


def register() -> None:
    import importlib.abc
    import importlib.util

    if _value() is None:
        return
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)  # one-shot: the module is imported once per process
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                _patch(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
