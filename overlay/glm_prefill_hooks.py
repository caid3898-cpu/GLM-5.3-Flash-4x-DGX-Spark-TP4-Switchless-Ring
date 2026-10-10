# SPDX-License-Identifier: Apache-2.0
"""Import hooks and drift guard for the GLM prefill adapters.

  glm_prefill_shard  GLM_PREFILL_SHARD=comm|1      mHC prefill sharding (port of jnardiello's E03)
  glm_prefill_sched  GLM_PREFILL_CADENCE=N, GLM_PREFILL_SHORT_TOKENS, GLM_PREFILL_CADENCE_WHEN_QUEUED,
                     GLM_END_DRAIN, GLM_IDLE_COALESCE_MS
                     decode-friendly prefill scheduling (port of jnardiello's E27 / E27b / E27c / E29)

register() is called from overlay/sitecustomize.py (one marked line) in every Python process of
the container. With every variable unset or 0 it registers nothing and imports nothing.
Otherwise it runs each adapter's install(module) right after the engine module it patches has
executed, in whichever process imports it (TP workers for the model, the engine core for the
scheduler). A failing install raises, so the import (and the boot) fails loudly instead of
serving a half-patched engine.

Drift guard: func_source() extracts one function's text by parsing the module file with ast
(decorators excluded, dedented), identically on the Mac (tests, hash generation) and in the
image; check_sources() compares sha256[:16] of each against the expected table.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.util
import os
import sys
import textwrap

SHARD_VARS = ("GLM_PREFILL_SHARD",)
SCHED_VARS = ("GLM_PREFILL_CADENCE", "GLM_PREFILL_SHORT_TOKENS", "GLM_PREFILL_CADENCE_WHEN_QUEUED",
              "GLM_END_DRAIN", "GLM_IDLE_COALESCE_MS")


def _on(env, name) -> bool:
    return env.get(name, "0").strip().lower() not in ("", "0", "off", "false", "no")


def wanted(env=None) -> dict:
    env = os.environ if env is None else env
    return {"shard": any(_on(env, v) for v in SHARD_VARS),
            "sched": any(_on(env, v) for v in SCHED_VARS)}


# ------------------------------------------------------------------------------------------
# drift guard
# ------------------------------------------------------------------------------------------
def func_source(path: str, qualname: str) -> str:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    node = ast.parse(src)
    for part in qualname.split("."):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == part:
                node = child
                break
        else:
            raise KeyError(f"{qualname} not found in {path}")
    lines = src.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]
    return textwrap.dedent("".join(lines))


def src_hash(path: str, qualname: str) -> str:
    return hashlib.sha256(func_source(path, qualname).encode()).hexdigest()[:16]


def module_path(modname: str) -> str:
    mod = sys.modules.get(modname)
    if mod is not None and getattr(mod, "__file__", None):
        return mod.__file__
    spec = importlib.util.find_spec(modname)
    if spec is None or not spec.origin:
        raise ImportError(modname)
    return spec.origin


def current_hashes(expected: dict) -> dict:
    out = {}
    for key in expected:
        modname, qual = key.split(":", 1)
        try:
            out[key] = src_hash(module_path(modname), qual)
        except (ImportError, KeyError, OSError) as exc:
            out[key] = f"missing ({type(exc).__name__})"
    return out


def hashes_at(root: str, expected: dict) -> dict:
    """Like current_hashes, but from files under root (no import): root/vllm/... as installed."""
    out = {}
    for key in expected:
        modname, qual = key.split(":", 1)
        try:
            out[key] = src_hash(os.path.join(root, *modname.split(".")) + ".py", qual)
        except (KeyError, OSError) as exc:
            out[key] = f"missing ({type(exc).__name__})"
    return out


def vllm_root() -> str:
    """Directory holding the installed `vllm` package (read by path; nothing extra is imported)."""
    spec = importlib.util.find_spec("vllm")
    return os.path.dirname(os.path.dirname(spec.origin))


def check_sources(expected: dict, who: str) -> None:
    got = hashes_at(vllm_root(), expected)
    bad = {k: (expected[k], got[k]) for k in expected if got[k] != expected[k]}
    if bad:
        lines = "\n".join(f"  {k}: expected {e}, image {g}" for k, (e, g) in sorted(bad.items()))
        raise RuntimeError(f"{who}: engine sources differ from the qualified image; refusing to "
                           f"patch (unset the flag to run stock):\n{lines}")


# ------------------------------------------------------------------------------------------
# after-import hooks
# ------------------------------------------------------------------------------------------
class _AfterImport(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.hooks: dict[str, list] = {}
        self._busy: set[str] = set()

    def add(self, modname, fn):
        mod = sys.modules.get(modname)
        if mod is not None:          # already imported: patch now
            fn(mod)
            return
        self.hooks.setdefault(modname, []).append(fn)

    def find_spec(self, name, path, target=None):
        if name not in self.hooks or name in self._busy:
            return None
        self._busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._busy.discard(name)
        if spec is None or spec.loader is None:
            return None
        callbacks = self.hooks.pop(name)
        exec_module = spec.loader.exec_module

        def patched_exec(module, _exec=exec_module, _cbs=callbacks):
            _exec(module)
            for cb in _cbs:
                cb(module)

        spec.loader.exec_module = patched_exec
        return spec


_FINDER = None


def after_import(modname: str, fn) -> None:
    global _FINDER
    if _FINDER is None:
        _FINDER = _AfterImport()
        sys.meta_path.insert(0, _FINDER)
    _FINDER.add(modname, fn)


def register(env=None) -> dict:
    """Entry point from sitecustomize. Returns what was registered (for tests and logs)."""
    w = wanted(env)
    if w["shard"]:
        def _shard(mod):
            import glm_prefill_shard
            glm_prefill_shard.install(mod)
        after_import("vllm.models.glm5next.nvidia.model", _shard)
    if w["sched"]:
        def _sched(mod):
            import glm_prefill_sched
            glm_prefill_sched.install_scheduler(mod)

        def _core(mod):
            import glm_prefill_sched
            glm_prefill_sched.install_core(mod)
        after_import("vllm.v1.core.sched.scheduler", _sched)
        after_import("vllm.v1.engine.core", _core)
    return w


def _cli(argv) -> int:
    """python3 glm_prefill_hooks.py --check ROOT   (ROOT = the image's dist-packages; no GPU needed)"""
    if len(argv) != 2 or argv[0] != "--check":
        print(_cli.__doc__)
        return 2
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import glm_prefill_sched
    import glm_prefill_shard
    bad = 0
    for mod in (glm_prefill_shard, glm_prefill_sched):
        got = hashes_at(argv[1], mod.EXPECTED)
        for k, v in mod.EXPECTED.items():
            ok = got[k] == v
            bad += not ok
            print(f"{'OK  ' if ok else 'DIFF'} {mod.__name__:18s} {k}  {got[k]}{'' if ok else '  expected ' + v}")
    print(f"{bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
