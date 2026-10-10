# SPDX-License-Identifier: Apache-2.0
"""In-boot A/B of GLM adapter flags: one set of target CUDA graphs per variant, switched at runtime. TEST ONLY.

Modelled on the DS4.1 harness (ds41 adapter/ab_variant.py + scripts/ab_inboot.py, ours), ported to the vLLM V2
model runner of the Tony v11 image (vLLM 0.1.dev20051+g487ecf187). Off unless GLM_AB_VARIANTS >= 2; never set it
in a serving profile (it multiplies target graph capture time and needs VLLM_SERVER_DEV_MODE=1).

  GLM_AB_VARIANTS=3
  GLM_AB_V0="GLM_TARGET_VOCAB_ARGMAX=1+GLM_KDA_STASH=1"   spell the baseline out; empty = the base env
  GLM_AB_V1="GLM_TARGET_VOCAB_ARGMAX=1+GLM_KDA_STASH=1"   (an A/A needs GLM_AB_ALLOW_AA=1)
  GLM_AB_V2="GLM_TARGET_VOCAB_ARGMAX=0+GLM_KDA_STASH=1"
  GLM_AB_ALLOW_AA=1        allow variants with identical effective configs (A/A control)
  GLM_AB_DRAFT_SETS=1      also capture the drafter's graphs once per variant (default: once, shared; no
                           switchable flag touches the drafter)
Separators inside a spec: '+', ';' or whitespace ('+' survives start.sh's EXTRA_ENV and the ssh command line).

Switchable keys (KNOWN). The adapters read them through env(name) at call time:
  GLM_TARGET_VOCAB_ARGMAX  0|1|check   glm_target_argmax: GPUModelRunner.sample (eager, outside every graph)
  GLM_KDA_NOCOPY           0|1         glm_kda_nocopy dispatcher (inside the target decode/verify graphs)
  GLM_KDA_STASH            0|1         glm_kda_stash dispatcher + its _forward flag (inside the target graphs)
  GLM_ROUTER_FP32OUT       0|1         glm_kda_stash.install_router GateLinear (inside the target graphs)
  GLM_GATE_GEMV            0|1         glm_small_gemv Indexer head gate (inside the target graphs)
  GLM_ROUTER_GEMV          0|1|fp32    glm_small_gemv GateLinear (inside the target graphs)
Install gates are read once per process (sitecustomize, glm_exact_hooks.register, glm_kda_stash.register), so
configure() switches each key ON in os.environ when any variant enables it (the "union"): the code is installed
everywhere and dispatches per call. The pre-union base values travel to child processes in GLM_AB_BASE.

Graphs. The V2 runner captures through CudaGraphManager.capture (warm-up, then capture, per descriptor) and replays
through run_fullgraph -> self.graphs[desc].replay(). For the target manager (ModelCudaGraphManager) the hook runs
the whole capture once per variant with that variant current, into its own dict; everything else the runner owns
(input buffers, attention metadata built per capture, the persistent hidden_states output, the global graph pool)
is shared, exactly as it already is between the ~40 shapes of one set. A switch points every manager's .graphs at
the variant's dict. Eager steps (prefill, mixed batches) dispatch on the runtime variant.

Rank invariance. The switch is a collective_rpc (POST /collective_rpc, method glm_ab_switch; needs
VLLM_SERVER_DEV_MODE=1). The engine core puts it on the same broadcast queue as execute_model, so every TP rank
switches between the same two steps. The engine core refuses it while any request is unfinished (a KDA-stash
record left by a verify step must never be read by the stock kernel of another variant). After a switch every
rank MAX-all-reduces [v, -v, seq, -seq, h, -h] over the TP CPU group and returns the result; the first target
capture all-reduces the config hash the same way, so nodes with different specs or adapter files refuse to boot.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.abc
import importlib.util
import json
import os
import re
import sys
import time

KNOWN = {
    "GLM_TARGET_VOCAB_ARGMAX": "mode",
    "GLM_KDA_NOCOPY": "bool",
    "GLM_KDA_STASH": "bool",
    "GLM_ROUTER_FP32OUT": "bool",
    "GLM_GATE_GEMV": "raw",
    "GLM_ROUTER_GEMV": "raw",
    "GLM_CERT_HEAD": "mode",            # glm_cert_head: GPUModelRunner.sample (eager, outside every graph)
    "GLM_CERT_HEAD_MINTOK": "bool",     # glm_cert_head: certified path on min_tokens steps (eager, per call)
    "GLM_DS_DRAFT_HEAD_FP4": "mode",    # glm_ds_draft_fp4: drafter compute_candidates (needs GLM_AB_DRAFT_SETS=1)
}
HASHED_SOURCES = ("glm_ab", "glm_target_argmax", "glm_kda_nocopy", "glm_kda_stash", "kda_stash",
                  "glm_exact_hooks", "glm_small_gemv", "sitecustomize", "glm_cert_head", "glm_ds_draft_fp4")
CUDAGRAPH = "vllm.v1.worker.gpu.cudagraph_utils"
WORKER = "vllm.v1.worker.gpu_worker"
CORE = "vllm.v1.engine.core"
_OFF = ("", "0", "off", "false", "no")
# Host-side state that follows the runtime variant but lives outside every graph (e.g. glm_roce_proxy_pin's
# thread placement): callables(variant) run on every rank right after a switch; a failing hook is logged, not fatal.
SWITCH_HOOKS: list = []

ACTIVE = False
N = 1
CONFIG_HASH = ""
SOURCES: dict = {}
_specs: list = [{}]
_base: dict = {}
_capture = None
_runtime = 0
_opts = {"allow_aa": False, "draft_sets": False, "capture": "whole"}
_state = {"seq": 0, "token": None, "managers": [], "replays": {}, "profiling": False, "rank_checked": False,
          "installed": set(), "last_switch": None}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-ab: {msg}\n")


def truthy(value) -> bool:
    return value is not None and str(value).strip().lower() not in _OFF


def norm_value(key: str, value):
    """Effective behaviour of one key's value."""
    kind = KNOWN[key]
    if kind == "bool":
        return truthy(value)
    raw = "" if value is None else str(value).strip().lower()
    if raw in _OFF:
        return "0"
    if kind == "raw":  # adapter-defined value set, validated by the adapter
        return raw
    if raw == "check":
        return "check"
    if raw in ("1", "on", "true"):
        return "1"
    raise RuntimeError(f"GLM_AB: {key}={value!r} is not 0|1|check")


def parse_spec(text: str) -> dict:
    out = {}
    for item in re.split(r"[;+\s]+", (text or "").strip()):
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(f"GLM_AB: '{item}' is not KEY=VALUE")
        key, value = (x.strip() for x in item.split("=", 1))
        if key not in KNOWN:
            raise RuntimeError(f"GLM_AB: {key} is not switchable in-boot; known: {sorted(KNOWN)}")
        norm_value(key, value)  # validates
        out[key] = value
    return out


def effective(spec: dict, base: dict) -> dict:
    return {k: norm_value(k, spec.get(k, base.get(k))) for k in sorted(KNOWN)}


def _source_hashes() -> dict:
    out = {}
    for name in HASHED_SOURCES:
        path = __file__ if name == "glm_ab" else None
        if path is None:
            try:
                spec = importlib.util.find_spec(name)
                path = spec.origin if spec is not None else None
            except (ImportError, ValueError):
                path = None
        try:
            with open(path, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()[:16]
        except (OSError, TypeError):
            out[name] = "missing"
    return out


def configure(environ=None) -> bool:
    """Parse GLM_AB_*; True when armed. Must run before any install gate is read (top of sitecustomize)."""
    global ACTIVE, N, CONFIG_HASH, SOURCES, _specs, _base, _capture, _runtime
    environ = os.environ if environ is None else environ
    raw = environ.get("GLM_AB_VARIANTS", "").strip()
    n = int(raw) if raw else 1
    if n < 2:
        ACTIVE, N, _specs, _base = False, 1, [{}], {}
        return False
    if n > 10:  # 2026-09-28 speed screen: 9 sets (5 sets x 15 shapes took 2.61 GiB of graph memory vs 2.57 release)
        raise RuntimeError(f"GLM_AB_VARIANTS={n}: at most 10 graph sets")
    extra = sorted(k for k in environ if re.fullmatch(r"GLM_AB_V\d+", k) and int(k[8:]) >= n)
    if extra:
        raise RuntimeError(f"GLM_AB: {extra} set but GLM_AB_VARIANTS={n}")
    specs = [parse_spec(environ.get(f"GLM_AB_V{i}", "")) for i in range(n)]
    base = (json.loads(environ["GLM_AB_BASE"]) if "GLM_AB_BASE" in environ
            else {k: environ.get(k) for k in sorted(KNOWN)})
    eff = [effective(s, base) for s in specs]
    same = [(i, j) for i in range(n) for j in range(i + 1, n) if eff[i] == eff[j]]
    allow_aa = truthy(environ.get("GLM_AB_ALLOW_AA"))
    if same and not allow_aa:
        raise RuntimeError(f"GLM_AB: variants {same} resolve to the same effective config (an A/A); spell the "
                           "baseline out in GLM_AB_V0, or set GLM_AB_ALLOW_AA=1 for a deliberate A/A")
    _opts["allow_aa"] = allow_aa
    _opts["draft_sets"] = truthy(environ.get("GLM_AB_DRAFT_SETS"))
    _opts["capture"] = environ.get("GLM_AB_CAPTURE", "whole").strip().lower() or "whole"
    if _opts["capture"] not in ("interleave", "whole"):
        raise RuntimeError(f"GLM_AB_CAPTURE={_opts['capture']!r} is not interleave|whole")
    environ.setdefault("GLM_AB_BASE", json.dumps(base, sort_keys=True))
    # union of the install gates over every variant
    for key, kind in KNOWN.items():
        if any((e[key] if kind == "bool" else e[key] != "0") for e in eff):
            environ[key] = "1"
    ACTIVE, N, _specs, _base = True, n, specs, base
    _capture, _runtime = None, 0
    SOURCES = _source_hashes()
    CONFIG_HASH = hashlib.sha256(json.dumps([n, specs, base, _opts, SOURCES], sort_keys=True)
                                 .encode()).hexdigest()[:16]
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    return True


def describe() -> str:
    lines = [f"GLM_AB armed (TEST ONLY): {N} variants, config {CONFIG_HASH}, sources {SOURCES}"]
    for i, s in enumerate(_specs):
        lines.append(f"  v{i}: overrides {s or '{}'} -> effective {effective(s, _base)}")
    return "\n".join(lines)


# -- flag reads ------------------------------------------------------------------------------------

def current() -> int:
    return _capture if _capture is not None else _runtime


def env_for(variant: int, name: str, default=None):
    spec = _specs[variant]
    if name in spec:
        return spec[name]
    value = _base.get(name)
    return default if value is None else value


def env(name: str, default=None):
    """os.environ.get(name, default), except for KNOWN keys while armed: the value of the variant being captured
    (inside a capture) or replayed/run (the runtime variant)."""
    if not ACTIVE or name not in KNOWN:
        return os.environ.get(name, default)
    return env_for(current(), name, default)


def flag(name: str) -> bool:
    """Per-call gate for an adapter that is already installed: True when harness is off (install == on)."""
    if not ACTIVE:
        return True
    return truthy(env(name))


@contextlib.contextmanager
def capturing(variant: int):
    global _capture
    prev, _capture = _capture, variant
    try:
        yield
    finally:
        _capture = prev


# -- rank checks -----------------------------------------------------------------------------------

def _tp():
    try:
        from vllm.distributed.parallel_state import get_tp_group
        g = get_tp_group()
        return g.cpu_group, g.world_size, g.rank_in_group
    except Exception:  # noqa: BLE001
        return None, 1, 0


def _max_reduce(values: list[int]) -> list[int] | None:
    group, world, _ = _tp()
    if group is None or world <= 1:
        return None
    import torch
    import torch.distributed as dist
    t = torch.tensor(values, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return [int(x) for x in t]


def _hash_int() -> int:
    return int(CONFIG_HASH, 16) & ((1 << 62) - 1)


def check_ranks_config() -> None:
    h = _hash_int()
    got = _max_reduce([h, -h])
    if got is not None and (got[0] != h or -got[1] != h):
        raise RuntimeError("GLM_AB: TP ranks disagree on GLM_AB_* / adapter sources (config hash differs); use "
                           "the same env and overlay files on every node")
    _state["rank_checked"] = True


def ranks_agree() -> dict | None:
    v, s, h = _runtime, _state["seq"], _hash_int()
    got = _max_reduce([v, -v, s, -s, h, -h])
    if got is None:
        return None
    return {"variant": [-got[1], got[0]], "seq": [-got[3], got[2]],
            "agree": got[0] == -got[1] == v and got[2] == -got[3] == s and got[4] == -got[5] == h}


# -- graphs ----------------------------------------------------------------------------------------

def _multi(mgr) -> bool:
    return _opts["draft_sets"] or type(mgr).__name__ == "ModelCudaGraphManager"


def _kind(mgr) -> str:
    return "target" if type(mgr).__name__ == "ModelCudaGraphManager" else type(mgr).__name__


def set_runtime(variant: int) -> None:
    global _runtime
    if not 0 <= variant < N:
        raise ValueError(f"variant {variant} not in [0, {N})")
    _runtime = variant
    for mgr in _state["managers"]:
        sets = mgr.__dict__.get("_glm_ab_sets")
        if sets is not None:
            mgr.graphs = sets[variant]


def install_cudagraph(module) -> None:
    cls = module.CudaGraphManager
    if getattr(cls, "_glm_ab", False):
        return
    cls._glm_ab = True
    orig_capture, orig_run, orig_profile = cls.capture, cls.run_fullgraph, cls.profile_memory

    def capture(self, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"):
        if not ACTIVE or _state["profiling"] or not _multi(self):
            out = orig_capture(self, create_forward_fn, progress_bar_desc)
            if ACTIVE and not _state["profiling"] and self not in _state["managers"]:
                _state["managers"].append(self)
            return out
        if not _state["rank_checked"]:
            check_ranks_config()
        sets = [dict() for _ in range(N)]
        t0 = time.perf_counter()
        try:
            if _opts["capture"] == "whole":
                # Default: the whole stock loop once per variant. Fleet-proven: KX-AB (glm-kernels, 4 sets x 46
                # shapes in 36 s on every rank, 2026-09-26 07:47). The INB boot (5 sets, GLM_DIAG_STACKS=2 sampler
                # thread) failed on the three workers at the first CUDA call of the second pass (torch.arange,
                # "invalid argument"); cause not isolated. GLM_AB_CAPTURE=interleave is the untested alternative.
                for v in range(N):
                    self.graphs = sets[v]
                    with capturing(v):
                        orig_capture(self, create_forward_fn, f"{progress_bar_desc} [glm-ab set {v}]")
            else:
                capture_interleaved(self, create_forward_fn, sets, progress_bar_desc)
        finally:
            self.graphs = sets[_runtime]
        self._glm_ab_sets = sets
        if self not in _state["managers"]:
            _state["managers"].append(self)
        _log(f"{_kind(self)}: captured {N} graph sets x {len(sets[0])} shapes in "
             f"{time.perf_counter() - t0:.1f} s (config {CONFIG_HASH})")

    def capture_interleaved(self, create_forward_fn, sets, progress_bar_desc):
        """The stock CudaGraphManager.capture loop (vllm/v1/worker/gpu/cudagraph_utils.py, vLLM 487ecf187,
        Apache-2.0) with one change: every descriptor is warmed up and captured once per variant, back to back,
        inside ONE graph_capture() context. The DS4.1 harness does the same per shape (capture_one per variant)
        and ran on the same RoCE runtime; the whole-loop-per-variant v1 failed on the worker ranks."""
        m = module
        torch = m.torch
        mode_full = m.CUDAGraphMode.FULL
        if m.CUDAGraphMode.PIECEWISE in self._capture_descs:
            raise RuntimeError("GLM_AB: interleaved capture supports FULL graphs only (cudagraph_mode "
                               "FULL_DECODE_ONLY); set GLM_AB_CAPTURE=whole to try the v1 loop")
        with torch.inference_mode(), m.graph_capture(device=self.device):
            descs = self._capture_descs.get(mode_full, [])
            if m.is_global_first_rank():
                descs = m.tqdm(descs, desc=f"{progress_bar_desc} [glm-ab {N} sets interleaved] (FULL)")
            for desc in descs:
                for v in range(N):
                    with capturing(v):
                        forward_fn = create_forward_fn(desc, warmup=True)
                        forward_fn(m.CUDAGraphMode.NONE)
                        forward_fn = create_forward_fn(desc, warmup=False)
                        graph = torch.cuda.CUDAGraph()
                        m.get_offloader().sync_prev_onload()
                        m.set_graph_pool_id(self.pool if self.pool is not None
                                            else m.current_platform.graph_pool_handle())
                        with torch.cuda.graph(graph, self.pool):
                            forward_fn(m.CUDAGraphMode.NONE)
                            m.get_offloader().join_after_forward()
                        sets[v][desc] = graph
                        m.compilation_counter.num_cudagraph_captured += 1
        self._graphs_captured = True

    def run_fullgraph(self, desc):
        if ACTIVE:
            kind = _kind(self)
            counts = _state["replays"].setdefault(kind, [0] * (N + 1))
            sets = self.__dict__.get("_glm_ab_sets")
            if sets is None:
                counts[N] += 1                      # shared (captured once)
            else:
                if self.graphs is not sets[_runtime]:
                    raise RuntimeError(f"GLM_AB: {kind} manager replays a set other than the runtime variant")
                counts[_runtime] += 1
        return orig_run(self, desc)

    def profile_memory(self, *args, **kwargs):
        prev, _state["profiling"] = _state["profiling"], True
        try:
            return orig_profile(self, *args, **kwargs)
        finally:
            _state["profiling"] = prev

    functools.update_wrapper(capture, orig_capture)
    functools.update_wrapper(run_fullgraph, orig_run)
    functools.update_wrapper(profile_memory, orig_profile)
    cls.capture, cls.run_fullgraph, cls.profile_memory = capture, run_fullgraph, profile_memory
    _log(f"CudaGraphManager: target graphs captured once per variant ({N} variants){' + drafter' if _opts['draft_sets'] else ''}")


# -- worker RPC (runs on every rank at the same position of the broadcast queue) ----------------------

def _argmax_stats() -> dict | None:
    """glm_target_argmax counters (fast / stock steps; with =check, steps compared on the same logits and the
    mismatches): the tensor-level exactness readout. Greedy text is not repeatable on this stack even within one
    boot (diagnostics/glm-verify-cut), so text hashes cannot prove exactness."""
    ta = sys.modules.get("glm_target_argmax")
    if ta is None:
        return None
    st = ta.Stats
    return {"fast": st.fast, "full": st.full, "checked": st.checked, "mismatch": st.mismatch}


def _gumbel_stats() -> dict | None:
    """glm_gumbel_coupled counters (coupled verify steps; with =check, steps compared with the stock Sampler on the
    same logits and the mismatches)."""
    gc = sys.modules.get("glm_gumbel_coupled")
    return None if gc is None else gc.stats()


def status() -> dict:
    _, _, rank = _tp()
    return {"rank": rank, "argmax": _argmax_stats(), "gumbel": _gumbel_stats(), "armed": ACTIVE, "variant": _runtime, "seq": _state["seq"], "token": _state["token"],
            "config": CONFIG_HASH, "variants": N, "replays": {k: list(v) for k, v in _state["replays"].items()},
            "sets": {_kind(m): len(m.__dict__.get("_glm_ab_sets") or [None]) for m in _state["managers"]},
            "effective": effective(_specs[_runtime], _base) if ACTIVE else None}


def worker_switch(variant, token="") -> dict:
    if not ACTIVE:
        return {"error": "GLM_AB not armed (GLM_AB_VARIANTS < 2)"}
    prev = _runtime
    set_runtime(int(variant))
    for fn in SWITCH_HOOKS:
        try:
            fn(_runtime)
        except Exception as exc:  # noqa: BLE001
            _log(f"switch hook {getattr(fn, '__qualname__', fn)} failed: {exc!r}")
    _state["seq"] += 1
    _state["token"] = token
    _state["last_switch"] = time.time()
    out = status()
    out["prev"] = prev
    out["ranks"] = ranks_agree()
    if out["ranks"] is not None and not out["ranks"]["agree"]:
        raise RuntimeError(f"GLM_AB: TP ranks disagree after a switch: {out['ranks']}")
    if out["rank"] == 0:
        _log(f"variant {prev} -> {_runtime} (seq {_state['seq']}, token {token})")
    return out


def install_worker(module) -> None:
    cls = module.Worker
    if getattr(cls, "_glm_ab", False):
        return
    cls._glm_ab = True
    cls.glm_ab_switch = lambda self, variant, token="": worker_switch(variant, token)
    cls.glm_ab_status = lambda self: status()


def install_core(module) -> None:
    """Engine core: refuse a switch while any request is unfinished (checked once, in the one scheduler, so every
    rank gets the same answer: the RPC is either broadcast to all ranks or to none)."""
    cls = module.EngineCore
    if getattr(cls, "_glm_ab", False):
        return
    cls._glm_ab = True
    orig = cls.collective_rpc

    @functools.wraps(orig)
    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if method == "glm_ab_switch":
            sched = self.scheduler
            busy = max(int(sched.get_num_unfinished_requests()), len(getattr(sched, "running", ()) or ()))
            if busy:
                return [{"busy": busy}]
        return orig(self, method, timeout, args, kwargs)

    cls.collective_rpc = collective_rpc


_HOOKS = {CUDAGRAPH: install_cudagraph, WORKER: install_worker, CORE: install_core}


def _install(module) -> None:
    name = module.__name__
    if name in _state["installed"]:
        return
    _state["installed"].add(name)
    _HOOKS[name](module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _HOOKS or name in _state["installed"]:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            _install(module)

        spec.loader.exec_module = exec_module
        return spec


def register() -> None:
    """Called from sitecustomize before any other adapter block. Fatal on a config error: a half-armed harness
    would silently measure the wrong thing."""
    try:
        if configure():
            for name in list(_HOOKS):
                if name in sys.modules:
                    _install(sys.modules[name])
            if os.environ.get("GLM_AB_QUIET") != "1" and "vllm" not in sys.modules:
                _log(describe())
    except BaseException as exc:  # noqa: BLE001
        msg = f"GLM_AB: refusing to start: {exc!r}"
        print(msg, flush=True)
        sys.stderr.write(msg + "\n")
        os._exit(1)
