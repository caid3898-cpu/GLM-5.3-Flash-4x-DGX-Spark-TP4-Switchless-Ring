# SPDX-License-Identifier: Apache-2.0
"""GLM_EARLY_PLAN=1: move the per-step host sync of the SM90 sparse-MLA planner from "after the draft" to
"after the verify". Exact (the planned lengths are the same integers; nothing numerical changes).

Target: Tony v11 image (vLLM 0.1.dev20051+g487ecf187), V2 model runner, async scheduling, DFlash2 drafter.

Why. Under async scheduling `FlashInferMLASparseSM90Builder._kv_lens_host` reads `positions[:n].cpu()`
(flashinfer_mla_sparse_sm90.py:348). `positions` for step N+1 are written by `_prepare_pos_seq_lens_kernel`, which
the stream runs only after drafter N, so the host blocks until drafter N ends and then does ~0.7 ms of preparation
plus a ~0.7 ms `cudaGraphLaunch` while the GPU idles (glm-steplevel-20260928: 1.57 ms draft-end -> target-start at
c1, 1.69 ms at c4, plus ~0.25 ms of start skew absorbed by the first in-graph all-reduce). The lengths only depend
on the device `num_computed_tokens` after step N's `post_update`, which is final ~3.6 ms before drafter N ends
(post_update runs before the drafter FC, the context-KV precompute and the draft graph).

How.
  1. Right after `postprocess_sampled` (post_update + postprocess_state) of every real step, a <=1 KiB D2H of
     `req_states.num_computed_tokens.gpu` goes into a ring of 3 pinned slots and a CUDA event is recorded
     ("snapshot"). Stream-ordered before the drafter, so it fires at verify time.
  2. In the next step's `_kv_lens_host` the host waits on that event (not on the whole stream) and builds the exact
     per-row context: ctx = base(req) + offset_in_request + 1, with base = snapshot[state index] for continuing
     requests and the host mirror for requests (re)added since the snapshot (add_request stages exactly that value
     to the device). Same lens formula as the deployed body. The host then plans, prepares and launches the target
     graph while drafter N still runs.
  3. Plan fence (the 2026-09-28 pinned-buffer audit, glm-audit-pinned-20260928): FlashInfer's single pinned int
     workspace is rewritten by every plan() and uploaded with cudaMemcpyAsync. An event is recorded after every
     plan() and waited (query first) before the next one, so no upload can be overwritten while pending. The
     three plan inputs (_qo_cpu/_kv_cpu/_lens_cpu) are staged through two pinned slots (GLM_EARLY_PLAN_PINNED,
     default 1) so their H2D copies are truly asynchronous (a pageable cudaMemcpyAsync "may synchronize with the
     stream", which would re-serialise the host behind the drafter); each slot has its own reuse event.

Fallback. Any step the mirror cannot vouch for takes the deployed body verbatim (one sync, exactly today's
behaviour): no batch noted for this step, no snapshot from the immediately preceding real step, row/qsl mismatch
between the InputBatch and the metadata, filtered rows (idx < 0), CUDA-graph capture, PCP, adaptive verification,
PP > 1, or async scheduling off (then the deployed body is already sync-free). Rank invariance: every rank
computes the same integers from the same device state; a rank that falls back computes them with a sync.

Knobs (read per call; switchable in-boot through overlay/glm_ab.py when it is armed):
  GLM_EARLY_PLAN=0|1               install + activate (default 0: register() does nothing, torch is not imported)
  GLM_EARLY_PLAN_DEBUG=0|1         every early build also runs the deployed device path, compares, keeps the device
                                   result (reintroduces the sync; the exactness gate)
  GLM_EARLY_PLAN_CHECK_EVERY=N     spot-check every N-th early build the same way (0 = never; canary: 1024)
  GLM_EARLY_PLAN_PINNED=0|1        pinned double-buffered plan inputs (default 1)
  GLM_EARLY_PLAN_SPIN=0|1          spin instead of sleep while waiting on the snapshot event (default 0)
  GLM_EARLY_PLAN_GAPLOG=N          every N steps log the GPU gap drafter-end -> next forward start and the step
                                   period, from CUDA events (0 = off)
  GLM_EARLY_PLAN_LOG_EVERY=N       status line every N builds (default 20000)
Mutually exclusive with GLM_KVLENS_EXACT (glm_kv_lens_exact.py: same purpose, but it polls the event with query(),
which never fires in time in the pipelined steady state, so it always fell back; see REPORT.md).

Credits: vLLM V2 model runner (post_update / prepare_pos_seq_lens semantics, AsyncOutput copy-then-propose
ordering); FlashInfer MLA plan; FujitsuPolycom (local-inference-lab/vllm#923) for the pinned-staging race class
that the plan fence closes; our glm_kv_lens_exact.py for the fallback discipline this module keeps.
"""
from __future__ import annotations

import collections
import importlib.abc
import importlib.util
import os
import statistics
import sys
import time

_OFF = ("", "0", "off", "false", "no")

TARGET_MLA = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90"
TARGET_RUNNER = "vllm.v1.worker.gpu.model_runner"
TARGET_STATES = "vllm.v1.worker.gpu.states"
TARGET_ASYNC = "vllm.v1.worker.gpu.async_utils"


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-early-plan: {msg}\n")


def env(name: str, default=None):
    """os.environ, or the in-boot A/B variant value while overlay/glm_ab.py is armed."""
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def _truthy(v) -> bool:
    return v is not None and str(v).strip().lower() not in _OFF


def _int(name: str, default: int) -> int:
    try:
        return int(float(env(name, str(default)) or default))
    except (TypeError, ValueError):
        return default


def installed() -> bool:
    return _truthy(os.environ.get("GLM_EARLY_PLAN"))


def active() -> bool:
    return _truthy(env("GLM_EARLY_PLAN", "0"))


# =====================================================================================================
# Pure core (no torch, no numpy): unit-tested on a CPU-only machine by tests/test_glm_early_plan.py
# =====================================================================================================

def lens_from_bases(bases, qsl, index_topk: int, index_kpool: int) -> list:
    """Exact per-row sparse-MLA kv lens from per-request device num_computed_tokens.

    Row j of request r attends ctx = bases[r] + (j - qsl[r]) + 1 tokens (positions are contiguous per request and
    start at num_computed_tokens, input_batch.py:_prepare_pos_seq_lens_kernel). The indexer bound is the deployed
    one: ctx if ctx <= index_topk else index_topk + ctx % index_kpool.
    """
    topk = int(index_topk)
    kpool = max(int(index_kpool), 1)
    out = []
    for r in range(len(bases)):
        b = int(bases[r])
        for j in range(int(qsl[r]), int(qsl[r + 1])):
            ctx = b + (j - int(qsl[r])) + 1
            out.append(ctx if ctx <= topk else topk + ctx % kpool)
    return out


class EarlyPlanCore:
    """Host-side bookkeeping for one TP worker. Single-threaded by construction (the worker's main thread runs
    execute_model, sample_tokens and every metadata build), so there is no lock."""

    def __init__(self, waiter=None) -> None:
        self.serial = 0                 # real (token-carrying, non-dummy) execute_model calls so far
        self.batch = None               # (serial, input_batch) noted by prepare_inputs of the current real step
        self.snap = None                # (serial, event, reader) taken after postprocess_sampled of a real step
        self.added: set = set()         # req_ids whose device value was (re)set by add_request after self.snap
        self.desync = False             # runner configuration the mirror cannot model
        self.in_real_step = False
        self.stats = collections.Counter()
        self.wait_ms: list = []
        self._waiter = waiter or (lambda ev: ev.synchronize())

    # -- hooks ----------------------------------------------------------------------------------------
    def begin_step(self, real: bool) -> None:
        self.batch = None
        self.in_real_step = bool(real)
        if real:
            self.serial += 1

    def note_added(self, req_id) -> None:
        self.added.add(req_id)

    def note_batch(self, input_batch) -> None:
        if self.in_real_step:
            self.batch = (self.serial, input_batch)

    def note_snapshot(self, event, reader) -> None:
        """Device num_computed_tokens after this real step's post_update (valid once `event` fired)."""
        if not self.in_real_step:
            return
        self.snap = (self.serial, event, reader)
        self.added = set()  # the snapshot already contains every value add_request staged before it

    def drop_snapshot(self) -> None:
        self.snap = None

    # -- the decision -------------------------------------------------------------------------------------
    def plan_bases(self, cam_num_reqs: int, cam_qsl: list):
        """(bases, None) with the exact per-request device num_computed_tokens of the batch being built, or
        (None, reason) when the caller must run the deployed device path. Blocks on the snapshot event."""
        if self.desync:
            return None, "desync"
        b = self.batch
        if b is None or b[0] != self.serial:
            return None, "no_batch"
        s = self.snap
        if s is None or s[0] != self.serial - 1:
            return None, "no_snapshot"
        ib = b[1]
        try:
            num_reqs = int(ib.num_reqs)
            if num_reqs != int(cam_num_reqs):
                return None, "row_mismatch"
            qsl_ib = [int(x) for x in _seq(ib.query_start_loc_np)[: num_reqs + 1]]
            if qsl_ib != [int(x) for x in cam_qsl[: num_reqs + 1]]:
                return None, "qsl_mismatch"
            req_ids = list(ib.req_ids)[:num_reqs]
            idx = [int(x) for x in _seq(ib.idx_mapping_np)[:num_reqs]]
            host = [int(x) for x in _seq(ib.num_computed_tokens_np)[:num_reqs]]
        except Exception:  # noqa: BLE001  duck-typing failure: never guess
            return None, "batch_shape"
        if len(req_ids) != num_reqs or len(idx) != num_reqs or len(host) != num_reqs:
            return None, "batch_shape"
        if any(i < 0 for i in idx):
            return None, "filtered_row"
        # Always wait: the snapshot event is also the proof that step N's forward (the reader of step N's plan
        # upload) and post_update are done. Waiting is cheap when it already fired.
        t0 = time.perf_counter()
        self._waiter(s[1])
        self.wait_ms.append((time.perf_counter() - t0) * 1e3)
        dev = None
        bases = []
        for rid, i, h in zip(req_ids, idx, host):
            if rid in self.added:
                bases.append(h)
                continue
            if dev is None:
                dev = [int(x) for x in s[2]()]
            if i >= len(dev):
                return None, "idx_range"
            bases.append(dev[i])
        return bases, None


def _seq(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


class PlanFence:
    """Orders host rewrites of plan staging memory after the GPU consumed the previous contents.

    `last` is the event recorded after the previous plan() (covers FlashInfer's single pinned int workspace);
    `slot_events[i]` is the event of the last plan that used pinned input slot i."""

    def __init__(self, nslots: int = 2, waiter=None) -> None:
        self.nslots = nslots
        self.i = 0
        self.last = None
        self.slot_events = [None] * nslots
        self.stats = collections.Counter()
        self._waiter = waiter or (lambda ev: ev.synchronize())

    def _drain(self, ev, key: str) -> None:
        if ev is None:
            return
        try:
            done = bool(ev.query())
        except Exception:  # noqa: BLE001
            done = False
        if not done:
            self.stats[key] += 1
            self._waiter(ev)

    def before(self) -> int:
        """Wait until the previous upload and the chosen input slot are free; return the slot index."""
        slot = self.i % self.nslots
        self._drain(self.last, "fence_waits")
        self._drain(self.slot_events[slot], "slot_waits")
        return slot

    def after(self, slot: int, event) -> None:
        self.last = event
        self.slot_events[slot] = event
        self.i += 1
        self.stats["plans"] += 1


class GapLog:
    """GPU gap drafter-end -> next forward-start and step period from CUDA timing events, read lagged."""

    def __init__(self) -> None:
        self.pending_end = None
        self.prev_start = None
        self.pairs: collections.deque = collections.deque()
        self.gaps: list = []
        self.periods: list = []
        self.steps = 0

    def end(self, ev) -> None:
        self.pending_end = ev

    def start(self, ev) -> None:
        self.steps += 1
        self.pairs.append((self.pending_end, self.prev_start, ev))
        self.pending_end = None
        self.prev_start = ev
        while self.pairs:
            e, p, s = self.pairs[0]
            try:
                if not s.query():
                    break
            except Exception:  # noqa: BLE001
                break
            self.pairs.popleft()
            if e is not None:
                self.gaps.append(e.elapsed_time(s))
            if p is not None:
                self.periods.append(p.elapsed_time(s))

    def report(self) -> str:
        def q(v):
            if not v:
                return "n/a"
            v = sorted(v)
            return (f"median {statistics.median(v):.3f} p10 {v[len(v) // 10]:.3f} p90 {v[(9 * len(v)) // 10]:.3f}"
                    f" n {len(v)}")
        s = f"gap draft-end->forward-start ms: {q(self.gaps)} | step period ms: {q(self.periods)}"
        self.gaps, self.periods = [], []
        return s


CORE = EarlyPlanCore()
FENCE = PlanFence()
GAP = GapLog()
_STATE = {"orig_kv_lens_host": None, "orig_plan": None, "ring": None, "ring_i": 0, "builds": 0,
          "mismatches": 0, "cache": None, "installed": set(), "warned": set()}


def status() -> dict:
    w = CORE.wait_ms[-2000:]
    return {
        "installed": sorted(_STATE["installed"]), "active": active(), "builds": _STATE["builds"],
        "stats": dict(CORE.stats), "fence": dict(FENCE.stats), "mismatches": _STATE["mismatches"],
        "wait_ms_median": statistics.median(w) if w else None,
    }


def _maybe_status() -> None:
    every = _int("GLM_EARLY_PLAN_LOG_EVERY", 20000)
    if every and _STATE["builds"] % every == 0:
        _log(f"status {status()}")


# =====================================================================================================
# Torch adapters (the only torch-touching code; replaced by fakes in the CPU tests)
# =====================================================================================================

def _capturing() -> bool:
    import torch
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _record_event(timing: bool = False):
    """A CUDA event recorded on the current stream."""
    import torch
    ev = torch.cuda.Event(enable_timing=timing, blocking=not _truthy(env("GLM_EARLY_PLAN_SPIN", "0")))
    ev.record()
    return ev


def _lens_tensor(lens: list):
    import torch
    return torch.tensor(lens, dtype=torch.int32)


def _snapshot(runner):
    """Enqueue the D2H of the device num_computed_tokens into the next pinned ring slot; return (event, reader)."""
    import torch
    nct = runner.req_states.num_computed_tokens.gpu
    ring = _STATE["ring"]
    if ring is None or ring[0].shape != nct.shape or ring[0].dtype != nct.dtype:
        ring = [torch.empty(nct.shape, dtype=nct.dtype, device="cpu", pin_memory=True) for _ in range(3)]
        _STATE["ring"] = ring
    slot = ring[_STATE["ring_i"] % 3]
    _STATE["ring_i"] += 1
    slot.copy_(nct, non_blocking=True)
    return _record_event(), slot.tolist


def _pinned_slots(state):
    """Two pinned copies of the three SM90 plan input buffers (same shapes/dtypes as the deployed lazy init)."""
    import torch
    slots = state.__dict__.get("_glm_ep_slots")
    if slots is None:
        if getattr(state, "_arange_cpu", None) is None:
            state._arange_cpu = torch.arange(state.max_tokens + 1, dtype=torch.int32)  # host-only, pageable is fine
        slots = []
        for _ in range(FENCE.nslots):
            slots.append((
                torch.empty(state.max_tokens + 1, dtype=torch.int32, pin_memory=True),
                torch.empty(state.max_tokens + 1, dtype=torch.int32, pin_memory=True),
                torch.full((state.max_tokens,), state.topk_width, dtype=torch.int32, pin_memory=True),
            ))
        state.__dict__["_glm_ep_slots"] = slots
    return slots


# =====================================================================================================
# Patched entry points
# =====================================================================================================

def patched_kv_lens_host(self, cam):
    orig = _STATE["orig_kv_lens_host"]
    if not active() or not getattr(self, "_async_scheduling", False) or _capturing():
        return orig(self, cam)
    cache = _STATE["cache"]
    if cache is not None and cache[0] is cam and cache[1] == CORE.serial:
        return cache[2]  # the same metadata object built twice in one step
    _STATE["builds"] += 1
    num_reqs = int(cam.num_reqs)
    qsl = [int(x) for x in _seq(cam.query_start_loc_cpu[: num_reqs + 1])]
    bases, reason = CORE.plan_bases(num_reqs, qsl)
    if bases is None:
        CORE.stats["fallback_" + reason] += 1
        result = orig(self, cam)
    else:
        lens = lens_from_bases(bases, qsl, self._index_topk, self._index_kpool)
        num_rows = qsl[num_reqs] if qsl else 0
        every = _int("GLM_EARLY_PLAN_CHECK_EVERY", 0)
        if _truthy(env("GLM_EARLY_PLAN_DEBUG", "0")) or (every and _STATE["builds"] % every == 0):
            d_rows, d_lens = orig(self, cam)
            d = [int(x) for x in _seq(d_lens)]
            CORE.stats["checked"] += 1
            if int(d_rows) != num_rows or d != lens:
                _STATE["mismatches"] += 1
                bad = [k for k in range(min(len(d), len(lens))) if d[k] != lens[k]][:8]
                _log(f"GLM_EARLY_PLAN_MISMATCH #{_STATE['mismatches']}: rows {d_rows} vs {num_rows}, first diffs "
                     f"{bad}, bases {bases[:8]}, qsl {qsl[:9]}")
            result = (d_rows, d_lens)  # keep the device result
        else:
            result = (num_rows, _lens_tensor(lens))
        CORE.stats["early"] += 1
    _STATE["cache"] = (cam, CORE.serial, result)
    _maybe_status()
    return result


def patched_plan(self, num_tokens, kv_lens):
    orig = _STATE["orig_plan"]
    if not active():
        return orig(self, num_tokens, kv_lens)
    slot = FENCE.before()
    if _truthy(env("GLM_EARLY_PLAN_PINNED", "1")):
        self._qo_cpu, self._kv_cpu, self._lens_cpu = _pinned_slots(self)[slot]
    out = orig(self, num_tokens, kv_lens)
    FENCE.after(slot, _record_event())
    return out


def _install_mla(mod) -> None:
    cls = mod.FlashInferMLASparseSM90Builder
    if getattr(cls, "_glm_early_plan", False):
        return
    _STATE["orig_kv_lens_host"] = cls._kv_lens_host
    cls._kv_lens_host = patched_kv_lens_host
    cls._glm_early_plan = True
    st = mod._SM90State
    _STATE["orig_plan"] = st.plan
    st.plan = patched_plan
    _STATE["installed"].add("mla")
    _log("SM90 _kv_lens_host and _SM90State.plan wrapped")


def _real_step(so, args, kwargs) -> bool:
    dummy = kwargs.get("dummy_run", False)
    if not dummy and len(args) >= 2 and isinstance(args[1], bool):  # (intermediate_tensors, dummy_run, ...)
        dummy = args[1]
    return (not dummy) and int(getattr(so, "total_num_scheduled_tokens", 0) or 0) > 0


def _install_runner(mod) -> None:
    cls = mod.GPUModelRunner
    if getattr(cls, "_glm_early_plan", False):
        return
    cls._glm_early_plan = True
    o_exec, o_prep, o_post, o_sample = (cls.execute_model, cls.prepare_inputs, cls.postprocess_sampled,
                                       cls.sample_tokens)

    def execute_model(self, scheduler_output, *args, **kwargs):
        try:
            CORE.begin_step(_real_step(scheduler_output, args, kwargs))
            CORE.desync = (getattr(self, "pcp_manager", None) is not None
                           or getattr(self, "adaptive_verification", None) is not None
                           or not getattr(self, "is_last_pp_rank", True)
                           or getattr(self, "pp_handler", None) is not None)
        except Exception:  # noqa: BLE001
            CORE.batch = None
        return o_exec(self, scheduler_output, *args, **kwargs)

    def prepare_inputs(self, *args, **kwargs):
        ib = o_prep(self, *args, **kwargs)
        CORE.note_batch(ib)
        return ib

    def postprocess_sampled(self, *args, **kwargs):
        out = o_post(self, *args, **kwargs)
        if active() and CORE.in_real_step and not _capturing():
            try:
                CORE.note_snapshot(*_snapshot(self))
            except Exception as exc:  # noqa: BLE001
                CORE.drop_snapshot()
                if "snap" not in _STATE["warned"]:
                    _STATE["warned"].add("snap")
                    _log(f"snapshot failed, device path until the next good snapshot: {exc!r}")
        return out

    def sample_tokens(self, *args, **kwargs):
        out = o_sample(self, *args, **kwargs)
        if _int("GLM_EARLY_PLAN_GAPLOG", 0) and CORE.in_real_step and not _capturing():
            GAP.end(_record_event(timing=True))
        return out

    cls.execute_model, cls.prepare_inputs = execute_model, prepare_inputs
    cls.postprocess_sampled, cls.sample_tokens = postprocess_sampled, sample_tokens
    _STATE["installed"].add("runner")


def _install_states(mod) -> None:
    cls = mod.RequestState
    if getattr(cls, "_glm_early_plan", False):
        return
    cls._glm_early_plan = True
    o_add = cls.add_request

    def add_request(self, req_id, *args, **kwargs):
        CORE.note_added(req_id)
        return o_add(self, req_id, *args, **kwargs)

    cls.add_request = add_request
    _STATE["installed"].add("states")


def _install_async(mod) -> None:
    cls = mod.StepTimingCollector
    if getattr(cls, "_glm_early_plan", False):
        return
    cls._glm_early_plan = True
    o_fs = cls.forward_start

    def forward_start(self):
        o_fs(self)
        every = _int("GLM_EARLY_PLAN_GAPLOG", 0)
        if every and CORE.in_real_step and CORE.batch is not None and not _capturing():
            GAP.start(_record_event(timing=True))
            if GAP.steps % every == 0:
                _log(GAP.report() + f" | {status()}")

    cls.forward_start = forward_start
    _STATE["installed"].add("async")


HOOKS = {TARGET_MLA: _install_mla, TARGET_RUNNER: _install_runner, TARGET_STATES: _install_states,
         TARGET_ASYNC: _install_async}


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
    """Idempotent. No-op unless GLM_EARLY_PLAN is set (default off)."""
    if not installed():
        return
    if _truthy(os.environ.get("GLM_KVLENS_EXACT")):
        _log("GLM_KVLENS_EXACT is set: both wrap _kv_lens_host; refusing to install (unset one of them)")
        return
    for name, fn in HOOKS.items():  # a module imported before us is patched in place
        if name in sys.modules:
            fn(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    _log("registered (default-off switch GLM_EARLY_PLAN is on)")
