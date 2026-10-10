# SPDX-License-Identifier: Apache-2.0
"""Decode-friendly prefill scheduling for GLM-5.3-Flash on the tonyd2wild v11 image (vLLM 487ecf187).

Credits
  * Jacopo Nardiello (jnardiello), https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless:
      E27  prefill cadence 8 while others decode            (commit 7e5976a, 2026-09-24)
      E27b short-prefill bypass on deferred steps            (promoted inside E27c, commit 68cb598)
      E27c cadence kept while requests are queued            (commit 68cb598, 2026-09-25)
      E29  end-drain + idle coalescing                       (commit e7ba1e6, 2026-09-25)
    Patches: scripts/node/experiments/e03/{long-prefill-cadence,queued-cadence,end-drain}/*.patch
  * FujitsuPolycom / SparkRing R10 image: the non-DP EngineCore._should_throttle_prefills that makes
    vLLM's --prefill-schedule-interval work without data parallelism (his E27 uses it natively).
  * vLLM contributors: the DP prefill-balancing gates (prefill_schedule_interval, defer_prefills,
    prefill_capacity_bound) that all of the above build on.
This file re-expresses those patches for vLLM 487ecf187's scheduler. The scheduler text edits below
are applied to the image's own Scheduler.schedule at import time, hash-pinned; the helper logic is ours.

Knobs (all default off; unset = stock scheduler, nothing installed)
  GLM_PREFILL_CADENCE=N           E27: while a request is decoding, admit prefill work on one engine
                                  step in N (N=8 in E27..E29). 0/unset: use --prefill-schedule-interval.
  GLM_PREFILL_SHORT_TOKENS=T      E27b: on a deferred step still admit prefills with < T remaining
                                  uncached tokens (counted after the prefix-cache lookup), at most T
                                  per step; long ones keep waiting in order. 2048 in E27b.
  GLM_PREFILL_CADENCE_WHEN_QUEUED=1  E27c: the cadence keeps deferring while requests wait in the
                                  queue (the vendor capacity latch no longer switches it off).
  GLM_END_DRAIN=1                 E29: a running request whose committed outputs plus in-flight
                                  placeholders reach max_tokens gets no further speculative step
                                  until that output returns (async scheduling).
  GLM_IDLE_COALESCE_MS=W          E29: when the engine was idle and requests arrive, keep taking
                                  arrivals for W ms (0 < W <= 5) before the first schedule.

Numerics: none of this changes a model kernel. It changes which rows share a forward (batch shape),
which in this engine is already not bit-invariant (see RESULTS.md, exactness section).
"""
from __future__ import annotations

import __future__ as _future
import logging
import os
import queue
import sys
import time

try:
    import glm_prefill_hooks as _hooks
except ImportError:  # pragma: no cover
    from . import glm_prefill_hooks as _hooks  # type: ignore

logger = logging.getLogger("vllm.glm_prefill_sched")


def _int(name: str, default: int = 0) -> int:
    v = int(os.environ.get(name, str(default)).strip() or default)
    if v < 0:
        raise ValueError(f"{name} must not be negative")
    return v


def _flag(name: str) -> bool:
    v = os.environ.get(name, "0").strip()
    if v not in ("0", "1", ""):
        raise ValueError(f"{name} must be 0 or 1")
    return v == "1"


CADENCE = _int("GLM_PREFILL_CADENCE")
SHORT_TOKENS = _int("GLM_PREFILL_SHORT_TOKENS")
WHEN_QUEUED = _flag("GLM_PREFILL_CADENCE_WHEN_QUEUED")
END_DRAIN = _flag("GLM_END_DRAIN")
COALESCE_S = float(os.environ.get("GLM_IDLE_COALESCE_MS", "0") or 0) / 1000.0
if not 0.0 <= COALESCE_S <= 0.005:
    raise ValueError("GLM_IDLE_COALESCE_MS must be within 0-5")

# [git] vLLM 487ecf187 sources (the pinned image did not have these files extracted; the image
# reports 0.1.dev20051+g487ecf187 and tonyd2wild's patch scripts do not touch them). Fleet step 0
# prints the image's hashes; any difference refuses the install.
EXPECTED = {
    "vllm.v1.core.sched.scheduler:Scheduler.schedule": "6a6bcd100d7e4715",
    "vllm.v1.core.sched.async_scheduler:AsyncScheduler._update_after_schedule": "0da19ddc560cc7a1",
    "vllm.v1.engine.core:EngineCoreProc._process_input_queue": "2c3091b60af8807c",
    "vllm.v1.engine.core:EngineCore._should_throttle_prefills": "039ae6f2f51ac992",
}

# ------------------------------------------------------------------------------------------
# text edits of Scheduler.schedule (dedented source); each old string must occur exactly once
# ------------------------------------------------------------------------------------------
TAG = "# GLM_PREFILL_SCHED"
EDITS = [
    # E27: the non-DP engine core never throttles; the cadence is computed here, from the step
    # counter before this call's increment (as SparkRing's EngineCore._should_throttle_prefills).
    ("    self.current_step += 1\n",
     f"    throttle_prefills = throttle_prefills or _glm_ps.cadence_throttles(self)  {TAG} E27\n"
     f"    _glm_ps.begin_step(self)  {TAG}\n"
     "    self.current_step += 1\n"),
    # E27c + E29: the latch override and the decode-eligibility check without held requests.
    ("    defer_prefills = (\n"
     "        throttle_prefills and not self.prefill_capacity_bound\n"
     "    ) and any(not r.is_prefill_chunk for r in self.running)\n",
     f"    defer_prefills = _glm_ps.defer_prefills(self, throttle_prefills)  {TAG} E27c/E29\n"),
    # E29: hold a request whose step in flight may already finish it by length.
    ("        if self.current_step < request.next_decode_eligible_step:\n",
     f"        if _glm_ps.holds(self, request, note=True):  {TAG} E29\n"
     "            req_index += 1\n"
     "            continue\n"
     "\n"
     "        if self.current_step < request.next_decode_eligible_step:\n"),
    # E27b, running prefill chunks.
    ("        if defer_prefills and request.is_prefill_chunk:\n",
     "        if (\n"
     "            defer_prefills\n"
     "            and request.is_prefill_chunk\n"
     "            and not _glm_ps.admit_short(\n"
     "                self, request.num_tokens - request.num_computed_tokens\n"
     "            )\n"
     f"        ):  {TAG} E27b\n"),
    # E27b, waiting requests: a queue for long prefills set aside on this step.
    ("        step_skipped_waiting = create_request_queue(self.policy)\n",
     "        step_skipped_waiting = create_request_queue(self.policy)\n"
     f"        glm_deferred = create_request_queue(self.policy)  {TAG} E27b\n"),
    # E27b, waiting requests: remaining uncached work is known here (after the prefix lookup).
    ("            elif defer_prefills and num_computed_tokens < request.num_tokens - 1:\n",
     "            elif (\n"
     "                defer_prefills\n"
     "                and num_computed_tokens < request.num_tokens - 1\n"
     "                and not _glm_ps.admit_short(\n"
     "                    self, request.num_tokens - num_computed_tokens\n"
     "                )\n"
     f"            ):  {TAG} E27b\n"
     "                if _glm_ps.SHORT_TOKENS:\n"
     "                    request_queue.pop_request()\n"
     "                    glm_deferred.prepend_request(request)\n"
     "                    _glm_ps.note_deferred(self, request)\n"
     "                    continue\n"),
    # E27b: long prefills set aside go back to the front of the waiting queue, in order.
    ("        # DP prefill balancing: on a step that admitted prefills (release),\n",
     f"        if glm_deferred:  {TAG} E27b\n"
     "            self.waiting.prepend_requests(glm_deferred)\n"
     "\n"
     "        # DP prefill balancing: on a step that admitted prefills (release),\n"),
]


def patch_schedule_source(src: str) -> str:
    for old, new in EDITS:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"glm-prefill-sched: edit anchor found {n} times: {old.strip()[:80]!r}")
        src = src.replace(old, new)
    return src


# ------------------------------------------------------------------------------------------
# helpers called from the patched Scheduler.schedule (module global `_glm_ps` there)
# ------------------------------------------------------------------------------------------
_stats = {"deferred_steps": 0, "short_admitted": 0, "long_deferred": 0, "latch_overrides": 0, "holds": 0}


def _log(key: str, **kw) -> None:
    _stats[key] += 1
    n = _stats[key]
    if n == 1 or n % 1000 == 0:
        logger.info("GLM_PREFILL_SCHED %s count=%d %s", key, n,
                    " ".join(f"{k}={v}" for k, v in kw.items()))


def interval(sched) -> int:
    iv = getattr(sched, "_glm_interval", None)
    if iv is None:
        iv = CADENCE or int(getattr(sched.scheduler_config, "prefill_schedule_interval", 1) or 1)
        dp = sched.vllm_config.parallel_config.data_parallel_size
        if iv > 1 and dp > 1:
            raise RuntimeError("glm-prefill-sched: the cadence is for DP1 (the DP core has its own)")
        sched._glm_interval = iv
    return iv


def cadence_throttles(sched) -> bool:
    iv = interval(sched)
    return iv > 1 and sched.current_step % iv != 0


def begin_step(sched) -> None:
    sched._glm_short_used = 0


def holds(sched, request, note: bool = False) -> bool:
    """E29: the step in flight may already finish this request by length. Structured output keeps
    the vendor pipeline (its bitmask for retained drafts is validated only while a step is in
    flight), and EOS / stop strings are unchanged."""
    if not END_DRAIN:
        return False
    max_tokens = getattr(request, "max_tokens", None)
    held = (not request.is_prefill_chunk
            and not request.use_structured_output
            and request.num_output_placeholders > 0
            and max_tokens is not None
            and request.num_output_tokens + request.num_output_placeholders >= max_tokens)
    if held and note:
        _log("holds", req=request.request_id)
    return held


def defer_prefills(sched, throttle: bool) -> bool:
    if not throttle:
        return False
    latched = sched.prefill_capacity_bound
    if latched and not WHEN_QUEUED:
        return False
    eligible = any(
        not r.is_prefill_chunk
        and sched.current_step >= r.next_decode_eligible_step
        and not holds(sched, r)
        for r in sched.running
    )
    if eligible:
        _log("deferred_steps", interval=interval(sched))
        if latched:
            _log("latch_overrides", waiting=len(sched.waiting))
    return eligible


def admits(remaining: int, used: int, limit: int) -> bool:
    """E27b predicate: a short prefill fits the per-step short budget."""
    return 0 < remaining < limit and used + remaining <= limit


def admit_short(sched, remaining: int) -> bool:
    used = getattr(sched, "_glm_short_used", 0)
    if not admits(remaining, used, SHORT_TOKENS):
        return False
    sched._glm_short_used = used + remaining
    _log("short_admitted", remaining=remaining)
    return True


def note_deferred(sched, request) -> None:
    _log("long_deferred", remaining=request.num_tokens - request.num_computed_tokens)


# ------------------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------------------
def build_schedule(mod, path: str | None = None):
    """Compile the edited Scheduler.schedule against the scheduler module's globals."""
    src = _hooks.func_source(path or mod.__file__, "Scheduler.schedule")
    new = patch_schedule_source(src)
    code = compile(new, "<glm_prefill_sched: Scheduler.schedule>", "exec",
                   flags=_future.annotations.compiler_flag, dont_inherit=True)
    ns: dict = {}
    exec(code, mod.__dict__, ns)  # noqa: S102 - hash-pinned engine source + the edits above
    fn = ns["schedule"]
    fn.__qualname__ = "Scheduler.schedule"
    fn.__module__ = mod.__name__
    fn.__glm_prefill_sched__ = True
    return fn


def install_scheduler(mod) -> None:
    scheduling = bool(CADENCE or SHORT_TOKENS or WHEN_QUEUED or END_DRAIN)
    if not scheduling:
        return
    if (SHORT_TOKENS or WHEN_QUEUED) and not CADENCE:
        logger.warning("glm-prefill-sched: GLM_PREFILL_SHORT_TOKENS / _CADENCE_WHEN_QUEUED act only on "
                       "cadence steps; GLM_PREFILL_CADENCE is unset (falls back to --prefill-schedule-interval)")
    _hooks.check_sources({k: v for k, v in EXPECTED.items() if k.startswith("vllm.v1.core.sched")},
                         "glm-prefill-sched")
    if getattr(mod.Scheduler.schedule, "__glm_prefill_sched__", False):
        return
    mod._glm_ps = sys.modules[__name__]
    mod.Scheduler.schedule = build_schedule(mod)
    logger.warning("GLM_PREFILL_SCHED_READY cadence=%d short_tokens=%d when_queued=%d end_drain=%d",
                   CADENCE, SHORT_TOKENS, int(WHEN_QUEUED), int(END_DRAIN))


def make_process_input_queue(orig, dp_cls):
    """E29 idle coalescing around EngineCoreProc._process_input_queue.

    Stock returns as soon as the first arrival (plus whatever is already queued) is handled. If the
    engine was idle before the call and now has work, keep taking arrivals until one fixed deadline
    W after the stock drain, so requests released together are prefilled in one step. The window
    is measured from the end of the stock drain (jnardiello's from the first ADD); the difference
    is the drain time, microseconds.
    """

    def _process_input_queue(self):
        idle = (COALESCE_S > 0 and not isinstance(self, dp_cls)
                and self.process_input_queue_block and not self.has_work())
        orig(self)
        if not idle or not self.scheduler.has_unfinished_requests():
            return
        deadline = time.monotonic() + COALESCE_S
        taken = 0
        while self.is_running() and self.scheduler.has_unfinished_requests():
            ps = getattr(self.scheduler, "pause_state", None)
            if ps is not None and getattr(ps, "name", "UNPAUSED") != "UNPAUSED":
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = self.input_queue.get(timeout=remaining)
            except queue.Empty:
                break
            self._handle_client_request(*req)
            taken += 1
        if taken:
            _stats.setdefault("coalesced", 0)
            _stats["coalesced"] += taken

    _process_input_queue.__glm_prefill_sched__ = True
    return _process_input_queue


def install_core(mod) -> None:
    if COALESCE_S <= 0:
        return
    _hooks.check_sources({k: v for k, v in EXPECTED.items() if k.startswith("vllm.v1.engine.core")},
                         "glm-prefill-sched")
    cls = mod.EngineCoreProc
    if getattr(cls._process_input_queue, "__glm_prefill_sched__", False):
        return
    cls._process_input_queue = make_process_input_queue(cls._process_input_queue, mod.DPEngineCoreProc)
    logger.warning("GLM_IDLE_COALESCE_READY ms=%g", COALESCE_S * 1000)
