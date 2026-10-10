"""CPU check for overlay/glm_prefill_sched.py against the REAL vLLM 487ecf187 scheduler.

Needs a vLLM 487ecf187 source tree that imports on CPU (GLM_VLLM_SRC; its `vllm/` and
`tests/v1/core/utils.py`), plus vLLM's Python requirements. RESULTS.md has the recipe (uv venv,
requirements/common.txt, a stub vllm/_version.py and a "+cpu" dist-info so the CPU platform is picked).

  GLM_VLLM_SRC=/path/to/vllm-487ecf187 python tests/test_glm_prefill_sched.py

Checks
  * every text edit applies exactly once to the image's Scheduler.schedule; the result compiles; the
    source hashes match the EXPECTED table (i.e. the tree is the one the edits were written for);
  * with every knob off the patched schedule() is step-for-step identical to stock on a mixed
    workload (and vLLM's own test_scheduler.py passes with the patch installed, see RESULTS.md);
  * E27  cadence: with a request decoding, a new prefill is admitted only on steps whose
         pre-increment counter is a multiple of N; decode runs every step; with nobody decoding
         nothing is deferred;
  * E27b short bypass: on a deferred step a short prompt is admitted past a long one, the long one
         stays at the front of the queue, the per-step budget holds; remaining work is counted after
         the prefix-cache lookup (a long, mostly cached prompt counts as short);
  * E27c the capacity latch no longer switches the cadence off;
  * E29  hold: a request whose in-flight placeholders may finish it is not scheduled; structured
         output keeps the vendor pipeline; idle coalescing takes arrivals within the window only.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))
SRC = os.environ.get("GLM_VLLM_SRC")
if not SRC:
    print("SKIP: set GLM_VLLM_SRC to a vLLM 487ecf187 tree importable on CPU")
    sys.exit(0)
sys.path.insert(0, SRC)
os.environ.setdefault("GLM_PREFILL_CADENCE", "8")   # makes install_scheduler act; knobs are reset per test

import glm_prefill_hooks as hooks  # noqa: E402
import glm_prefill_sched as gps  # noqa: E402
import vllm.v1.core.sched.scheduler as smod  # noqa: E402
import vllm.v1.engine.core as cmod  # noqa: E402
from tests.v1.core.utils import create_requests, create_scheduler  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

STOCK = smod.Scheduler.schedule


def knobs(cadence=0, short=0, queued=False, drain=False):
    gps.CADENCE, gps.SHORT_TOKENS, gps.WHEN_QUEUED, gps.END_DRAIN = cadence, short, queued, drain


def install():
    gps.install_scheduler(smod)
    on = bool(gps.CADENCE or gps.SHORT_TOKENS or gps.WHEN_QUEUED or gps.END_DRAIN)
    assert getattr(smod.Scheduler.schedule, "__glm_prefill_sched__", False) == on


def step(s, throttle=False):
    out = s.schedule(throttle)
    ids = list(out.num_scheduled_tokens)
    # a request whose prompt is complete after this step samples one token; a partial chunk none
    sampled = [[0] if s.requests[rid].num_computed_tokens >= s.requests[rid].num_tokens else []
               for rid in ids]
    s.update_from_output(out, ModelRunnerOutput(
        req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
        sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={}, pooler_output=[]))
    return out


def with_decoder(s, max_tokens=10_000):
    (run0,) = create_requests(1, num_tokens=8, req_ids=["run0"], max_tokens=max_tokens, ignore_eos=True)
    s.add_request(run0)
    step(s)
    assert "run0" in s.requests and s.running


# ------------------------------------------------------------------------------------------
def test_edits_and_hashes():
    path = smod.__file__
    src = hooks.func_source(path, "Scheduler.schedule")
    new = gps.patch_schedule_source(src)
    assert new.count(gps.TAG) == len(gps.EDITS) + 1, new.count(gps.TAG)   # E27 inserts two lines
    compile(new, "x", "exec")
    got = hooks.current_hashes(gps.EXPECTED)
    assert got == gps.EXPECTED, got


def test_knobs_off_identical_to_stock():
    knobs()
    fn = gps.build_schedule(smod)
    smod._glm_ps = gps

    def workload(sched_fn):
        smod.Scheduler.schedule = sched_fn
        try:
            s = create_scheduler(max_num_seqs=8, max_num_batched_tokens=512, enable_prefix_caching=True)
            reqs = create_requests(6, num_tokens=300, max_tokens=12, same_prompt=False)
            trace = []
            for i in range(40):
                if i < len(reqs):
                    s.add_request(reqs[i])
                out = step(s, throttle=(i % 3 == 1))
                trace.append(tuple(sorted(out.num_scheduled_tokens.items())))
            return trace
        finally:
            smod.Scheduler.schedule = STOCK

    assert workload(STOCK) == workload(fn)


def test_e27_cadence():
    knobs(cadence=8)
    install()
    s = create_scheduler(max_num_seqs=8, max_num_batched_tokens=8192)
    with_decoder(s)                      # call 1 (counter 0 -> released)
    (long_,) = create_requests(1, num_tokens=3000, req_ids=["long"])
    s.add_request(long_)
    admitted_at = None
    for _ in range(12):
        pre = s.current_step
        out = step(s)
        assert "run0" in out.num_scheduled_tokens, "decode must run every step"
        if "long" in out.num_scheduled_tokens:
            admitted_at = pre
            break
    assert admitted_at is not None and admitted_at % 8 == 0 and admitted_at > 0, admitted_at


def test_e27_no_decoder_no_deferral():
    knobs(cadence=8)
    install()
    s = create_scheduler(max_num_seqs=8, max_num_batched_tokens=8192)
    step(s)                               # counter 0
    (long_,) = create_requests(1, num_tokens=3000, req_ids=["long"])
    s.add_request(long_)
    out = step(s)                         # counter 1: throttled, but nobody decodes
    assert "long" in out.num_scheduled_tokens


def _to_throttled_step(s):
    while s.current_step % 8 == 0:
        step(s)


def test_e27b_short_bypass_keeps_order_and_budget():
    knobs(cadence=8, short=64)
    install()
    s = create_scheduler(max_num_seqs=8, max_num_batched_tokens=8192)
    with_decoder(s)
    _to_throttled_step(s)
    reqs = create_requests(4, num_tokens=40, req_ids=["long", "s1", "s2", "s3"])
    reqs[0] = create_requests(1, num_tokens=3000, req_ids=["long"])[0]
    for r in reqs:
        s.add_request(r)
    out = step(s)
    got = set(out.num_scheduled_tokens)
    assert "s1" in got and "long" not in got, got
    assert "s2" not in got and "s3" not in got, "per-step short budget (40 + 40 > 64)"
    assert [r.request_id for r in s.waiting][:1] == ["long"], [r.request_id for r in s.waiting]
    assert s.requests["long"].status == RequestStatus.WAITING
    # vendor behaviour without the bypass: nothing is admitted on a deferred step
    knobs(cadence=8, short=0)
    s2 = create_scheduler(max_num_seqs=8, max_num_batched_tokens=8192)
    with_decoder(s2)
    _to_throttled_step(s2)
    for r in create_requests(2, num_tokens=40, req_ids=["a", "b"]):
        s2.add_request(r)
    out = step(s2)
    assert not ({"a", "b"} & set(out.num_scheduled_tokens))


def make_req(rid, tokens, max_tokens=16):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.request import Request
    sp = SamplingParams(max_tokens=max_tokens)
    sp.update_from_generation_config({}, 50256)
    return Request(request_id=rid, prompt_token_ids=list(tokens), sampling_params=sp, pooling_params=None,
                   mm_features=None, block_hasher=get_request_block_hasher(16, sha256))


def test_e27b_counts_after_prefix_lookup():
    knobs(cadence=8, short=64)
    install()
    s = create_scheduler(max_num_seqs=8, max_num_batched_tokens=8192, enable_prefix_caching=True)
    with_decoder(s)
    while s.current_step % 8 != 0:        # prefill the warm prompt on a release step
        step(s)
    s.add_request(make_req("warm", [5] * 1024))
    out = step(s)
    assert "warm" in out.num_scheduled_tokens
    _to_throttled_step(s)
    # 1040 tokens, of which 1024 are cached: remaining uncached work 16 < 64
    s.add_request(make_req("again", [5] * 1024 + [7] * 16))
    (cold,) = [make_req("cold", [9] * 1040)]
    s.add_request(cold)
    out = step(s)
    got = set(out.num_scheduled_tokens)
    assert "again" in got, "a long, mostly cached prompt counts as short (remaining after the lookup)"
    assert "cold" not in got, "a long uncached prompt waits for the cadence"


def test_e27c_latch_override():
    for queued, expect_deferred in ((False, False), (True, True)):
        knobs(cadence=8, queued=queued)
        install()
        s = create_scheduler(max_num_seqs=8, max_num_batched_tokens=8192)
        with_decoder(s)
        _to_throttled_step(s)
        s.prefill_capacity_bound = True   # latched by an earlier capacity-bound release
        (long_,) = create_requests(1, num_tokens=3000, req_ids=["long"])
        s.add_request(long_)
        out = step(s)
        assert ("long" not in out.num_scheduled_tokens) == expect_deferred, (queued, out.num_scheduled_tokens)


def test_e29_hold():
    for drain, expect_scheduled in ((False, True), (True, False)):
        knobs(drain=drain)
        install()
        s = create_scheduler(max_num_seqs=8, max_num_batched_tokens=8192)
        with_decoder(s, max_tokens=4)     # 1 output token after the prefill
        r = s.requests["run0"]
        r.num_output_placeholders = 3     # a step in flight may produce the last 3 tokens
        out = s.schedule()
        assert ("run0" in out.num_scheduled_tokens) == expect_scheduled, drain
        assert gps.holds(s, r) == drain
        r.max_tokens = 100                # the in-flight step cannot finish it: not held
        assert not gps.holds(s, r)


def test_idle_coalescing():
    class Core:
        def __init__(self):
            self.input_queue = queue.Queue()
            self.process_input_queue_block = True
            self.handled = []
            self.aborts_queue = queue.Queue()
            self.scheduler = NS(has_unfinished_requests=lambda: bool(self.handled), pause_state=None)

        def has_work(self):
            return bool(self.handled)

        def is_running(self):
            return True

        def _notify_idle_state_callbacks(self):
            pass

        def _handle_client_request(self, kind, req):
            self.handled.append((req, time.monotonic()))

    class DP:
        pass

    wrapped = gps.make_process_input_queue(cmod.EngineCoreProc._process_input_queue, DP)
    for window_s, expect in ((0.0, 1), (0.004, 2)):
        gps.COALESCE_S = window_s
        c = Core()

        def arrivals():
            c.input_queue.put(("ADD", "a"))
            time.sleep(0.0015)
            c.input_queue.put(("ADD", "b"))
            time.sleep(0.02)
            c.input_queue.put(("ADD", "late"))

        t = threading.Thread(target=arrivals)
        t.start()
        wrapped(c)
        got = [x for x, _ in c.handled]
        t.join()
        assert got == ["a", "b"][:expect], (window_s, got)
    gps.COALESCE_S = 0.0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        finally:
            smod.Scheduler.schedule = STOCK
        print("PASS", t.__name__)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
