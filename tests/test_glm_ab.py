"""CPU checks for overlay/glm_ab.py (in-boot A/B harness) and the per-call gates it drives. Pure Python, no torch:
the engine classes are small fakes with the same call shapes as vLLM 487ecf187's V2 runner.

  python3 tests/test_glm_ab.py

The real-class check (hooks land on the image's CudaGraphManager / Worker / EngineCore / Triton classes) is
tests/test_glm_ab_image.py, run inside the image.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overlay"))

RESULTS = []


def case(fn):
    def run():
        for k in [k for k in os.environ if k.startswith("GLM_AB") or k in (
                "GLM_TARGET_VOCAB_ARGMAX", "GLM_KDA_NOCOPY", "GLM_KDA_STASH", "GLM_ROUTER_FP32OUT")]:
            del os.environ[k]
        for m in ("glm_ab", "glm_kda_nocopy", "glm_target_argmax"):
            sys.modules.pop(m, None)
        sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != "glm_ab"]
        try:
            fn()
            RESULTS.append((fn.__name__, "PASS", ""))
        except Exception as exc:  # noqa: BLE001
            import traceback
            RESULTS.append((fn.__name__, "FAIL", traceback.format_exc()))
    return run


def fresh(env: dict):
    os.environ.update(env)
    import glm_ab
    importlib.reload(glm_ab)
    return glm_ab


def raises(fn, text):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        assert text in str(exc), f"expected {text!r} in {exc!r}"
        return
    raise AssertionError(f"no exception (expected {text!r})")


# ---- fake engine classes ------------------------------------------------------------------------------------

def fake_cudagraph_module(ab):
    """Mimics vllm/v1/worker/gpu/cudagraph_utils.py: capture() warms up + captures every desc into self.graphs,
    run_fullgraph() replays self.graphs[desc]. A 'graph' records the flags current at capture time."""
    mod = types.ModuleType("vllm.v1.worker.gpu.cudagraph_utils")

    class Graph:
        def __init__(self, tag):
            self.tag = tag
            self.replays = 0

        def replay(self):
            self.replays += 1
            return self.tag

    class CudaGraphManager:
        def __init__(self, descs):
            self.descs = descs
            self.graphs = {}
            self._graphs_captured = False
            self.warmups = []

        def capture(self, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"):
            for d in self.descs:
                fwd = create_forward_fn(d, warmup=True)
                self.warmups.append(fwd())
                fwd = create_forward_fn(d, warmup=False)
                assert d not in self.graphs, f"Graph already captured for {d}"
                self.graphs[d] = Graph(fwd())
            self._graphs_captured = True

        def profile_memory(self, create_forward_fn):
            CudaGraphManager.capture(self, create_forward_fn, "profile")
            n = len(self.graphs)
            self.graphs.clear()
            return n

        def run_fullgraph(self, desc):
            return self.graphs[desc].replay()

    class ModelCudaGraphManager(CudaGraphManager):
        def capture(self, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"):
            return super().capture(create_forward_fn, progress_bar_desc)

        def run_fullgraph(self, desc):
            return super().run_fullgraph(desc)

    class DFlashCudaGraphManager(CudaGraphManager):
        pass

    mod.CudaGraphManager = CudaGraphManager
    mod.ModelCudaGraphManager = ModelCudaGraphManager
    mod.DFlashCudaGraphManager = DFlashCudaGraphManager
    mod.__name__ = "vllm.v1.worker.gpu.cudagraph_utils"
    ab.install_cudagraph(mod)
    return mod


def forward_factory(ab):
    def create(desc, warmup):
        return lambda: (desc, ab.env("GLM_KDA_STASH"), ab.env("GLM_TARGET_VOCAB_ARGMAX"), ab.current())
    return create


# ---- cases ----------------------------------------------------------------------------------------------------

@case
def off_by_default():
    ab = fresh({})
    assert ab.configure() is False and not ab.ACTIVE
    assert ab.flag("GLM_KDA_STASH") is True          # off: installed == on
    os.environ["GLM_KDA_STASH"] = "1"
    assert ab.env("GLM_KDA_STASH") == "1"
    ab = fresh({"GLM_AB_VARIANTS": "1"})
    assert ab.configure() is False


@case
def spec_union_base_and_refusals():
    env = {"GLM_AB_VARIANTS": "3", "GLM_TARGET_VOCAB_ARGMAX": "1", "GLM_KDA_STASH": "1",
           "GLM_AB_V0": "GLM_TARGET_VOCAB_ARGMAX=1+GLM_KDA_STASH=1",
           "GLM_AB_V1": "GLM_TARGET_VOCAB_ARGMAX=0",
           "GLM_AB_V2": "GLM_KDA_NOCOPY=1 GLM_KDA_STASH=0"}
    ab = fresh(env)
    assert ab.configure()
    assert ab.N == 3 and len(ab.CONFIG_HASH) == 16
    # union: nocopy installed because v2 wants it; stash / argmax stay 1
    assert os.environ["GLM_KDA_NOCOPY"] == "1" and os.environ["GLM_KDA_STASH"] == "1"
    assert os.environ["GLM_TARGET_VOCAB_ARGMAX"] == "1"
    base = json.loads(os.environ["GLM_AB_BASE"])
    assert base["GLM_KDA_NOCOPY"] is None and base["GLM_KDA_STASH"] == "1"   # pre-union
    # per-variant values, base where not overridden
    ab.set_runtime(1)
    assert ab.env("GLM_TARGET_VOCAB_ARGMAX") == "0" and ab.env("GLM_KDA_STASH") == "1"
    assert ab.flag("GLM_KDA_NOCOPY") is False
    ab.set_runtime(2)
    assert ab.flag("GLM_KDA_NOCOPY") and not ab.flag("GLM_KDA_STASH") and ab.env("GLM_TARGET_VOCAB_ARGMAX") == "1"
    with ab.capturing(1):
        assert ab.env("GLM_TARGET_VOCAB_ARGMAX") == "0" and ab.current() == 1
    assert ab.current() == 2
    # a child process re-configures from the unioned env + GLM_AB_BASE and gets the same hash
    h = ab.CONFIG_HASH
    ab2 = fresh({})
    assert ab2.configure() and ab2.CONFIG_HASH == h
    ab2.set_runtime(0)
    assert ab2.flag("GLM_KDA_NOCOPY") is False        # base None, not the unioned 1


@case
def aa_guard():
    env = {"GLM_AB_VARIANTS": "2", "GLM_KDA_STASH": "1", "GLM_AB_V0": "", "GLM_AB_V1": "GLM_KDA_STASH=1"}
    ab = fresh(env)
    raises(ab.configure, "same effective config")
    os.environ["GLM_AB_V1"] = "GLM_KDA_STASH=on"
    raises(ab.configure, "same effective config")      # on == 1
    os.environ["GLM_AB_ALLOW_AA"] = "1"
    assert ab.configure()
    os.environ["GLM_AB_V1"] = "GLM_TARGET_VOCAB_ARGMAX=off"
    os.environ.pop("GLM_AB_BASE")
    assert ab.configure()                              # base unset == off: still an A/A, allowed


@case
def bad_specs():
    ab = fresh({"GLM_AB_VARIANTS": "2", "GLM_AB_V1": "GLM_DS_SPLIT=1"})
    raises(ab.configure, "not switchable")
    ab = fresh({"GLM_AB_VARIANTS": "2", "GLM_AB_V1": "GLM_TARGET_VOCAB_ARGMAX=2"})
    raises(ab.configure, "0|1|check")
    ab = fresh({"GLM_AB_VARIANTS": "2", "GLM_AB_V2": "GLM_KDA_STASH=1"})
    raises(ab.configure, "GLM_AB_VARIANTS=2")
    ab = fresh({"GLM_AB_VARIANTS": "11"})
    raises(ab.configure, "at most")


@case
def graph_sets_capture_and_replay():
    ab = fresh({"GLM_AB_VARIANTS": "2", "GLM_KDA_STASH": "1", "GLM_AB_V0": "GLM_KDA_STASH=1",
                "GLM_AB_V1": "GLM_KDA_STASH=0", "GLM_AB_CAPTURE": "whole"})
    assert ab.configure()
    mod = fake_cudagraph_module(ab)
    tgt = mod.ModelCudaGraphManager(["d8", "d16"])
    drf = mod.DFlashCudaGraphManager(["q8"])
    tgt.capture(forward_factory(ab))
    drf.capture(forward_factory(ab))
    sets = tgt._glm_ab_sets
    assert len(sets) == 2 and set(sets[0]) == set(sets[1]) == {"d8", "d16"}
    assert tgt.graphs is sets[0]
    # set v was captured (and warmed up) with variant v current
    assert sets[0]["d8"].tag[1:] == ("1", None, 0) and sets[1]["d8"].tag[1:] == ("0", None, 1)
    assert [w[3] for w in tgt.warmups] == [0, 0, 1, 1]
    assert "_glm_ab_sets" not in drf.__dict__ and len(drf.graphs) == 1   # drafter: one shared set
    assert tgt.run_fullgraph("d8")[3] == 0
    ab.set_runtime(1)
    assert tgt.graphs is sets[1] and tgt.run_fullgraph("d8")[3] == 1 and tgt.run_fullgraph("d16")[1] == "0"
    drf.run_fullgraph("q8")
    st = ab.status()
    assert st["replays"]["target"] == [1, 2, 0] and st["replays"]["DFlashCudaGraphManager"] == [0, 0, 1]
    # a manager left pointing at the wrong set is caught at replay
    tgt.graphs = sets[0]
    raises(lambda: tgt.run_fullgraph("d8"), "other than the runtime variant")
    tgt.graphs = sets[1]
    # memory profiling captures once and leaves no extra sets behind
    prof = mod.ModelCudaGraphManager(["d8"])
    assert prof.profile_memory(forward_factory(ab)) == 1 and "_glm_ab_sets" not in prof.__dict__
    raises(lambda: ab.set_runtime(2), "not in [0, 2)")


def fake_interleave_module(ab, log):
    """Adds the module globals the interleaved capture uses (names as imported by cudagraph_utils.py)."""
    import contextlib
    mod = fake_cudagraph_module(ab)

    class Mode:
        NONE, PIECEWISE, FULL = "NONE", "PIECEWISE", "FULL"

    class FakeGraph:
        def __init__(self):
            self.tag = None
            self.replays = 0

        def replay(self):
            self.replays += 1
            return self.tag

    cur = {"graph": None}

    @contextlib.contextmanager
    def graph(g, pool=None):
        cur["graph"] = g
        log.append(("begin", ab.current()))
        yield
        log.append(("end", ab.current()))
        cur["graph"] = None

    @contextlib.contextmanager
    def graph_capture(device=None):
        log.append(("ctx-enter",))
        yield
        log.append(("ctx-exit",))

    torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext,
                                  cuda=types.SimpleNamespace(CUDAGraph=FakeGraph, graph=graph))
    mod.torch, mod.CUDAGraphMode, mod.graph_capture = torch, Mode, graph_capture
    mod.is_global_first_rank = lambda: False
    mod.tqdm = lambda it, desc=None: it
    off = types.SimpleNamespace(sync_prev_onload=lambda: None, join_after_forward=lambda: None)
    mod.get_offloader = lambda: off
    mod.set_graph_pool_id = lambda pid: None
    mod.current_platform = types.SimpleNamespace(graph_pool_handle=lambda: "pool")
    mod.compilation_counter = types.SimpleNamespace(num_cudagraph_captured=0)
    mod._cur = cur
    return mod


@case
def interleaved_capture():
    ab = fresh({"GLM_AB_VARIANTS": "3", "GLM_KDA_STASH": "1", "GLM_AB_ALLOW_AA": "1", "GLM_AB_V0": "GLM_KDA_STASH=1",
                "GLM_AB_V1": "GLM_KDA_STASH=1", "GLM_AB_V2": "GLM_KDA_STASH=0"})
    os.environ["GLM_AB_CAPTURE"] = "interleave"
    assert ab.configure() and ab._opts["capture"] == "interleave"
    log = []
    mod = fake_interleave_module(ab, log)
    tgt = mod.ModelCudaGraphManager(["d16", "d8"])
    tgt._capture_descs = {"FULL": ["d16", "d8"]}
    tgt.pool, tgt.device = None, "cuda:0"

    def create(desc, warmup):
        def fwd(mode):
            log.append(("fwd", desc, warmup, ab.current(), ab.env("GLM_KDA_STASH")))
            g = mod._cur["graph"]
            if g is not None:
                g.tag = (desc, ab.current(), ab.env("GLM_KDA_STASH"))
        return fwd
    tgt.capture(create)
    assert [e for e in log if e[0].startswith("ctx")] == [("ctx-enter",), ("ctx-exit",)]   # ONE context
    order = [(e[1], e[3]) for e in log if e[0] == "fwd" and e[2]]                            # warm-ups
    assert order == [("d16", 0), ("d16", 1), ("d16", 2), ("d8", 0), ("d8", 1), ("d8", 2)], order
    sets = tgt._glm_ab_sets
    assert sets[2]["d8"].tag == ("d8", 2, "0") and sets[0]["d16"].tag == ("d16", 0, "1")
    assert tgt._graphs_captured and mod.compilation_counter.num_cudagraph_captured == 6
    ab.set_runtime(2)
    assert tgt.run_fullgraph("d16") == ("d16", 2, "0")
    tgt._capture_descs = {"PIECEWISE": ["p"], "FULL": ["d8"]}
    tgt2 = mod.ModelCudaGraphManager(["d8"])
    tgt2._capture_descs, tgt2.pool, tgt2.device = {"PIECEWISE": ["p"]}, None, "cuda:0"
    raises(lambda: tgt2.capture(create), "FULL graphs only")


@case
def draft_sets_optional():
    ab = fresh({"GLM_AB_VARIANTS": "2", "GLM_AB_V0": "GLM_KDA_STASH=1", "GLM_AB_V1": "GLM_KDA_STASH=0",
                "GLM_AB_DRAFT_SETS": "1", "GLM_AB_CAPTURE": "whole"})
    assert ab.configure()
    mod = fake_cudagraph_module(ab)
    drf = mod.DFlashCudaGraphManager(["q8"])
    drf.capture(forward_factory(ab))
    assert len(drf._glm_ab_sets) == 2


@case
def worker_switch_and_core_guard():
    ab = fresh({"GLM_AB_VARIANTS": "2", "GLM_AB_V0": "GLM_KDA_STASH=1", "GLM_AB_V1": "GLM_KDA_STASH=0"})
    assert ab.configure()
    wmod = types.ModuleType(ab.WORKER)

    class Worker:
        pass
    wmod.Worker = Worker
    ab.install_worker(wmod)
    w = Worker()
    out = w.glm_ab_switch("1", "tok")
    assert out["variant"] == 1 and out["token"] == "tok" and out["seq"] == 1 and out["prev"] == 0
    assert out["ranks"] is None                          # no TP group here
    assert w.glm_ab_status()["effective"]["GLM_KDA_STASH"] is False

    cmod = types.ModuleType(ab.CORE)
    calls = []

    class Sched:
        def __init__(self):
            self.running, self.n = [], 0

        def get_num_unfinished_requests(self):
            return self.n

    class EngineCore:
        def __init__(self):
            self.scheduler = Sched()

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            calls.append(method)
            return ["ok"]
    cmod.EngineCore = EngineCore
    ab.install_core(cmod)
    core = EngineCore()
    core.scheduler.n = 2
    assert core.collective_rpc("glm_ab_switch", None, ("0", "t")) == [{"busy": 2}] and calls == []
    assert core.collective_rpc("glm_ab_status") == ["ok"]          # reads are never refused
    core.scheduler.n, core.scheduler.running = 0, ["r"]
    assert core.collective_rpc("glm_ab_switch", None, ("0", "t")) == [{"busy": 1}]
    core.scheduler.running = []
    assert core.collective_rpc("glm_ab_switch", None, ("0", "t")) == ["ok"]
    import inspect
    assert list(inspect.signature(EngineCore.collective_rpc).parameters) == [
        "self", "method", "timeout", "args", "kwargs"]


@case
def not_armed_worker():
    ab = fresh({})
    assert ab.worker_switch(1)["error"].startswith("GLM_AB not armed")


@case
def nocopy_gate_follows_variant():
    ab = fresh({"GLM_AB_VARIANTS": "3", "GLM_KDA_STASH": "1",
                "GLM_AB_V0": "GLM_KDA_STASH=1+GLM_KDA_NOCOPY=0", "GLM_AB_V1": "GLM_KDA_STASH=1+GLM_KDA_NOCOPY=1",
                "GLM_AB_V2": "GLM_KDA_STASH=0+GLM_KDA_NOCOPY=1"})
    assert ab.configure()
    import glm_kda_nocopy as nc
    seen = []
    nc.wants_strided = lambda kw, prev_is_stock: seen.append(prev_is_stock) or ("S",)
    nc.fused_recurrent_kda_strided = lambda _strides=None, **kw: ("strided", _strides)
    pristine = lambda *a, **kw: "stock"                 # noqa: E731
    stash = lambda *a, **kw: "stash-dispatch"           # noqa: E731
    d = nc.make_dispatch(stash, pristine)
    ab.set_runtime(0)
    assert d(q=1, num_accepted_tokens=1) == "stash-dispatch" and seen == []
    ab.set_runtime(1)
    assert d(q=1, num_accepted_tokens=1) == ("strided", ("S",)) and seen == [False]   # stash owns verify
    ab.set_runtime(2)
    assert d(q=1, num_accepted_tokens=1) == ("strided", ("S",)) and seen == [False, True]  # stash off: nocopy does
    # harness off: nocopy installed == on, stash installed == on
    for k in [k for k in os.environ if k.startswith("GLM_AB")]:
        del os.environ[k]
    ab2 = fresh({})
    ab2.configure()
    sys.modules["glm_ab"] = ab2
    seen.clear()
    d(q=1)
    assert seen == [False]


@case
def argmax_mode_per_call():
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    os.environ["GLM_TARGET_VOCAB_ARGMAX"] = "1"
    import glm_target_argmax as ta
    importlib.reload(ta)
    assert ta._mode() == "1"                            # harness off: import-time value
    ab = fresh({"GLM_AB_VARIANTS": "3", "GLM_AB_V0": "GLM_TARGET_VOCAB_ARGMAX=1",
                "GLM_AB_V1": "GLM_TARGET_VOCAB_ARGMAX=0", "GLM_AB_V2": "GLM_TARGET_VOCAB_ARGMAX=check"})
    assert ab.configure()
    for v, want in ((0, "1"), (1, "0"), (2, "check")):
        ab.set_runtime(v)
        assert ta._mode() == want, (v, ta._mode())


def main():
    for name, fn in list(globals().items()):
        if callable(fn) and getattr(fn, "__name__", "") == "run":
            fn()
    for name, res, tb in RESULTS:
        print(f"{res} {name}")
        if tb:
            print(tb)
    ok = all(r == "PASS" for _, r, _ in RESULTS)
    print(f"{sum(r == 'PASS' for _, r, _ in RESULTS)}/{len(RESULTS)} PASS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
