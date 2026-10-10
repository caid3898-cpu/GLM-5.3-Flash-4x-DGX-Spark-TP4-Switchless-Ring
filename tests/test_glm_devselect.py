"""CPU test of overlay/glm_devselect.py host logic (DS_OVL overrides the overlay dir): eligibility (c1, 7 drafts, same request, no structured output, no
logprobs, a parent for the current graph set, glm_ab switch), debug-step scheduling, the go flag lifecycle.
  python tests/test_glm_devselect.py"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("DS_OVL", os.path.join(os.path.dirname(HERE), "overlay")))
VAR = {"GLM_DRAFT_TRUNC": "1", "GLM_DEVSELECT": "1"}
m = types.ModuleType("glm_ab"); m.ACTIVE = True; m.current = lambda: 0; m.env = lambda k, d=None: VAR.get(k, d)
sys.modules["glm_ab"] = m
import glm_devselect as ds  # noqa: E402


def so(spec, has_so=False):
    return types.SimpleNamespace(scheduled_spec_decode_tokens={r: [-1] * k for r, k in spec.items()},
                                 num_scheduled_tokens={r: k + 1 for r, k in spec.items()},
                                 total_num_scheduled_tokens=sum(k + 1 for k in spec.values()),
                                 has_structured_output_requests=has_so)


graphs = {}
lp = np.full(8, -1, dtype=np.int32)
runner = types.SimpleNamespace(
    req_states=types.SimpleNamespace(req_id_to_index={"a": 3, "b": 4}),
    sampler=types.SimpleNamespace(sampling_states=types.SimpleNamespace(num_logprobs=lp)),
    cudagraph_manager=types.SimpleNamespace(graphs=graphs))
pend = (None, 1, ("a",), 0)

assert not ds.eligible(runner, so({"a": 7}), pend)                 # not built
ds._D.built = True
assert not ds.eligible(runner, so({"a": 7}), pend)                 # no parent for this graph set
ds._D.sets[id(graphs)] = {}
assert ds.eligible(runner, so({"a": 7}), pend)
assert not ds.eligible(runner, so({"a": 3}), pend)                 # scheduled k != 7
assert not ds.eligible(runner, so({"a": 7}, has_so=True), pend)    # structured output
assert not ds.eligible(runner, so({"a": 7, "b": 7}), (None, 2, ("a", "b"), 0))   # c2
assert not ds.eligible(runner, so({"b": 7}), pend)                 # a new request (no draft of its own yet)
assert not ds.eligible(runner, so({"a": 7}), None)                 # no draft recorded
assert not ds.eligible(runner, so({"a": 7}), (None, 2, ("a", "b"), 0))
lp[3] = 5
assert not ds.eligible(runner, so({"a": 7}), pend)                 # logprobs requested
lp[3] = -1
VAR["GLM_DEVSELECT"] = "0"
assert not ds.eligible(runner, so({"a": 7}), pend)                 # variant without device select
VAR["GLM_DEVSELECT"] = "1"
runner.cudagraph_manager.graphs = {}                               # glm_ab switched to a set without a parent
assert not ds.eligible(runner, so({"a": 7}), pend)
runner.cudagraph_manager.graphs = graphs
ds._D.failed = "x"
assert not ds.eligible(runner, so({"a": 7}), pend)                 # build failed -> never
ds._D.failed = None

os.environ["GLM_DEVSELECT_DEBUG_STEPS"] = "2"
os.environ["GLM_DEVSELECT_DEBUG_EVERY"] = "5"
due = []
for step in range(12):
    ds._D.steps = step
    due.append(ds._debug_due())
assert due == [True, True, False, False, True, False, False, False, False, True, False, False], due
ds._D.steps = 0
s = so({"a": 7})
ds.begin_step(runner, s, True)
assert ds._D.go and ds._D.so is s
ds.begin_step(runner, s, False)
assert not ds._D.go and ds._D.so is None

ib = types.SimpleNamespace(num_reqs=1, cu_num_logits="fresh")
assert ds.after_prepare_inputs(ib).cu_num_logits == "fresh"       # go off -> untouched
print("ALL DEVSELECT CPU CHECKS PASSED")

# ---- overlay4: fused fixups, host side (offset table) + a numpy emulation of fix_kernel's MLA valid-region copy
if hasattr(ds, "_fused_prepare"):
    import torch
    rng = np.random.default_rng(0)
    NBX, NBY = 4, 12                                   # num_sm = 48, num_clusters = 12
    base = 4096
    offs_b = [base + 4 * 2048 * i for i in range(14)]  # 14 int arrays, 2048 ints of capacity each
    pi = [NBX, NBY] + offs_b + [0, 0]
    nbytes = base + 4 * 2048 * 15

    class W_:                                          # wrapper stand-in
        pass
    w = W_()
    w._int_workspace_buffer = torch.zeros(nbytes, dtype=torch.uint8)
    w._qo_indptr_buf = torch.zeros(9, dtype=torch.int32)
    w._kv_indptr_buf = torch.zeros(9, dtype=torch.int32)
    w._kv_len_arr_buf = torch.zeros(8, dtype=torch.int32)
    S = {}
    for M in range(2, 8):
        src = torch.from_numpy(rng.integers(0, 255, nbytes, dtype=np.uint8))
        wi = offs_b[13] // 4
        v = src.view(torch.int32)
        Wn = int(rng.integers(1, 200))
        v[wi: wi + NBY + 1] = torch.tensor(sorted(rng.integers(0, Wn, NBY + 1).tolist())[:-1] + [Wn], dtype=torch.int32)
        S[M] = types.SimpleNamespace(int_dev=src, plan_info=pi, qo_dev=torch.arange(9, dtype=torch.int32),
                                     kv_dev=torch.arange(9, dtype=torch.int32) * 3, len_dev=torch.ones(8, dtype=torch.int32))
    T = types.SimpleNamespace(gdn=[torch.zeros(3, dtype=torch.int32)] * 4, indexers=[], slots=torch.zeros(7, 16, dtype=torch.int64),
                              ctx=torch.zeros(1, 16, dtype=torch.int64), qsl=torch.zeros(9, dtype=torch.int32),
                              seq_lens=torch.zeros(8, dtype=torch.int32), cu=torch.zeros(9, dtype=torch.int32), wrapper=w)
    F = ds._fused_prepare(T, S)

    def emulate_copy(F, M):                            # fix_kernel's MLA part, line by line
        p = F["per_m"][M]
        src, dst, offs = p["src"], F["dst"], p["offs"].tolist()
        wi = offs[13]
        Wv = int(src[wi + p["NCL"]])
        for a in range(8):
            dst[offs[a]: offs[a] + Wv] = src[offs[a]: offs[a] + Wv]
        for a in range(8, 13):
            dst[offs[a]: offs[a] + p["NSM"]] = src[offs[a]: offs[a] + p["NSM"]]
        dst[wi: wi + p["NCL"] + 1] = src[wi: wi + p["NCL"] + 1]

    for M in range(2, 8):
        w._int_workspace_buffer.fill_(0x5A)
        emulate_copy(F, M)
        ok, Wv, field = ds.mla_valid_equal(S[M].int_dev, w._int_workspace_buffer, pi)
        assert ok, (M, Wv, field)
        assert F["per_m"][M]["NCL"] == NBY and F["per_m"][M]["NSM"] == NBX * NBY
        # bytes outside the valid region stay untouched (the stale tail is never read by the kernel)
        touched = int((w._int_workspace_buffer != 0x5A).sum())
        assert touched <= 4 * (8 * Wv + 5 * NBX * NBY + NBY + 1), touched
    # a broken layout (unaligned offset) is refused -> legacy fixups
    S2 = {2: types.SimpleNamespace(**{**vars(S[2]), "plan_info": [NBX, NBY, 4097] + offs_b[1:] + [0, 0]})}
    try:
        ds._fused_prepare(T, S2)
        raise AssertionError("unaligned plan accepted")
    except RuntimeError:
        pass
    T5 = types.SimpleNamespace(**{**vars(T), "gdn": [torch.zeros(3, dtype=torch.int32)] * 5})
    try:
        ds._fused_prepare(T5, S)
        raise AssertionError("5 GDN buffers accepted")
    except RuntimeError:
        pass
    print("FUSED FIXUP HOST CHECKS PASSED")

# ---- overlay4: live timeline bookkeeping (fake events)
if hasattr(ds, "_TL"):
    class Ev:
        t = 0.0
        def __init__(self):
            Ev.t += 1.0
            self.at = Ev.t
        def query(self):
            return True
        def elapsed_time(self, o):
            return o.at - self.at
    ds._tl_event = lambda: Ev()
    ds._TL.on = True
    os.environ["GLM_DEVSELECT_TIMELINE_EVERY"] = "0"
    for i in range(5):
        ds._tl_target_start(True)
        ds._tl_target_end(True)
        ds._tl_draft_end()
    ds._tl_target_start(False)                         # closes the 5th cycle
    st = ds._TL.stats
    assert list(st) == [(0, True)] and len(st[(0, True)]) == 5, st
    assert st[(0, True)][0] == (1.0, 1.0, 1.0), st
    ds._tl_other()                                     # an unrelated replay drops the open cycle
    ds._tl_target_start(False)
    assert len(st[(0, True)]) == 5
    ds._tl_report()
    print("TIMELINE CHECKS PASSED")
