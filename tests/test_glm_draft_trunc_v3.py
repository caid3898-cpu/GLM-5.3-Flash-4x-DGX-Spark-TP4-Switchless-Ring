"""CPU test of overlay/glm_draft_trunc.py (v2/v3: gate / early skip / ring; the v1 checks first): the policy (E(L) from the table, cost-priced uniform L, lam update), the
in-place truncation of a SchedulerOutput-like object, the skips, and the scheduler default_k property.
  python tests/test_glm_draft_trunc_v3.py"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.environ.get("TR_OVL", os.path.join(os.path.dirname(HERE), "overlay"))
sys.path.insert(0, OVERLAY)
VAR = {"GLM_DRAFT_TRUNC": "1"}
m = types.ModuleType("glm_ab"); m.ACTIVE = True; m.current = lambda: 0; m.env = lambda k, d=None: VAR.get(k, d); sys.modules["glm_ab"] = m
import glm_draft_trunc as t  # noqa: E402

table = json.load(open(os.path.join(OVERLAY, "glm_bav_table_seg.json")))
P = t.Policy(table)
sure, unsure = [-0.001] * 7, [-2.6] * 7
E_s, E_u = P.expect(sure), P.expect(unsure)
assert E_s[7] > 5 and E_u[1] <= 1.5, (E_s, E_u)
assert P.choose([sure], 7) == 7 and P.choose([unsure], 7) == 1
mixed = [-0.001, -0.001, -2.6, -2.6, -2.6, -2.6, -2.6]
L = L_mixed = P.choose([mixed], 7)
assert 2 <= L <= 3, L


def so(spec, has_so=False):
    return types.SimpleNamespace(scheduled_spec_decode_tokens={r: [-1] * k for r, k in spec.items()},
                                 num_scheduled_tokens={r: k + 1 for r, k in spec.items()},
                                 total_num_scheduled_tokens=sum(k + 1 for k in spec.values()),
                                 has_structured_output_requests=has_so)


s = so({"a": 7})
assert t.truncate(s, {"a": unsure}, P) == 1
assert s.scheduled_spec_decode_tokens["a"] == [-1] and s.num_scheduled_tokens["a"] == 2 and s.total_num_scheduled_tokens == 2
s = so({"a": 7})
assert t.truncate(s, {"a": sure}, P) == 7 and s.total_num_scheduled_tokens == 8          # untouched
s = so({"a": 7, "b": 7})                                                                   # c2: one uniform L
L = t.truncate(s, {"a": unsure, "b": sure}, P)
assert L is not None and s.num_scheduled_tokens["a"] == s.num_scheduled_tokens["b"] == L + 1
assert s.total_num_scheduled_tokens == 2 * (L + 1)
assert t.truncate(so({"a": 7, "b": 7}), {"a": sure}, P) is None                           # a request without confidence
assert t.truncate(so({"a": 7}, True), {"a": unsure}, P) is None                            # structured output
assert t.truncate(so({"a": 7, "b": 3}), {"a": unsure, "b": unsure}, P) is None             # mixed scheduled k
# prefill rows (no spec) are left alone
s = so({"a": 7}); s.num_scheduled_tokens["p"] = 300; s.total_num_scheduled_tokens += 300; s.scheduled_spec_decode_tokens["p"] = []
assert t.truncate(s, {"a": unsure}, P) == 1 and s.num_scheduled_tokens["p"] == 300 and s.total_num_scheduled_tokens == 302
# cost table override
P2 = t.Policy(table, {"c1": {"2": 20, "3": 20.5, "4": 21, "5": 21.5, "6": 22, "7": 22.5, "8": 23}, "row_ms": 0.5})
assert P2.choose([mixed], 7) >= L_mixed                                                          # cheap rows -> longer shapes
# scheduler property: default_k = engine_k only while the variant is active
class SpecProbeScheduler:
    def __init__(self):
        self._ak_engine_k = 7
        self._probe_default_k = None
t._patch_sched(types.SimpleNamespace(SpecProbeScheduler=SpecProbeScheduler))
sch = SpecProbeScheduler()
assert sch._probe_default_k == 7
VAR["GLM_DRAFT_TRUNC"] = "0"; assert sch._probe_default_k is None
sch._probe_default_k = 3; assert sch._probe_default_k == 3
VAR["GLM_DRAFT_TRUNC"] = "1"; assert sch._probe_default_k == 7
print("lam:", P.lam)


# ------------------------------------------------------------------------------------------ v2: the runner flow
assert P.last_kmax == 7 and isinstance(P.last_margin, float)
P3 = t.Policy(table); P3.choose([sure], 7); assert P3.last_margin > 0.1
P3.choose([unsure], 7); assert P3.last_margin < 0


class Ev:
    def __init__(self, name): self.name, self.synced = name, 0
    def synchronize(self): self.synced += 1
    def query(self): return True


class Buf:
    def __init__(self): self.rows = []
    def __getitem__(self, sl): return types.SimpleNamespace(tolist=lambda: [list(r) for r in self.rows[sl]])


class Runner:
    def __init__(self, *a, **k): pass
    def execute_model(self, so, *a, **k):
        self.seen = (dict(so.num_scheduled_tokens), so.total_num_scheduled_tokens)
        return None


mod = types.SimpleNamespace(GPUModelRunner=Runner)
t._patch_runner(mod)
t._W.policy = t.Policy(table); t._W.rank0 = True
t._W.ring = [Buf() for _ in range(t.RING)]
R = Runner.__new__(Runner)
step = [0]


def draft(confs, rids=("a",)):
    """what the propose hook leaves behind: confidences in the next ring slot + an event"""
    slot = t._W.slot_i % t.RING; t._W.slot_i += 1
    t._W.ring[slot].rows = [list(c) for c in confs]
    ev = Ev(f"d{step[0]}"); step[0] += 1
    t._W.pend = (ev, len(confs), tuple(rids), slot)
    return ev


def run(spec=None, has_so=False):
    s = so(spec or {"a": 7}, has_so)
    R.execute_model(s)
    return R.seen


def reset(gate="0", early="0"):
    VAR.update({"GLM_DRAFT_TRUNC_GATE": gate, "GLM_DRAFT_TRUNC_EARLYSKIP": early})
    t._W.pols, t._W.st, t._W.unscored, t._W.last, t._W.pend = {}, {}, None, None, None


# (a) knobs off = 2209: every step waits on its own draft, truncation from it
reset()
e1 = draft([sure]); assert run()[1] == 8 and e1.synced == 1
e2 = draft([unsure]); assert run()[1] == 2 and e2.synced == 1
st = t._W.st[0]; assert st["waited"] == 2 and st["gated"] == 0 and st["hist"][7] == 1 and st["hist"][1] == 1

# (b) gate: after a clear full-shape decision the next step keeps k = 7 without waiting; the gated draft is scored
#     post hoc (it decides the NEXT gate), exactly the replay's semantics
reset(gate="0.1")
e1 = draft([sure]); assert run()[1] == 8 and e1.synced == 1                 # waited, L = 7 clearly
e2 = draft([unsure]); assert run()[1] == 8 and e2.synced == 0               # gated: full shape, no wait
assert t._W.unscored is not None and t._W.st[0]["gated"] == 1
e3 = draft([sure]); r = run()                                               # e2 scored post hoc -> L = 1: no gate
assert e2.synced == 1 and e3.synced == 1 and r[1] == 8 and t._W.st[0]["posthoc"] == 1
e4 = draft([sure]); assert run()[1] == 8 and e4.synced == 0                 # previous decision 7 again: gated
e5 = draft([unsure]); assert run()[1] == 8 and e5.synced == 0 and e4.synced == 1   # e4 post hoc = 7: gated
e6 = draft([unsure]); r = run(); assert e5.synced == 1 and e6.synced == 1 and r[1] == 2   # e5 = 1: wait, truncate
# a borderline full-shape win (margin below the threshold) does not gate
reset(gate="1000")
e1 = draft([sure]); run(); e2 = draft([sure]); run(); assert e2.synced == 1 and t._W.st[0]["gated"] == 0
# c2: the batch decision gates the batch; the post-hoc score uses the live requests in spec order
reset(gate="0.1")
e1 = draft([sure, sure], ("a", "b")); assert run({"a": 7, "b": 7})[1] == 16
e2 = draft([unsure, sure], ("b", "a")); assert run({"a": 7, "b": 7})[1] == 16 and e2.synced == 0
idx = t._W.unscored[2]; assert idx == [1, 0], idx                         # spec order a, b -> ring rows 1, 0
e3 = draft([sure, sure], ("a", "b")); run({"a": 7, "b": 7}); assert e2.synced == 1
# a skipped step (structured output) clears the gate state
reset(gate="0.1")
draft([sure]); run(); assert t._W.last[0] == 7
e2 = draft([sure]); run(has_so=True)                                        # ineligible: never gated
assert t._W.st[0]["gated"] == 0 and t._W.last is None and e2.synced == 1
e3 = draft([sure]); run(); assert e3.synced == 1                            # so the next step waits again

# (c) early skip: a step the policy would skip does not wait
reset(early="1")
e1 = draft([sure]); run(has_so=True); assert e1.synced == 0 and t._W.st[0]["early"] == 1
e2 = draft([sure], ("new",)); run(); assert e2.synced == 0                 # request "a" has no confidence record
reset(early="0")
e1 = draft([sure]); run(has_so=True); assert e1.synced == 1                 # 2209 behaviour: waited, then skipped

# (d) per-variant policy state and a variant switch clears the gate
reset(gate="0.1")
draft([sure]); run(); draft([sure]); run(); assert t._W.unscored is not None
m.current = lambda: 1
draft([sure]); run(); assert t._W.st[1]["gated"] == 0 and 1 in t._W.pols and 0 in t._W.pols
m.current = lambda: 0

# (e) eligible() mirrors truncate()'s skips
assert t.eligible(so({"a": 7}), ("a",)) == 7
assert t.eligible(so({"a": 7}, True), ("a",)) is None
assert t.eligible(so({"a": 7, "b": 3}), ("a", "b")) is None
assert t.eligible(so({"a": 7, "b": 7}), ("a",)) is None
# (f) periods: consecutive draft ends, attributed to the step whose draft ended; idle gaps dropped
class TEv(Ev):
    def __init__(self, t_ms): super().__init__("t"); self.t = t_ms
    def elapsed_time(self, other): return other.t - self.t
t._W.timeline, t._W.per = None, {}
clock = 0.0
for i in range(16):
    t._W.cur_meta = (0, 1, 4 if i % 2 else 8, "w" if i % 2 else "g")
    clock += 36.0 if i % 2 else 49.0
    if i == 6:
        clock += 5000.0                                                     # idle between requests
    t._note_period(TEv(clock))
rep = t._period_report(0)
assert "1x4w: 36.00" in rep and "1x8g: 49.00" in rep, rep
print("periods:", rep)
print("ALL CHECKS PASSED (2209 + v2)")
