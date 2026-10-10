"""CPU test of overlay/glm_draft_trunc.py: the policy (E(L) from the table, cost-priced uniform L, lam update), the
in-place truncation of a SchedulerOutput-like object, the skips, and the scheduler default_k property.
  python tests/test_glm_draft_trunc.py   (numpy not needed; stdlib only)"""
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.join(os.path.dirname(HERE), "overlay")
sys.path.insert(0, OVERLAY)
VAR = {"GLM_DRAFT_TRUNC": "1"}
m = types.ModuleType("glm_ab"); m.ACTIVE = True; m.env = lambda k, d=None: VAR.get(k, d); sys.modules["glm_ab"] = m
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
print("ALL CHECKS PASSED")
