#!/usr/bin/env python3
"""CPU test of the trunc cost JSON overlay/glm_draft_trunc_cost_c4fit.json (row_ms 2.0, the profile default).
  1. data-only: identical to glm_draft_trunc_cost.json (row_ms 1.2) except row_ms 1.2 -> 2.0 and its source note; c1 block equal
  2. c1 invariance: the Policy chooses the same L at n = 1 with either JSON on 20k random drafts
  3. n = 4 plan costs differ (the knob acts only at batch > 1), and the JSON loads via _load_policy."""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OVL = os.path.join(os.path.dirname(HERE), "overlay")
sys.path.insert(0, OVL)
os.environ["GLM_DRAFT_TRUNC_TABLE"] = os.path.join(OVL, "glm_bav_table_seg.json")
os.environ["GLM_DRAFT_TRUNC_COST"] = os.path.join(OVL, "glm_draft_trunc_cost_c4fit.json")
import glm_draft_trunc as T  # noqa: E402

served = json.load(open(os.path.join(OVL, "glm_draft_trunc_cost.json")))
c4 = json.load(open(os.path.join(OVL, "glm_draft_trunc_cost_c4fit.json")))
extra = {k for k in set(served) | set(c4) if served.get(k) != c4.get(k)}
assert extra <= {"row_ms", "source", "note"} and "row_ms" in extra, extra
assert served["c1"] == c4["c1"] and served["row_ms"] == 1.2 and c4["row_ms"] == 2.0
print("1. data-only: differs from glm_draft_trunc_cost.json only in row_ms (1.2 -> 2.0) and its note; c1 block equal")
table = json.load(open(os.environ["GLM_DRAFT_TRUNC_TABLE"]))
rng = random.Random(0)


def draft():
    return [max(-2.77, min(0.0, -abs(rng.gauss(0, 0.6)) * rng.choice([0.2, 1, 3]))) for _ in range(7)]


pa, pb = T.Policy(table, served), T.Policy(table, c4)
d1 = sum(pa.choose([c], 7) != pb.choose([c], 7) for c in (draft() for _ in range(20000)))
assert d1 == 0, f"c1 decisions differ on {d1} drafts"
print("2. c1 invariance: 0 of 20000 decisions differ")
ca, cb = [round(pa.cost(4, q), 2) for q in range(2, 9)], [round(pb.cost(4, q), 2) for q in range(2, 9)]
assert ca != cb and all(pa.cost(1, q) == pb.cost(1, q) for q in range(2, 9))
d4 = sum(pa.choose(cs, 7) != pb.choose(cs, 7) for cs in ([draft() for _ in range(4)] for _ in range(4000)))
lp = T._load_policy()
assert lp.row_ms == 2.0 and lp.t1 == pa.t1
print(f"3. n=4 plan costs served {ca} vs c4fit {cb}; n=4 choices differ on {d4}/4000; _load_policy row_ms {lp.row_ms}")
assert d4 > 0
print("ALL PASS")
