#!/usr/bin/env python3
"""CPU test: GLM_DRAFT_TRUNC (overlay/glm_draft_trunc.py) composed with GLM_GUMBEL_COUPLED (overlay/glm_gumbel_coupled.py).

  uv run --no-project --with torch --with numpy python tests/test_trunc_gumbel_compose.py

What the final stack does per decode step at T > 0 (c = 1..4 requests, every request scheduled with 7 drafts):
  1. draft    the coupled walk proposes d_1..d_7 with the verify rows' noise keys (pos = sample_pos - 1);
  2. decide   glm_draft_trunc reads the per-depth confidences and truncate() cuts EVERY request's draft list to one
              uniform L (in place on a SchedulerOutput-like object: spec lists, num_scheduled_tokens, total);
  3. verify   L + 1 rows per request; t_i = argmax(processed_i + g(seed, P_i, .)) on every row (stock gumbel_sample);
              glm_target_argmax.greedy_verify commits t_0 .. t_n (n = leading drafts equal to t).
Claim: the committed stream of every request equals plain (non-speculative) sampling with the same seed, token for
token, whatever rule picks L (the real policy, a random one, and an adversarial one that looks at the noise). The
shortened scheduler output must stay self-consistent (rows = L + 1 per request = num_scheduled_tokens).

Toy target / drafter / noise are the ones of tests/test_glm_gumbel_coupled.py (V = 12, T = 0.7, top-p 0.9, numpy port
of the image's Philox noise).
"""
import json
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OVL = os.path.join(HERE, "..", "overlay")
sys.path.insert(0, OVL)
sys.path.insert(0, HERE)
import test_glm_gumbel_coupled as G  # noqa: E402  (toy target, drafter, noise; its checks do not run on import)
import glm_draft_trunc as tr  # noqa: E402
import glm_target_argmax as ta  # noqa: E402
import glm_gumbel_coupled as gc  # noqa: E402

KMAX = 7
FAIL = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def confidences(seq, seed, k, temp):
    """Per-depth log max softmax of the toy drafter's scores along its own path (what trunc reads from
    _selector_scores): uses the same fake lattice as G.drafter."""
    out, ctx = [], list(seq)
    d = G.drafter("coupled", seq, seed, k, temp)
    for j in range(k):
        q = G.target_logits(ctx) + torch.from_numpy(np.random.default_rng(len(ctx) * 7 + seed % 97).normal(0, 0.8, G.V))
        out.append(float(torch.log_softmax(q / max(temp, 1e-6) if temp else q, -1).max()))
        ctx.append(max(d[j], 0))
    return d, out


def make_so(reqs):
    return types.SimpleNamespace(
        scheduled_spec_decode_tokens={r: list(x["draft"]) for r, x in reqs.items()},
        num_scheduled_tokens={r: len(x["draft"]) + 1 for r, x in reqs.items()},
        total_num_scheduled_tokens=sum(len(x["draft"]) + 1 for x in reqs.values()),
        has_structured_output_requests=False)


def run_batch(seeds, prompts, n, rule, policy, temp=G.TEMP, stats=None):
    """Decode len(seeds) requests together; every step: 7 coupled drafts, one uniform L (rule), coupled verify."""
    seqs = {i: list(p) for i, p in enumerate(prompts)}
    goal = {i: len(p) + n for i, p in enumerate(prompts)}
    rng = np.random.default_rng((abs(seeds[0]) + 7919 * len(seeds)) % 2 ** 32)
    while any(len(seqs[i]) < goal[i] for i in seqs):
        live = [i for i in seqs if len(seqs[i]) < goal[i]]
        reqs, confs = {}, {}
        for i in live:
            d, c = confidences(seqs[i], seeds[i], KMAX, temp)
            reqs[i] = {"draft": d}
            confs[i] = c
        so = make_so(reqs)
        if rule == "policy":
            L = tr.truncate(so, confs, policy)
        else:
            if rule == "random":
                L = int(rng.integers(1, KMAX + 1))
            elif rule == "adversarial":
                # looks at the noise: stop right before the first draft the target would ACCEPT (worst case for
                # an unbiased-looking stream if the rule could leak into the output)
                i0 = live[0]
                s, d = seqs[i0], reqs[i0]["draft"]
                L = KMAX
                for j in range(KMAX):
                    if G.draw(s + [max(x, 0) for x in d[:j]], seeds[i0], len(s) - 1 + j, temp) == d[j]:
                        L = max(1, j)
                        break
            pol = types.SimpleNamespace(choose=lambda cs, k, _L=L: min(_L, k))
            L = tr.truncate(so, confs, pol)
        if L is None:
            raise AssertionError("truncate() skipped a plain decode batch")
        # the shortened scheduler output must be self-consistent
        for i in live:
            if not (len(so.scheduled_spec_decode_tokens[i]) == min(L, KMAX)
                    and so.num_scheduled_tokens[i] == len(so.scheduled_spec_decode_tokens[i]) + 1):
                raise AssertionError(f"inconsistent truncated scheduler output {so}")
        if so.total_num_scheduled_tokens != sum(so.num_scheduled_tokens.values()):
            raise AssertionError("total_num_scheduled_tokens out of sync")
        # verify: rows per request = its (truncated) drafts + 1, concatenated like the runner's logits rows
        targets, draft_col, cu, refs = [], [], [0], []
        for i in live:
            s, d = seqs[i], so.scheduled_spec_decode_tokens[i]
            ctx = list(s)
            for j in range(len(d) + 1):
                if j > 0:
                    ctx = ctx + [max(d[j - 1], 0)]
                targets.append(G.draw(ctx, seeds[i], len(ctx) - 1, temp))
            refs.extend(targets[-(len(d) + 1):])
            draft_col.extend([s[-1]] + list(d))
            cu.append(cu[-1] + len(d) + 1)
        t = torch.tensor(targets, dtype=torch.int64)
        dc = torch.tensor(draft_col, dtype=torch.int64)
        cut = torch.tensor(cu, dtype=torch.int64)
        sampled, ns = ta.greedy_verify_torch(t, dc, cut, len(live), KMAX + 1)
        res = gc.compare(torch.tensor(refs, dtype=torch.int64), t, dc, cut, sampled, ns,
                         torch.tensor([temp > 0] * len(targets)))
        if stats is not None:
            stats["bad"] += res["bad"]
            stats["steps"] += 1
            stats["L"][min(L, KMAX)] += 1
            stats["acc"] += int(ns.sum()) - len(live)
        for r, i in enumerate(live):
            seqs[i].extend(int(x) for x in sampled[r, :int(ns[r])])
    return {i: seqs[i][:goal[i]] for i in seqs}


def main():
    table = json.load(open(os.path.join(OVL, "glm_bav_table_seg.json")))
    cost = json.load(open(os.path.join(OVL, "glm_draft_trunc_cost.json")))
    policy = tr.Policy(table, cost)
    for temp in (G.TEMP, 1.0, 0.0):
        for rule in ("policy", "random", "adversarial"):
            n_seq = bad_seq = 0
            st = {"bad": 0, "steps": 0, "L": [0] * (KMAX + 1), "acc": 0}
            for c in (1, 2, 4):
                for base in (1, 17, 2 ** 40 + 3):
                    seeds = [base + 101 * i for i in range(c)]
                    prompts = [[3], [1, 4, 1, 5], [2, 7], [9]][:c]
                    got = run_batch(seeds, prompts, 20, rule, policy, temp, st)
                    for i in range(c):
                        ref = G.decode_plain(seeds[i], prompts[i], 20, temp) if temp else \
                            G.decode_plain(seeds[i], prompts[i], 20, 0.0)
                        n_seq += 1
                        bad_seq += got[i] != ref
            check(bad_seq == 0 and st["bad"] == 0,
                  f"T={temp} rule={rule:11s}: truncated coupled speculation == plain sampling "
                  f"({n_seq - bad_seq}/{n_seq} sequences, {st['steps']} steps, compare() mismatches {st['bad']}, "
                  f"L hist {st['L'][1:]}, accepted drafts {st['acc']})")
    # the real policy does truncate here (the test is not vacuous) and the multi-request cut is uniform
    so = make_so({"a": {"draft": [5] * 7}, "b": {"draft": [6] * 7}})
    L = tr.truncate(so, {"a": [-2.6] * 7, "b": [-2.6] * 7}, policy)
    check(L is not None and L < 7 and all(len(v) == L for v in so.scheduled_spec_decode_tokens.values())
          and so.total_num_scheduled_tokens == 2 * (L + 1),
          f"policy: unsure drafts at c=2 are cut to one uniform L={L}, totals consistent")
    print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
