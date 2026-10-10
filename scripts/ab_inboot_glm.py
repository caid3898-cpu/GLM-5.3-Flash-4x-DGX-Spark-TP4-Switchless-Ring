#!/usr/bin/env python3
"""Drive an in-boot A/B of GLM adapter flags (overlay/glm_ab.py; engine booted with GLM_AB_VARIANTS >= 2 and
VLLM_SERVER_DEV_MODE=1). Run on the head node. Stdlib only. TEST ONLY.

Port of the DS4.1 driver (ds41 scripts/ab_inboot.py, ours) to vLLM:
  * the switch is POST /collective_rpc glm_ab_switch: the engine core refuses it while any request is unfinished
    ({"busy": n}, retried here), otherwise every TP rank switches at the same point of the step queue and
    answers with its variant, sequence number, config hash and a MAX all-reduce agreement check;
  * warm pass: every variant runs every prompt once at the full --max-tokens (JIT of variant-only eager kernels,
    prefix cache, adaptive-draft state), off the clock;
  * rounds: ABBA order by default (round r reversed when r is odd), greedy c=1, streamed: TTFT = first streamed
    token, decode = total - TTFT, steps = the delta of vllm:spec_decode_num_drafts (one draft per step at c=1),
    step_ms = decode / steps, tok/s = (tokens - 1) / decode;
  * per request, vllm:request_success must move by exactly 1 (else another client was running: the row is
    marked foreign and left out of the timing statistics);
  * per block, the target graph replays of every rank must come from the block's own set only.
Reports per variant median step_ms / tok/s / tokens-per-step and, for every variant vs v0, the per-round mean of
the paired (same prompt) differences with a 95 % bootstrap CI over rounds.
Exactness is read at tensor level, not from text: greedy output on this stack is not repeatable even within one
boot (adaptive draft length changes the verify shape; diagnostics/glm-verify-cut measured 1 of 8 prompts
identical T=0 vs T=0). A variant with GLM_TARGET_VOCAB_ARGMAX=check runs the vocab-parallel path AND the stock
sampler on the same logits every eligible step, on every rank; the driver sums the compared steps and the
mismatches per block. Identical-text fractions are still reported, as information (the A/A pair is the baseline).
FAIL (exit 1): rank disagreement, a block replaying another set, no valid rows, any argmax check mismatch, or
(with --require-identical) a greedy text that differs between variants.

--conc N (N >= 2; added for the 2026-09-28 speed screen): every row is a BATCH of N concurrent greedy requests
(the prompts cycled into batches of N) instead of one request. steps = the target CUDA-graph replays of rank 0
during the batch (glm_ab_status before / after: one replay per decode/verify step; the eager prefill steps are not
counted; falls back to drafts / N, marked in the row, if the counter did not move); step_ms = (last finish - the
slowest first token) / steps; tps = aggregate tokens / (first send to last finish); tok_per_step = tokens / steps.
Pairing, rounds, CI and replay checks are unchanged. (vllm:iteration_tokens_total is not exported by this image.)

usage: python3 ab_inboot_glm.py LABEL [--rounds 6] [--drop-rounds 1] [--max-tokens 256] [--prompts builtin|FILE]
                                 [--conc N] [--abab] [--require-identical] [--base URL]
Raw rows: ~/glm-inboot/LABEL-<time>.jsonl; summary appended to ~/glm-inboot/summary.jsonl.
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bench"))


def builtin_prompts():
    import conc_bench as cb
    return [("prose", p) for p in cb.PROSE[:3]] + [("code", p) for p in cb.CODE[:3]]


def load_prompts(spec):
    if spec == "builtin":
        return builtin_prompts()
    out = []
    for line in open(spec):
        line = line.strip()
        if line:
            rec = json.loads(line)
            out.append((rec.get("kind", "file"), rec["prompt"]))
    return out


class Engine:
    def __init__(self, base, model, effort, switch_timeout):
        self.base, self.model, self.effort, self.switch_timeout = base, model, effort, switch_timeout

    def _post(self, path, body, timeout=900):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    def rpc(self, method, args=()):
        r = json.load(self._post("/collective_rpc", {"method": method, "args": [str(a) for a in args]}, timeout=120))
        return r.get("results") or []

    def switch(self, variant):
        token = f"{os.getpid()}-{time.monotonic_ns()}"
        deadline = time.time() + self.switch_timeout
        waited = 0
        while True:
            res = self.rpc("glm_ab_switch", (variant, token))
            if res and isinstance(res[0], dict) and "busy" in res[0]:
                if time.time() > deadline:
                    raise RuntimeError(f"switch to v{variant}: engine still busy ({res[0]['busy']} unfinished) "
                                       f"after {self.switch_timeout} s")
                waited += 1
                time.sleep(2)
                continue
            break
        if waited:
            print(f"  (switch to v{variant} waited {2 * waited} s for other requests to finish)", flush=True)
        bad = [r for r in res if not isinstance(r, dict) or r.get("error") or r.get("variant") != variant
               or r.get("token") != token]
        if bad or not res:
            raise RuntimeError(f"switch to v{variant} failed: {res}")
        if len({r["config"] for r in res}) != 1 or len({r["seq"] for r in res}) != 1:
            raise RuntimeError(f"ranks disagree on config / seq after the switch: {res}")
        for r in res:
            ranks = r.get("ranks")
            if ranks is not None and not ranks.get("agree"):
                raise RuntimeError(f"ranks disagree after switching to v{variant}: {ranks}")
        return res

    def status(self):
        return self.rpc("glm_ab_status")

    def metrics(self):
        text = urllib.request.urlopen(self.base + "/metrics", timeout=30).read().decode()
        out = {"drafts": 0.0, "accepted": 0.0, "success": 0.0, "running": 0.0}
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            m = re.match(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
            if not m:
                continue
            name, value = m.group(1), float(m.group(3))
            if name == "vllm:spec_decode_num_drafts_total":
                out["drafts"] += value
            elif name == "vllm:spec_decode_num_accepted_tokens_total":
                out["accepted"] += value
            elif name == "vllm:request_success_total":
                out["success"] += value
            elif name == "vllm:num_requests_running":
                out["running"] += value
        return out

    def chat(self, prompt, max_tokens):
        body = dict(model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
                    temperature=0, stream=True, stream_options={"include_usage": True},
                    chat_template_kwargs={"reasoning_effort": self.effort})
        t0 = time.monotonic()
        first = None
        usage = None
        parts = {"reasoning": [], "content": []}
        with self._post("/v1/chat/completions", body) as r:
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                p = line[6:].strip()
                if p == b"[DONE]":
                    break
                e = json.loads(p)
                if e.get("usage"):
                    usage = e["usage"]
                for ch in e.get("choices") or []:
                    d = ch.get("delta") or {}
                    rc = d.get("reasoning_content") or d.get("reasoning")
                    c = d.get("content")
                    if (rc or c) and first is None:
                        first = time.monotonic()
                    if rc:
                        parts["reasoning"].append(rc)
                    if c:
                        parts["content"].append(c)
        total = time.monotonic() - t0
        text = "".join(parts["reasoning"]) + "\x00" + "".join(parts["content"])
        return {"ttft": (first or time.monotonic()) - t0, "total": total,
                "tokens": (usage or {}).get("completion_tokens"), "text": text}


def run_one(eng, p, max_tokens):
    """One request (p a string) or a batch of concurrent requests (p a list): returns the chat() fields, for a
    batch total = first send to last finish, ttft = the slowest first token, tokens summed, texts joined."""
    if not isinstance(p, list):
        return eng.chat(p, max_tokens)
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.monotonic()

    def one(q):
        r = eng.chat(q, max_tokens)
        r["end"] = time.monotonic()
        return r

    with ThreadPoolExecutor(len(p)) as ex:
        rs = list(ex.map(one, p))
    return {"ttft": max(r["ttft"] for r in rs), "total": max(r["end"] for r in rs) - t0,
            "tokens": sum(r["tokens"] or 0 for r in rs), "text": "\x01".join(r["text"] for r in rs)}


def replay_delta(before, after):
    """Per rank: target replays per set (index N = shared) between two status snapshots."""
    out = []
    for a, b in zip(before, after):
        ra = (a.get("replays") or {}).get("target") or []
        rb = (b.get("replays") or {}).get("target") or []
        if len(ra) < len(rb):
            ra = ra + [0] * (len(rb) - len(ra))
        out.append([y - x for x, y in zip(ra, rb)])
    return out


def check_replays(delta, v, nvar):
    bad = []
    for rank, d in enumerate(delta):
        if not d or d[v] <= 0:
            bad.append(f"rank {rank}: no target replays from set {v} ({d})")
        elif any(n for j, n in enumerate(d[:nvar]) if j != v):
            bad.append(f"rank {rank}: target replayed another set: {d}")
    if len({tuple(d) for d in delta}) > 1:
        bad.append(f"ranks replayed different counts: {delta}")
    return bad


def bootstrap_ci(values, n=5000, seed=12345):
    rng = random.Random(seed)
    k = len(values)
    vals = sorted(statistics.mean([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def _ok(x):
    return x.get("steps") and x.get("step_ms") is not None and not x.get("foreign")


def summarize(rows, nvar):
    out = {"variants": {}, "vs_v0": {}}
    for v in range(nvar):
        r = [x for x in rows if x["variant"] == v and _ok(x)]
        if not r:
            continue
        kinds = sorted({x["kind"] for x in r})
        out["variants"][v] = {
            "n": len(r),
            "step_ms_median": round(statistics.median(x["step_ms"] for x in r), 3),
            "tps_median": round(statistics.median(x["tps"] for x in r), 2),
            "tok_per_step_mean": round(statistics.mean(x["tok_per_step"] for x in r), 4),
            "by_kind": {k: {"step_ms_median": round(statistics.median(x["step_ms"] for x in r if x["kind"] == k), 3),
                            "tps_median": round(statistics.median(x["tps"] for x in r if x["kind"] == k), 2)}
                        for k in kinds},
        }
    base_all = {(x["round"], x["prompt"]): x for x in rows if x["variant"] == 0}
    base = {k: x for k, x in base_all.items() if _ok(x)}
    for v in range(1, nvar):
        res = {}
        allpairs = [(base_all[(x["round"], x["prompt"])], x) for x in rows
                    if x["variant"] == v and (x["round"], x["prompt"]) in base_all]
        same = sum(a["text_sha"] == b["text_sha"] for a, b in allpairs)
        res["identical_outputs"] = f"{same}/{len(allpairs)}"
        res["output_diffs"] = len(allpairs) - same
        pairs = [(base[(x["round"], x["prompt"])], x) for x in rows
                 if x["variant"] == v and _ok(x) and (x["round"], x["prompt"]) in base]
        rounds = sorted({b["round"] for _, b in pairs})
        res["pairs"], res["rounds"] = len(pairs), len(rounds)
        if len(rounds) < 2:
            res["error"] = "fewer than 2 rounds"
            out["vs_v0"][v] = res
            continue
        for metric in ("step_ms", "tps"):
            per_round = [statistics.mean(b[metric] - a[metric] for a, b in pairs if b["round"] == r) for r in rounds]
            rel_round = [statistics.mean((b[metric] - a[metric]) / a[metric] * 100 for a, b in pairs
                                         if b["round"] == r) for r in rounds]
            lo, hi = bootstrap_ci(per_round)
            rlo, rhi = bootstrap_ci(rel_round)
            res[metric] = {"mean_diff": round(statistics.mean(per_round), 4), "ci95_rounds": [round(lo, 4), round(hi, 4)],
                           "per_round": [round(x, 4) for x in per_round],
                           "median_pair_diff": round(statistics.median(b[metric] - a[metric] for a, b in pairs), 4),
                           "mean_rel_pct": round(statistics.mean(rel_round), 3),
                           "rel_ci95_rounds": [round(rlo, 3), round(rhi, 3)]}
        res["tok_per_step_diff"] = round(statistics.mean(b["tok_per_step"] - a["tok_per_step"] for a, b in pairs), 4)
        out["vs_v0"][v] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--drop-rounds", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompts", default="builtin")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--conc", type=int, default=1, help="requests per row, sent concurrently (default 1)")
    ap.add_argument("--abab", action="store_true")
    ap.add_argument("--require-identical", action="store_true",
                    help="FAIL on any greedy text difference between variants (not meaningful on GLM KSN)")
    ap.add_argument("--base", default="http://127.0.0.1:8093")
    ap.add_argument("--model", default="GLM-5.3-Flash-FP8")
    ap.add_argument("--switch-timeout", type=float, default=300.0)
    args = ap.parse_args()

    eng = Engine(args.base, args.model, args.effort, args.switch_timeout)
    prompts = load_prompts(args.prompts)
    conc = max(1, args.conc)
    if conc > 1:   # cycle the prompts into batches of `conc`; kind = the batch's kinds
        nb = max(1, -(-len(prompts) // conc))
        flat = [prompts[i % len(prompts)] for i in range(nb * conc)]
        prompts = [("+".join(sorted({k for k, _ in flat[b * conc:(b + 1) * conc]})) + f"@c{conc}",
                    [p for _, p in flat[b * conc:(b + 1) * conc]]) for b in range(nb)]
    outdir = os.path.expanduser("~/glm-inboot")
    os.makedirs(outdir, exist_ok=True)
    raw_path = os.path.join(outdir, f"{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")

    st = eng.status()
    if not st or not all(isinstance(s, dict) and s.get("armed") for s in st):
        sys.exit(f"harness not armed on every rank: {st}")
    nvar = int(st[0]["variants"])
    config = st[0]["config"]
    print(f"armed: {nvar} variants, config {config}, {len(st)} ranks, {len(prompts)} prompts, rounds {args.rounds}",
          flush=True)
    for v in range(nvar):
        print(f"  v{v}: {eng.switch(v)[0]['effective']}", flush=True)

    rows = []
    argmax_blocks = []
    t_start = time.time()
    with open(raw_path, "w") as raw:
        raw.write(json.dumps({"meta": {"label": args.label, "config": config, "variants": nvar, "status": st,
                                       "args": vars(args)}}) + "\n")
        for v in range(nvar):                    # warm pass, every variant, off the clock
            eng.switch(v)
            for _, p in prompts:
                run_one(eng, p, args.max_tokens)
        print(f"warm pass done in {time.time() - t_start:.0f} s", flush=True)
        try:
            for rnd in range(args.rounds):
                order = list(range(nvar))
                if not args.abab and rnd % 2:
                    order.reverse()
                for v in order:
                    eng.switch(v)
                    before = eng.status()
                    for i, (kind, p) in enumerate(prompts):
                        nreq = len(p) if isinstance(p, list) else 1
                        s0 = eng.status() if nreq > 1 else None
                        m0 = eng.metrics()
                        r = run_one(eng, p, args.max_tokens)
                        time.sleep(0.05)
                        m1 = eng.metrics()
                        steps = int(round(m1["drafts"] - m0["drafts"]))
                        dec = r["total"] - r["ttft"]
                        if nreq > 1:
                            d = replay_delta(s0, eng.status())
                            rep = sum(d[0]) if d else 0
                            steps = rep if rep > 0 else int(round(steps / nreq))
                            r["steps_src"] = "replays" if rep > 0 else "drafts/n"
                        toks = r["tokens"] or 0
                        row = {"round": rnd, "variant": v, "prompt": i, "kind": kind, "ttft": round(r["ttft"], 4),
                               "decode_s": round(dec, 4), "tokens": toks, "steps": steps,
                               "accepted": int(round(m1["accepted"] - m0["accepted"])),
                               "foreign": int(round(m1["success"] - m0["success"])) != nreq,
                               "step_ms": round(dec / steps * 1000, 3) if steps > 0 and dec > 0 else None,
                               "tps": (round((toks - 1) / dec, 2) if dec > 0 and toks > 1 else None) if nreq == 1 else
                                      (round(toks / r["total"], 2) if r["total"] > 0 and toks else None),
                               "tok_per_step": round((toks - nreq) / steps, 4) if steps > 0 else None,
                               "conc": nreq, "steps_src": r.get("steps_src", "drafts"),
                               "text_sha": hashlib.sha1(r["text"].encode()).hexdigest()[:12],
                               "t": round(time.time(), 2), "text": r["text"]}
                        rows.append(row)
                        raw.write(json.dumps(row) + "\n")
                    after = eng.status()
                    delta = replay_delta(before, after)
                    am = {k: sum((a.get("argmax") or {}).get(k, 0) - (b.get("argmax") or {}).get(k, 0)
                                 for a, b in zip(after, before)) for k in ("fast", "full", "checked", "mismatch")}
                    argmax_blocks.append({"round": rnd, "variant": v, **am})
                    bad = check_replays(delta, v, nvar)
                    blk = [x for x in rows if x["round"] == rnd and x["variant"] == v and _ok(x)]
                    stats = (f"step_ms {statistics.median(x['step_ms'] for x in blk):.2f} "
                             f"tps {statistics.median(x['tps'] for x in blk):.1f}" if blk else "NO VALID ROWS")
                    nforeign = sum(1 for x in rows if x["round"] == rnd and x["variant"] == v and x["foreign"])
                    print(f"round {rnd} v{v}: {stats} replays(rank0) {delta[0] if delta else None}"
                          + (f" argmax checked {am['checked']} mismatch {am['mismatch']}" if am["checked"] else "")
                          + (f" foreign {nforeign}" if nforeign else "") + (f"  FAIL {bad}" if bad else ""), flush=True)
                    raw.write(json.dumps({"round": rnd, "variant": v, "replays": delta, "problems": bad}) + "\n")
                    if bad:
                        raise RuntimeError(f"round {rnd} v{v}: {bad}")
        finally:
            try:
                eng.switch(0)
            except RuntimeError as exc:
                print(f"could not restore variant 0: {exc}", flush=True)

    kept = [r for r in rows if r["round"] >= args.drop_rounds]
    summary = {"label": args.label, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "raw": raw_path, "config": config,
               "conc": conc,
               "rounds": args.rounds, "dropped_rounds": args.drop_rounds, "max_tokens": args.max_tokens,
               "prompts": args.prompts, "abba": not args.abab, "foreign_rows": sum(r["foreign"] for r in rows),
               "wall_s": round(time.time() - t_start, 1), **summarize(kept, nvar)}
    diffs_all = sum(r.get("output_diffs", 0) for r in summarize(rows, nvar)["vs_v0"].values())
    summary["output_diffs_all_rounds"] = diffs_all
    summary["argmax_check"] = {v: {k: sum(b[k] for b in argmax_blocks if b["variant"] == v)
                                   for k in ("fast", "full", "checked", "mismatch")} for v in range(nvar)}
    mismatches = sum(x["mismatch"] for x in summary["argmax_check"].values())
    no_rows = any(v not in summary["variants"] for v in range(nvar))
    failed = (diffs_all and args.require_identical) or no_rows or mismatches or \
        any("error" in r for r in summary["vs_v0"].values())
    summary["verdict"] = "FAIL" if failed else "ok"
    print(json.dumps(summary, indent=1), flush=True)
    with open(os.path.join(outdir, "summary.jsonl"), "a") as f:
        f.write(json.dumps(summary) + "\n")
    if failed:
        sys.exit("FAIL")


if __name__ == "__main__":
    main()
