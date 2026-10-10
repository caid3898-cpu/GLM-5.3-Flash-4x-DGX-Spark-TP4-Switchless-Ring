"""Acceptance and step-time probe for one drafter boot (run on the head, next to conc_bench.py).

python3 accept_probe.py LABEL [--base http://127.0.0.1:8093] [--model GLM-5.3-Flash-FP8]
        [--types prose,code,json] [--n 6] [--k 0,8,5,3] [--conc 1,4] [--max-tokens 512] [--effort low]
        [--temperature 0]

For every (type, k, conc) cell it sends --n distinct prompts (from conc_bench.PROSE/CODE/JSON, same
order every run) in waves of `conc`, each carrying {"vllm_xargs": {"spec_k": k}} (k=0 means "no
override": whatever the launch policy does). It needs spec_probe_scheduler.SpecProbeScheduler on the
server for the override; without it k is ignored and only the policy row is meaningful.

It separates the two things tok/s mixes (DS lesson 6):
  accept  = mean tokens per verify step (1 + accepted drafts / verify steps), from /metrics deltas
  pos     = per-position conditional acceptance from vllm:spec_decode_num_accepted_tokens_per_pos
  step_ms = decode wall time per verify step (c1: (last - first token) / steps of that request)
  tps     = per-stream decode tok/s, mean over the cell
From the k=K_max row alone, AL(k) for every smaller k is 1 + sum_{i<=k} P(pos i accepted) (greedy,
prefix-consistent drafts); the k-rows then confirm it and give step_ms(k).
Writes LABEL-accept.json. Metrics are global: run it with no other traffic on the endpoint.
"""
import argparse
import concurrent.futures
import json
import pathlib
import re
import threading
import time
import urllib.request

import conc_bench  # PROSE / CODE / JSON prompt lists (16 each)

PROMPTS = {"prose": conc_bench.PROSE, "code": conc_bench.CODE, "json": conc_bench.JSON}
METRIC_RE = re.compile(r'^vllm:(spec_decode_num_(?:accepted_tokens|draft_tokens|drafts)_total|'
                       r'spec_decode_num_accepted_tokens_per_pos_total)(\{[^}]*\})?\s+([0-9.e+]+)', re.M)


def scrape(base):
    txt = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    out = {}
    for name, labels, val in METRIC_RE.findall(txt):
        key = name
        if "per_pos" in name:
            m = re.search(r'position="(\d+)"', labels or "")
            key = f"pos{m.group(1)}" if m else name
        out[key] = out.get(key, 0.0) + float(val)
    return out


def one(base, model, prompt, k, max_tokens, effort, temperature, barrier):
    body = dict(model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature,
                max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
                chat_template_kwargs={"reasoning_effort": effort})
    if k > 0:
        body["vllm_xargs"] = {"spec_k": k}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait()
    t0 = time.monotonic(); first = last = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            p = line[6:].strip()
            if p == b"[DONE]":
                break
            e = json.loads(p); now = time.monotonic()
            if e.get("usage"):
                usage = e["usage"]
            for c in e.get("choices", []):
                d = c.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    first = first or now; last = now
    ct = usage["completion_tokens"] if usage else 0
    dec = (last - first) if (first and last and last > first) else None
    return dict(tokens=ct, decode_s=dec, ttft=first and first - t0)


def cell(a, kind, k, conc):
    prompts = [PROMPTS[kind][i % len(PROMPTS[kind])] for i in range(a.n)]
    m0 = scrape(a.base); rows = []
    for w in range(0, len(prompts), conc):
        wave = prompts[w:w + conc]
        barrier = threading.Barrier(len(wave)) if len(wave) > 1 else None
        with concurrent.futures.ThreadPoolExecutor(len(wave)) as ex:
            rows += list(ex.map(lambda p: one(a.base, a.model, p, k, a.max_tokens, a.effort, a.temperature, barrier), wave))
    m1 = scrape(a.base)
    d = {key: m1.get(key, 0) - m0.get(key, 0) for key in m1}
    steps = d.get("spec_decode_num_drafts_total", 0)
    acc = d.get("spec_decode_num_accepted_tokens_total", 0)
    pos = []
    for i in range(16):
        if f"pos{i}" not in d:
            break
        pos.append(d[f"pos{i}"])
    cond = [round(pos[i] / pos[i - 1], 3) if i and pos[i - 1] else (round(pos[0] / steps, 3) if steps else None)
            for i in range(len(pos))]
    tok = sum(r["tokens"] for r in rows); dec = [r for r in rows if r["decode_s"]]
    tps = [(r["tokens"] - 1) / r["decode_s"] for r in dec]
    res = dict(type=kind, k=k, conc=conc, n=len(rows), tokens=tok,
               accept=round(1 + acc / steps, 3) if steps else None,
               drafted_per_step=round(d.get("spec_decode_num_draft_tokens_total", 0) / steps, 2) if steps else None,
               pos_uncond=[round(p / steps, 3) for p in pos] if steps else None, pos_cond=cond,
               step_ms=round(1000 * sum(r["decode_s"] for r in dec) / (steps / conc), 2) if (steps and dec and conc == 1) else None,
               tps_mean=round(sum(tps) / len(tps), 1) if tps else None, tps_min=round(min(tps), 1) if tps else None)
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label"); ap.add_argument("--base", default="http://127.0.0.1:8093")
    ap.add_argument("--model", default="GLM-5.3-Flash-FP8"); ap.add_argument("--types", default="prose,code,json")
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--k", default="0")
    ap.add_argument("--conc", default="1"); ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--effort", default="low"); ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    out = [cell(a, t, int(k), int(c)) for c in a.conc.split(",") for t in a.types.split(",") for k in a.k.split(",")]
    pathlib.Path(f"{a.label}-accept.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
