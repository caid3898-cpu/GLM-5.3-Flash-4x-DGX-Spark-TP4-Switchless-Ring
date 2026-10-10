#!/usr/bin/env python3
"""Long-context checks for the DSA indexer kpool fixes (GLM_KPOOL_FIX; diagnostics/glm-kpool-audit). Stdlib only.
Both tests only mean something above index_topk = 2048 tokens of context, which qeval / KLD never reach.

  registry  vllm#57477-style cached-prompt repro (upstream request-level repro by JaredforReal): a ~14k-token
            registry prompt with 7 lookups, cold; then N filler prompts (150-1500 tokens) at c16; then the same
            registry prompt again (prefix-cache hit) and a cache_salt control (recomputed). Scores exact lookups.
  recall    decode-built pools (dropped tail slot map, vllm#58454): S streams each generate a ~4k-token numbered
            code list at T=0; the next turn asks for 8 codes from that list; scored against what the stream itself
            emitted. Run once at c1 (streams one after another) and once at cS (all streams concurrently).

python3 kpool_longctx.py LABEL [registry] [recall] [--streams 4] [--fillers 100] [--base http://127.0.0.1:8093]
Writes ~/glm-kpool/LABEL-<test>.json and prints one SUMMARY line per test.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import time
import urllib.request

MODEL = "GLM-5.3-Flash-FP8"
SYL = ["ka", "lo", "mi", "ren", "tu", "va", "sho", "ber", "qui", "dan", "fel", "gor", "hin", "jor", "pex", "zam",
       "nol", "tri", "wex", "yul"]


def chat(base, messages, max_tokens, salt=None, timeout=1800):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
            "stream": False, "chat_template_kwargs": {"reasoning_effort": "low"}}
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    msg = r["choices"][0]["message"]
    text = msg.get("content") or ""
    if not text.strip():
        text = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return text, r.get("usage", {}), round(time.time() - t0, 1)


def metric(base, name):
    try:
        txt = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return None
    tot = 0.0
    for line in txt.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            try:
                tot += float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return tot


def word(rng):
    return "".join(rng.choice(SYL) for _ in range(3))


# ------------------------------------------------------------------------------------------------ registry
def registry(base, label, fillers, out):
    rng = random.Random(57477)
    ids = rng.sample(range(1000, 9999), 1150)
    reg = {f"ID-{i}": f"{word(rng)}-{rng.randint(100, 999)}" for i in ids}
    keys = list(reg)
    lines = "\n".join(f"{k}: {v}" for k, v in reg.items())
    pick = [keys[j] for j in (3, 90, 260, 480, 700, 905, 1120)]  # early / middle / late entries
    q = ("Below is a registry. Answer only from it.\n\n" + lines + "\n\nGive the values for these IDs, one per line "
         "exactly as ID=value, nothing else: " + ", ".join(pick))
    msgs = [{"role": "user", "content": q}]

    def score(text):
        got = dict(re.findall(r"(ID-\d{4})\s*[=:]\s*([a-z]+-\d{3})", text))
        return sum(1 for k in pick if got.get(k) == reg[k])

    hits0 = metric(base, "vllm:prefix_cache_hits_total")
    t_cold, u_cold, s_cold = chat(base, msgs, 300)
    frng = random.Random(9)

    def filler(i):
        n = frng.randint(150, 1500) if False else random.Random(1000 + i).randint(150, 1500)
        body = " ".join(word(random.Random(i * 7919 + j)) for j in range(n // 3))
        return chat(base, [{"role": "user", "content": f"Filler {i}. Summarise in five words: {body}"}], 8)[2]

    t0 = time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(filler, range(fillers)))
    t_fill = round(time.time() - t0, 1)
    hits1 = metric(base, "vllm:prefix_cache_hits_total")
    t_hot, u_hot, s_hot = chat(base, msgs, 300)
    hits2 = metric(base, "vllm:prefix_cache_hits_total")
    t_ctl, u_ctl, s_ctl = chat(base, msgs, 300, salt=f"ctl-{time.time()}")
    res = dict(test="registry", label=label, prompt_tokens=u_cold.get("prompt_tokens"), lookups=len(pick),
               cold=score(t_cold), after_fillers_cached=score(t_hot), salted_control=score(t_ctl),
               fillers=fillers, filler_s=t_fill,
               cache_hit_tokens_on_rehit=(hits2 - hits1) if hits1 is not None and hits2 is not None else None,
               texts={"cold": t_cold[-400:], "hot": t_hot[-400:], "ctl": t_ctl[-400:]})
    json.dump(res, open(out, "w"), indent=1)
    print("SUMMARY " + json.dumps({k: v for k, v in res.items() if k != "texts"}), flush=True)
    return res


# ------------------------------------------------------------------------------------------------ recall
THEMES = ["kitchen tools", "garden plants", "museum exhibits", "spare car parts", "library books",
          "ship cargo", "laboratory samples", "festival stalls"]
LINE_RE = re.compile(r"^\s*(\d{1,3})[.)]\s*(.+?)\s*[-–:]+\s*(?:code\s*)?([A-Z]{4}\d{3})\b", re.M)
ASK = [12, 47, 95, 131, 150, 177, 203, 241]


def stream(base, s, max_tokens, n_lines=320, ask=None):
    theme = THEMES[s % len(THEMES)]
    p1 = (f"Write a numbered inventory list of exactly {n_lines} {theme}. One line per item, format exactly: "
          f"'<n>. <item name> - code <FOUR CAPITAL LETTERS><THREE DIGITS>'. Invent every name and code, never "
          f"repeat a code. Output only the {n_lines} lines.")
    m1 = [{"role": "user", "content": p1}]
    t1, u1, s1 = chat(base, m1, max_tokens)
    table = {int(n): c for n, _, c in LINE_RE.findall(t1)}
    asked = [n for n in (ask or ASK) if n in table]
    m2 = m1 + [{"role": "assistant", "content": t1},
               {"role": "user", "content": "From your list above, give the codes of lines "
                + ", ".join(map(str, asked)) + ". Answer with one line per requested line, in the order asked, formatted as "
                "<line number>=<code>, for example 12=ABCD123. Nothing else."}]
    t2, u2, s2 = chat(base, m2, 200)
    codes = re.findall(r"([A-Z]{4}\d{3})", t2)
    got = {int(n): c for n, c in re.findall(r"(\d{1,3})\s*[=:]\s*([A-Z]{4}\d{3})", t2)}
    by_num = sum(1 for n in asked if got.get(n) == table[n])
    by_pos = sum(1 for i, n in enumerate(asked) if i < len(codes) and codes[i] == table[n])
    ok = max(by_num, by_pos)  # the model sometimes writes "1=CODE" / "n=CODE": then the order is the key
    return dict(stream=s, theme=theme, list_lines=len(table), gen_tokens=u1.get("completion_tokens"),
                turn2_prompt_tokens=u2.get("prompt_tokens"), asked=len(asked), correct=ok, by_num=by_num,
                by_pos=by_pos, s1=s1, s2=s2, answer=t2, expected={n: table[n] for n in asked})


def recall(base, label, streams, out, max_tokens=6000, reps=1, n_lines=320, ask=None, test="recall"):
    res = {"test": test, "label": label, "n_lines": n_lines}
    rows, rows4 = [], []
    for rep in range(reps):
        off = rep * streams
        rows += [stream(base, off + s, max_tokens, n_lines, ask) for s in range(streams)]
        with cf.ThreadPoolExecutor(streams) as ex:
            rows4 += list(ex.map(lambda s: stream(base, off + s, max_tokens, n_lines, ask), range(streams)))
    res["c1"] = rows
    res[f"c{streams}"] = rows4
    for key in ("c1", f"c{streams}"):
        r = res[key]
        res[key + "_score"] = f"{sum(x['correct'] for x in r)}/{sum(x['asked'] for x in r)}"
    json.dump(res, open(out, "w"), indent=1)
    print("SUMMARY " + json.dumps({k: v for k, v in res.items() if k.endswith("score")} |
                                  {"label": label, "test": test, "gen_tokens": [x["gen_tokens"] for x in rows]}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("tests", nargs="*", default=["registry", "recall"], help="registry | recall | control")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--fillers", type=int, default=100)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--base", default="http://127.0.0.1:8093")
    a = ap.parse_args()
    d = os.path.expanduser("~/glm-kpool")
    os.makedirs(d, exist_ok=True)
    if "registry" in a.tests:
        registry(a.base, a.label, a.fillers, f"{d}/{a.label}-registry.json")
    if "control" in a.tests:  # short context (< 2048 tokens total): validates the task and the scorer
        recall(a.base, a.label, a.streams, f"{d}/{a.label}-control.json", max_tokens=1200, reps=a.reps,
               n_lines=60, ask=[5, 12, 23, 31, 38, 44, 52, 58], test="control")
    if "recall" in a.tests:
        recall(a.base, a.label, a.streams, f"{d}/{a.label}-recall.json", reps=a.reps)


if __name__ == "__main__":
    main()
