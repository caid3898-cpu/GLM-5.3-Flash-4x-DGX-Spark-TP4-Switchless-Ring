#!/usr/bin/env python3
"""vLLM lifetime token accumulator.

Polls vLLM /metrics every 5s, accumulates prompt/generation/cached token counters
into a persistent file, and serves them as Prometheus gauges on :9101.
Survives vLLM restarts (detects counter resets and keeps the running total).
"""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import urlopen

VLLM = "http://127.0.0.1:8093/metrics"
STATE = Path("/home/dada/monitoring/totals.json")
PORT = 9101
POLL_S = 5

COUNTERS = {
    "prompt": "vllm:prompt_tokens_total",
    "generation": "vllm:generation_tokens_total",
    "cached": "vllm:prompt_tokens_cached_total",
}

_pat = {k: re.compile(r"^" + re.escape(v) + r'\{[^}]*\}\s+([0-9.eE+-]+)', re.M)
        for k, v in COUNTERS.items()}

lock = threading.Lock()
totals = {"prompt": 0.0, "generation": 0.0, "cached": 0.0}
last_seen = {}


def load():
    global totals
    try:
        d = json.loads(STATE.read_text())
        for k in totals:
            totals[k] = float(d.get(k, 0.0))
    except Exception:
        pass


def save():
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(totals))
    tmp.replace(STATE)


def poll():
    global last_seen
    while True:
        try:
            text = urlopen(VLLM, timeout=4).read().decode("utf-8", "replace")
            with lock:
                for k, pat in _pat.items():
                    m = pat.search(text)
                    if not m:
                        continue
                    cur = float(m.group(1))
                    prev = last_seen.get(k)
                    if prev is None:
                        # 首次见到：把当前值当基线（本期启动以来已发生的）
                        totals[k] += cur
                    elif cur < prev:
                        # vLLM 重启了：计数归零，新值即本周期总量
                        totals[k] += cur
                    else:
                        totals[k] += cur - prev
                    last_seen[k] = cur
                save()
        except Exception:
            pass
        time.sleep(POLL_S)


def metrics_text():
    with lock:
        lines = []
        for k, v in totals.items():
            lines.append(f"# TYPE glm_lifetime_{k}_tokens gauge")
            lines.append(f"glm_lifetime_{k}_tokens {v:.0f}")
        return "\n".join(lines) + "\n"


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = metrics_text().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    load()
    threading.Thread(target=poll, daemon=True).start()
    print(f"serving lifetime totals on :{PORT}, state={STATE}")
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()
