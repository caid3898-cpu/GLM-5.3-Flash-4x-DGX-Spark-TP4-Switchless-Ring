#!/usr/bin/env python3
"""Helpers for run_rigmark.sh (stdlib only; runs on Spark_01 or the Mac).

  preflight --base URL [--wait S]         health, model id from /v1/models, idle check
  meta --base URL --preflight F --out F   RigMark appliance metadata, built from docker inspect
  ab-pin / ab-check / ab-restore          pin one in-boot glm_ab variant for a run and prove it held
  summary RECEIPT.json [...]              one line per workload + c4 aggregate
  scrub FILE [...]                        fail if an address, path or credential leaked

Nothing here writes endpoint URLs, IPs, home paths or credentials into metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SECRET_KEY = re.compile(r"TOKEN|KEY|SECRET|PASS|AUTH|COOKIE|CREDENTIAL", re.I)
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Flags that carry addresses/ports of this deployment, never published.
DROP_FLAGS = {"--master-addr", "--master-port", "--host", "--port", "--node-rank"}
ENV_PREFIXES = ("GLM_", "VLLM_ADAPTIVE_K_", "B12X_ROCE_", "NCCL_", "QMIX_")
ENV_EXACT = {"VLLM_DISABLED_KERNELS", "VLLM_TEST_FORCE_FP8_MARLIN", "PYTORCH_CUDA_ALLOC_CONF",
             "VLLM_BUILD_COMMIT", "VLLM_IMAGE_TAG"}


def get(base: str, path: str, timeout: float = 10.0):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
        return r.status, r.read()


def queue_depth(base: str) -> float:
    _, body = get(base, "/metrics")
    total = 0.0
    for line in body.decode().splitlines():
        if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
            total += float(line.rsplit(" ", 1)[1])
    return total


def preflight(a) -> None:
    base = a.base.rstrip("/").removesuffix("/v1")
    status, _ = get(base, "/health")
    if status != 200:
        sys.exit(f"preflight: /health returned {status}")
    _, body = get(base, "/v1/models")
    data = json.loads(body).get("data") or []
    if len(data) != 1:
        sys.exit(f"preflight: expected exactly one model, got {len(data)}")
    model = data[0]
    deadline = time.time() + a.wait
    depth = queue_depth(base)
    while depth > 0 and time.time() < deadline:
        time.sleep(5)
        depth = queue_depth(base)
    json.dump({"model": model["id"], "max_model_len": model.get("max_model_len"),
               "queue_depth_at_start": depth, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
              sys.stdout)
    print()


def docker_json(*args: str):
    return json.loads(subprocess.check_output(["docker", *args], text=True))


def head_container(port: str):
    names = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).split()
    for name in names:
        info = docker_json("inspect", name)[0]
        args = info.get("Args") or []
        flags = dict(zip(args, args[1:]))
        if flags.get("--port") == port and flags.get("--node-rank", "0") == "0":
            return info
    return None


def flag_value(args: list[str], flag: str):
    return args[args.index(flag) + 1] if flag in args else None


def public_flags(args: list[str]) -> list[str]:
    out, skip = [], False
    for i, tok in enumerate(args):
        if skip:
            skip = False
            continue
        if tok in DROP_FLAGS:
            skip = True
            continue
        out.append(IPV4.sub("<ip>", tok))
    return out


def public_env(env: list[str]) -> dict[str, str]:
    out = {}
    for item in env:
        key, _, value = item.partition("=")
        if SECRET_KEY.search(key):
            continue
        if not (key.startswith(ENV_PREFIXES) or key in ENV_EXACT):
            continue
        if "/" in value:
            value = "<path>"
        out[key] = IPV4.sub("<ip>", value)
    return dict(sorted(out.items()))


def manifest_sha(directory: Path, small: tuple[str, ...]) -> str:
    """Fingerprint of a weight directory: small metadata files by content, shards by name+size."""
    h = hashlib.sha256()
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        h.update(p.name.encode() + b"\0")
        if p.name in small or p.suffix in (".json", ".jinja"):
            h.update(p.read_bytes())
        else:
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def tree_sha(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(directory)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "unknown"


def read(p: str) -> str:
    try:
        return Path(p).read_text().strip()
    except OSError:
        return "unknown"


def meta(a) -> None:
    pre = json.loads(Path(a.preflight).read_text())
    info = head_container(a.port)
    if info is None:
        sys.exit(f"meta: no running container serving --port {a.port} as node-rank 0")
    args = info.get("Args") or []
    mounts = {m["Destination"]: Path(m["Source"]) for m in info.get("Mounts", [])}
    image_id = info.get("Image", "unknown")
    image_tag = info.get("Config", {}).get("Image", "unknown")
    spec = json.loads(flag_value(args, "--speculative-config") or "{}")
    comp = json.loads(flag_value(args, "--compilation-config") or "{}")
    env = public_env(info.get("Config", {}).get("Env") or [])

    model_dir, draft_dir, deploy_dir = mounts.get("/model"), mounts.get("/draft"), mounts.get("/overlay")
    model_fp = manifest_sha(model_dir, ()) if model_dir and model_dir.is_dir() else "unknown"
    draft_fp = manifest_sha(draft_dir, ()) if draft_dir and draft_dir.is_dir() else "unknown"
    overlay_fp = (tree_sha(deploy_dir / "overlay")
                  if deploy_dir and (deploy_dir / "overlay").is_dir() else "unknown")
    profile_fp = (file_sha(deploy_dir / "profiles" / "current.env")
                  if deploy_dir else "unknown")
    template_fp = file_sha(model_dir / "chat_template.jinja") if model_dir else "unknown"
    qmix = file_sha(model_dir / "qmix-manifest.json") if model_dir else "unknown"

    speeds = []
    for iface in a.ifaces.split(","):
        s = read(f"/sys/class/net/{iface}/speed")
        speeds.append(f"{int(s) // 1000} Gb/s" if s.isdigit() else "unknown")
    driver = read("/proc/driver/nvidia/version").split("\n")[0]
    m = re.search(r"\s(\d+\.\d+\.\d+)\s", driver)
    ctx = pre.get("max_model_len") or flag_value(args, "--max-model-len")
    depth = pre.get("queue_depth_at_start", 0)

    md = {
        "hardware": "4x NVIDIA DGX Spark (GB10, 128 GB unified memory each)",
        "topology": "TP4 across 4 nodes, MikroTik CRS812 switch, two ConnectX-7 RoCE rails per node",
        "negotiated_link_speed": f"head node sysfs: {' + '.join(speeds)} ({a.ifaces.count(',') + 1} fabric ports)",
        "model": "zai-org/GLM-5.3-Flash: nvidia/GLM-5.3-Flash-NVFP4 routed experts + dense layers "
                 "re-encoded to 8-bit grids (glm53-flash-4x-spark scripts/build_lossless8.sh, 'lossless8')",
        "model_revision": f"local build; manifest sha256 {model_fp} (json/jinja by content, shards by name+size); "
                          f"qmix-manifest sha256 {qmix}",
        "quantisation": "routed experts NVFP4 weights, W4A16 on Marlin; attention/KDA/shared/dense MLP on "
                        "MXFP8 / block-FP8 8-bit grids; BF16 activations",
        "kv_cache_dtype": f"{flag_value(args, '--kv-cache-dtype') or 'auto'}, "
                          f"--kv-cache-memory-bytes {flag_value(args, '--kv-cache-memory-bytes') or 'unset'} per rank",
        "serving_engine": f"vLLM {env.get('VLLM_BUILD_COMMIT') if env.get('VLLM_BUILD_COMMIT') not in (None, '', 'unknown') else '487ecf187'} (tonyd2wild vllm-glm53-flash v11 base) "
                          f"+ glm53-flash-4x-spark overlays sha256 {overlay_fp}",
        "serving_image": f"{image_tag} {image_id}",
        "drafter": f"{spec.get('method', 'none')} incoai/GLM-5.3-Flash-DFlash2 (block-FP8 linears), "
                   f"num_speculative_tokens {spec.get('num_speculative_tokens')}, per batch size "
                   f"{spec.get('num_speculative_tokens_per_batch_size')}, rejection "
                   f"{spec.get('rejection_sample_method')}, drafter manifest sha256 {draft_fp}",
        "context_limit": int(ctx),
        "scheduler": f"max-num-seqs {flag_value(args, '--max-num-seqs')}, max-num-batched-tokens "
                     f"{flag_value(args, '--max-num-batched-tokens')}, block-size {flag_value(args, '--block-size')}, "
                     f"scheduler-cls {flag_value(args, '--scheduler-cls') or 'default'}, chunked prefill, "
                     f"prefix caching, cudagraph {comp.get('cudagraph_mode')} sizes {comp.get('cudagraph_capture_sizes')}",
        "competing_traffic": ("none: vllm num_requests_running+waiting = 0 at start (checked by run_rigmark.sh)"
                              if depth == 0 else
                              f"NOT IDLE: vllm num_requests_running+waiting = {depth} at start"),
        "chat_template": f"model-bundled chat_template.jinja sha256 {template_fp}; server default "
                         f"reasoning_effort overridden per request by --extra-body",
        "profile_sha256": profile_fp,
        "kernel": os.uname().release,
        "nvidia_driver": m.group(1) if m else "unknown",
        "container_started_at": info.get("State", {}).get("StartedAt", "unknown"),
        "server_flags": public_flags(args),
        "server_env": env,
        "recipe": "knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 (deployment copy; see overlay/profile fingerprints)",
    }
    if a.ab_pin:
        pin = json.loads(Path(a.ab_pin).read_text())
        md["ab_variant"] = (f"in-boot glm_ab variant v{pin['variant']} of {pin['variants']} pinned for the whole run "
                            f"(config {pin['config']}); effective flags "
                            + IPV4.sub("<ip>", json.dumps(pin.get("effective"), sort_keys=True)))
        md["drafter"] += "; NOTE: glm_ab test harness armed (one CUDA-graph set per variant)"
    Path(a.out).write_text(json.dumps(md, indent=2) + "\n")
    print(f"metadata: {a.out} (container started {md['container_started_at']})")


def ab_rpc(base: str, method: str, args=(), timeout: float = 120):
    req = urllib.request.Request(base.rstrip("/") + "/collective_rpc",
                                 json.dumps({"method": method, "args": [str(x) for x in args]}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r).get("results") or []
    except urllib.error.HTTPError as e:
        sys.exit(f"ab: POST /collective_rpc {method} -> HTTP {e.code}; the server needs VLLM_SERVER_DEV_MODE=1 "
                 f"and GLM_AB_VARIANTS>=2 (overlay/glm_ab.py)")


def ab_pin(a) -> None:
    """Switch the in-boot glm_ab harness to one variant for the whole run (retries while requests are open)."""
    base = a.base.rstrip("/").removesuffix("/v1")
    before = ab_rpc(base, "glm_ab_status")
    if not before or not isinstance(before[0], dict) or not before[0].get("armed"):
        sys.exit(f"ab-pin: glm_ab not armed on this boot: {str(before)[:300]}")
    n = before[0].get("variants")
    if not 0 <= a.variant < int(n):
        sys.exit(f"ab-pin: variant {a.variant} not in [0, {n})")
    token = f"rigmark-{os.getpid()}-{time.monotonic_ns()}"
    deadline = time.time() + a.timeout
    while True:
        res = ab_rpc(base, "glm_ab_switch", (a.variant, token))
        if res and isinstance(res[0], dict) and "busy" in res[0]:
            if time.time() > deadline:
                sys.exit(f"ab-pin: engine still busy ({res[0]['busy']} unfinished) after {a.timeout} s")
            time.sleep(2)
            continue
        break
    bad = [r for r in res if not isinstance(r, dict) or r.get("error") or r.get("variant") != a.variant
           or r.get("token") != token or (r.get("ranks") is not None and not r["ranks"].get("agree"))]
    if bad or not res or len({r["config"] for r in res}) != 1 or len({r["seq"] for r in res}) != 1:
        sys.exit(f"ab-pin: switch to v{a.variant} failed or ranks disagree: {str(res)[:600]}")
    status = ab_rpc(base, "glm_ab_status")
    pin = {"variant": a.variant, "variants": n, "token": token, "seq": res[0]["seq"], "config": res[0]["config"],
           "prev_variant": res[0].get("prev"), "ranks": len(res), "effective": res[0].get("effective"),
           "replays": [r.get("replays") for r in status]}
    Path(a.out).write_text(json.dumps(pin, indent=1) + "\n")
    print(f"ab-pin: all {len(res)} ranks on v{a.variant} of {n} (prev v{pin['prev_variant']}, seq {pin['seq']}, "
          f"config {pin['config']}); effective {json.dumps(pin['effective'], sort_keys=True)}")


def _replay_deltas(old, new):
    """{kind: [per-set counts]} per rank -> list of (rank, kind, set, delta)."""
    out = []
    for rank, (o, nw) in enumerate(zip(old, new)):
        for kind, counts in (nw or {}).items():
            prev = (o or {}).get(kind) or [0] * len(counts)
            for i, c in enumerate(counts):
                out.append((rank, kind, i, c - (prev[i] if i < len(prev) else 0)))
    return out


def ab_check(a) -> None:
    """Verify nobody switched variant during the run and only the pinned graph set replayed."""
    base = a.base.rstrip("/").removesuffix("/v1")
    pin = json.loads(Path(a.pin).read_text())
    status = ab_rpc(base, "glm_ab_status")
    problems = []
    for r in status:
        if r.get("variant") != pin["variant"] or r.get("seq") != pin["seq"] or r.get("token") != pin["token"]:
            problems.append(f"rank {r.get('rank')}: variant {r.get('variant')} seq {r.get('seq')} "
                            f"token {r.get('token')} (pinned v{pin['variant']} seq {pin['seq']})")
    deltas = _replay_deltas(pin["replays"], [r.get("replays") for r in status])
    foreign = [d for d in deltas if d[2] != pin["variant"] and d[3] != 0]
    own = sum(d[3] for d in deltas if d[2] == pin["variant"])
    if foreign:
        problems.append(f"graph sets other than v{pin['variant']} replayed: {foreign[:8]}")
    record = {"stage": a.stage, "ok": not problems, "problems": problems, "own_set_replays": own,
              "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(a.log, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"ab-check [{a.stage}]: {'OK' if not problems else 'FAIL'}: v{pin['variant']} held on every rank, "
          f"{own} replays of its graph set" + ("" if not problems else "; " + " | ".join(problems)))
    if problems:
        sys.exit(1)


def ab_restore(a) -> None:
    base = a.base.rstrip("/").removesuffix("/v1")
    pin = json.loads(Path(a.pin).read_text())
    prev = pin.get("prev_variant")
    if prev is None or prev == pin["variant"]:
        print(f"ab-restore: nothing to do (was v{prev})")
        return
    deadline = time.time() + a.timeout
    while True:
        res = ab_rpc(base, "glm_ab_switch", (prev, f"rigmark-restore-{os.getpid()}"))
        if res and isinstance(res[0], dict) and "busy" in res[0] and time.time() < deadline:
            time.sleep(2)
            continue
        break
    ok = res and all(isinstance(r, dict) and r.get("variant") == prev for r in res)
    print(f"ab-restore: back to v{prev}: {'OK' if ok else 'FAILED ' + str(res)[:300]}")


def summary(a) -> None:
    for path in a.receipts:
        d = json.loads(Path(path).read_text())
        p = d["protocol"]
        print(f"== {Path(path).name}  protocol {p['version']}  git {p.get('repository_revision', '?')[:12]}"
              f"  dirty={p.get('repository_dirty')}  comparison_id {d['run']['comparison_id']}"
              f"  body {json.dumps(d['settings']['extra_body'], separators=(',', ':'))}")
        if "error" in d:
            print(f"   ERROR: {d['error']}")
        for w in ("code", "prose", "structured"):
            x = d.get("decode", {}).get(w)
            if not x:
                continue
            t, g = x["decode_tokens_per_second"], x["completion_gate"]
            ttft = x.get("ttft_seconds", {}).get("median")
            print(f"   {w:<10} decode median {t['median']:7.1f} tok/s  range {t['minimum']:.1f}-{t['maximum']:.1f}"
                  f"  ttft {ttft}s  gate {g['passed']}/{g['total']}")
        for depth, x in sorted(d.get("prefill", {}).items(), key=lambda kv: int(kv[0])):
            c = x["cold"]["effective_prefill_tokens_per_second"]["median"]
            r = x["warm_replay"]["effective_prefill_tokens_per_second"]["median"]
            print(f"   prefill {int(depth) // 1024:>3}K cold {c:8.0f} tok/s  replay {r:9.0f} tok/s")
        conc = d.get("concurrency", {})
        if conc:
            cells = "  ".join(
                f"C{k} {v['aggregate_end_to_end_tokens_per_second']['median']:.1f}"
                for k, v in sorted(conc.items(), key=lambda kv: int(kv[0])))
            print(f"   aggregate end-to-end (256-token code cap): {cells} tok/s")
            if "4" in conc:
                c4 = conc["4"]
                print(f"   C4 aggregate median {c4['aggregate_end_to_end_tokens_per_second']['median']:.1f} tok/s"
                      f"  per-stream decode {c4['per_stream_decode_tokens_per_second']['median']:.1f}"
                      f"  per-stream ttft {c4['per_stream_ttft_seconds']['median']}s")


LEAK_PATTERNS = [
    (re.compile(r"127\.0\.0\.1|localhost:\d+|:8093\b"), "loopback endpoint"),
    (re.compile(r"\b10\.100\.9[67]\.\d+\b"), "fabric IP"),
    (re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b"), "tailnet IP"),
    (re.compile(r"\b192\.168\.\d+\.\d+\b"), "LAN IP"),
    (re.compile(r"\.ts\.net|spark-0\d\.local", re.I), "hostname"),
    (re.compile(r"/home/|/Users/"), "home path"),
    (re.compile(r"\bknapcio\b(?!/GLM-5\.3-Flash-4x-DGX-Spark-TP4)|\b%s\b" % re.escape(os.environ.get("USER") or "knapcio")),
     "username"),
    (re.compile(r"Bearer\s+\S+|sk-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{20,}"), "credential"),
]
if os.environ.get("GLM_SCRUB_EXTRA"):          # e.g. your tailnet name or other site-specific identifiers (a regex)
    LEAK_PATTERNS.append((re.compile(os.environ["GLM_SCRUB_EXTRA"], re.I), "site-specific"))


def scrub(a) -> None:
    bad = 0
    for path in a.files:
        text = Path(path).read_text(errors="replace")
        for rx, what in LEAK_PATTERNS:
            hits = rx.findall(text)
            if hits:
                bad += 1
                print(f"scrub: {path}: {what}: {len(hits)} hit(s), e.g. {hits[0]!r}")
    if bad:
        sys.exit(1)
    print(f"scrub: clean ({len(a.files)} files: no endpoint, IP, hostname, home path or credential)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--base", required=True); p.add_argument("--wait", type=float, default=120)
    p = sub.add_parser("meta"); p.add_argument("--base", required=True); p.add_argument("--preflight", required=True)
    p.add_argument("--out", required=True); p.add_argument("--port", default="8093")
    p.add_argument("--ifaces", default="enp1s0f0np0,enP2p1s0f0np0"); p.add_argument("--ab-pin")
    p = sub.add_parser("ab-pin"); p.add_argument("--base", required=True); p.add_argument("--variant", type=int, required=True)
    p.add_argument("--out", required=True); p.add_argument("--timeout", type=float, default=300)
    p = sub.add_parser("ab-check"); p.add_argument("--base", required=True); p.add_argument("--pin", required=True)
    p.add_argument("--stage", required=True); p.add_argument("--log", required=True)
    p = sub.add_parser("ab-restore"); p.add_argument("--base", required=True); p.add_argument("--pin", required=True)
    p.add_argument("--timeout", type=float, default=300)
    p = sub.add_parser("summary"); p.add_argument("receipts", nargs="+")
    p = sub.add_parser("scrub"); p.add_argument("files", nargs="+")
    a = ap.parse_args()
    {"preflight": preflight, "meta": meta, "summary": summary, "scrub": scrub,
     "ab-pin": ab_pin, "ab-check": ab_check, "ab-restore": ab_restore}[a.cmd](a)


if __name__ == "__main__":
    main()
