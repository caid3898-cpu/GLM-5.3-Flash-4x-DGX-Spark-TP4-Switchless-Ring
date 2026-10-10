# GLM-5.3-Flash · 4× DGX Spark · TP4 · Switchless 200G Ring

**switchless** = no switch at all: four DGX Sparks cabled ring0→1→2→3→0, SparkRing-patched NCCL
running pure ring algorithms over 8 logical 200G RoCE links. Real endpoint of this repo:
`http://<head>:8002/v1` (OpenAI-compatible), full 1,048,576-token context.

**Production deployment of [knapcio's GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4) recipe on a four-node DGX Spark cluster cabled as a switchless 200G RoCE ring — the first end-to-end hardware qualification of the upstream `TRANSPORT=switchless` mode on a ring, plus the ops tooling a real deployment needs (front-door LB, monitoring, cable patrol, systemd units) and a tuned serving profile with the full 1,048,576-token context.**

> Serve GLM-5.3-Flash (NVFP4 experts + lossless8 dense + DFlash2 speculative drafting) with vLLM across
> 4× NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory each), OpenAI-compatible API, full 1M-token
> context per request, tool calling, structured output, `reasoning_effort` low/high/max.
> Everything here builds on the upstream recipe by **knapcio** — see [CREDITS.md](CREDITS.md) for the
> complete lineage (SparkRing, othexmr, tonyd2wild, jnardiello, b12x/RoCEnante, vLLM/MiaAI-Lab and more).

---

## Results (this ring, measured)

| Metric | Value | Note |
|---|---|---|
| qeval quality gate | **73–75 / 75** | matches the upstream-declared 74–75 window |
| Prefill (cold), 8k–128k | **3,064–3,470 tok/s** | above the upstream README figure @2200 MHz |
| Decode, single request | prose c1 71.3 / code c1 115.1 tok/s | upstream switched fleet: 71.9 / 119.8 → ~95%, **code on par** |
| Decode, aggregate | prose c4 148 / c8 219 / c16 279 tok/s | |
| Context per request | **1,048,576 tokens** | full model context |
| KV pool | 28 GiB FP8/rank ≈ 4.49M tokens | sized after an OOM war story, see [profiles/current.env](profiles/current.env) |
| Ring cable health | 8×rail ≈ 203 Gb/s bidirectional | patrolled every 6h, threshold 190 Gb/s |

## What this fork adds to upstream

1. **Ring qualification of `TRANSPORT=switchless`** — upstream shipped it (community PR #1 by @othexmr)
   untested on rings; we ran it in production. Key enablers: parameterized
   `SWITCHLESS_ADDR_RANGE=10.200.0.0/16` + `SWITCHLESS_SUBNET_PREFIX_LEN=24` +
   `NCCL_IB_SUBNET_AWARE_ROUTING=1`, all four RDMA functions listed in `IB_HCA`
   (no hardcoded subnets — this NCCL build carries no literal `/24`s from the author's fleet).
2. **4-port dual-PCI-domain NCCL 2.30.7** build (SparkRing patch lineage), pinned by SHA256 —
   prefill **+11%** vs the 2-port build, decode par. GPU clocks pinned at 2550 MHz (+2–3%).
3. **Ops tooling** (`ops/`, `monitoring/`): session-sticky front-door LB on :8002
   (OpenAI-compatible, health-checked backends), Prometheus + Grafana stack with a vLLM dashboard,
   bidirectional `ib_write_bw` cable patrol every 6h (textfile collector → Prometheus; skips rounds
   while vLLM is busy), hardened healthcheck, daily smoke, node monitors, systemd units.
4. **Tuned serving profile** ([profiles/current.env](profiles/current.env)): 1M context, KV pool sized
   against a real OOM incident, prefill chunks aligned to the 2304-token KDA block.
5. **Full Chinese playbooks** under [docs/zh/](docs/zh/) (deployment scheme, API usage, community survey)
   alongside this English README.

## Deployment artifacts — exact addresses

Everything a replication needs, with the pinned address actually used here:

| Artifact | Address | Notes |
|---|---|---|
| Base vLLM image (SM121 ARM64) | `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6` | tag `sm121-v11-dflash2`; also the `ARG BASE_IMAGE` default in [Dockerfile.roce](Dockerfile.roce) |
| RoCE serving image | built from [Dockerfile.roce](Dockerfile.roce) in this repo → `glm53-roce:v11` | build with `--build-arg BASE_IMAGE=<local-tag>` (docker load drops digests) |
| NCCL | [NVIDIA/nccl 2.30.7-1](https://github.com/NVIDIA/nccl/tree/73cf112295c33aee2b895f329f592f2a9b4b0f97) + **[SparkRing dual-PCI-domain patch](https://github.com/FujitsuPolycom/sparkring/blob/ae38b0f11df4dece04673ba8549befc7e527bdf5/spark_transport/nccl/nccl-2.30.7-dual-pci-domain.patch)** + hardening from [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl); launcher side = [othexmr PR #1](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/pull/1) (in this repo: `scripts/transport.sh`, `scripts/check_switchless_nccl.py`, `docs/switchless.md`) | our build pinned `SWITCHLESS_NCCL_SHA256=9ddd77c2…` in `.env.switchless.ring` |
| Target weights | [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) @ revision `09b04e5e` (44 files, 204.5 GB) | verify every file vs HF API (`scripts/verify_weights.sh`) |
| Drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) → convert with `scripts/drafter_fp8.py` → `GLM-5.3-Flash-DFlash2-fp8blk` | CC BY-NC-ND; weights not redistributed |
| Dense conversion | `scripts/build_lossless8.sh` (in-repo) | lossless8 dense + hardlink layout, one filesystem |
| Monitoring images | `prom/prometheus:v2.54.1` · `grafana/grafana:11.2.2` · `prom/node-exporter:v1.8.2` · `utkuozdemir/nvidia_gpu_exporter:1.3.0` | [monitoring/docker-compose.yml](monitoring/docker-compose.yml) |
| Dashboards & configs | in-repo: [monitoring/grafana/dashboards/vllm-tp4.json](monitoring/grafana/dashboards/vllm-tp4.json), prometheus.yml, provisioning | |
| Front door / gateways / ops | in-repo: [ops/](ops/) (LB + MemoryMax guard, gateway-8003, concurrency-8001, fabric-check, healthcheck, systemd units) | |
| Model license | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | weights have own licences; nothing redistributed |

## Hardware & cabling

- 4× DGX Spark (GB10, sm_121, 128 GB unified). Each has **2 physical QSFP sockets** implemented as
  Socket-Direct: each socket draws its two functions from *different* PCIe roots → per node
  `rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, `roceP2p1s0f1` (4 RDMA functions, 1 port each —
  verified with `ibdev2netdev`).
- Cabled **rank0 → rank1 → rank2 → rank3 → rank0**; at each node the `f0` port faces the next rank
  and `f1` faces the previous one. **Every Socket-Direct rail of every cable is used**
  (4 cables → 8 logical links; all four functions per node carry IP). Running only the primary-domain
  rail (2-port NCCL) also works but prefill is ~11% slower.
- Addressing: one /30 per rail, each cable's rails in **different /24s**
  (`SWITCHLESS_SUBNET_PREFIX_LEN=24` distinguishes cables), MTU 9000, `SWITCHLESS_ADDR_RANGE=10.200.0.0/16`.
  Management LAN (192.168.31.0/24 here) is only used for NCCL bootstrap/ssh — data plane never touches it.

## Quick start

Prerequisites: four Sparks with Docker, passwordless ssh between them **in ring order**, NCCL built
from source, weights on local disk. Full command-level detail: [docs/zh/方案总结-TP4环网.md](docs/zh/方案总结-TP4环网.md).

```bash
# 0) NCCL 2.30.7 + dual-PCI-domain patch (SparkRing lineage), ~3 min per node
make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
#    pin the build: sha256sum the lib (SWITCHLESS_NCCL_SHA256 in .env.switchless)

# 1) clone this repo on all four nodes; fill .env.switchless (hosts, IPs, HCAs, paths)
#    → use .env.switchless.ring as the reference for a 4-node ring

# 2) base image (ghcr may 403 some nodes → docker save|load between nodes)
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6

# 3) RoCE image on every node — ⚠️ docker load drops RepoDigests, so build with
docker build -f Dockerfile.roce --build-arg BASE_IMAGE=<local-tag> -t glm53-roce:v11 .

# 4) weights: nvidia/GLM-5.3-Flash-NVFP4 (44 files, 204.5 GB) — verify EVERY file against the
#    HF API size (scripts/verify_weights.sh); the download script's [ -s file ] check skips
#    truncated files silently. Then convert on each node (same filesystem for hardlinks):
bash scripts/build_lossless8.sh && python scripts/drafter_fp8.py

# 5) head node: ENV_FILE=.env.switchless ./start.sh serve
#    ⚠️ ./start.sh DRY mode does no real ssh — dry-run PASS ≠ real run works
#    cold start ~9.5 min → watch http://<head>:8093/health (LB exposes :8002)

# 6) front door + gateway (optional, ops/): session-sticky LB, Anthropic-compat gateway
# 7) monitoring (monitoring/): docker compose up -d → Grafana :3000, vLLM dashboard provisioned
# 8) fabric patrol (ops/): glm-fabric-check.service+timer → ib_write_bw 8 rails, Prometheus textfile
```

Client usage (model id, endpoints, streaming, tool calls, 1M-context notes):
[docs/zh/集群调用说明.md](docs/zh/集群调用说明.md).

## Monitoring & ops

- `monitoring/docker-compose.yml`: Prometheus + Grafana + exporters; the vLLM dashboard
  (`monitoring/grafana/dashboards/vllm-tp4.json`) tracks throughput, TTFT, running/waiting, KV usage,
  prefix-cache hit rate, speculative accept length. **Lesson learned**: Grafana idles at ~96 MB +
  43% CPU polling — on a host with only ~20 GB of headroom next to the KV pool it got the front door
  OOM-killed repeatedly; the fix was shrinking the KV pool (40→28 GiB), not deleting monitors
  ([profiles/current.env](profiles/current.env) comments).
- `ops/scripts/fabric-check.sh` + `glm-fabric-check.{service,timer}`: bidirectional per-rail
  `ib_write_bw`, threshold 190 Gb/s, writes `glm_fabric_bw_gbps{edge,rail,dev}` for Prometheus.
  Cables degrade **silently** (observed 213→26 Gb/s with all-zero RoCE counters) and unidirectional
  tests miss it.
- `ops/frontdoor/glm53-lb.py`: session-sticky LB with backend health; `GLM_BACKENDS` JSON env
  (quote it single-style in systemd — double quotes get swallowed).
- `ops/systemd/`: GPU clock pinning (2550 MHz), optional Anthropic-compat gateway example.

## Tuning — what we measured so you don't have to

| Change | Verdict |
|---|---|
| 4-port dual-PCI-domain NCCL | **keep** (prefill +11%, decode par) |
| SparkRing mesh + RoCEnante (hairpin all-to-all) | ported & green in microbench (25 µs vs NCCL 51 µs), but **serving one-shot 84 µs > NCCL 59 µs** (arrival-jitter sensitive) → off in production, `MESH_ROCE=0/1` switchable |
| NCCL 2.31.2 self-build | identical to 2.30.7 → reverted |
| MIN/MAX_NCHANNELS=4 | par → default |
| CUDA_MODULE_LOADING=EAGER (devselect fix) | CUDA 13 on this platform can't capture the conditional graph; EAGER also +20 min startup → dropped (forfeits README's +1.9% prose) |
| GPU 2400→2550 MHz | **keep** (+2–3%) |
| KV pool 40→28 GiB | **keep** — host OOM-killer was reaping the front door; 4.49M KV tokens ≈ 4.28 full-1M contexts, plenty for single-user serving |

Methodology note that saved us: **no conclusion without a profiler reading** — before we measured
"96 allreduces × 59 µs = 5.7 ms/step" we wrongly blamed bandwidth, clocks and acceptance in turn.

## Gotchas (each one cost real time)

1. Weight download "completes" with truncated files (`[ -s file ]` skip) — verify every file against
   the HF API (`scripts/verify_weights.sh`).
2. `start.sh` DRY mode does no real ssh — dry-run all-PASS proves nothing.
3. `docker load` drops RepoDigests → builds that `FROM image@sha256` fail at registry →
   `--build-arg BASE_IMAGE=<local-tag>`.
4. HF cache snapshots are symlinks; conversion needs real files on one filesystem
   (`mv` weights in, `cp -rL` the drafter).
5. rsync to a missing parent dir fails the whole hop — `mkdir -p` first.
6. Node ssh must exist in ring order; missing `.pub` can be regenerated with `ssh-keygen -y`.
7. No-sudo life: privileged container mounting host `/usr/sbin` (mind the `/usr/bin/ip` symlink),
   `/proc/sys` sysctls, `SYSTEMD_IGNORE_CHROOT=1` + `--pid host` for systemd --user.
8. The conversion container refuses an existing output dir — rename the stale one (`*.vNfail`), never
   `rm -rf` into it.
9. systemd `Environment=` swallows JSON double quotes — wrap the whole value in single quotes.
10. Restart cycle: `stop` + `docker rm` before `serve`, or preflight rejects on container-name clash.

## Repository map

```
.env.switchless.ring   our production ring config (hosts/rails/HCAs) — adapt to your cabling
start.sh, Dockerfile.roce, scripts/, roce/, overlay/, bench/, profiles/, tests/   upstream recipe
ops/frontdoor/         session-sticky LB (:8002) + systemd unit + drop-ins (incl. MemoryMax guard)
ops/gateway-8003/      Responses/Anthropic-compat gateway (auth + model alias + reasoning mirror)
ops/concurrency-8001/  traffic governance proxy (concurrency/TTFT budget/heartbeat/dedup)
ops/systemd/           GPU clock pinning, fabric patrol units, gateway example
ops/scripts/           fabric-check, hardened healthcheck, daily smoke, node monitors
monitoring/            Prometheus + Grafana + dashboards (compose, provisioning, vllm-tp4.json)
docs/zh/               Chinese playbooks: deployment scheme · API usage · survey · overview
README.upstream.md     the upstream README, unmodified
```

## References & upstream

- Base recipe: **[knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)** (upstream README: [README.upstream.md](README.upstream.md))
- Switchless mode: **othexmr**, [PR #1](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/pull/1) · NCCL lineage: [FujitsuPolycom/SparkRing](https://github.com/FujitsuPolycom/sparkring), [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl)
- Weights: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) · drafter [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC BY-NC-ND — weights are **not** distributed here)
- Alternative 2-Spark path: [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
- Community 1M-context ring references: yunwei37 (993K needle @1,048,576), METIQ-Solutions, sethforprivacy, ZackO2o — survey with links in [docs/zh/环网与TP4全网调研.md](docs/zh/环网与TP4全网调研.md)
- Full contributor lineage: [CREDITS.md](CREDITS.md)

## License

Upstream recipe is MIT (© knapcio) — see [LICENSE](LICENSE), [NOTICE](NOTICE) and component licences in
[CREDITS.md](CREDITS.md). This fork's deployment tooling and docs follow the same MIT terms.
Model weights have their own licences (see links above) and are **not** included.
