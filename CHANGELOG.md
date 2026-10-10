# Changelog

## 2026-10-10 (this fork: switchless ring production deployment)
- **First production deployment & hardware qualification of `TRANSPORT=switchless` on a four-node
  SWITCHLESS 200G RoCE ring** (no switch at all). Upstream shipped the mode untested on rings; this
  deployment proves it end-to-end: qeval 73-75/75, cold prefill 3,064-3,470 tok/s across 8k-128k,
  front door served ~95% of the switched-fleet decode numbers (code c1 on par).
- **4-port dual-PCI-domain NCCL 2.30.7** build pinned by SHA256 (prefill +11% vs 2-port, decode par).
  GPU clocks pinned at 2550 MHz (systemd unit, +2~3%).
- **Ops tooling that upstream doesn't ship** (`ops/`, `monitoring/`): session-sticky front-door LB
  with a MemoryMax guard after the OOM-killer war story, responses/concurrency gateways, Prometheus +
  Grafana stack with a vLLM dashboard, bidirectional `ib_write_bw` fabric patrol every 6h (cables
  degrade silently - 213->26 Gb/s seen with all-zero RoCE counters), hardened healthcheck, daily smoke.
- **Tuned serving profile** (`profiles/current.env`): MAX_MODEL_LEN 1,048,576; KV pool 28 GiB FP8/rank
  after the front-door OOM war story; prefill chunks aligned to the 2304-token KDA block.
- English replication guide (README) + full Chinese playbooks under `docs/zh/`.


## 2026-09-29 (switchless ring, community PR #1)
- Optional `TRANSPORT=switchless` for four Sparks cabled as a ring without a switch (patched NCCL pinned by
  SHA256, RoCEnante off), contributed by @othexmr. Not tested on our switched fleet. The default switched launch
  is byte-identical; the container preflight now reads only names and mounts from `docker inspect`.
  [docs/switchless.md](docs/switchless.md).

## 2026-09-29 (hotfix, 3edfbc9)
- KDA state checkpoints now align to the 2304-token block (`GLM_MAMBA_ALIGN_FIX=1`, `BATCHED_TOKENS=6919`). With prefix
  caching on, a cache hit could resume the KDA recurrent state 1,152 tokens early (issue #2); affected every profile
  since 2026-09-19. No speed change. [Details](docs/results/2026-09-29-mamba-align-fix.md).
- New release gate `bench/prefix_scan.py`: c8 shared-long-prefix scan at T > 0, cold vs warm.

## 2026-09-29 (release, d80f4fd / 982e258 / 8f49b5c)
- Draft truncation with the verify length chosen on the GPU at one request (shared lambda with the host policy),
  c4 cost table refit, certified LM head with min_tokens, Gumbel-coupled drafting at T > 0.
- Prefill: routed-MoE prefill kernels (fused SiLU epilogue, down tile, exact sum+shared add), large dim-0 gathers
  over NCCL, prefill package (sharded mHC, FlashKDA, Triton sparse MLA, dense 8-bit prefill GEMMs).
- Corrected L2 prefetch tables; qeval number-extractor fix and a three-run quality rule.
- [Results](docs/results/2026-09-29-release.md); comparison with 2026-09-28 in [docs/history.md](docs/history.md).

## 2026-09-28 (release, adb55e3 / 1f5b9eb)
- Lossless 8-bit dense layers, RoCEnante image, decode kernels, LeversScheduler; the repository becomes the serving
  production stack.

## 2026-09-18 (first release)
- NVFP4 experts on Marlin, adaptive draft length, kpool patch; [archived README](docs/history-2026-09-18.md).
