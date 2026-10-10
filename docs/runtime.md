# Runtime provenance and publication boundary

## Current release (2026-09-29)

The serving profile is `profiles/current.env`. On top of the 2026-09-28 release (below) it turns on:

- decode kernels: head-gate GEMV (`GLM_GATE_GEMV`), early attention plan (`GLM_EARLY_PLAN`), Marlin launch tuning
  (`GLM_MARLIN_TUNE_ON` with the SHA-pinned table), dense 8-bit decode kernels (`GLM_DENSE_FAST`), L2 windows for
  MLA and the drafter, the BF16 mHC hc function (`GLM_MHC_BF16W`), the fused drafter conv and the RoCE proxy pin;
- the certified target head (`GLM_CERT_HEAD`, also for `min_tokens` requests);
- draft-shape truncation (`GLM_DRAFT_TRUNC`, 7 drafts scheduled, verify length chosen per step) with every verify
  family k = 1..7 captured (`SPEC_TABLE`, `CAPTURE_SIZES`), device-side selection at one request (`DEVSELECT=1`) and
  the batch > 1 cost refit (`TRUNC_COST=c4fit`);
- Gumbel-coupled drafting at T > 0 (`GLM_GUMBEL_COUPLED`);
- the prefill package: mHC sharding with row padding, KDA conv split, FlashKDA, Triton sparse MLA, dense 8-bit
  prefill GEMMs, routed-MoE prefill kernels (`PF3_ARM=samemath`) and large row gathers over NCCL (`GATHER_ROUTE=1`);
- corrected L2 prefetch tables (`L2PF_V2=1`) and memory compaction before boot (`COMPACT_MEM=1`).

Knob values and what each one turns off are documented at the end of the profile. Every module is default-off and
inert when its switch is off; modules that build CUDA code (dense 8-bit kernels, FlashKDA, routed-MoE kernels,
device-side selection) fall back to the stock path when the build or their self-check fails, and log it.

### 2026-09-28 base

The 2026-09-28 profile: the LVKP-S-L2 base (lossless8 target, block-128 FP8
incoai DFlash2 drafter, batch-uniform adaptive 3/7, kpool fixes, KDA stash with the 2026-09-27
[boundary repair](results/2026-09-27-kda-boundary-fix.md), L2 prefetch), the GDN metadata fast path
and router dedup, plus three additions:
- `GLM_ARGMAX_CLAMP=1`: vLLM #50843's padded-vocab clamp in the Gumbel and rejection samplers
  (`overlay/gumbel.py`, `overlay/rejection_sampler_utils.py`, mounted over the image files) and in
  `overlay/glm_target_argmax.py`. Bit-exact for every valid token id.
- The prefill scheduler: `GLM_PREFILL_CADENCE=8 GLM_PREFILL_SHORT_TOKENS=2048
  GLM_PREFILL_CADENCE_WHEN_QUEUED=1 GLM_END_DRAIN=1 GLM_IDLE_COALESCE_MS=4`
  (`overlay/glm_prefill_sched.py`, `glm_prefill_hooks.py`). Only rank 0 runs the engine core and
  scheduler in this TP4-over-4-nodes layout, so its ready lines appear in the rank-0 log only.
- The draft-length policy ships as `profiles/levers_policy.json` and is read from
  `/overlay/profiles/levers_policy.json`. Without the file the scheduler silently falls back to
  the launch table.

`overlay/sitecustomize.py` is the file the fleet serves. It also carries default-off registrations
for experiments that were measured and not adopted (in-boot A/B harness, verify cut, draft-context
graph, fused mHC, Marlin M=32, KV-lens exactness check, context-lookup drafter). Each is inert unless its
environment switch is set; `profiles/current.env` sets none of them.

## How the 2026-09-28 release was checked (2026-09-29: see results/2026-09-29-release.md)

- **Launch identity:** a `DRY=1` launch of this checkout was compared with `docker inspect` of the
  containers serving before the release. Arguments and bind mounts were identical; the environment
  differed only by the scheduler switches and the policy path above.
- **Fresh-clone boot:** the image was built from `Dockerfile.roce` in a fresh clone on all four
  nodes, and `start.sh serve` from that clone booted in 441 s with cold FlashInfer / Triton /
  TileLang caches. The rank-0 log showed every enabled piece installing (router dedup, L2 windows,
  RoCEnante, prefill scheduler, policy file).
- **Measurements on that boot:** sparkDash, `bench/accept_probe.py` step time, `bench/qeval.py`
  (75/75), `bench/kld_probe.py` + `bench/compare_kld_strict.py` (0.0293 over 6618 positions, same
  panel and reference as the earlier releases). Numbers are in the README.
- **Tests:** the CPU tests pass inside the image (`tests/test_glm_ab.py`, `test_glm_fast_load.py`,
  `test_glm_prefill_hooks.py` drift tables against the image's vLLM, `test_glm_prefill_shard.py`,
  `bench/test_*.py`). `tests/test_glm_prefill_sched.py` needs a vLLM source checkout
  (`GLM_VLLM_SRC`); the `*_gpu.py` tests need a free GPU and were run when their pieces were admitted.

## Source boundaries

The RoCEnante Docker layer pins its base image digest and the vendored b12x commit; source and
licence provenance is in `roce/`. The shim is based on Local Inference Lab's vLLM #597 via
tonyd2wild's port, with b12x/RoCEnante by Luke Alonso and Jason Cook. The base image contains the
vLLM / torch / CUDA / compiler stack; this repository provides the source-pinned derivative build,
not a from-source rebuild of every package in it.

`runtime-source-manifest.json` records the hashes of the 2026-09-27 publication; the files changed
since are listed in this release's commit. The launcher's `stop` preserves containers and signals
only verified auxiliary PIDs.

Cold caches are disposable compilation artifacts, not hidden model dependencies. A fresh checkout,
image and weight conversion must still pass transport, quality and end-to-end measurement checks
on your hardware before claiming the published performance.
