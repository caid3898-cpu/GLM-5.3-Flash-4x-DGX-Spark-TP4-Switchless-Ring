# Measurement and recipe history

[Current recipe and results](../README.md) · [Archived 2026-09-18 README](history-2026-09-18.md)

Numbers below retain their original benchmark, sample count and configuration. Different benchmark prompts, reasoning settings, quantizations and boot conditions are not interchangeable baselines.

## 2026-09-29: KDA checkpoint alignment fix (issue #2)

With prefix caching on, a cache hit could resume the KDA recurrent state from 1,152 tokens too early. Prefill chunk
ends were aligned to the 1,152-token drafter block, not the 2,304-token KDA state block, and the 2026-09-19
coordinator repair exposed the stale checkpoint. `GLM_MAMBA_ALIGN_FIX=1` aligns chunk ends to the KDA block and
materializes the last full block. `BATCHED_TOKENS` goes from 6912 to 6919, which keeps prefill at 6,912-token
chunks.

- **Cold/warm drift:** 0.25-0.36 → 0.02-0.04, against a cold-vs-cold floor of 0.04.
- **Cold prefill, 99k tokens:** 30.60 → 30.50 s.
- **sparkDash:** prose c1 85.0, code c1 126.9.

New release gate: `bench/prefix_scan.py`. [Results](results/2026-09-29-mamba-align-fix.md).

## 2026-09-29: comparison with the 2026-09-28 release (moved from the README)

Both columns from one gate window (gate 0003, 2026-09-29 00:43-01:50), same boot order and benchmark code, GPU clock
cap 2200 MHz; the right column is the 2026-09-29 stack before the release additions (device-side draft length,
routed-MoE prefill kernels, gather route, L2 tables, c4 cost table, KDA checkpoint fix).

**Decode, per-stream tok/s (aggregate in brackets)**

| prompt type | 2026-09-28 release | 2026-09-29 stack, gate 0003 | change |
|---|---:|---:|---:|
| prose c1 | 72.3 | 83.8 | +16.0 % |
| code c1 | 109.5 | 125.2 | +14.4 % |
| JSON c1 | 99.7 | 114.8 | +15.1 % |
| prose c4 | 42.0 (161.4) | 39.5 (154.2) | -6.0 % |
| code c4 | 55.8 (211.5) | 56.9 (210.8) | +1.9 % |
| JSON c4 | 65.6 (252.0) | 72.8 (281.0) | +11.0 % |
| prose c16 (1 run) | 22.1 (335.6) | 21.7 (326.1) | -1.9 % |

**Prefill, cold**

| prompt | 2026-09-28 release | 2026-09-29 stack, gate 0003 | change |
|---|---:|---:|---:|
| 32k tokens | 2150 tok/s, TTFT 15.3 s | 3053 tok/s, TTFT 10.7 s | +42.0 % |
| 128k tokens | 2127 tok/s, TTFT 61.6 s | 3036 tok/s, TTFT 43.2 s | +42.7 % |

**RigMark 1.0.0 decode screen, tok/s**

| workload | 2026-09-28 release | 2026-09-29 stack, gate 0003 | change |
|---|---:|---:|---:|
| code | 95.5 | 110.4 | +15.6 % |
| prose | 51.7 | 57.6 | +11.4 % |
| structured | 140.5 | 146.4 | +4.2 % |

## 2026-09-28: release results table (moved from the README on 2026-09-29)

sparkDash 1.8.8, 256 new tokens, temperature 0, thinking off, idle endpoint, same boot as the gates
below. Two warm-up prose c1 runs discarded before each series.

Commit 1f5b9eb / beca637, fresh clone. GPU clocks were not capped then (about 2450-2550 MHz; the fleet runs at a
2200 MHz cap since the evening of 2026-09-28), so these rows are not directly comparable with the 2026-09-29 tables.
The same stack re-measured at the cap in the 2026-09-29 gate window read sparkDash prose c1 72.3, code c1 109.5.

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **71.8** | 107.6 (54.1) | **156.9** (40.6) | 233.2 (30.2) | 329.2 (21.7) |
| code | 114.8 | 159.7 (83.1) | 208.9 (56.5) | 297.3 (40.5) | 342.7 (24.5) |
| structured | 158.4 | 259.3 (129.7) | 356.6 (92.2) | 645.9 (82.8) | 821.3 (53.5) |
| JSON | 104.0 | 173.7 (86.8) | 260.2 (68.7) | 358.7 (48.7) | 504.0 (35.0) |

Prose c1 is the median of three runs (73.1 / 69.3 / 71.8) and prose c4 the median of three
(143.8 / 156.9 / 176.4); every other cell is one run. The gate series on the same boot half an hour
earlier read prose c1 71.4 (74.3 / 71.4 / 62.9) and c4 151.0. Acceptance moves prose by 3-5 % run to
run, and sparkDash uses different prompts at each concurrency, so per-stream values are not
comparable across columns. The stable number is the decode step: `bench/accept_probe.py` at c1 gives
prose 39.0 ms per step at 2.2-2.3 tokens per step, code 48.9 ms at 4.5-4.7, JSON 50.1 ms at 6.3 (two
runs each, identical within 0.2 ms).

**Prefill, cold, tok/s by prompt length** (sparkDash prefill bench, one pass, same boot; 256k is
258,073 tokens, the longest prompt that fits 262,144 with the reply)

| 4k | 16k | 32k | 64k | 128k | 256k |
|---:|---:|---:|---:|---:|---:|
| 2098 | 2243 | 2255 | 2264 | 2229 | 2150 |

Time to first token at 32k is 14.5 s, at 128k 58.8 s, at 256k 120 s. sparkDash's prefill filler is
one repeated token (unique prefix per size, so the prefix cache does not apply); on varied random-word
text `bench/prefill_checked.py` measured 2,202 / 2,209 / 2,197 tok/s at 16k / 32k / 64k on the
2026-09-27 stack, within 2 % of these. Prefill is not optimised yet: in a prefill step MoE takes 35 %, attention
16 %, all-reduce 13 % and mHC 11 %.

- **Quality:** `bench/qeval.py` 75/75 (75 auto-scored checks: code run against hidden asserts, JSON
  schema, numeric answers, format constraints, degeneration). KL divergence 0.0293 over 6618
  teacher-forced positions against a BF16-attention reference (`bench/kld_probe.py`,
  `bench/compare_kld_strict.py`).
- **Boot:** ~2 min to `/health` 200 with warm JIT caches; 7.4 min on the first boot of a fresh clone (cold FlashInfer / Triton / TileLang caches).

These numbers include the 2026-09-27 KDA speculative-block-boundary fix ([notes](results/2026-09-27-kda-boundary-fix.md)).

## 2026-09-28: fresh-clone release (argmax clamp, prefill scheduler, shipped policy)

The first release booted from a fresh clone, including the image build. Adds the vLLM #50843 argmax
clamp and jnardiello's E27/E29 prefill scheduler, and ships the draft-length policy in
`profiles/`. sparkDash prose c1 71.39 / c4 151.03 / c16 330.92, step time prose 39.0 ms, qeval
75/75, KL 0.0293. [Results](results/2026-09-28-release.md).

## 2026-09-27: KDA speculative-boundary correctness fix

Fixed compact verification-state records being copied into a full-state slot
when the next speculative window crosses a block boundary. This could turn
subsequent output into repeated tokens with non-finite log probabilities.
The fix stores full states early enough; recurrence arithmetic and quantization
are unchanged. Native tensor checks reproduce the old failure and confirm
byte-exact corrected boundary states against a full-state reference. Two fresh
55k-context request replays completed without the failure. No new performance,
qeval or KLD result is claimed. [Validation and limits](results/2026-09-27-kda-boundary-fix.md).

## 2026-09-27: GDN metadata fusion and router deduplication

The 262k GDN metadata + router-dedup stack passed the predeclared
qualification protocol and was deployed on the four-node fleet. The complete [20-cell sparkDash table](results/2026-09-27-gdn-router-admitted.md)
retains prose c1 **71.64 tok/s** versus prior accepted **70.19**, but prose c4
aggregate **149.44 tok/s** versus **152.89**. The c1 qeval score fell to
**72/75** from **75/75**, with one truncation; c4 stayed **75/75**. Mean
teacher-forced KL was **0.029189** versus **0.028837**. Admission accepts the
declared floor while recording the loss; it is not a whole-model equivalence
claim. Fresh client prefill medians at nominal 16k/32k/64k were approximately
2,202/2,209/2,197 input tok/s to first observable generated delta including
reasoning. No 1M or optional optimization was admitted.

## 2026-09-27: earlier candidate snapshot

This checkpoint predates the completed qualification above; pending states below are historical.

A later strict 192-round in-boot target-start CUDA-event screen passed its own
A/A and structural gates for GDN metadata fusion plus router dedup. That screen
was not a sparkDash throughput result. The complete selected 262k stack then
booted and its final client/guard completed: qeval c1 **72/75** versus accepted
**75/75** (one truncated; three failed IDs), qeval c4 **75/75**, mean
teacher-forced KL **0.029189** versus **0.028837**, and sparkDash prose c1
**71.64** versus **70.19 tok/s** (median of 5) while prose c4 aggregate
**149.44** versus **152.89 tok/s** (median of 3). The c1 qeval and c4
throughput regressions remain visible. The qeval floor passed, but manual
numerical-quality and production promotion decisions are pending. A separate
fresh prefill client reported medians 2,202 / 2,209 / 2,197 input tok/s at
nominal 16k / 32k / 64k to the first observable generated delta (including
reasoning), with independent result review still pending. No 1M capacity result
is inferred. See the [candidate panel](results/2026-09-27-gdn-router-candidate.md).

## 2026-09-26: accepted LVKP-S-L2

The previously published warm-cache boot measurement was **129 seconds**. It is not a new boot measurement of the GDN/router release.

The accepted profile adds read-only L2 prefetch to LVKP-S. In the 72-round in-boot qualification, `inboot-target-start-period-cuda-events` measured savings of **0.610 ms at c1** (95% CI [0.548, 0.671]) and **0.727 ms at c4** ([0.617, 0.827]). The duplicate-baseline A/A intervals were [-0.147, 0.029] ms and [-0.127, 0.064] ms, inside the predefined ±0.2 ms band.

The following full sparkDash qualification measured prose c1 **70.84 tok/s** (median of 5: 72.02, 68.86, 70.84, 66.90, 72.21) and c4 **151.41 tok/s aggregate** (median of 3: 146.62, 153.85, 151.41). The preceding accepted LVKP-S sparkDash medians were 68.35 and 149.55 tok/s respectively. Those measurements imply +3.64% and +1.24% for that comparison; they are not the result of the later evening screen.

L2 quality qualification: qeval 75/75 at c1 and c4; KLD 0.0288366 over 17 items / 6,618 teacher-forced positions against the retained BF16-attention reference. A separate approximately 9.6k-token registry sanity test passed 32/32 lookups at each concurrency. No full-context quality claim follows from that test.

The original accepted containers were restored after the evening experiments. A fresh full sparkDash measurement is recorded in [release validation](validation.md), separately from the promotion measurement above.

## 2026-09-26 evening: GDN and router experiments, not promoted

Two independent in-boot experiments found positive target-step contrasts for GDN metadata optimization combined with router deduplication. Neither qualified for deployment under the predefined duplicate-baseline control gate.

The final 32-round run used four active arms in one diagnostic boot: baseline L2, duplicate L2, GDN+router, and GDN. Common-shape coverage passed the 95% gate (minimum 96.078431%); structural and tensor-check-counter reviews passed. The benchmark was `inboot-target-start-period-cuda-events`, not sparkDash:

| Contrast | c1 saving, ms (95% CI) | c4 saving, ms (95% CI) |
|---|---:|---:|
| GDN+router vs L2 | 0.813 [0.663, 0.959] | 0.663 [0.375, 0.909] |
| GDN vs L2 | 0.563 [0.427, 0.695] | 0.121 [-0.437, 0.518] |
| Duplicate baseline A/A | 0.073 [-0.105, 0.233] | -0.151 [-0.381, 0.063] |

Both A/A intervals extend outside ±0.2 ms. The result is **insufficient control precision for promotion**, not evidence that the positive candidate contrast is zero, nor a demonstrated quality regression. The thresholds were retained. No candidate sparkDash, qeval or KLD release run followed, and none of these experimental gains is included in the README throughput.

An earlier 36-round experiment also failed the A/A gate. Its six-arm coverage failure involved CPU-placement arms; a separately documented primary-arm analysis retained the same primary estimator but still did not qualify. The two runs were not pooled, and no failed blocks were removed to manufacture a pass.

## 2026-09-26: bounded dense / MoE investigations

The dense experiment passed 4,496 finite and byte-comparison checks. Eliminating original-workspace zeroing saved approximately 0.4–0.7 microseconds per component call, but the full integration candidate did not beat the installed stock path: one tested cell was slower and the other was within noise. This does not rule out different tiling or a different integration.

The MoE workspace-reuse experiment passed 600 finite and byte-comparison checks. It measured only workspace-zero reuse with fixed alignment and automatic tiles, not a replacement MoE implementation. Some component contrasts were positive, but duplicate graph controls showed bias in other cells. No model throughput claim or production change followed.

A draft-gather qualification tool failed before any tensor test because its staged path was shallower than the generator expected. The repaired tool passed 9 CPU tests and independent source review; GPU qualification and performance remain unmeasured. It is not part of this release.

## Earlier recipes

The [2026-09-18 snapshot](history-2026-09-18.md) preserves the previous FP8 / NVFP4 / EXL3 / SGLang comparisons, high/low reasoning tests, task-time experiments and original boot measurements. They use different settings from the current thinking-off sparkDash table. Historical statements about which ideas were still open or which switches were defaults apply only to that snapshot.
