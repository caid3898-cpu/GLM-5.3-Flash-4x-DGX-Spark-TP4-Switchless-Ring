# GDN/router 262k results — 27 September 2026

The four-node GDN metadata + router-dedup stack passed the predeclared decode, tensor, full-stack and operational gates. These measurements come from the running fleet; a fresh-clone build and boot of this portable recipe have not been tested. [Machine-readable results and source hashes](2026-09-27-gdn-router-admitted.json).

The strict 192-round in-boot CUDA-event screen measured target-start-period savings of 0.829073 ms at c1 and 0.928958 ms at c4, with duplicate-baseline A/A controls inside ±0.2 ms. It did not measure overall tok/s. The final sparkDash DecodeBench panel used 256 completion tokens/255 decode tokens, thinking off, and completed all jobs and streams. Prose c1 is a five-run median; prose c4 is a three-run median; every other cell is one observation. Eight cells came from a later same-boot supplement with exact source/identity/guard review.

| Prompt | c1 tok/s | c2 aggregate (per-stream) | c4 aggregate (per-stream) | c8 aggregate (per-stream) | c16 aggregate (per-stream) |
|---|---:|---:|---:|---:|---:|
| Prose | **71.64** | 102.43 (53.09) | **149.44** (39.41) | 214.70 (28.06) | 305.59 (20.29) |
| Code | 123.93 | 151.40 (84.87) | 168.71 (50.28) | 243.43 (33.42) | 289.58 (21.67) |
| Structured | 163.07 | 144.88 (83.15) | 194.03 (52.05) | 270.84 (38.69) | 300.21 (21.98) |
| JSON | 126.16 | 126.36 (64.96) | 208.55 (54.33) | 297.32 (38.31) | 439.12 (30.46) |

The prior accepted L2 serving panel had prose c1 70.19 tok/s and c4 aggregate 152.89 tok/s. The new cross-session contrasts are **+2.07% at c1** and **−2.26% at c4 aggregate**. They have no confidence interval or isolated causal interpretation, and do not establish an all-concurrency improvement or 75 tok/s at c1.

Qeval returned **72/75 at c1**, below the accepted **75/75**, with one truncated answer and failures on `code_two_sum`, `math_m9`, `reason_r4`; c4 remained **75/75**. The c1 run met the declared floor of 72. Teacher-forced mean KL was **0.029189162** versus accepted **0.028836616** on the 17-item, 6,618-position private panel; p99 KL was 0.267795 versus 0.283625, and top-1 agreement 0.923995 versus 0.925204. These results meet the declared gates, but do not establish empirical quality equivalence or whole-model bit identity. Private panel prompts are not distributed here.

Fresh prefill client medians were **2,202.12**, **2,209.32**, **2,196.90 input tok/s** at nominal 16k, 32k, 64k, with TTFT **7.587**, **15.198**, **30.638 s**. Each is a median of three scored requests after two warmups. The metric divides actual API prompt tokens by elapsed time to the first observable generated delta, **including reasoning**. Cached-token counts were absent; the result does not isolate GPU prefill work or prove long-context retrieval, KV-pool gain, or 1M usability.

Optional gather/L2, E03, dense, scratch, indexer, and metadata-reuse candidates were not promoted. The separate 1M capacity ladder stopped under sustained host memory pressure before any large stage. The operator decision and full raw evidence remain in the private workspace; users of this portable file should treat the data JSON's SHA pins as provenance, not as a claim that private files ship with the repository.
