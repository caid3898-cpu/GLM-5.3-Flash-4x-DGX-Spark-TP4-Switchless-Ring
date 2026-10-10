# KDA checkpoint alignment fix (issue #2), 2026-09-29

## Symptom

With prefix caching on, a request that hit the cache read the shared prefix differently from a cold request.

Asked to scan a 2,600-line synthetic log (99.4k tokens) for malformed lines, the model reported a line that does
not exist: "Job 36559 … `t=179` … held then released", with jobs 36560-36590 "missing".

| | runs reporting the phantom line |
|---|---|
| cold (fresh cache salt), T=0 and T=0.8 | 0 of 9 |
| warm (99,072 cached tokens) | 7 of 7 |

- At T=0, the mean |Δlogprob| cold-vs-warm was 0.25-0.36, against 0.04-0.09 cold-vs-cold.
- Retrieval of random codes inside the affected span was unaffected (12/12 cold, 24/24 warm). So the attention KV
  was intact, and the damage was in the KDA recurrent state.

## Cause

vLLM 487ecf187 `Scheduler._mamba_block_aligned_split` aligns prefill chunk ends to `cache_config.block_size`.

- In this model that value is 1,152: `EngineCore` sets it to the smallest KV group block, which is the DFlash2
  drafter's group.
- A KDA state slot covers `mamba_block_size` = 2,304 tokens (`--block-size`).
- The KDA prefill stores only the final state of a chunk, into slot `(end - 1) // 2304`.
- A chunk ending on an odd multiple of 1,152 therefore leaves slot *i* holding the state after `end` tokens, while
  its prefix-cache hash claims `(i + 1) * 2304 = end + 1152`.

For the 99,447-token prompt:

- The chunks end at 92160, 97920 and 99447, with 5,760-token chunks from the 6,905-token budget.
- Slot 42 keeps S(97920) under the hash of 99,072.

With a drafter, stock vLLM also backs the hit off far enough that a repeated prompt does not reach such a slot: the
image's own coordinator hits 92,160 on this prompt. Upstream is still exposed through the earlier checkpoints, and
the same scheduler code is in upstream vLLM 487ecf187. Our coordinator repair (`overlay/kv_cache_coordinator.py`, 2026-09-19)
keeps the drafter group from shrinking the hit. So the hit reached 99,072, and the KDA layers resumed 1,152 tokens
early.

**Exposure:** every profile since 2026-09-19 with prefix caching on. With 5,760-token chunks, every other prefill
checkpoint is stale: the ones written at odd multiples of 1,152 (5,760, 17,280, 28,800, …). The replay covers 2,662
prompt lengths from 4k to 262k; 2,631 of them publish at least one stale checkpoint. Two cases:

- **Repeating the same prompt**, as in the issue probe: the hit lands on the last full block. It is stale exactly
  when `P mod 2304` is in [1, 1151].
- **A later request that shares only part of the prompt**, such as the next turn of a conversation or a different
  question on the same document: the hit can land on any earlier checkpoint, and about half of those are stale.

## Bisect

Issue prompt, T=0, thinking off, 3 cold/warm pairs, 250 tokens with logprobs.

| stack | warm hit (tokens) | warm phantom | cold-vs-warm drift | cold-vs-cold |
|---|---|---|---|---|
| release (glm53-rel0929d) | 99,072 | 3/3 | 0.27 / 0.25 / 0.36 | 0.065 |
| prefix caching off | 0 | 0/3 | 0.05 / 0.09 / 0.05 | 0.050 |
| image's own coordinator | 92,160 | 0/3 | 0.04 / 0.07 / 0.03 | 0.086 |
| KDA stash off | 99,072 | 3/3 | 0.15 / 0.28 / 0.36 | 0.034 |
| **GLM_MAMBA_ALIGN_FIX** | 99,072 | 0/3 | 0.02 / 0.02 / 0.04 | 0.044 |
| **fix + BATCHED_TOKENS 6919** | 99,072 | 0/1 | 0.03 | 0.036 |

## Fix

`overlay/glm_mamba_align_fix.py`, with `GLM_MAMBA_ALIGN_FIX=1` (default on). It applies two hash-pinned text edits
to the image's method:

1. Align chunk ends to `mamba_block_size`.
2. Add the prompt's last full block (`num_tokens // 2304 * 2304`) to the mandatory stops. The Eagle back-off
   checkpoint is kept.

The chunks for the 99,447-token prompt become 96768 → 99072 → 99447.

`tests/test_glm_mamba_align_fix.py` replays the vendored image method (sha-checked) over 31 prompt lengths, five
budgets and warm starts:

- The stock method leaves a wrongly positioned checkpoint on 27 of the 31 lengths.
- The fix leaves none, and it always materializes the block the coordinator can hit.

Aligned to 2,304, a 6,905-token budget gives 4,608-token chunks. `BATCHED_TOKENS` is therefore raised from 6912 to
6919, so the chunks are 6,912 tokens (three KDA blocks). Before the fix they were 5,760.

## Measurements (4x Spark, clock cap 2200 MHz)

| | release | fix, 6912 | fix, 6919 |
|---|---|---|---|
| cold prefill, 99.4k tokens (median of 4) | 30.60 s | 31.88 s | **30.50 s** |
| sparkDash prose c1 (median of 3) / code c1 | 80.3 / 126.3 (release gate) | 84.1 / 124.5 | 85.0 / 126.9 |
| `bench/prefix_scan.py` | **FAIL** | PASS | **PASS** |

`prefix_scan` details:

- **Release: FAIL.** Drift 0.378 against a 0.197 limit. The warm scan flagged Records 2944-2976, the stale half
  block [95,616, 96,768).
- **Fix, 6912: PASS.** Drift 0.135 against a 0.127 floor.
- **Fix, 6919: PASS.**
  - Drift 0.064 against a 0.153 floor, with no warm-only anomalies.
  - 0 degenerate outputs in 48 T=0.8 responses.
  - Warm/cold incorrect 5/4.

Release quality gates on the fix stack (`BATCHED_TOKENS=6919`, same boot):

- **Long-prompt KL** (`bench/final_bench.py kldlong`), against the rel0928 reference of gate 0003:
  - **pooled mean 0.00971 over 98,500 teacher-forced positions: PASS** (limit ≤ 0.035; release 0801 measured
    about 0.0098);
  - p99 0.122, top-1 agreement 95.99 %, last-2048 mean 0.00867, continuation mean 0.00414;
  - per text: prose-docs 0.0115, code-runner 0.0071, code-sched 0.0114, mixed-long 0.0091.
- **T > 0 scan** (`bench/final_bench.py tscan`): **0 garbled of 35 outputs**, 0 errors, 0 bad JSON.

## Limits

- Measured on one boot per stack.
- The 6,618-position KLD panel and qeval were not rerun for this change; the long-prompt KL and the T > 0 scan
  above were. The fix changes only where prefill chunks end, i.e. the batch shape, not any kernel.
- The repeated-token runs in issue #2 did not reproduce on the 2026-09-29 release: 0 in about 64 responses,
  including the reporter's c8 shape.
