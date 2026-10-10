# KDA speculative-boundary correctness fix

The verification-state stash could produce NaNs and repeated output when a
speculative window crossed a Mamba state block boundary. This was reproduced
through the serving API without a desktop client, with both sampling and greedy
decoding, including a fresh prompt-cache salt.

## Cause and repair

The stash represents some intermediate states as compact replay records with a
NaN marker. Before a forward pass, the runtime can migrate an accepted state to
the next block based on the *next scheduled window*. The previous FULL flag only
covered the current window. A compact record could therefore be copied into a
slot expected to contain a full matrix, and the next recurrence consumed the
marker as numerical state.

Both `glm_kda_stash.py` and `glm_kda_stash_fast.py` now extend the FULL predicate
by the maximum speculative lookahead, taken from the metadata state-column
width. This preserves full states before migration. Recurrence arithmetic,
weights and quantization are unchanged; the shared flag cache includes the width.

This repairs the repository's stash integration. The underlying recurrence
comes from vLLM / Flash Linear Attention by Songlin Yang and Yu Zhang; existing
[credits and licence notices](../../CREDITS.md) remain unchanged.

## Validation

- CPU regression: 87,264 lifecycle cases; 6,285 unsafe migrations with the old
  predicate and zero with the fix. The published test evaluates the actual
  slow-wrapper predicate without importing Torch or CUDA.
- Independent wrapper review: 270,200 cases; 29,965 unsafe before, zero after.
- Native GPU gate: 123 recorded checks, including 48 fused-flag cases and five
  CUDA graph replay cases. Two old NaN failures were reproduced. Six candidate
  lifecycles remained finite; four boundary output/state pairs were byte-exact
  against the original recurrence forced to store full states. Interior behavior
  remained byte-exact against the old implementation.
- The native gate ran the actual migration-decision/reset kernel; its temporal
  state copy used an equivalent whole-matrix copy. Full runtime integration was
  subsequently exercised by serving replays.
- Two fresh 55,044-token requests, greedy and sampled, completed with 3,871 and
  3,485 output tokens respectively. Both crossed the 55,296 and 57,600 block
  boundaries without repeated output or invalid observed log probabilities.
  Each stream returned one fewer token log probability than its completion-token
  usage count.
- A padded replay timed out after 130 seconds and is **incomplete**, despite no
  observed invalid log probabilities or repetition. Its final paired sampled
  case was not run. These are not additional passes.

The request contents are private and are not published. The portable CPU test
is in `tests/test_kda_stash_boundary.py`; the native gate and serving observations
above are reported results, not a complete public reproduction bundle.

**No new sparkDash, prefill, qeval or KLD measurements were collected for this
fix.** The README retains the earlier measured baseline with that limitation.
Extra full-state storage near boundaries may have a performance cost; zero
throughput impact has not been established.

## Existing deployments

Updating files in a running process is insufficient: its caches and captured
graphs can still contain old state. Stop all four serving ranks, then deploy the
updated checkout using fresh container names (`CTN`) and an unused overlay path
(`OVERLAY_REMOTE`) as described in [installation](../install.md). Preserve the
old containers and runtime directory; the launcher deliberately refuses to
replace their bind-mounted files. Keep the existing model paths and quantization.
Coordinate the restart across all ranks and validate the restarted API before use.
