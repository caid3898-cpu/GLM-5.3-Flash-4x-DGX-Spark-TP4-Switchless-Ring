# Validation

**2026-09-28 release:** the current profile was booted from a fresh clone and gated on that boot:
qeval 75/75 at c1, KL 0.0293 on the same panel, step time unchanged against the previous
production stack. See [2026-09-28 results](results/2026-09-28-release.md). The paragraphs below
describe the 2026-09-27 GDN/router admission.

The 2026-09-27 recipe added GDN metadata fusion and router deduplication to LVKP-S-L2. Quantization remains NVFP4 for routed experts
and the existing `lossless8` profile for 8-bit non-expert weights. `lossless8`
is a profile name, not a mathematically lossless conversion. A component speed result does not
qualify a serving change.

The current validation panel is summarized in
[2026-09-27 results](results/2026-09-27-gdn-router-admitted.md). It met the
predeclared qeval floor but c1 was 72/75 versus the prior accepted 75/75;
prose c4 aggregate sparkDash was lower. This does not establish quality equivalence or a uniform
throughput gain. The earlier L2 results remain historical comparators.

## KDA boundary regression

The current recipe includes the [KDA state-migration repair](results/2026-09-27-kda-boundary-fix.md).
The broad quality/throughput panel above predates that repair. Run the portable
CPU regression without a model or GPU:

```bash
python3 -m unittest discover -s tests -p 'test_kda_stash_boundary.py' -v
```

This checks the actual wrapper predicate against state-migration timelines; it
does not replace native tensor checks or long-context serving validation.

## Decode performance

Use sparkDash DecodeBench with its original generic prose/code/structured/JSON
prompts and thinking disabled. `bench/final_sparkdash.py` runs the same matrix as
`glm_prodbench.sh`, with additional prose samples to reach the required counts:

1. Two discarded prose c1 warmups.
2. Three prose c1 runs, code c1, prose c2/c4/c8/c16, code c4/c16,
   structured c1/c16, and JSON c1/c16.
3. Two additional prose c1 and two additional prose c4 runs.

This produces 20 jobs, 18 scored. Report prose c1 median of 5 and prose c4 aggregate
decode throughput median of 3, both in tokens/s. `meanDecodeTps` is per stream;
`aggregateDecodeTps` is the concurrency-wide metric. Do not confuse either with
request throughput including prefill or a component CUDA-event benchmark.

```bash
python3 -B bench/final_sparkdash.py YOUR_NEW_LABEL \
  --base http://localhost:5555/api/sparks/spark-01/llm \
  --model GLM-5.3-Flash-FP8 --out YOUR_NEW_RESULT.json
```

Set `--base` to your dashboard endpoint. The historical API model alias in this
command does not describe the weight format: this recipe retains NVFP4+8-bit.
The collector checks the dashboard's `spark-01` logical identifier and model port
8093. Match these configured identifiers before running. It creates a new output
exclusively, refuses an already-active benchmark, polls each returned job ID,
rejects reused/stale IDs, and never substitutes the dashboard's last result.
There is no automatic retry or cancellation after timeout; inspect the active
job before starting more work. Use an idle service with no competing requests.

A completed result requires `COMPLETE_VALID_MEASUREMENT`, `complete=true`, and
20 unique valid jobs. Every stream must complete 256 tokens, report 255 decode
tokens and zero reasoning chunks, with no stream errors or early-EOS substitute.
Partial output and partial summary fields are diagnostic evidence only.

CPU API-contract tests (no model/GPU):

```bash
cd bench
python3 -B -S -m unittest -v test_final_sparkdash_cpu
```

Credit: the original `glm_prodbench.sh` matrix and sparkDash DecodeBench
contributors. This wrapper adds strict collection/validation and enough repeated
prose samples; it does not replace the dashboard's measurement implementation.

## Quality and changes

Run the bundled qeval panel against your own idle endpoint, preserving both full
75-task runs and their output files. Run from a new output directory because the
producer writes `qeval-LABEL.json` to its current working directory:

```bash
RECIPE_ROOT="$PWD"
mkdir quality-NEW_LABEL
cd quality-NEW_LABEL
python3 -B "$RECIPE_ROOT/bench/qeval.py" run NEW_LABEL-c1 \
  --url http://127.0.0.1:8093/v1/chat/completions --concurrency 1
python3 -B "$RECIPE_ROOT/bench/qeval.py" run NEW_LABEL-c4 \
  --url http://127.0.0.1:8093/v1/chat/completions --concurrency 4
```

The producer uses the historical model alias `GLM-5.3-Flash-FP8`. Do not use
`--only` or `--limit` for the full quality gate, and do not replace failed tasks
with selective retries. Retain all task outputs, score and truncation counts.

Require qeval >=72/75 at c1 and c4, teacher-forced KLD around 0.03, and retained
long-context sanity evidence.

Since 2026-09-29 the release rule is: the long-prompt KL panel (`bench/final_bench.py kldlong`, four 16.5k-40k-token
prompts, <= 0.035 and on the A/A floor of about 0.009-0.010 against the previous release) and the 6618-position
teacher-forced KLD are the primary quality gates; qeval is run three times with the fixed number extractor and
compared with a reference stack measured in the same window, never against a fixed single-run floor. qeval is
noisy at temperature 0 on this stack: greedy output is not reproducible in a boot, and the previous release scored
73 / 75 / 72 in three consecutive runs of one boot. Before 2026-09-29 `extract_final_number` took the last number in
the reply, so a correct answer followed by a restatement ("... 3 ... so 9 minus 6 is 3" style) could be scored as
wrong; it now prefers the last line that holds only a number, which is the form the prompts ask for. Rescoring 360
repeated runs of the flaky tasks moved reason_r12, reason_r4 and math_m4 to zero misses; the remaining misses
(math_m3, math_m9, json_count) occur at the same rate on the 2026-09-28-based stack and this release. KLD must contain all 17 expected calibration items
and 6618 teacher-forced positions with matching per-item lengths and prompt
identity; greedy text equality or a truncated zip is not a substitute. The
reported top-20 folded-tail estimate is not full-vocabulary KL divergence.

The exact retained KLD panel includes private operational material and is not
distributed here. The published hashes identify the original evidence; they do
not make that specific numerical KLD result independently reproducible from this
repository alone. A new public panel needs a separately recorded reference from
the chosen reference configuration and identical prompts/tokenization for each
candidate. Changing or redacting the panel creates a different measurement and
cannot reproduce the published 0.0288366 value. Do not substitute generated
continuation matching for teacher-forced log-probability comparisons.

For your own calibration panel, use `bench/kld_probe.py` unchanged. Supply a JSON
array of objects with `id`, `kind` and `text`; keep the panel, model/tokenizer,
prompt formatting and top-K fixed across the two collections. The reference
endpoint must be a separately qualified reference configuration. These commands
do not build or qualify that reference for you:

```bash
test ! -e reference.json && test ! -e candidate.json
python3 -B "$RECIPE_ROOT/bench/kld_probe.py" collect \
  --url http://REFERENCE_HOST:8093 --model GLM-5.3-Flash-FP8 \
  --texts YOUR_PANEL.json --out reference.json --k 20 --gen 256
python3 -B "$RECIPE_ROOT/bench/kld_probe.py" collect \
  --url http://127.0.0.1:8093 --model GLM-5.3-Flash-FP8 \
  --texts YOUR_PANEL.json --out candidate.json --k 20 --gen 256
python3 -B "$RECIPE_ROOT/bench/compare_kld_strict.py" \
  --texts YOUR_PANEL.json --reference reference.json --candidate candidate.json
```

The original collector can continue with greedy output if prompt log probabilities
are unavailable. The strict comparison refuses that fallback, incomplete item
grids, unequal per-item lengths, invalid probability support and nonfinite values.
It emits artifact hashes and the actual lengths/position count. Confirm the
reference's full lengths against your tokenizer and preserve collection commands
and calibration hashes: equal lengths alone cannot prove the two files came from
the same complete prompts. `STRUCTURAL_PASS` is not a numerical quality gate.
No KLD value is claimed here for an operator-supplied panel.

### Shared-prefix concurrency gate (since 2026-09-29)

qeval and KLD run at c1/c4 with fresh prompts, so they never read state back out of the prefix cache. Issue #2 was
reported at c8 with eight requests sharing a ~99k-token prefix at temperature 0.8. The investigation found that a
prefix-cache hit could resume the KDA recurrent state from 1,152 tokens too early (see
[the fix](results/2026-09-29-mamba-align-fix.md)). Every release now also runs `bench/prefix_scan.py` on the idle,
gated boot:

```bash
python3 -B bench/prefix_scan.py run NEW_LABEL --base http://127.0.0.1:8093 --out prefix-scan-NEW_LABEL.json
```

- **Prefix:** synthetic records, each with a random 5-letter code, so a correct quote cannot be rebuilt from a
  pattern. It is sized through `/tokenize` to about 97-99k tokens, with every prompt at P mod 2304 in 300-900. That
  is the phase at which the 2026-09-29 stale checkpoint was reachable.
- **Sampled rounds:** 8 concurrent distinct tasks, temperature 0.8, top-p 0.95, 1,200 tokens, thinking on, 3
  rounds. Each round runs cold (a fresh `cache_salt`) and then warm (same salt).
- **Drift test:** 3 repetitions of cold A / cold B / warm A at temperature 0, thinking off, with logprobs.
- **Solo lookups:** reported only.

The gate tests the cache, not the model. GLM misreads near-duplicate records, and its greedy output is not
reproducible within a boot, both cold and warm. So absolute correctness and cold/warm identity are reported, but
not required. Pass requires all of:

1. **Zero degenerate outputs:** no run of 8 or more identical tokens, no periodic loop of 48 or more tokens, no
   non-finite logprobs.
2. **Warm answers no worse than cold:** warm incorrect, misquoting and false-anomaly counts are each at most cold + 2.
3. **Drift within the noise:** median cold-vs-warm mean |Δlogprob| over the common token prefix is at most
   max(1.5 × median cold-vs-cold, 0.06). Every warm drift request must hit the cache (`/metrics`).
4. **No warm-only scan anomalies:** no record that the warm scan flags and neither cold scan flags.

Calibration on 2026-09-29, same prompt set and one boot per stack:
- **The release without the fix FAILS.** Drift was 0.378 against a limit of 0.197, and the warm scan flagged
  Records 2944-2976, the stale half block. On the issue's own prompt, drift was 0.25-0.36 against 0.065, with a
  warm-only phantom record in 3 of 3 repetitions.
- **With `GLM_MAMBA_ALIGN_FIX` and `BATCHED_TOKENS=6919` it PASSES:**
  - drift 0.064 against a 0.153 floor, with no warm-only anomalies;
  - warm/cold incorrect 5/4, 0 misquotes, 0 degenerate outputs in 48.

  With the fix at 6912 it also passes: drift 0.135 against 0.127.

Retain the JSON output. CPU tests (a fake server, no model; they include the saved release and fix numbers):

```bash
cd bench && python3 -B -S -m unittest -v test_prefix_scan_cpu
python3 -B tests/test_glm_mamba_align_fix.py
```

Bundled source SHA256: qeval `a83ccf00919153cae3acfaee49390be57d2f447b30e12ba4b4cbb9ac364a694e`,
qeval tasks `54719522d26996198c870264dfe5a93e2dd23f33436626c2477f1ac71206ffd2` (fixed extractor, 2026-09-29),
KLD probe `75a2adbb16500fbafe86b01c19680ab64e9282522490e608fdc34559a7a5005e`.

For a new optimization, first use one-boot balanced A/B plus a duplicate baseline
A/A control, at least 20–30 retained rounds. Keep the numerical estimator and
coverage gates declared in advance. The owner gate is a savings confidence
interval wholly above 0.2 ms with A/A within ±0.2 ms at both c1 and c4. Qualify
arithmetic on tensors (bit-exact or ULP), not greedy response text. Run the full
sparkDash/quality gates only for the selected combined stack.

The dated result page distinguishes the fresh throughput rerun from retained
quality qualification. An unchanged recipe can vary between sessions; a difference
from an earlier run alone does not establish a causal speedup or regression.
