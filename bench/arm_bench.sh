#!/usr/bin/env bash
# arm_bench.sh LABEL [MODE] : benchmark the GLM endpoint on the head, results into $OUT (default ~/glm-mega-logs/LABEL).
#   MODE=full   warm-up, single-stream matrix, c1 x3, concurrency 1-16 (distinct prompts), prefill 8k/32k/64k,
#               sparkDash prose/code, qeval (75 checks), hardset if HARDSET=1
#   MODE=screen warm-up, c1 prose/code x3, c16 prose/code, qeval
set -u
L=$1; MODE=${2:-full}; B=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$HOME/glm-mega-logs/$L}; mkdir -p "$OUT"
BASE=${BASE:-http://127.0.0.1:8093}; PORT=${BASE##*:}; DASH=http://127.0.0.1:5555/api/sparks/spark-01/llm
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/bench.log"; }
metrics() { curl -s -m 10 $BASE/metrics | grep -E '^vllm:(spec_decode_num_(accepted|draft)_tokens_total|spec_decode_num_drafts_total|prefix_cache_(hits|queries)_total|num_preemptions_total)' ; }
cd "$B"
log "=== $L $MODE start"; metrics > "$OUT/metrics-start.txt"
# warm-up: first traffic after a boot is invalid (autotune/JIT); two short streams per type plus one c4 wave
python3 conc_bench.py "$L-warm" --conc 1,4 --types prose,code --max-tokens 256 > "$OUT/warm.log" 2>&1
python3 conc_bench.py "$L-warm2" --conc 1,16 --types prose --max-tokens 128 >> "$OUT/warm.log" 2>&1
# longer ramp (2026-09-26): the adaptive draft EMA / late JIT made the first full-length c1 runs read 10-30 % low;
# 6 full-length c1 requests (3 prose + 3 code) and one c4 burst before anything is measured
python3 conc_bench.py "$L-warm3" --conc 1 --types prose,code --repeat 3 >> "$OUT/warm.log" 2>&1
python3 conc_bench.py "$L-warm4" --conc 4 --types mix --max-tokens 512 >> "$OUT/warm.log" 2>&1
log "warm-up done"
kld() { python3 "$HOME/glm-quant-mix/tools/kld_probe.py" collect --url "$BASE" --model GLM-5.3-Flash-FP8 \
          --texts "$HOME/glm-quant-mix/data/calib.json" --out "$OUT/kld-$L.json" > "$OUT/kld.log" 2>&1; log "kld collected"; }
if [[ $MODE == kld ]]; then kld; docker logs --tail 20000 "${CTN:-glm53-nvfp4}-r0" > "$OUT/rank0-end.log" 2>&1   # counters printed during the bench
log "=== $L $MODE done"; exit 0; fi
python3 conc_bench.py "$L-c1" --conc 1 --types prose,code --repeat ${C1_REPEAT:-3} | tee "$OUT/c1.log"
if [[ $MODE == full ]]; then
  python3 bench_matrix.py "$L" > "$OUT/matrix.log" 2>&1; log "matrix done"
  python3 conc_bench.py "$L" --conc 1,2,4,8,16 --types prose,code,json,mix | tee "$OUT/conc.log"
  # the first long cold prefill of a boot runs 25-45 % slow (diagnostics/glm-inboot); three long cold prompts
  # (unique nonce each) first, so prefill_bench measures warm-engine cold prefill, not the boot's first one
  python3 prefill_bench.py "$L-pfwarm" --sizes 8192,16384,32768 --repeat 1 > "$OUT/pfwarm.log" 2>&1
  python3 prefill_bench.py "$L" --sizes 8192,32768,65536 --repeat 2 | tee "$OUT/prefill.log"
  # sparkDash (same tool and settings as the DS4.1 prodbench, so the two model families line up)
  bash "$B/sparkdash_bench.sh" "$OUT" "$PORT"
  log "sparkDash done"
else
  python3 conc_bench.py "$L-c16" --conc 16 --types prose,code,mix | tee "$OUT/c16.log"
fi
metrics > "$OUT/metrics-end.txt"
python3 qeval.py run "$L" > "$OUT/qeval.log" 2>&1; tail -12 "$OUT/qeval.log" | tee -a "$OUT/bench.log"
[[ ${QEVAL_C4:-0} == 1 ]] && { python3 qeval.py run "$L-c4" --concurrency 4 > "$OUT/qeval-c4.log" 2>&1; tail -6 "$OUT/qeval-c4.log" | tee -a "$OUT/bench.log"; }
[[ ${KLD:-0} == 1 ]] && kld
[[ ${HARDSET:-0} == 1 ]] && { python3 hardset.py "$L" > "$OUT/hardset.log" 2>&1; log "hardset done"; }
cp -f "$B/$L"*.json "$B/qeval-$L"*.json "$OUT/" 2>/dev/null
docker logs --tail 20000 "${CTN:-glm53-nvfp4}-r0" > "$OUT/rank0-end.log" 2>&1   # counters printed during the bench
log "=== $L $MODE done"
