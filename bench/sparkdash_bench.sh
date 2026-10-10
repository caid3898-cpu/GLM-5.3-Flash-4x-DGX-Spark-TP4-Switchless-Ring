#!/usr/bin/env bash
# sparkdash_bench.sh OUTDIR [PORT] : sparkDash decode benches on the head (same tool and settings as the DS4.1
# prodbench: maxTokens 256, lab prompts, thinking off where the model has a switch), appended to OUTDIR/sparkdash.txt.
OUT=$1; PORT=${2:-8093}; DASH=http://127.0.0.1:5555/api/sparks/spark-01/llm
sd() { id=$(curl -s -X POST "$DASH/bench" -H 'Content-Type: application/json' -d "{\"port\":$PORT,\"concurrencies\":[$2],\"maxTokens\":256,\"promptType\":\"$1\"}" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("benchId",""))')
  [[ -z $id ]] && { echo "sparkDash refused $1 c$2" | tee -a "$OUT/sparkdash.txt"; return; }
  for i in $(seq 1 150); do sleep 4; curl -s "$DASH/bench" | python3 -c 'import json,sys; sys.exit(1 if json.load(sys.stdin)["active"] else 0)' && break; done
  curl -s "$DASH/bench" | ID=$id python3 -c 'import json,os,sys; d=json.load(sys.stdin)["last"]; assert d["benchId"]==os.environ["ID"], "stale"; [print(d["config"]["promptType"], "c%d"%r["concurrency"], "per", r["meanDecodeTps"], "agg", r["aggregateDecodeTps"]) for r in d["results"]]' 2>&1 | tail -3 | tee -a "$OUT/sparkdash.txt"; }
sd prose 1 > /dev/null; sd prose 1; sd prose 1; for c in 1 2 4 8 16; do sd prose $c; done; for c in 1 8 16; do sd code $c; done
