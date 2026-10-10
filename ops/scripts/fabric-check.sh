#!/usr/bin/env bash
# GLM TP4 环网双向带宽巡检（2026-10-07 部署）
# 每根线 × 每条 Socket-Direct rail 各测一次 ib_write_bw -b 双向带宽，阈值 190 Gb/s（SparkRing 健康线）。
# 背景：环上单 rail 双向和健康值 ~203 Gb/s（实测），线缆会"静默劣化"（曾见 213→26 Gb/s 且 RoCE 计数器全 0，
# 单向测试测不出，只能双向发现）。
# 输出：journal + ~/glm53-ops/fabric-check/{latest.json,history.log,prom/glm_fabric.prom}
# 说明：测试会短暂占用环带宽（每 rail 约 10 秒）；vLLM 有在跑请求时自动跳过本轮。
set -u
THRESH=190
OUT="$HOME/glm53-ops/fabric-check"
PROM="$OUT/prom"
mkdir -p "$PROM"
TS=$(date +%s); STAMP=$(date "+%F %T")

# ---- 忙时跳过 ----
RUNNING=$(curl -sf --max-time 3 http://127.0.0.1:8093/metrics 2>/dev/null | awk '/^vllm:num_requests_running /{print $2; exit}')
if [ -n "${RUNNING:-}" ] && awk "BEGIN{exit !($RUNNING > 0)}" 2>/dev/null; then
  echo "$STAMP skip: vLLM busy (running=$RUNNING)" >> "$OUT/history.log"
  exit 0
fi

# ---- 环拓扑（rank0=bf94，f0 口朝下一跳、f1 口朝上一跳；每根线 rail1=p1s0 奇数/24，rail2=P2p1s0 偶数/24）----
# 列：edge 名 | 客户端主机(.1侧) | 客户端设备 | 服务端主机 | 服务端设备 | 服务端 IP
EDGES=(
"bf94-0167|r1|spark-bf94|rocep1s0f0|spark-0167|rocep1s0f1|10.200.1.2"
"bf94-0167|r2|spark-bf94|roceP2p1s0f0|spark-0167|roceP2p1s0f1|10.200.2.2"
"0167-6e8e|r1|spark-0167|rocep1s0f0|spark-6e8e|rocep1s0f1|10.200.3.2"
"0167-6e8e|r2|spark-0167|roceP2p1s0f0|spark-6e8e|roceP2p1s0f1|10.200.4.2"
"6e8e-bfe5|r1|spark-6e8e|rocep1s0f0|spark-bfe5|rocep1s0f1|10.200.5.2"
"6e8e-bfe5|r2|spark-6e8e|roceP2p1s0f0|spark-bfe5|roceP2p1s0f1|10.200.6.2"
"bfe5-bf94|r1|spark-bfe5|rocep1s0f0|spark-bf94|rocep1s0f1|10.200.7.2"
"bfe5-bf94|r2|spark-bfe5|roceP2p1s0f0|spark-bf94|rocep1s0f1|10.200.8.2"
)

FAILS=0
JSON_RESULTS=()
PROM_LINES=()
LOG_LINES=()

for row in "${EDGES[@]}"; do
  IFS='|' read -r edge rail chost cdev shost sdev sip <<< "$row"
  ssh -o ConnectTimeout=10 -o BatchMode=yes -f "$shost" \
    "ib_write_bw -b -d $sdev -x 3 -s 1048576 -D 8 > /tmp/ibsrv.log 2>&1"
  sleep 2
  if [ "$chost" = "spark-bf94" ]; then
    RAW=$(ib_write_bw -b -d "$cdev" -x 3 -s 1048576 -D 8 "$sip" 2>&1)
  else
    RAW=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$chost" \
      "ib_write_bw -b -d $cdev -x 3 -s 1048576 -D 8 $sip" 2>&1)
  fi
  ssh -o BatchMode=yes "$shost" 'pkill -x ib_write_bw' 2>/dev/null

  MB=$(printf '%s\n' "$RAW" | awk '/1048576/{print $4; exit}')
  if [ -z "${MB:-}" ] || [ "$MB" = "0.00" ]; then
    GB=0; OK=False; FAILS=$((FAILS+1)); NOTE="no-result"
  else
    GB=$(awk "BEGIN{printf \"%.1f\", $MB*8/1000}")
    if awk "BEGIN{exit !($GB >= $THRESH)}"; then OK=True; NOTE="ok"; else OK=False; FAILS=$((FAILS+1)); NOTE="below-threshold"; fi
  fi
  LOG_LINES+=("$STAMP $edge $rail $cdev <-> $shost:$sdev ${GB}Gb/s $NOTE")
  JSON_RESULTS+=("{\"edge\":\"$edge\",\"rail\":\"$rail\",\"client_dev\":\"$cdev\",\"gbps\":$GB,\"ok\":$OK}")
  PROM_LINES+=("glm_fabric_bw_gbps{edge=\"$edge\",rail=\"$rail\",dev=\"$cdev\"} $GB")
done

printf '%s\n' "${PROM_LINES[@]}" > "$PROM/glm_fabric.prom.tmp"
printf 'glm_fabric_check_failures %d\nglm_fabric_check_last_run_seconds %d\n' "$FAILS" "$TS" >> "$PROM/glm_fabric.prom.tmp"
mv "$PROM/glm_fabric.prom.tmp" "$PROM/glm_fabric.prom"
printf '{"timestamp":"%s","epoch":%s,"threshold_gbps":%d,"all_ok":%s,"results":[%s]}\n' \
  "$STAMP" "$TS" "$THRESH" "$([ $FAILS -eq 0 ] && echo true || echo false)" \
  "$(IFS=,; printf '%s' "${JSON_RESULTS[*]}")" \
  > "$OUT/latest.json"
printf '%s\n' "${LOG_LINES[@]}" >> "$OUT/history.log"
echo "fabric-check done: failures=$FAILS"
exit $([ $FAILS -eq 0 ] && echo 0 || echo 1)
