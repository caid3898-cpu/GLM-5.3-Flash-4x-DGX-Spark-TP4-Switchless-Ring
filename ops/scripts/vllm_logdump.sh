#!/bin/bash
# =============================================================
# SCRIPT: vllm_logdump.sh
# ROLE: vLLM TP4 容器日志落盘守护 (crash forensics — 容器被删后日志仍留存)
# 机制: 对本机每台 vllm-tp4-rank[0-9] 运行容器维持一个 `docker logs -f` 跟随进程,
#       追加写入 $LOGDUMP_DIR/<container>.log; 每 ${LOGDUMP_INTERVAL}s 巡检一次
#       (新容器拉起跟随 / 消失容器停跟随); 单文件超 200MB 做 .1/.2/.3.gz 轮转
#       (copytruncate + gzip, 保留 3 代)。
#       docker daemon 不可达时跳过本轮巡检, 不动既有跟随进程(防误杀后全量重灌)。
# env (由 systemd/vllm-logdump.service 注入, 缺省值与 unit 一致):
#   LOGDUMP_DIR      落盘目录   (默认 /opt/luz045/backup/vllm-logdump)
#   LOGDUMP_INTERVAL 巡检周期秒 (默认 5)
# 部署: /opt/luz045/scripts/vllm_logdump.sh (与 vllm-logdump.service ExecStart 一致)
# =============================================================
set -uo pipefail
LOGDUMP_DIR="${LOGDUMP_DIR:-/opt/luz045/backup/vllm-logdump}"
INTERVAL="${LOGDUMP_INTERVAL:-5}"
MAX_BYTES=$((200*1024*1024))
mkdir -p "$LOGDUMP_DIR"

declare -A FOLLOW_PID=()

rotate() {  # $1=logfile: 超 200MB 则 copytruncate + gzip, .1.gz->.2.gz->.3.gz 保留 3 代
  local f="$1" sz
  [ -f "$f" ] || return 0
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "${sz:-0}" -lt "$MAX_BYTES" ] && return 0
  [ -f "$f.2.gz" ] && mv -f "$f.2.gz" "$f.3.gz"
  [ -f "$f.1.gz" ] && mv -f "$f.1.gz" "$f.2.gz"
  if cp "$f" "$f.1" 2>/dev/null && : > "$f"; then
    gzip -f "$f.1"
    echo "[logdump] rotated $f (sz=$sz)" 
  fi
}

stop_all() {
  for p in "${FOLLOW_PID[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
  exit 0
}
trap stop_all TERM INT

echo "[logdump] start dir=$LOGDUMP_DIR interval=${INTERVAL}s max=$((MAX_BYTES/1024/1024))MBx3"
while :; do
  names=$(docker ps --format '{{.Names}}' 2>/dev/null)
  if [ $? -ne 0 ]; then
    echo "[logdump][warn] docker ps 失败, 跳过本轮巡检" >&2
    sleep "$INTERVAL"; continue
  fi
  seen=""
  for name in $(printf '%s\n' "$names" | grep -E '^vllm-tp4-rank[0-9]+$'); do
    seen="$seen $name"
    out="$LOGDUMP_DIR/${name}.log"
    pid="${FOLLOW_PID[$name]:-}"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      docker logs -f --timestamps "$name" >> "$out" 2>&1 &
      FOLLOW_PID[$name]=$!
      echo "[logdump] follow $name -> $out (pid ${FOLLOW_PID[$name]})"
    fi
    rotate "$out"
  done
  for name in "${!FOLLOW_PID[@]}"; do
    case " $seen " in
      *" $name "*) : ;;
      *) kill "${FOLLOW_PID[$name]}" 2>/dev/null; unset 'FOLLOW_PID[$name]'
         echo "[logdump] unfollow $name (容器已消失)" ;;
    esac
  done
  sleep "$INTERVAL"
done
