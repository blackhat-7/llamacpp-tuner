#!/bin/bash
# usage: probe_start.sh MODEL PRESET [extra loom-serve flags...]
LCT_HOME=${LCT_HOME:-$(cd "$(dirname "$0")/../../.." && pwd)/tmp}
LOOM=$LCT_HOME/loom                          # venv and packages
OUT=$LCT_HOME/experiments/inkling-loom; mkdir -p "$OUT"   # logs and results
M=$1; P=$2; shift 2
$LOOM/venv/bin/loom-serve --device=amdgpu --model=$M --preset=$P --listen=127.0.0.1:8093 "$@" > $OUT/serve-kv8-probe.log 2>&1 &
PID=$!
for i in $(seq 1 150); do
  sleep 2
  s=$(curl -s http://127.0.0.1:8093/v1/status)
  if echo "$s" | grep -q '"ready"\|"failed"'; then break; fi
done
echo "$s" | head -c 1500; echo
curl -s -m 120 http://127.0.0.1:8093/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Inkling-2-Mini","messages":[{"role":"user","content":"Say hi."}],"max_tokens":20,"temperature":0}' | head -c 600; echo
grep -c driver_failure $OUT/serve-kv8-probe.log
kill $PID; wait $PID
