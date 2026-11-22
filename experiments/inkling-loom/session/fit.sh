#!/bin/bash
# usage: fit.sh MODEL PRESET LABEL : start with fixed backing, one short request, report VRAM and draft metrics
LCT_HOME=${LCT_HOME:-$(cd "$(dirname "$0")/../../.." && pwd)/tmp}
LOOM=$LCT_HOME/loom                          # venv and packages
OUT=$LCT_HOME/experiments/inkling-loom; mkdir -p "$OUT"   # logs and results
V=/sys/class/drm/card2/device/mem_info_vram_used
echo "idle vram $(cat $V)"
$LOOM/venv/bin/loom-serve --device=amdgpu --pool_backing=fixed --model=$1 --preset=$2 --listen=127.0.0.1:8093 > $OUT/serve-kv8-fit-$3.log 2>&1 &
PID=$!
for i in $(seq 1 200); do sleep 2; curl -s http://127.0.0.1:8093/v1/models | grep -q '"registration":"ready"' && break; kill -0 $PID 2>/dev/null || break; done
curl -s http://127.0.0.1:8093/v1/models | grep -o '"registration":"[a-z]*"'
echo "ready vram $(cat $V)"
curl -s -m 300 http://127.0.0.1:8093/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Inkling-2-Mini","messages":[{"role":"user","content":"Write a C function that reverses a string in place."}],"max_tokens":300,"temperature":0}' | head -c 300; echo
echo "after-request vram $(cat $V)"
curl -s http://127.0.0.1:8093/metrics | grep -iE "draft|accept|propos" | head -12
grep -c driver_failure $OUT/serve-kv8-fit-$3.log
kill $PID; wait $PID
