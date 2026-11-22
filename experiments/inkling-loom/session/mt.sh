#!/bin/bash
LCT_HOME=${LCT_HOME:-$(cd "$(dirname "$0")/../../.." && pwd)/tmp}
LOOM=$LCT_HOME/loom                          # venv and packages
OUT=$LCT_HOME/experiments/inkling-loom; mkdir -p "$OUT"   # logs and results
$LOOM/venv/bin/loom-serve --device=amdgpu --pool_backing=fixed --model=$1 --preset=$2 --listen=127.0.0.1:8093 > $OUT/serve-kv8-mt.log 2>&1 &
PID=$!
for i in $(seq 1 200); do sleep 2; curl -s http://127.0.0.1:8093/v1/models | grep -q '"registration":"ready"' && break; done
LCT_HOME=$LCT_HOME python3 -I "$(dirname "$0")/multiturn.py"
echo "vram $(cat /sys/class/drm/card2/device/mem_info_vram_used)"
kill $PID; wait $PID
