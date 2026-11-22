#!/bin/bash
# Serve kv8w4096 (preset xtx-kv8-196k) under the GPU lock and run probe_loom.py code/prose/recall at 195K,
# recording draft counters and VRAM around each request.
LCT_HOME=${LCT_HOME:-$(cd "$(dirname "$0")/../.." && pwd)/tmp}
LOOM=$LCT_HOME/loom                          # venv and packages
OUT=$LCT_HOME/experiments/inkling-loom; mkdir -p "$OUT"   # logs and results
here=$(cd "$(dirname "$0")" && pwd)
export KINDS RECALL_MAX LCT_HOME LOOM OUT here
exec flock /tmp/xtx-gpu.lock bash -c '
  log=$OUT/serve-headline.log
  $LOOM/venv/bin/loom-serve --model=$LOOM/kv8w4096 --preset=xtx-kv8-196k "--device=amdgpu://?completion_spin_ns=4294967295" --pool_backing=fixed --listen=127.0.0.1:8097 > $log 2>&1 &
  pid=$!
  for i in $(seq 900); do curl -s http://127.0.0.1:8097/v1/models | grep -q "\"id\"" && break; kill -0 $pid 2>/dev/null || { echo died; tail -5 $log; exit 1; }; sleep 1; done
  for k in ${KINDS:-code prose recall}; do
    curl -s http://127.0.0.1:8097/metrics | grep -E "draft_(proposed_tokens|accepted_inputs)" | grep -v "^#" > /tmp/hm0
    $LOOM/venv/bin/python $here/probe_loom.py run --server http://127.0.0.1:8097 --model Inkling-2-Mini --prompt $LCT_HOME/model-hunt/prompt_inkling_195000.txt --label ink-loom-kv8w4096-draft --kinds $k | cut -c1-400
    curl -s http://127.0.0.1:8097/metrics | grep -E "draft_(proposed_tokens|accepted_inputs)" | grep -v "^#" > /tmp/hm1
    echo "METRICS $k: $(paste /tmp/hm0 /tmp/hm1 | awk "{split(\$1,a,\"field=\\\"\"); split(a[2],b,\"\\\"\"); printf \"%s=%d \", b[1], \$4-\$2}") VRAM=$(cat /sys/class/drm/card*/device/mem_info_vram_used | sort -n | tail -1)"
  done
  kill $pid; wait $pid; echo "server $pid stopped"
'
