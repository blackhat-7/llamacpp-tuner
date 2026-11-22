#!/bin/bash
# Interleaved e2e A/B: base (kv8w4096-d) vs moe-opt (moe_opt.py applied to a hard-link copy of kv8w4096-d).
# Presets: xtx-kv8-196k (depth 1), d2 (span 3, shape 4), d3, d4 (span 5, shape 8). Short jobs x2, 64K x1, profile x1.
export LCT_HOME=/home/illusion/Documents/projects/llamacpp-tuner/tmp
cd /home/illusion/Documents/projects/llamacpp-tuner-inkling
M=$LCT_HOME/experiments/inkling-loom/moe; J=$LCT_HOME/loom/moe/deliver
run() { # pkg preset jobs label [extra serve args]
  local pkg=$1 preset=$2 jobs=$3 label=$4; shift 4
  $LCT_HOME/loom/notes/gpu-run $LCT_HOME/loom/venv/bin/python -I experiments/inkling-loom/session/session.py --model $LCT_HOME/loom/$pkg --preset $preset --jobs $J/$jobs --out $M/e2e.jsonl --label $label --port 8131 "$@" >> $M/e2e.out 2>&1
  journalctl -k --since "-10min" | grep -i amdgpu | grep -i -E "fault|timeout|reset" && { echo "AMDGPU FAULT after $label" >> $M/e2e.out; exit 1; }
}
for rep in 1 2; do
  for p in xtx-kv8-196k d2 d3 d4; do
    run kv8w4096-d $p jobs_short.json base-$p-short
    run moe-opt $p jobs_short.json opt-$p-short
  done
done
for p in xtx-kv8-196k d3; do
  run moe-opt $p jobs_64k.json opt-$p-64k
  run kv8w4096-d $p jobs_64k.json base-$p-64k
done
PROF="--serve-arg=--device_profiling_mode=dispatch-events,executable-metadata --serve-arg=--device_profiling_flush_interval_ms=5 --serve-arg=--print_device_statistics"
for p in d2 d4; do run moe-opt $p jobs_short.json moeprof-opt-$p $PROF; done   # logs: serve-kv8-moeprof-opt-*.log
echo DONE >> $M/e2e.out
