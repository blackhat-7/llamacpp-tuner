#!/bin/bash
# usage: run2.sh DIR -> device time per kernel for each benchmark of DIR/bench.loom
LCT_HOME=${LCT_HOME:-$(cd "$(dirname "$0")/../.." && pwd)/tmp}
LOOM=$LCT_HOME/loom                          # venv and packages
OUT=$LCT_HOME/experiments/inkling-loom; mkdir -p "$OUT"   # logs and results
d=$1
flock /tmp/xtx-gpu.lock $LOOM/venv/bin/loom-benchmark $d/bench.loom --device=amdgpu --measure=dispatch_complete --batch-size=${BATCH:-8} --profile-final-batch=true --output-format=jsonl --output=$d/r.jsonl > $d/r.log 2>&1 || { tail -5 $d/r.log; exit 1; }
jq -r 'select(.row=="benchmark") | "B \(.candidate_id) \(.benchmark_result.benchmark)", (select(.row=="profile_replay") as $r | $r.profile_replay.summary.rows[]? | select(.type=="dispatch_function") | "P \($r.candidate_id) \(.function_name) \(.timing.mean_ns)")' $d/r.jsonl 2>/dev/null > /dev/null
jq -r '(select(.row=="benchmark") | "B \(.candidate_id) \(.benchmark_result.benchmark)"), (select(.row=="profile_replay") as $r | $r.profile_replay.summary.rows[]? | select(.type=="dispatch_function") | "P \($r.candidate_id) \(.function_name) \(.timing.mean_ns)")' $d/r.jsonl | awk '$1=="B"{n[$2]=$3} $1=="P"{t[$2]+=$4; s[$2]=s[$2]" "$3"="int($4/1000)"us"} END {for (c in t) printf "%-12s%s total=%dus\n", n[c], s[c], t[c]/1000}' | sort
