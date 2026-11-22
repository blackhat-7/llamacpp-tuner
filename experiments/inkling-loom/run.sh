#!/bin/bash
# usage: run.sh DIR  -> benchmark DIR/bench.loom, print device time per kernel and GB/s of the split+combine
LCT_HOME=${LCT_HOME:-$(cd "$(dirname "$0")/../.." && pwd)/tmp}
LOOM=$LCT_HOME/loom                          # venv and packages
OUT=$LCT_HOME/experiments/inkling-loom; mkdir -p "$OUT"   # logs and results
d=$1
flock /tmp/xtx-gpu.lock $LOOM/venv/bin/loom-benchmark $d/bench.loom --device=amdgpu --measure=dispatch_complete --batch-size=${BATCH:-8} --profile-final-batch=true --output-format=jsonl --output=$d/r.jsonl > $d/r.log 2>&1 || { tail -5 $d/r.log; exit 1; }
jq -r 'select(.row=="profile_replay") as $r | $r.profile_replay.summary.rows[]? | select(.type=="dispatch_function") | "\($r.candidate_id) \(.function_name) \(.timing.mean_ns)"' $d/r.jsonl | awk '{t[$1]+=$3; s[$1]=s[$1]" "$2"="int($3/1000)"us"} END {for (c in t) {split(ENVIRON["DEPTHS"]?ENVIRON["DEPTHS"]:"64000,195000",D,","); b=D[substr(c,2)+1]; printf "%s depth %d:%s total=%dus  %.0f GB/s\n", c, b, s[c], t[c]/1000, b*(ENVIRON["PB"]?ENVIRON["PB"]:2048)*2/t[c]}}' | sort
