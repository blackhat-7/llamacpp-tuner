#!/bin/bash
# usage: run.sh DIR -> per case: mean device us per launch of each kernel (L launches per case)
L=${L:-16}
d=$1; LOOM=/home/illusion/Documents/projects/llamacpp-tuner/tmp/loom
/home/illusion/Documents/projects/llamacpp-tuner/tmp/loom/notes/gpu-run $LOOM/venv/bin/loom-benchmark $d/bench.loom --device=amdgpu --measure=dispatch_complete --batch-size=${BATCH:-8} --profile-final-batch=true --output-format=jsonl --output=$d/r.jsonl > $d/r.log 2>&1 || { tail -20 $d/r.log; exit 1; }
jq -r '(select(.row=="benchmark") | "B \(.candidate_id) \(.benchmark_result.benchmark)"), (select(.row=="profile_replay") as $r | $r.profile_replay.summary.rows[]? | select(.type=="dispatch_function") | "P \($r.candidate_id) \(.function_name) \(.timing.mean_ns) \(.count // .timing.count // 0)")' $d/r.jsonl > $d/rows.txt
awk '$1=="B"{n[$2]=$3} $1=="P"{t[$2]+=$4; s[$2]=s[$2]" "$3"="sprintf("%.1f",$4/1000)} END {for (c in t) printf "%-10s%s\n", n[c], s[c]}' $d/rows.txt | sort
