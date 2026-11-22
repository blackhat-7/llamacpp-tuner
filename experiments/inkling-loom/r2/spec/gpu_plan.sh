#!/bin/bash
# GPU runs still needed by the spec workstream, in order. Each step takes the GPU through notes/gpu-run.
# usage: gpu_plan.sh STEP   (1..4)
set -e
L=/home/illusion/Documents/projects/llamacpp-tuner/tmp
S=$L/experiments/inkling-loom/spec
G=$L/loom/notes/gpu-run
case $1 in
1) # draft-head kernel test: cases match and numpy must pass, neg must fail
   $G $L/loom/venv/bin/loom-benchmark $L/loom/spec/dhtest/test.loom --device=amdgpu --output=$L/loom/spec/dhtest/r.json > $L/loom/spec/dhtest/r.log 2>&1 || true
   jq -c '.summary' $L/loom/spec/dhtest/r.json; grep -i "fail\|pass" $L/loom/spec/dhtest/r.log | head ;;
2) # short A/B: draft head alone at static d3 (same verify widths -> same text: clean acceptance/speed A/B),
   # then dynspan (rebuilt by the final tool) with and without the draft head, the d7 two-threshold variant,
   # and three more single-threshold policies
   $S/run_short.sh r3 kv8w4096-d:d3 spec-dh32k:d3 spec-dyn2-t90s3:d3 spec-dyn2-t90s3-dh32k:d3 spec-dyn2-d7-t90-42:d7 \
     spec-dyn-t110s3:d3 spec-dyn-t90s4:d3 spec-dyn-t100s2:d3 ;;
3) # long context for the lead (64K; 195K code + recall; 195K prose), then the 195K recall baseline at d1 and d3
   PKG=$L/loom/spec-dyn2-t90s3-dh32k $S/run_long.sh lead d3
   JOBS=195a $S/run_long.sh b xtx-kv8-196k d3 ;;
4) # summaries
   python3 $L/loom/spec/tools/summ.py $S/short.jsonl; python3 $L/loom/spec/tools/summ.py $S/long.jsonl ;;
esac
