#!/bin/bash
# usage: run_long.sh TAG PRESET... ; runs 64K, 195Ka, 195Kb sessions per preset, each under the GPU lock separately
cd /home/illusion/Documents/projects/llamacpp-tuner-inkling
export LCT_HOME=/home/illusion/Documents/projects/llamacpp-tuner/tmp
S=$LCT_HOME/experiments/inkling-loom/spec
PKG=${PKG:-$LCT_HOME/loom/kv8w4096-d}
tag=$1; shift
for jobs in ${JOBS:-64 195a 195b}; do
 for p in "$@"; do
  /home/illusion/Documents/projects/llamacpp-tuner/tmp/loom/notes/gpu-run $LCT_HOME/loom/venv/bin/python -I experiments/inkling-loom/session/session.py --model $PKG --preset $p --jobs $S/jobs_$jobs.json --out $S/long.jsonl --label spec-$tag-$p --port 8097
 done
done
