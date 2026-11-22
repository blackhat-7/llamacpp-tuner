#!/bin/bash
# usage: run_short.sh TAG "PKG:PRESET" ...   (one GPU-locked session each, jobs_short.json)
cd /home/illusion/Documents/projects/llamacpp-tuner-inkling
export LCT_HOME=/home/illusion/Documents/projects/llamacpp-tuner/tmp
S=$LCT_HOME/experiments/inkling-loom/spec
tag=$1; shift
for pp in "$@"; do
  pkg=${pp%%:*}; preset=${pp##*:}
  /home/illusion/Documents/projects/llamacpp-tuner/tmp/loom/notes/gpu-run $LCT_HOME/loom/venv/bin/python -I experiments/inkling-loom/session/session.py --model $LCT_HOME/loom/$pkg --preset $preset --jobs ${JOBS:-$S/jobs_short.json} --out $S/short.jsonl --label $tag-$pkg-$preset --port 8098
  journalctl -k --since "-5min" | grep -i amdgpu | tail -2
done
