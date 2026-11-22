#!/usr/bin/env bash
# chain A (2026-10-18): dump the first 67 transcripts (d1) -> parity engine (210ebbdd + bc4cab96, built in r5-upstream,
# then the worktree is restored) + one parity run -> smoke train on d1. Every GPU step takes gpu.lock.
set -u
M=$(dirname "$(readlink -f "$0")"); P=$M/..; D=$M/data
PY=~/Documents/projects/Strata/.venv/bin/python
W=~/Documents/projects/Strata-wt/r5-upstream
echo "== $(date +%T) dump d1"
$PY -I $M/gen_data.py dump $D/transcripts.jsonl --out $D/d1 > $D/d1.out 2>&1; tail -2 $D/d1.out
$PY -I $M/gen_data.py check $D/d1
echo "== $(date +%T) parity build"
git -C $W apply $M/parity_commit.diff && $P/build.sh $W && cp $W/build-hip/strata $P/bin/r9-parity
git -C $W checkout -- src/core/mtp.cpp
echo "== $(date +%T) parity run"
rm -f $D/par.bin $D/par_r.bin
$P/r8_run.sh r9par_prose prose.ids $P/bin/r9-parity STRATA_MTP_PARITY=$D/par.bin STRATA_PREFILL_DUMP_R_ALL=$D/par_r.bin \
  STRATA_PREFILL_CPU_SHARE=0 -- --short-read 0 --prompt-cache 0
ls -la $D/par.bin $D/par_r.bin
echo "== $(date +%T) smoke train"
$M/gpu_hold.sh $M/.venv/bin/python $M/train.py --data $D/d1 --out $D/rt-smoke --ckpt $D/smoke.pt --minutes 8 \
  --eval-every 50 --n-held 8 > $D/smoke.out 2>&1; tail -15 $D/smoke.out
echo "== $(date +%T) CHAIN A DONE"
