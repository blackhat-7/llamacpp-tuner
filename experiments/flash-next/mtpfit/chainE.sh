#!/usr/bin/env bash
# chain E (2026-10-18): more capacity / longer fits on d1 + d2, from sd2's recipe (KL + TV, lr 2e-5).
#   sd4: + trainable expert scales      sd5: TV weight 2, 3 epochs
set -u
M=$(dirname "$(readlink -f "$0")"); cd "$M"
run() { name=$1; shift; echo "== $(date +%T) $name $*"
  ./gpu_hold.sh .venv/bin/python train.py --data data/d1 data/d2 --out data/rt-$name --ckpt data/$name.pt --n-held 69 \
    --minutes 60 "$@" > data/$name.out 2>&1
  grep -E 'held-out|early' data/$name.out | tail -1; }
run sd4 --tv 1.0 --train-scales
run sd5 --tv 2.0 --epochs 3
echo "== $(date +%T) CHAIN E DONE"
