#!/usr/bin/env bash
# chain C (2026-10-18): training variants on d1 + d2, one after another under gpu.lock (grouped-MoE train.py).
#   sd2: KL + TV (w 1), lr 2e-5      sd3: KL, lr 5e-5
set -u
M=$(dirname "$(readlink -f "$0")"); cd "$M"
run() { name=$1; shift; echo "== $(date +%T) $name $*"
  ./gpu_hold.sh .venv/bin/python train.py --data data/d1 data/d2 --out data/rt-$name --ckpt data/$name.pt --n-held 69 \
    --minutes 60 "$@" > data/$name.out 2>&1
  grep -E 'held-out|early' data/$name.out | tail -2; }
run sd2 --tv 1.0
run sd3 --lr 5e-5
echo "== $(date +%T) CHAIN C DONE"
