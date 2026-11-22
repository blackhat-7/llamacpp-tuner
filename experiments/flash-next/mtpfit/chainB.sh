#!/usr/bin/env bash
# chain B (2026-10-18): the remaining transcripts (same flags as chunk 1) -> dump them into d2 (skipping d1's lines).
set -u
M=$(dirname "$(readlink -f "$0")"); D=$M/data
PY=~/Documents/projects/Strata/.venv/bin/python
echo "== $(date +%T) transcripts"
$PY -I $M/gen_data.py transcripts $D/prompts.jsonl --out $D/transcripts.jsonl --max-new 768 --think 30 --minutes ${MINUTES:-110} \
  > $D/gen2.out 2>&1; tail -2 $D/gen2.out
n1=$(grep -c . $D/d1/index.jsonl)
echo "== $(date +%T) dump d2 (skip $n1)"
$PY -I $M/gen_data.py dump $D/transcripts.jsonl --out $D/d2 --skip $n1 > $D/d2.out 2>&1; tail -1 $D/d2.out
$PY -I $M/gen_data.py check $D/d2
echo "== $(date +%T) CHAIN B DONE"
