#!/usr/bin/env bash
# chain D (2026-10-18): transcripts for prompts2.jsonl's new prompts (chain B's flags) -> dump them into d3.
set -u
M=$(dirname "$(readlink -f "$0")"); D=$M/data
PY=~/Documents/projects/Strata/.venv/bin/python
echo "== $(date +%T) transcripts (prompts2)"
$PY -I $M/gen_data.py transcripts $D/prompts2.jsonl --out $D/transcripts.jsonl --max-new 768 --think 30 --minutes ${MINUTES:-110} \
  > $D/gen3.out 2>&1; tail -2 $D/gen3.out
n=$(( $(grep -c . $D/d1/index.jsonl) + $(grep -c . $D/d2/index.jsonl) ))
echo "== $(date +%T) dump d3 (skip $n)"
$PY -I $M/gen_data.py dump $D/transcripts.jsonl --out $D/d3 --skip $n > $D/d3.out 2>&1; tail -1 $D/d3.out
$PY -I $M/gen_data.py check $D/d3
echo "== $(date +%T) CHAIN D DONE"
