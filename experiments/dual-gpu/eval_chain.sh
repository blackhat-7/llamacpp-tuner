#!/usr/bin/env bash
# usage: eval_chain.sh MODEL_ID ARM=STACK [ARM=STACK ...]   e.g. eval_chain.sh swift-1.5-iq3_xxs mixd=local-flash-mixd base=local-flash
# Agent replay A/B: for each arm, `lct up STACK`, wait for :6868, replay tasks 1-5 once and tasks 1-3 a second time
# through the agent replay harness (run_eval.py in AB_EVAL, default $LCT_HOME/ab-eval), `lct down STACK`.
# Holds the shared gpu.lock for the whole chain. Results append to $AB_EVAL/eval/results.jsonl, one line per run.
set -u
P=$(dirname "$(readlink -f "$0")")
R=$(cd "$P/../.." && pwd)
LCT_HOME=${LCT_HOME:-$R/tmp}
E=${AB_EVAL:-$LCT_HOME/ab-eval}
model=$1; shift
mkdir -p "$LCT_HOME/experiments"
exec 9>"$LCT_HOME/experiments/gpu.lock"; flock 9
for pair in "$@"; do
  arm=${pair%%=*}; stack=${pair#*=}
  cd "$R" && uv run lct up "$stack"   # a broken side server fails `up`; the model server is checked below
  ok=0
  for _ in $(seq 1 200); do curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:6868/v1/models | grep -q 200 && { ok=1; break; }; sleep 3; done
  if [ $ok = 0 ]; then echo "$(date +%T) $arm: server did not answer"; cd "$R" && uv run lct down "$stack"; continue; fi
  echo "$(date +%T) $arm up"
  cd "$E" && python3 -I run_eval.py "$arm" "$model" 1 1 2 3 4 5
  cd "$E" && START_RUN=2 python3 -I run_eval.py "$arm" "$model" 2 1 2 3
  cd "$R" && uv run lct down "$stack"
  echo "$(date +%T) $arm done"
done
echo CHAIN_DONE
