#!/bin/bash
# Real Claude Code sessions taking turns: N `claude-local --offline -p` runs at once against lct
# on :6890 (read-only tools), each on its own task in this repo. Writes answers and timings.
cd "$(dirname "$0")/../../.."  # the repo root
out=${LCT_HOME:-tmp}/multiagent/claude
rm -rf "$out"; mkdir -p "$out"
tasks=(
  "Read llamacpp_tuner/parking.py. In three short bullets: how is a request matched to a parked chat?"
  "Read llamacpp_tuner/backends/loom.py. List the Loom flags and the memory-lock rule lct applies. Short."
  "Read docs/inkling-loom-research.md. List its open leads in three bullets."
  "Read engines/loom/README.md. One line each: what do changes 6, 7, 8 and 9 do?"
)
n=${1:-${#tasks[@]}}
for ((i = 0; i < n; i++)); do
  (
    start=$(date +%s)
    LOCAL_MODEL_BASE_URL=http://127.0.0.1:6890 timeout 1800 claude-local --offline -p "${tasks[$i]}" \
      --allowedTools "Read,Grep,Glob" \
      --disallowedTools "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Task" \
      >"$out/session-$i.txt" 2>"$out/session-$i.err"
    echo "session $i exit $? $(($(date +%s) - start)) s" >>"$out/summary.txt"
  ) &
done
wait
cat "$out/summary.txt"
