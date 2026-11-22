#!/bin/bash
# usage: e2e.sh PKG_DIR LABEL MAX_TOKENS PROMPT_FILE... ; serves PKG_DIR (preset xtx-nodraft) under the GPU lock,
# sends each prompt once, appends results to e2e.jsonl, and stops the server.
LCT_HOME=${LCT_HOME:-$(cd "$(dirname "$0")/../.." && pwd)/tmp}
LOOM=$LCT_HOME/loom                          # venv and packages
OUT=$LCT_HOME/experiments/inkling-loom; mkdir -p "$OUT"   # logs and results
pkg=$1; label=$2; mt=$3; shift 3
here=$(cd "$(dirname "$0")" && pwd)
export EXTRA PRESET KINDS LOOM OUT here
exec flock /tmp/xtx-gpu.lock bash -c '
  pkg=$1; label=$2; mt=$3; shift 3
  log=$OUT/serve-$label.log
  $LOOM/venv/bin/loom-serve --model=$pkg --preset=${PRESET:-xtx-nodraft} "--device=amdgpu://?completion_spin_ns=4294967295" --pool_backing=fixed --listen=127.0.0.1:8097 $EXTRA > $log 2>&1 &
  pid=$!; echo $pid > $OUT/serve-$label.pid
  for i in $(seq 600); do grep -q "\"ready\"" $log && break; kill -0 $pid 2>/dev/null || { echo "server died"; tail -5 $log; exit 1; }; sleep 1; done
  for p in "$@"; do for k in ${KINDS:-prose}; do $LOOM/venv/bin/python $here/e2e_client.py http://127.0.0.1:8097 $p $mt "$label:$(basename $p)" $k | tee -a $OUT/e2e.jsonl | cut -c1-300; done; done
  kill $pid; wait $pid; echo "server $pid stopped"
' _ "$pkg" "$label" "$mt" "$@"
