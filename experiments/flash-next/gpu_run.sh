#!/usr/bin/env bash
# usage: gpu_run.sh CMD [ARGS...]
# Runs any GPU program (parity tests, microbenchmarks) under the shared gpu.lock, after the CPU cools below 80 C,
# with a 10-minute timeout. gen.sh / ab.sh already take the same lock themselves.
set -u
P=$(dirname "$(readlink -f "$0")")
LCT_HOME=${LCT_HOME:-$(cd "$P/../.." && pwd)/tmp}
X=$LCT_HOME/experiments
mkdir -p "$X"
exec 9>"$X/gpu.lock"; flock 9
[ -e "$X/STOP" ] && { echo "STOP file present: not starting" >&2; exit 3; }
while [ "$(sensors 2>/dev/null | awk '/Package id 0/ {gsub(/[+°C]/,"",$4); print int($4)}')" -ge 80 ]; do sleep 5; done
LD_LIBRARY_PATH=$LCT_HOME/rocr-7.2.4-fix/lib:/opt/rocm/lib timeout 600 "$@"
