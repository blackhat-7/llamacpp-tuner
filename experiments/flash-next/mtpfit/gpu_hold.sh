#!/usr/bin/env bash
# usage: gpu_hold.sh CMD [ARGS...]
# ../gpu_run.sh without its 10-minute timeout, for jobs that bound themselves (train.py --minutes): the shared
# gpu.lock, the STOP file, and a CPU below 80 C before starting.
set -u
P=$(dirname "$(readlink -f "$0")")
LCT_HOME=${LCT_HOME:-$(cd "$P/../../.." && pwd)/tmp}
X=$LCT_HOME/experiments
mkdir -p "$X"
exec 9>"$X/gpu.lock"; flock 9
[ -e "$X/STOP" ] && { echo "STOP file present: not starting" >&2; exit 3; }
while [ "$(sensors 2>/dev/null | awk '/Package id 0/ {gsub(/[+°C]/,"",$4); print int($4)}')" -ge 80 ]; do sleep 5; done
LD_LIBRARY_PATH=$LCT_HOME/rocr-7.2.4-fix/lib:/opt/rocm/lib "$@"
