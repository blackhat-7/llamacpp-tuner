#!/usr/bin/env bash
# usage: prof.sh ENGINE NAME IDS MAX_NEW [extra strata args...]
# gen.sh's run under rocprofv3 --kernel-trace; writes logs/prof_NAME/ (kernel_stats.csv, kernel_trace.csv).
# ROCPROF picks the profiler (default /opt/rocm/bin/rocprofv3; Arch's ROCm 7.2.4 ships none).
set -u
P=$(dirname "$(readlink -f "$0")")
LCT_HOME=${LCT_HOME:-$(cd "$P/../.." && pwd)/tmp}
export LCT_HOME
L=$LCT_HOME/experiments/flash-next/logs
eng=$1; name=$2; shift 2
mkdir -p "$L"
wrap="$L/prof_$name.sh"
printf '#!/usr/bin/env bash\nexec %q --kernel-trace --stats --output-format csv -d %q -o run -- %q "$@"\n' \
  "${ROCPROF:-/opt/rocm/bin/rocprofv3}" "$L/prof_$name" "$eng" > "$wrap"
chmod +x "$wrap"
"$P/gen.sh" "$wrap" "prof_$name" "$@"
