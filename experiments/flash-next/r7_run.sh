#!/usr/bin/env bash
# usage: r7_run.sh NAME IDS ENGINE [K=V ...] -- [strata args ...]
# One gen.sh run (600 tokens) with the round-6 decode setup: the user's flags (reserve 2000, agent draft
# vocabulary, mmap PLE, Swift profile, mtp-window 8192) and env (cumulative draft stop t4). K=V overrides the env.
# Logs in $LCT_HOME/experiments/flash-next/logs: NAME.log, NAME.hashes, NAME.spec (per-window spec log).
set -u
P=$(dirname "$(readlink -f "$0")")
LCT_HOME=${LCT_HOME:-$(cd "$P/../.." && pwd)/tmp}
export LCT_HOME
L=$LCT_HOME/experiments/flash-next/logs
D=${STRATA_DATA:?set STRATA_DATA to the Strata data folder}
name=$1; ids=$2; eng=$3; shift 3
envs=(STRATA_SPEC_CUM=0.35 STRATA_SPEC_CUM_ACC=0.12,0.26,0.43,0.51,0.59,0.66,0.79,0.96)
while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
[ $# -gt 0 ] && shift
B="--vram-reserve-mib 2000 --mtp-draft-vocab $D/draft_vocab_agent.bin --ple-io mmap --expert-profile $D/profiles/swift-1.5-iq3_xxs.bin --mtp-window 8192"
mkdir -p "$L"
rm -f "$L/$name.spec"
env "${envs[@]}" STRATA_SPEC_LOG="$L/$name.spec" "$P/gen.sh" "$eng" "$name" "$ids" 600 $B "$@"
