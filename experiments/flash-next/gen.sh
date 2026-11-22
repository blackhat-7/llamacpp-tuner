#!/usr/bin/env bash
# usage: gen.sh ENGINE NAME IDS MAX_NEW [extra strata args...]
# One greedy Strata generate run with the tuned serving flags (adaptive tier off so runs are repeatable).
# With a FOLLOW file next to IDS (code.ids -> code.follow) the run verifies that fixed continuation and writes
# logs/NAME.hashes: one hash per verify window, identical between builds that do the same arithmetic.
# Writes $LCT_HOME/experiments/flash-next/logs/NAME.log. Holds the shared gpu.lock so two runs never share the GPU.
# Env: STRATA_DATA, STRATA_SRC (required); GGUF, PACK, MTP, PROFILE override the model files.
set -u
P=$(dirname "$(readlink -f "$0")")
LCT_HOME=${LCT_HOME:-$(cd "$P/../.." && pwd)/tmp}
X=$LCT_HOME/experiments; O=$X/flash-next
D=${STRATA_DATA:?set STRATA_DATA to the Strata data folder}
S=${STRATA_SRC:?set STRATA_SRC to the Strata source checkout}
eng=$1; name=$2; ids=$3; n=$4; shift 4
G=${GGUF:-$LCT_HOME/models/ukisai%2FSwift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
mkdir -p "$O/logs"
follow=""
f="$P/${ids%.ids}.follow"
[ -f "$f" ] && follow="--spec-follow $f --window-hashes $O/logs/$name.hashes"
cd "$(dirname "$eng")/.." || exit
exec 9>"$X/gpu.lock"; flock 9
# Wind-down switch: a STOP file makes queued GPU jobs exit instead of starting.
[ -e "$X/STOP" ] && { echo "STOP file present: not starting" >&2; exit 3; }
# Each run loads 40 GB with every core busy; start cool so load bursts do not stack up to the 100 C limit.
while [ "$(sensors 2>/dev/null | awk '/Package id 0/ {gsub(/[+°C]/,"",$4); print int($4)}')" -ge 80 ]; do sleep 5; done
STRATA_HIPBLASLT_TUNING=$S/tools/hip/gfx1100-hipblaslt-100202.txt \
STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_SWITCH_MIN_T=4096 \
STRATA_FUSE_WAIT=${STRATA_FUSE_WAIT:-1} STRATA_FUSE_HITS=${STRATA_FUSE_HITS:-1} STRATA_QFUSE_HC=${STRATA_QFUSE_HC:-1} STRATA_GR_NORM_FAST=${STRATA_GR_NORM_FAST:-1} \
LD_LIBRARY_PATH=$LCT_HOME/rocr-7.2.4-fix/lib:/opt/rocm/lib timeout 900 "$eng" \
  --pack "${PACK:-$D/packs/swift-iq3_xxs}" --native "$G" --ple-gguf "$G" \
  --expert-profile "${PROFILE:-$D/profiles/swift-1.5-iq3_xxs.bin}" --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.5 --mtp "${MTP:-$D/mtp/rt}" --max-context 262144 --kv int8 --kv-resident 32768 \
  --vram-reserve-mib 4000 --pcie-frac 0.35 --pcie-mode dma --ple-io ram --pool-workers 13 --adapt-every 0 --stats \
  --tokens-file "$P/$ids" --max-new "$n" $follow "$@" > "$O/logs/$name.log" 2>&1
echo "$name exit $? $(grep -E '^decode' "$O/logs/$name.log" | awk '{print $(NF-1)}') tok/s"
