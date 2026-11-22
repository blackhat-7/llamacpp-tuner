#!/usr/bin/env bash
# usage: run.sh NAME PROMPT [K=V ...] -- [strata args ...]
# One greedy generate run (600 tokens, follow text + window hashes when PROMPT.follow exists) with the default
# local-flash setup of 2026-11-16 evening: mixB pack + the 4070 helper (strata-swift-mixb-dual.json's args and env),
# learned mixB profile (read only), --adapt-every 4 as the server runs, STRATA_PREFILL_CPU_SHARE=0 for exactness.
# ENG / PLUG pick the engine and the CUDA helper plugin (default: installed ac085da1). NOHELPER=1 drops the 4070.
# GGUF / PACK / PROFILE pick the model files. K=V words override env switches; strata args after -- override
# earlier ones. Prompts (PROMPT.ids, PROMPT.follow) come from ../flash-next. Holds the shared gpu.lock + build.lock.
# Logs: $LCT_HOME/experiments/dual-gpu/logs/NAME.{log,hashes,spec}. RPROF="rocprofv3 options" with ROCPROF=path
# wraps the run in rocprofv3 (CSVs in logs/prof_NAME/).
# Env: STRATA_DATA, STRATA_SRC (required).
set -u
P=$(dirname "$(readlink -f "$0")")
I=$P/../flash-next
LCT_HOME=${LCT_HOME:-$(cd "$P/../.." && pwd)/tmp}
X=$LCT_HOME/experiments; L=$X/dual-gpu/logs
D=${STRATA_DATA:?set STRATA_DATA to the Strata data folder}
S=${STRATA_SRC:?set STRATA_SRC to the Strata source checkout}
name=$1; p=$2; shift 2
eng=${ENG:-$D/engines/strata-0.2.9-gfx1100opt-nvhelper-ac085da1}
plug=${PLUG:-$D/engines/libstrata_nvhelper-ac085da1.so}
G=${GGUF:-$LCT_HOME/models/hybrid-swift-abl-iq4_xs/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
PACK=${PACK:-$D/packs/swift-abl-mixB}; PROF=${PROFILE:-$D/profiles/swift-1.5-mixb-learned-frozen13.bin}
envs=(STRATA_DISJOINT_ADAPT=1 STRATA_HIPBLASLT_TUNING=$S/tools/hip/gfx1100-hipblaslt-100202.txt
  STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_SWITCH_MIN_T=4096 STRATA_SPEC_COUPLED=1 STRATA_SPEC_GUMBEL=1
  STRATA_FUSE_WAIT=1 STRATA_FUSE_HITS=1 STRATA_QFUSE_HC=1 STRATA_GR_NORM_FAST=1 STRATA_ADAPT_LAG=2 STRATA_HIP_WMMA=1
  STRATA_SELECT_WMMA=1 STRATA_HC_UPMIX=1 STRATA_GDN_HEAD=1 STRATA_GDN_PP=2 STRATA_GDN_CONVL2=1 STRATA_GDN_NOY=1
  STRATA_PF_PAD=1 STRATA_SPEC_CUM=0.35 STRATA_SPEC_CUM_ACC=0.29,0.46,0.60,0.70,0.79,0.87,0.90,0.99
  STRATA_PREFILL_CPU_SHARE=0 STRATA_PREFILL_CPU_SHARE_MAX=4096 STRATA_SH_GATE_TAIL=1 STRATA_COMBINE_MAPPED=1
  STRATA_QSA_GATE_Q8=1 STRATA_VERIFY_QDEDUP=1 STRATA_QFUSE=1 STRATA_DF_PLE=1 STRATA_MAPPED_FAST=1 STRATA_T1_FUSE=1
  STRATA_QSA_FOLD=1 STRATA_ROUTER_TAIL_ROWS=4 STRATA_MTP_NO_RCOPY=1 STRATA_MTP_MIX_FUSED=1 STRATA_PREFILL_EQUAL=1
  STRATA_QSA_NR3=1 STRATA_Z_CONVAB=1 STRATA_QSA_KV_ROLE=1 STRATA_WAIT_PF=1 STRATA_ROUTE_TAIL_SKIP=7 STRATA_HC_Q8=1)
helper=(--expert-cache-device1 auto)
if [ -n "${NOHELPER:-}" ]; then helper=(); else envs+=(STRATA_NV_HELPER="$plug"); fi
while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
[ $# -gt 0 ] && shift
ids=$I/$p.ids
follow=""; [ -f "$I/$p.follow" ] && follow="--spec-follow $I/$p.follow --window-hashes $L/$name.hashes"
mkdir -p "$L"
cd "$S" || exit   # the engine runs from a Strata source tree, as when it was measured
exec 9>"$X/gpu.lock"; flock 9
exec 8>"$X/build.lock"; flock 8
[ -e "$X/STOP" ] && { echo "STOP file present: not starting" >&2; exit 3; }
while [ "$(sensors 2>/dev/null | awk '/Package id 0/ {gsub(/[+°C]/,"",$4); print int($4)}')" -ge 80 ] ||
      [ "$(cat /sys/class/drm/card*/device/hwmon/hwmon*/temp2_input 2>/dev/null | sort -n | tail -1)" -ge 80000 ]; do sleep 5; done
rm -f "$L/$name.spec"
wrap=()
if [ -n "${RPROF:-}" ]; then
  rm -rf "$L/prof_$name"
  # shellcheck disable=SC2206
  wrap=("${ROCPROF:-/opt/rocm/bin/rocprofv3}" $RPROF --output-format csv -d "$L/prof_$name" -o run --)
fi
env "${envs[@]}" STRATA_SPEC_LOG="$L/$name.spec" LD_LIBRARY_PATH=$LCT_HOME/rocr-7.2.4-fix/lib:/opt/rocm/lib \
  timeout "${TMO:-590}" "${wrap[@]}" "$eng" \
  --pack "$PACK" --native "$G" --ple-gguf "$G" --expert-profile "$PROF" \
  --expert-cache auto --prefill auto:16384 --spec 6 --spec-min-p 0.3 --mtp $D/mtp/rt-sd2-swift --max-context 262144 \
  --kv int8 --kv-resident 32768 --vram-reserve-mib 1744 --pcie-frac 0.35 --pcie-mode dma --ple-io mmap \
  --mtp-draft-vocab $D/draft_vocab_agent.bin --mtp-window 8192 --pool-workers 9 --adapt-every 4 --stats \
  "${helper[@]}" --tokens-file "$ids" --max-new "${NEW:-600}" $follow "$@" > "$L/$name.log" 2>&1
echo "$name exit $? $(grep -E '^decode' "$L/$name.log" | awk '{print $(NF-1)}') tok/s"
