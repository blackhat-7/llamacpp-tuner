#!/usr/bin/env bash
# usage: ab.sh TAG REPS PROMPTS ARMS...   e.g. ab.sh nv1 2 "code prose edit" base nv
# One greedy generate run per (rep, prompt, arm), interleaved, through run.sh (gpu.lock, temps, follow texts,
# window hashes). Arms: base = the engine with no helper; nv = + the 4070 (STRATA_NV_HELPER, --expert-cache-device1
# auto); nvN = + the 4070 with N slots; nvd = nv + STRATA_DISJOINT_ADAPT=1 (the XTX's tier never promotes the 4070's
# experts); name%K=V%K2=V2 = env switches for that arm only; arm.label = the same arm under another log name; arm
# words after ':' are extra engine args (e.g. nv:--pcie-frac:0.2).
# ENG / PLUG pick engine and plugin (see run.sh); RUN picks another runner with run.sh's arguments.
# run.sh's own env already sets STRATA_DISJOINT_ADAPT=1, so nv = nvd here; nv%STRATA_DISJOINT_ADAPT=0 is the
# old nv (the study ran this driver over a single-GPU runner without that switch).
# Logs: $LCT_HOME/experiments/dual-gpu/logs/r7d_TAG_<arm>_<prompt>_<rep>.{log,hashes,spec};
# price with engine_price.py TAG PROMPT ARMS.
set -u
P=$(dirname "$(readlink -f "$0")")
tag=$1; reps=$2; prompts=$3; shift 3
D=${STRATA_DATA:?set STRATA_DATA to the Strata data folder}
PLUG=${PLUG:-$D/engines/libstrata_nvhelper-ac085da1.so}
for rep in $(seq 1 "$reps"); do
  for p in $prompts; do
    for armspec in "$@"; do
      arm=${armspec%%:*}; extra=(); [ "$arm" != "$armspec" ] && IFS=: read -r -a extra <<< "${armspec#*:}"
      envs=(); args=(); armenv=()
      # "name%K=V%K2=V2" sets env switches for this arm only
      [ "$arm" != "${arm%%\%*}" ] && { IFS=% read -r -a armenv <<< "${arm#*%}"; arm=${arm%%\%*}; }
      kind=${arm%%.*}   # an arm "nv.label" runs as nv; the label only names its logs
      case "$kind" in
        base*) ;;
        nvd*) envs=(STRATA_NV_HELPER="$PLUG" STRATA_DISJOINT_ADAPT=1 ${NVENV:-}); n=${kind#nvd}; args=(--expert-cache-device1 "${n:-auto}") ;;
        nv*) envs=(STRATA_NV_HELPER="$PLUG" ${NVENV:-}); n=${kind#nv}; args=(--expert-cache-device1 "${n:-auto}") ;;
      esac
      name="r7d_${tag}_${arm}_${p}_${rep}"
      NOHELPER=1 "${RUN:-$P/run.sh}" "$name" "$p" "${envs[@]}" "${armenv[@]}" -- "${args[@]}" "${extra[@]}"
    done
  done
done
