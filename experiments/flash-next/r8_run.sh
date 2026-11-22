#!/usr/bin/env bash
# usage: r8_run.sh NAME IDS ENGINE [K=V ...] -- [strata args ...]
# One gen.sh run (600 tokens) with the round-7 decode setup: r7_run.sh's flags and env, plus the round-7 env
# switches and the Basalt/q2b drafter (--mtp $STRATA_DATA/mtp/rt-basalt-q2b). K=V overrides the env (later wins).
# Logs as r7_run.sh: NAME.log, .hashes (per-window ids + main-model argmaxes), .spec.
set -u
P=$(dirname "$(readlink -f "$0")")
D=${STRATA_DATA:?set STRATA_DATA to the Strata data folder}
name=$1; ids=$2; eng=$3; shift 3
envs=(STRATA_SH_GATE_TAIL=1 STRATA_COMBINE_MAPPED=1 STRATA_QSA_GATE_Q8=1 STRATA_VERIFY_QDEDUP=1 STRATA_QFUSE=1
      STRATA_DF_PLE=1 STRATA_MAPPED_FAST=1 STRATA_T1_FUSE=1 STRATA_QSA_FOLD=1 STRATA_ROUTER_TAIL_ROWS=4
      STRATA_MTP_NO_RCOPY=1 STRATA_MTP_MIX_FUSED=1)
while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
[ $# -gt 0 ] && shift
exec "$P/r7_run.sh" "$name" "$ids" "$eng" "${envs[@]}" -- --mtp "$D/mtp/rt-basalt-q2b" "$@"
