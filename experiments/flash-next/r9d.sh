#!/usr/bin/env bash
# usage: r9d.sh TAG ENGINE DRAFTER
# Drafter x draft-stop arms on the follow texts (code, prose, edit; greedy, 600 tokens) with the round-7 setup
# (r8_run.sh), named logs/r7d_TAG_<arm>_<prompt>_1 so r7d_std.py prices every arm at one shared round-ms-by-T
# table (removes machine noise). a = r8_run.sh as is; b = DRAFTER (an --mtp folder); c = DRAFTER + mid table +
# min-p 0.3; d = DRAFTER + its greedy table; e = DRAFTER + mid table + S 0.25 + min-p 0.
# Round 9 ran it with ENGINE = engines/strata-0.1.43-gfx1100opt-210ebbdd and DRAFTER = the fitted rt-sd2.
set -u
P=$(dirname "$(readlink -f "$0")")
tag=$1; E=$2; SD=$3
MID=0.29,0.46,0.60,0.70,0.79,0.87,0.90,0.99
OWN=0.25,0.42,0.60,0.75,0.84,0.92,0.97,1.00
for ids in code prose edit; do
  "$P/r8_run.sh" "r7d_${tag}_a_${ids}_1" "$ids.ids" "$E"
  "$P/r8_run.sh" "r7d_${tag}_b_${ids}_1" "$ids.ids" "$E" -- --mtp "$SD"
  "$P/r8_run.sh" "r7d_${tag}_c_${ids}_1" "$ids.ids" "$E" STRATA_SPEC_CUM_ACC=$MID -- --mtp "$SD" --spec-min-p 0.3
  "$P/r8_run.sh" "r7d_${tag}_d_${ids}_1" "$ids.ids" "$E" STRATA_SPEC_CUM_ACC=$OWN -- --mtp "$SD"
  "$P/r8_run.sh" "r7d_${tag}_e_${ids}_1" "$ids.ids" "$E" STRATA_SPEC_CUM_ACC=$MID STRATA_SPEC_CUM=0.25 -- --mtp "$SD" --spec-min-p 0
done
python3 "$P/r7d_std.py" "$tag" a b c d e
