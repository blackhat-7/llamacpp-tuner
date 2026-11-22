#!/usr/bin/env bash
# usage: ab.sh TAG ROUNDS "ENGINE_A [K=V ...]" "ENGINE_B [K=V ...]" [extra strata args...]
# Interleaved A/B in generate mode on code/edit/prose (600 tokens, greedy). Each arm is an engine binary plus
# optional environment switches. Prints whether B's token ids match A's, then the mean speed per prompt.
set -u
P=$(dirname "$(readlink -f "$0")")
LCT_HOME=${LCT_HOME:-$(cd "$P/../.." && pwd)/tmp}
export LCT_HOME
L=$LCT_HOME/experiments/flash-next/logs
tag=$1; rounds=$2; A=$3; B=$4; shift 4
run() { local spec=$1 name=$2 ids=$3; shift 3; read -r eng envs <<< "$spec"; env $envs "$P/gen.sh" "$eng" "$name" "$ids" 600 "$@" > /dev/null; }
for r in $(seq 1 "$rounds"); do
  for ids in code edit prose; do
    run "$A" "${tag}_A_${ids}_$r" "$ids.ids" "$@"
    run "$B" "${tag}_B_${ids}_$r" "$ids.ids" "$@"
    python3 "$P/cmp.py" "$L/${tag}_A_${ids}_$r.log" "$L/${tag}_B_${ids}_$r.log" | tail -1
  done
done
python3 - "$L" "$tag" "$rounds" <<'PY'
import re, sys
from pathlib import Path
logs, tag, rounds = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
def tps(p):
    m = re.search(r"^decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", p.read_text(errors="replace"), re.M)
    return float(m.group(1)) if m else float("nan")
for ids in ("code", "edit", "prose"):
    a = [tps(logs / f"{tag}_A_{ids}_{r}.log") for r in range(1, rounds + 1)]
    b = [tps(logs / f"{tag}_B_{ids}_{r}.log") for r in range(1, rounds + 1)]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    print(f"{ids:6s} A {ma:6.2f}  B {mb:6.2f}  B/A {mb / ma:.3f}")
PY
