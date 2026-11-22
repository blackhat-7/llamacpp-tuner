"""Engine speed on one window list: each arm's own round cost by window size T (the median round ms of its clean runs,
verify host <= 1 ms, from .spec) prices the same windows (every clean run of every arm, from .hashes).  Run-to-run
window differences (drafting noise) then drop out; what is left is the per-round cost of each engine.

usage: python3 -I engine_price.py TAG PROMPT [ARM ...]   ($LCT_HOME/experiments/dual-gpu/logs/r7d_TAG_<arm>_<prompt>_<n>;
default arms ours port)
"""
import os
import re
import statistics
import sys
from pathlib import Path

L = Path(os.environ.get("LCT_HOME") or Path(__file__).resolve().parents[2] / "tmp") / "experiments/dual-gpu/logs"
tag, prompt = sys.argv[1], sys.argv[2]
arms = sys.argv[3:] or ["ours", "port"]


def clean(log):
    m = re.search(r"host ([\d.]+)\s+commit", log.read_text(errors="replace"))
    return m is not None and float(m.group(1)) <= 1.0


def windows(h):   # (T, drafts accepted) per window; the first row is the prompt's token
    rows = [line.split() for line in h.read_text().splitlines() if line.strip()]
    return [(int(w[2]), int(n[1]) - int(w[1]) - 1) for w, n in zip(rows[1:], rows[2:])]


runs = {a: [h for h in sorted(L.glob(f"r7d_{tag}_{a}_{prompt}_*.hashes")) if clean(h.with_suffix(".log"))] for a in arms}
common = [w for a in arms for h in runs[a] for w in windows(h)]
if not common:
    sys.exit("no clean runs")
tokens = sum(acc + 1 for _, acc in common)
base = None
for a in arms:
    ms = {}
    for h in runs[a]:
        for line in h.with_suffix(".spec").read_text().splitlines()[1:]:   # the first window carries the warm-up
            f = line.split()
            if f and f[0] == "W":
                ms.setdefault(int(f[2]), []).append(float(f[7]))
    cost = {t: statistics.median(v) for t, v in ms.items() if len(v) >= 3}
    known = sorted(cost)
    for t in {t for t, _ in common} - set(cost):   # a size this arm rarely ran: linear in T (r7d_std.py's rule)
        lo = [u for u in known if u < t]
        hi = [u for u in known if u > t]
        x0, x1 = (lo[-1], hi[0]) if lo and hi else (known[-2], known[-1]) if lo else (known[0], known[1])
        cost[t] = cost[x0] + (cost[x1] - cost[x0]) * (t - x0) / (x1 - x0)
    tps = 1000.0 * tokens / sum(cost[t] for t, _ in common)
    base = base or tps
    print(f"{prompt} {a:5s} clean runs {len(runs[a])}: round ms by T " +
          " ".join(f"{t}:{cost[t]:.2f}" for t in sorted(cost) if t <= 6) + f"  -> {tps:.1f} t/s on {len(common)} windows"
          f"  x{tps / base:.3f}")
