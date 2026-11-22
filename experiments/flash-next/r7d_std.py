"""r7 draft lever: A/B runs priced at one shared cost table per prompt (removes run-to-run machine noise; r5sc_std.py
method). Windows (size T, accepted) come from each run's .hashes; a window of size T costs the median round ms of all
logged windows of that size in the clean runs (verify host <= 1 ms) of every arm of the tag.

usage: [PROMPTS="p ..."] python3 r7d_std.py TAG [ARM ...]      (runs $LCT_HOME/experiments/flash-next/logs/r7d_TAG_<arm>_<prompt>_<round>; default arms a b)
Prints per arm and prompt: runs, tokens per window, measured t/s (clean runs), modelled t/s and ratio to arm a.
"""
import os
import re
import sys
from pathlib import Path

HOME = os.environ.get("LCT_HOME") or Path(__file__).resolve().parents[2] / "tmp"
L = Path(HOME) / "experiments/flash-next/logs"
tag = sys.argv[1]
arms = sys.argv[2:] or ["a", "b"]


def host_ok(log):
    m = re.search(r"host ([\d.]+)\s+commit", log.read_text(errors="replace"))
    return m is not None and float(m.group(1)) <= 1.0


def tps(log):
    m = re.search(r"^decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", log.read_text(errors="replace"), re.M)
    return float(m.group(1)) if m else 0.0


def windows(h):
    rows = [line.split() for line in h.read_text().splitlines() if line.strip()]
    return [(int(w[2]), int(n[1]) - int(w[1]) - 1) for w, n in zip(rows[1:], rows[2:])]   # first: the prompt's token


for ids in os.environ.get("PROMPTS", "prose code edit").split():   # PROMPTS="long200k": other prompts
    runs = {arm: sorted(L.glob(f"r7d_{tag}_{arm}_{ids}_*.hashes")) for arm in arms}
    cost = {}
    for arm in arms:
        for h in runs[arm]:
            if not host_ok(h.with_suffix(".log")):
                continue
            for line in h.with_suffix(".spec").read_text().splitlines()[1:]:
                f = line.split()
                if f and f[0] == "W":
                    cost.setdefault(int(f[2]), []).append(float(f[7]))
    cost = {t: sorted(v)[len(v) // 2] for t, v in cost.items() if len(v) >= 5}
    if len(cost) < 2:
        continue
    known = sorted(cost)
    for t in range(1, 9):
        if t in cost:
            continue
        lo = [u for u in known if u < t]
        hi = [u for u in known if u > t]
        a, b = (lo[-1], hi[0]) if lo and hi else (known[-2], known[-1]) if lo else (known[0], known[1])
        cost[t] = cost[a] + (cost[b] - cost[a]) * (t - a) / (b - a)
    print(f"{ids}: round ms by T " + " ".join(f"{t}:{c:.1f}" for t, c in sorted(cost.items())))
    base = None
    for arm in arms:
        out = []
        for h in runs[arm]:
            W = windows(h)
            tok = sum(a + 1 for _, a in W)
            out.append((tok / len(W), 1000.0 * tok / sum(cost[t] for t, _ in W),
                        tps(h.with_suffix(".log")) if host_ok(h.with_suffix(".log")) else None))
        if not out:
            continue
        tpw = sum(o[0] for o in out) / len(out)
        mod = sum(o[1] for o in out) / len(out)
        meas = [o[2] for o in out if o[2]]
        base = base or mod
        print(f"  {arm:4s} n={len(out)}  {tpw:.2f} tok/window  measured {' '.join(f'{m:.1f}' for m in meas) or '-'} t/s"
              f"  modelled {mod:6.1f} t/s  x{mod / base:.3f}")
