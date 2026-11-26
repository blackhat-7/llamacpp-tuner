"""Score gpqa.py runs on the questions every run finished, with 95% intervals.

python3 -I score.py flash.jsonl inkling.jsonl
"""

import json
import math
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return mid - half, mid + half


runs = {
    Path(f).stem: {
        r["idx"]: r for r in map(json.loads, Path(f).read_text().splitlines())
    }
    for f in sys.argv[1:]
}
common = set.intersection(*(set(r) for r in runs.values()))
n = len(common)
print(f"{n} questions answered by every run")
for name, recs in runs.items():
    rs = [recs[i] for i in common]
    k = sum(r["correct"] for r in rs)
    lo, hi = wilson(k, n)
    toks = sorted(r["completion_tokens"] for r in rs)
    print(
        f"{name:10} {k}/{n} = {100 * k / n:.1f}%  (95% CI {100 * lo:.1f}-{100 * hi:.1f})  "
        f"median {toks[n // 2]} tok, mean {sum(toks) / n:.0f} tok, "
        f"mean {sum(r['seconds'] for r in rs) / n:.0f} s, "
        f"no answer {sum(r['pred'] is None for r in rs)}, cut at max_tokens {sum(r['finish'] == 'length' for r in rs)}"
    )
if len(runs) == 2:
    a, b = runs.values()
    only_a = sum(a[i]["correct"] and not b[i]["correct"] for i in common)
    only_b = sum(b[i]["correct"] and not a[i]["correct"] for i in common)
    # Exact two-sided McNemar test on the questions the two runs disagree on.
    m = only_a + only_b
    p = (
        min(
            1.0, 2 * sum(math.comb(m, j) for j in range(min(only_a, only_b) + 1)) / 2**m
        )
        if m
        else 1.0
    )
    names = list(runs)
    print(
        f"only {names[0]} right: {only_a}, only {names[1]} right: {only_b}, McNemar p = {p:.3f}"
    )
