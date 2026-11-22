"""Compare Strata generate logs: decode speed, tokens per round, and whether the output token ids match.

usage: python3 cmp.py BASE.log OTHER.log [...]
"""
import re
import sys
from pathlib import Path


def read(path):
    text = Path(path).read_text(errors="replace")
    out = re.search(r"^output\s*:\s*(.*)$", text, re.M)
    dec = re.search(r"^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", text, re.M)
    spec = re.search(r"([\d.]+) tokens per round", text)
    return {
        "ids": out.group(1).split() if out else [],
        "tps": float(dec.group(3)) if dec else 0.0,
        "tpr": spec.group(1) if spec else "?",
    }


base = read(sys.argv[1])
for p in sys.argv[1:]:
    r = read(p)
    same = next((i for i, (a, b) in enumerate(zip(base["ids"], r["ids"])) if a != b), None)
    match = "identical" if same is None and len(base["ids"]) == len(r["ids"]) else f"differs at token {same}"
    ha, hb = Path(sys.argv[1]).with_suffix(".hashes"), Path(p).with_suffix(".hashes")
    if ha.exists() and hb.exists():
        la, lb = ha.read_text().splitlines(), hb.read_text().splitlines()
        bad = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), None)
        if bad is None and len(la) == len(lb):
            match += ", windows bit-identical"
        else:
            # same window (index, position, size) but other argmaxes = a prediction flipped by rounding
            flips = sum(1 for x, y in zip(la, lb) if x.split()[:3] == y.split()[:3] and x.split()[4:] != y.split()[4:])
            match += f", window {bad} differs ({flips} of {min(len(la), len(lb))} windows flip a prediction)"
    print(f"{Path(p).stem:40s} {r['tps']:7.2f} tok/s  {r['tpr']:>5s} tok/round  {len(r['ids'])} ids  {match}")
