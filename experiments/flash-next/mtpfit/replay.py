"""replay.py - the draft stop's tokens per second, replayed offline on calib.py --save rows.

    python3 replay.py BASE.npz [OTHER.npz ...] [--mode greedy|coupled]

A window starts at a base: the drafter offers drafts 1..cap while each p >= min_p and the product of acc(p) stays at or
above S (include/strata/spec/draft_cum.hpp); the main model keeps the leading drafts that hold, plus its own token, and
the next window starts after them. Round cost by window size T is the A/B fit over temp-1 requests (data/ab_sd2.jsonl:
14.4 ms per round + 2.56 ms per offered draft, rel. error 5%; a 5th row costs 5 ms, measured in r5-spec-cum). The
oracle offers exactly the drafts that hold.
Every step's draft is the one calib.py made on the main model's tokens, as in a window whose earlier drafts held.
"""
import argparse

import numpy as np

EDGES = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
INSTALLED = np.array([0.12, 0.26, 0.43, 0.51, 0.59, 0.66, 0.79, 0.96])


def cost(t):
    return 14.4 + 2.56 * (t - 1) + (2.44 if t >= 5 else 0.0)


def walk(p, h, cap, min_p, s, tab):
    """Tokens and ms over one transcript: p, h [steps, bases]; tab None = the oracle."""
    tok, ms, j = 0, 0.0, 0
    while j < p.shape[1] and not np.isnan(p[0, j]):
        if tab is None:
            n = 0
            while n < cap and h[n, j] == 1:
                n += 1
        else:
            n, c = 0, 1.0
            while n < cap and not np.isnan(p[n, j]):
                c *= tab[np.searchsorted(EDGES, p[n, j], side="right")]
                if p[n, j] < min_p or c < s:
                    break
                n += 1
        a = 0
        while a < n and h[a, j] == 1:
            a += 1
        tok, ms, j = tok + a + 1, ms + cost(n + 1), j + a + 1
    return tok, ms


def rate(per, mode, cap, min_p, s, tab):
    """Tokens per second over every transcript."""
    tok = ms = 0.0
    for x in per:
        p, h = x[2 * mode], x[2 * mode + 1]
        t, m = walk(p, h, cap, min_p, s, tab)
        tok, ms = tok + t, ms + m
    return tok / ms * 1e3


def table(per, mode):
    """The per-band acceptance over all steps (calib.py's STRATA_SPEC_CUM_ACC)."""
    p = np.concatenate([x[2 * mode].ravel() for x in per])
    h = np.concatenate([x[2 * mode + 1].ravel() for x in per])
    keep = ~np.isnan(p)
    b = np.searchsorted(EDGES, p[keep], side="right")
    return np.array([h[keep][b == i].mean() if (b == i).any() else 0.0 for i in range(8)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--mode", choices=("greedy", "coupled"), default="greedy")
    a = ap.parse_args()
    mode = ("greedy", "coupled").index(a.mode)
    sets = {f: [z[k] for k in z.files] for f in a.rows for z in [np.load(f)]}
    ref = rate(sets[a.rows[0]], mode, 3, 0.5, 0.35, INSTALLED)
    print(f"{a.mode}: tokens/s relative to {a.rows[0]} with the installed stop (min_p 0.5, S 0.35, installed table)")
    for f, per in sets.items():
        own = table(per, mode)
        print(f"{f}: own table {','.join(f'{v:.2f}' for v in own)}")
        rows = [("installed", 0.5, 0.35, INSTALLED)] + [(f"own S {s}", mp, s, own) for mp in (0.5, 0.3, 0.0)
                                                         for s in (0.25, 0.35, 0.45, 0.55)]
        for cap in range(3, per[0].shape[1] + 1):   # cap 4 (--spec 5) when calib.py ran --steps 4
            for name, mp, s, tab in rows:
                print(f"   cap {cap} min_p {mp:.1f} {name:12} x{rate(per, mode, cap, mp, s, tab) / ref:.3f}")
            print(f"   cap {cap} oracle x{rate(per, mode, cap, 0, 0, None) / ref:.3f}")


if __name__ == "__main__":
    main()
