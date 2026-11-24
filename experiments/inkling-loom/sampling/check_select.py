#!/usr/bin/env python3
"""Host side of gen_select_test.py. Per (row, config): chi-square of the new kernel's draws against softmax over the
kept candidates, its per-draw agreement with a float32 numpy port, and its draw-for-draw match with the slow
(old top-k) kernel; then the negative controls.

usage: uv run --no-project --with numpy --with scipy python check_select.py OUTDIR
"""
import sys

import numpy as np
from scipy.stats import chi2

d = sys.argv[1]
M32 = np.uint64(0xFFFFFFFF)
GOLDEN, MIX_A, MIX_B, MIX_C = 0x9E3779B9, 0x85EBCA6B, 0x7FEB352D, 0x846CA68B
CONFIGS = {"c1": (1.0, 1.0, 0), "c2": (0.7, 0.9, 40), "c3": (1.5, 1.0, 10)}
FAST, SLOW = 12, 2
failures = []


def lowbias(h):
    h = h.astype(np.uint64)
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(MIX_B)) & M32
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(MIX_C)) & M32
    return h ^ (h >> np.uint64(16))


def kept(row, temperature, top_p, top_k):
    """g_head_select's candidates in float32 (exact top-k by sorting): ids, ds = (v - m) / T, kept prefix."""
    k = min(top_k, 64) if top_k > 0 else 64
    order = np.lexsort((np.arange(row.size), -row.astype(np.float64)))[:k]
    y = row[order] * np.float32(0.03333333333333333)
    v = (np.float32(1) - np.float32(2) / (np.exp(y + y) + np.float32(1))) * np.float32(30)
    inv_t = np.float32(1) / np.float32(temperature)
    ds = (v - v[0]) * inv_t
    w = np.exp(ds)
    total = np.float32(0)
    for x in w:
        total = np.float32(total + x)
    mass = np.float32(min(top_p, 1.0) if top_p > 0 else 1.0) * total
    acc, n = np.float32(0), 0
    for x in w:
        if acc >= mass:
            break
        acc, n = np.float32(acc + x), n + 1
    return order[:n].astype(np.int64), ds[:n]


def port(seeds, positions, ids, ds):
    hp = lowbias(((seeds.astype(np.uint64) * np.uint64(GOLDEN)) & M32) ^ ((positions.astype(np.uint64) * np.uint64(MIX_A)) & M32))
    g = lowbias((hp[:, None] + ids[None, :].astype(np.uint64) * np.uint64(GOLDEN)) & M32)
    u = ((g >> np.uint64(8)).astype(np.float32) + np.float32(0.5)) * np.float32(5.9604644775390625e-08)
    e = np.maximum(-np.log(u), np.float32(5.9604644775390625e-08))
    return ids[np.argmax(ds[None, :] - np.log(e), axis=1)]


def chisq(tokens, ids, ds):
    p = np.exp(ds.astype(np.float64))
    p /= p.sum()
    counts = np.array([(tokens == i).sum() for i in ids], np.float64)
    outside = int(tokens.size - counts.sum())
    exp = tokens.size * p
    big = exp >= 5  # pool the small-expectation tail into one bin
    obs_b, exp_b = np.append(counts[big], counts[~big].sum()), np.append(exp[big], exp[~big].sum())
    keep = exp_b > 0
    stat = ((obs_b[keep] - exp_b[keep]) ** 2 / exp_b[keep]).sum()
    dof = int(keep.sum() - 1)
    return stat, dof, (chi2.sf(stat, dof) if dof > 0 else 1.0), outside


def load(kernel, tag, launches):
    toks, seeds, positions = [], [], []
    for i in range(launches):
        m = np.load(f"{d}/meta_{tag}_{i}.npy")[4:].reshape(-1, 9)
        toks.append(np.load(f"{d}/tok_{kernel}_{tag}_{i}.npy"))
        seeds.append(m[:, 8].view(np.uint32))
        positions.append((m[:, 1] + m[:, 0]).view(np.uint32))
    return np.concatenate(toks), np.concatenate(seeds), np.concatenate(positions)


def expect(ok, label):
    print(f"    {label}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(label)


for row in "abcd":
    logits = np.load(f"{d}/row_{row}.npy")
    for config, (t, tp, tk) in CONFIGS.items():
        tag = f"{row}{config}"
        ids, ds = kept(logits, t, tp, tk)
        tok, seeds, positions = load("new", tag, FAST)
        stat, dof, pval, outside = chisq(tok, ids, ds)
        agree = (port(seeds, positions, ids, ds) == tok).mean()
        slow = load("slow", tag, SLOW)[0]
        same = (slow == tok[: slow.size]).mean()
        print(f"{tag}: {tok.size} draws, {ids.size} kept; chi2 {stat:.1f} dof {dof} p {pval:.3g}, outside {outside}; "
              f"port agreement {agree:.5f}; new == slow (old top-k) {same:.5f}")
        expect(pval > 1e-3 and outside == 0 and (tok >= 0).all(), f"{tag} chi-square")
        expect(agree == 1.0, f"{tag} matches the float32 port")
        expect(same == 1.0, f"{tag} fast top-k == old top-k")

print("negative controls (each must FAIL its test):")
ids, ds = kept(np.load(f"{d}/row_a.npy"), 1.0, 1.0, 0)
tok = load("new", "ac1", FAST)[0]
stat, dof, pval, _ = chisq(tok, *kept(np.load(f"{d}/row_a.npy"), 1.1, 1.0, 0))
print(f"  ac1 draws vs softmax at T 1.1: chi2 {stat:.1f} dof {dof} p {pval:.3g}")
expect(pval < 1e-6, "power: wrong temperature rejected")
stat, dof, pval, _ = chisq(load("neg", "ac1", FAST)[0], ids, ds)
print(f"  broken noise -log(u): chi2 {stat:.1f} dof {dof} p {pval:.3g}")
expect(pval < 1e-6, "broken noise rejected")
negtk = load("negtk", "cc1", SLOW)[0]
diff = (negtk != load("slow", "cc1", SLOW)[0]).mean()
print(f"  half the chosen chunks: differs from the old top-k on {diff:.4f} of draws")
expect(diff > 0.001, "broken top-k detected")
print("ALL PASS" if not failures else f"FAILED: {failures}")
