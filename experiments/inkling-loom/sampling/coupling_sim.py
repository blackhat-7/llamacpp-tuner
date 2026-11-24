#!/usr/bin/env python3
"""Speculative decoding at temperature > 0 on Loom: CPU exactness check and acceptance estimates.

Loom today (kv8-r2v): the drafter proposes its argmax (g_dsp_prepare writes temperature 0 into step_meta);
each verify row samples the target with the uniform keyed on (seed, position) that plain decode uses at that
position (g_spec_rows, g_head_select); g_spec_accept emits target samples while they equal the drafts.

Proposed: the drafter samples with the same public randomness as the target at the same position
(Daliri et al. 2024, arXiv 2408.07978, "drafter-invariant speculative decoding"). Two variants:
  cdf    - keep Loom's inverse-CDF draw (one u per position), drafter draws with the same u
  gumbel - Gumbel-max draw with one Gumbel per (seed, position, token id), for target and drafter

usage:
  uv run --no-project --with numpy python coupling_sim.py exact    # identity + chi-square checks (toy LM)
  uv run --no-project --with numpy python coupling_sim.py accept   # acceptance on calibrated synthetic p, q
"""

import math
import random
import sys
import zlib

M32 = 0xFFFFFFFF
GOLDEN, MIX_A, MIX_B, MIX_C = 0x9E3779B9, 0x85EBCA6B, 0x7FEB352D, 0x846CA68B


def lowbias(h: int) -> int:
    """The 32-bit finalizer in g_head_select (llm.loom 18242-18262)."""
    h ^= h >> 16
    h = (h * MIX_B) & M32
    h ^= h >> 15
    h = (h * MIX_C) & M32
    return h ^ (h >> 16)


def position_hash(seed: int, position: int) -> int:
    return lowbias((((seed & M32) * GOLDEN) & M32) ^ (((position & M32) * MIX_A) & M32))


def uniform(seed: int, position: int) -> float:
    return (position_hash(seed, position) >> 8) * 2.0**-24


def gumbel(seed: int, position: int, token: int) -> float:
    h = lowbias((position_hash(seed, position) + token * GOLDEN) & M32)
    u = ((h >> 8) + 0.5) * 2.0**-24
    return -math.log(-math.log(u))


def kernel_sample(logits, seed, position, temperature=1.0, top_p=1.0, top_k=0, noise=True):
    """g_head_select's T > 0 draw after engines/loom/tools/spec/gumbel.py, in float32 as the kernel computes it:
    top-k (k <= 64) by (logit desc, id asc), soft cap 30 (1 - 2 / (exp(2y) + 1)), w = exp((v - m) / T), keep the
    shortest prefix reaching top_p of the total, then argmax over the kept candidates of (v - m) / T + Gumbel,
    Gumbel = -log(max(-log(u), 2^-24)), u = ((h >> 8) + 0.5) * 2^-24, h = lowbias(position hash + id * GOLDEN).
    noise=False is a negative control (argmax of the kept candidates)."""
    import numpy as np

    F = np.float32
    x32 = [F(v) for v in logits]
    if temperature <= 0:
        return max(range(len(x32)), key=lambda i: (x32[i], -i))
    k = min(top_k, 64) if top_k > 0 else 64
    order = sorted(range(len(x32)), key=lambda i: (-x32[i], i))[:k]
    v = []
    for i in order:
        y = x32[i] * F(0.03333333333333333)
        e = np.exp(y + y, dtype=np.float32)
        v.append((F(1.0) - F(2.0) / (e + F(1.0))) * F(30.0))
    p = min(F(top_p), F(1.0)) if top_p > 0 else F(1.0)
    inv_t = F(1.0) / F(temperature)
    m = v[0]
    total = F(0.0)
    for c in v:
        total = total + np.exp((c - m) * inv_t, dtype=np.float32)
    mass = p * total
    h = position_hash(seed, position)
    acc, done, best, pick = F(0.0), False, F(-3.0e38), None
    for c, i in zip(v, order):
        ds = (c - m) * inv_t
        w = np.exp(ds, dtype=np.float32)
        g = lowbias((h + i * GOLDEN) & M32)
        u = (F(g >> 8) + F(0.5)) * F(5.9604644775390625e-08)
        e = max(-np.log(u, dtype=np.float32), F(5.9604644775390625e-08))
        score = ds - np.log(e, dtype=np.float32) if noise else ds
        if not done and score > best:
            best, pick = score, i
        nxt = acc if done else acc + w
        done = done or bool(nxt >= mass)
        acc = nxt
    return order[0] if pick is None else pick


def candidates(logits, temperature, top_p, top_k):
    """g_head_select's T > 0 candidate set: top-k (k <= 64) by (logit desc, id asc), soft cap 30, softmax at T,
    shortest prefix reaching top_p of the mass. Returns kept ids and their weights."""
    k = min(top_k, 64) if top_k > 0 else 64
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:k]
    v = [30.0 * math.tanh(logits[i] / 30.0) for i in order]
    w = [math.exp((x - v[0]) / temperature) for x in v]
    p = min(top_p, 1.0) if top_p > 0 else 1.0
    mass, acc, kept = p * sum(w), 0.0, 0
    for x in w:
        if acc >= mass:
            break
        acc += x
        kept += 1
    return order[:kept], w[:kept]


def sample(logits, seed, position, method, temperature=1.0, top_p=1.0, top_k=0):
    if method == "greedy":
        return max(range(len(logits)), key=lambda i: (logits[i], -i))
    if method == "kernel":
        return kernel_sample(logits, seed, position, temperature, top_p, top_k)
    if method == "kernel_nonoise":  # negative control: no noise, so not a sample of the kept softmax
        return kernel_sample(logits, seed, position, temperature, top_p, top_k, noise=False)
    ids, w = candidates(logits, temperature, top_p, top_k)
    if method == "cdf":
        target, acc = uniform(seed, position) * sum(w), 0.0
        for i, x in zip(ids, w):
            acc += x
            if acc > target:
                return i
        return ids[0]
    if method == "gumbel":
        return max(zip(ids, w), key=lambda t: math.log(t[1]) + gumbel(seed, position, t[0]))[0]
    if method == "nontruncated":  # negative control: ignores top-p
        ids, w = candidates(logits, temperature, 1.0, top_k)
        return max(zip(ids, w), key=lambda t: math.log(t[1]) + gumbel(seed, position, t[0]))[0]
    raise ValueError(method)


# ---- toy language model: logits are a fixed pseudo-random function of the last two tokens
V = 48


def target_logits(ctx):
    rnd = random.Random(zlib.crc32(bytes(ctx[-2:])))
    scale = rnd.choice([0.5, 1.5, 3.0, 6.0])  # mixed entropy
    return [scale * rnd.gauss(0, 1) for _ in range(V)]


def drafter_logits(ctx):
    rnd = random.Random(zlib.crc32(b"d" + bytes(ctx[-2:])))
    return [x + 0.8 * rnd.gauss(0, 1) for x in target_logits(ctx)]


def plain_decode(prompt, n, seed, method, **kw):
    ctx = list(prompt)
    for _ in range(n):
        ctx.append(sample(target_logits(ctx), seed, len(ctx), method, **kw))
    return ctx[len(prompt):]


def spec_decode(prompt, n, seed, depth, method, policy, shift=0, **kw):
    """g_spec_rows + g_spec_accept: row j draws at anchor position + j + 1; emit while the sample equals the draft."""
    ctx, out, proposed, accepted = list(prompt), [], 0, 0
    while len(out) < n:
        anchor = len(ctx) - 1
        drafts = []
        for s in range(1, depth + 1):
            drafts.append(sample(drafter_logits(ctx + drafts), seed, anchor + s, policy, **kw))
        emitted = []
        for j in range(depth + 1):
            y = sample(target_logits(ctx + drafts[:j]), seed, anchor + j + 1 + shift, method, **kw)
            emitted.append(y)
            if j == depth or y != drafts[j]:
                break
        proposed += depth
        accepted += len(emitted) - 1
        ctx += emitted
        out += emitted
    return out[:n], accepted, proposed


def exact():
    prompt, n, seeds = [3, 7], 48, range(400)
    kw = dict(temperature=1.0, top_p=0.95, top_k=0)
    cases = [
        ("cdf", "greedy", 0, "today: argmax drafts"),
        ("cdf", "cdf", 0, "drafter shares u"),
        ("gumbel", "greedy", 0, "gumbel target, argmax drafts"),
        ("gumbel", "gumbel", 0, "drafter shares Gumbel noise"),
        ("gumbel", "gumbel", 1, "NEGATIVE: rows draw one position late"),
        ("kernel", "greedy", 0, "kernel port: target, argmax drafts"),
        ("kernel", "kernel", 0, "kernel port: drafter shares the noise"),
        ("kernel", "kernel", 1, "NEGATIVE kernel port: rows one late"),
    ]
    for method, policy, shift, label in cases:
        same = acc = prop = 0
        for seed in seeds:
            ref = plain_decode(prompt, n, seed, method, **kw)
            for depth in (1, 3):
                got, a, p = spec_decode(prompt, n, seed, depth, method, policy, shift, **kw)
                same += got == ref
                if depth == 1:
                    acc, prop = acc + a, prop + p
        print(f"{label:40s} identical {same}/{2 * len(seeds)}  depth-1 acceptance {acc / prop:.3f}")

    # chi-square of each sampler against the exact kept softmax (fixed flat logits, many seeds)
    rnd = random.Random(1)
    logits = [1.2 * rnd.gauss(0, 1) for _ in range(V)]
    draws = 200_000
    for kw2 in (dict(temperature=0.8, top_p=0.9, top_k=20), dict(temperature=1.0, top_p=1.0, top_k=0)):
        ids, w = candidates(logits, **kw2)
        probs = {i: x / sum(w) for i, x in zip(ids, w)}
        for method in ("cdf", "gumbel", "nontruncated", "kernel", "kernel_nonoise"):
            counts = {}
            for seed in range(draws):
                t = sample(logits, seed, 1000 + seed % 7, method, **kw2)
                counts[t] = counts.get(t, 0) + 1
            outside = sum(c for t, c in counts.items() if t not in probs)
            chi = sum((counts.get(i, 0) - draws * p) ** 2 / (draws * p) for i, p in probs.items())
            dof = len(probs) - 1
            print(f"{kw2} sampler {method:12s} chi2 {chi:7.1f} (dof {dof}, 99.9% bound "
                  f"~{dof + 3.1 * math.sqrt(2 * dof):.0f})  draws outside the kept set {outside}")


def accept():
    import numpy as np

    rng = np.random.default_rng(7)
    vocab, k, n, mc, chunk = 256, 64, 3000, 48, 250

    def probs(x):
        idx = np.argsort(-x, axis=1)[:, :k]
        vals = np.take_along_axis(x, idx, 1)
        e = np.exp(vals - vals[:, :1])
        e /= e.sum(1, keepdims=True)
        p = np.zeros_like(x)
        np.put_along_axis(p, idx, e, 1)
        hi = np.zeros_like(x)
        np.put_along_axis(hi, idx, np.cumsum(e, 1), 1)
        return p, hi - p, hi  # probability, CDF interval [lo, hi) in own (logit desc) order

    def run(family, lo, hi, sigma):
        beta = np.exp(rng.uniform(np.log(lo), np.log(hi), (n, 1)))
        if family == "gumbel-logits":
            base = rng.gumbel(size=(n, vocab))
        else:  # zipf: p_r ~ r^-beta, ids shuffled
            base = -np.log(np.arange(1, vocab + 1))[None, :].repeat(n, 0)
            base = rng.permuted(base, axis=1)
        l = beta * base
        d = l + sigma * rng.standard_normal((n, vocab))
        p, plo, phi = probs(l)
        q, qlo, qhi = probs(d)
        a0 = np.mean(l.argmax(1) == d.argmax(1))
        greedy = np.mean(p[np.arange(n), d.argmax(1)])
        cdf = np.mean(np.clip(np.minimum(phi, qhi) - np.maximum(plo, qlo), 0, None).sum(1))
        best = np.mean(np.minimum(p, q).sum(1))
        indep = np.mean((p * q).sum(1))  # drafter samples with its own randomness
        hits = 0.0
        with np.errstate(divide="ignore"):
            lp, lq = np.log(p), np.log(q)
        for s in range(0, n, chunk):
            g = rng.gumbel(size=(min(chunk, n - s), mc, vocab))
            xp = (lp[s:s + chunk, None, :] + g).argmax(2)
            xq = (lq[s:s + chunk, None, :] + g).argmax(2)
            hits += (xp == xq).sum()
        gum = hits / (n * mc)
        return a0, greedy, indep, cdf, gum, best

    print("family         beta range  sigma | T=0 agree | T=1 acceptance: argmax draft  own-noise sample  shared-u CDF  Gumbel  1-TV")
    for family in ("gumbel-logits", "zipf"):
        for lo, hi in ((0.3, 6.0), (0.5, 10.0), (1.0, 16.0)):
            for sigma in (0.5, 1.0, 1.5, 2.0, 3.0):
                a0, g, ind, c, gu, b = run(family, lo, hi, sigma)
                mark = " <" if 0.39 <= g <= 0.47 and 0.6 <= a0 <= 0.8 else ""
                print(f"{family:14s} {lo:4.1f}-{hi:<5.1f} {sigma:5.1f} | {a0:9.3f} | {g:27.3f} {ind:17.3f} {c:13.3f} {gu:7.3f} {b:5.3f}{mark}")


if __name__ == "__main__":
    {"exact": exact, "accept": accept}[sys.argv[1]]()
