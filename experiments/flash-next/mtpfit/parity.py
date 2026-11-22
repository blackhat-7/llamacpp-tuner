"""parity.py - the reference draft step (mtp_ref.py) against the engine's own draft logits.

Engine side (README "Parity"): build r5-upstream with mtp_parity.patch, then one greedy `strata generate` run with
STRATA_MTP_PARITY=par.bin (every round: inputs, and per step the draft, its probability, the head logits, the
output residual) and STRATA_PREFILL_DUMP_R_ALL=r.bin (the prompt's residual rows, which build the drafter's K/V).

    python parity.py engine --par par.bin --r r.bin --prompt-ids prompt.json [--rounds 20]
    python parity.py self                     # CPU: train.py's torch forward against mtp_ref.py on the real weights

The drafter's K/V cell c holds (R_c, ids[c + 1]) (generate.cpp:12503-12505); a round's rows are cells p..p+T-1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_ref as ref  # noqa: E402
from mtp_ref import HC, N  # noqa: E402

HCN = HC * N


def rounds(path: Path):
    """Records written by mtp_parity.patch (all little-endian int32 / float32)."""
    b = np.fromfile(path, dtype=np.uint8)
    o = 0
    def take(n, dt):
        nonlocal o
        v = b[o:o + 4 * n].view(dt)
        o += 4 * n
        return v
    while o < b.size:
        p, a, T, steps, nv = take(5, "<i4").tolist()
        r = {"p": p, "a": a, "T": T, "tokens": take(T, "<i4").copy(), "R": take(T * HCN, "<f4").reshape(T, HCN).copy(), "steps": []}
        for _ in range(steps):
            r["steps"].append({"draft": int(take(1, "<i4")[0]), "prob": float(take(1, "<f4")[0]),
                               "logits": take(nv, "<f4").copy(), "R": take(HCN, "<f4").copy()})
        yield r


def engine(a) -> int:
    W, ex = ref.load_dense(a.rt), ref.Experts(a.rt)
    vocab = np.fromfile(a.vocab, dtype="<i4")
    main = ref.MainModel(a.gguf, vocab)
    head = main.head()
    pos_of = {int(t): i for i, t in enumerate(vocab)}
    ids = json.loads(Path(a.prompt_ids).read_text())
    R = ref.bf16(np.fromfile(a.r, dtype="<u2")).reshape(-1, HCN)
    cache = ref.Cache()
    for c in range(len(R)):                                   # the prompt's cells, from the BF16 dump
        _, _, _, k, v = ref.front(W, R[c], main.embed(int(ids[c + 1])), c)
        cache.K[c], cache.V[c] = k, v
    stats = []
    for n, r in enumerate(rounds(a.par)):
        if n >= a.rounds:
            break
        p, at = r["p"], r["a"]
        for t in range(r["T"]):                               # the round's rows: real cells p..p+T-1 (fp32 rows)
            Rt, mixed, inj, k, v = ref.front(W, r["R"][t], main.embed(int(r["tokens"][t])), p + t)
            cache.K[p + t], cache.V[p + t] = k, v
            if t == at:
                row = (Rt, mixed, inj)
        R_in, x = None, None
        for j, s in enumerate(r["steps"]):
            cell = p + at + j
            if j == 0:
                Rt, mixed, inj = row
                K, V = cache.kv(cell)
                R_out, logits, _, _ = ref.rest(W, ex, Rt, mixed, inj, cell, K, V, head)
            else:                                             # the chain follows the engine's own drafts
                R_out, logits, _, _ = ref.step(W, ex, cache, R_in.reshape(-1), main.embed(x), cell, head)
            e = s["logits"]
            q = np.exp(logits - logits.max()); q /= q.sum()
            pe = np.exp(e - e.max()); pe /= pe.sum()
            stats.append((n, j, int(vocab[np.argmax(logits)]) == s["draft"], float(np.abs(logits - e).max()),
                          float(logits @ e / np.linalg.norm(logits) / np.linalg.norm(e)),
                          float((pe * (np.log(pe + 1e-30) - np.log(q + 1e-30))).sum()),
                          s["prob"], float(q[pos_of[s["draft"]]]), ref.rel(R_out.reshape(-1), s["R"])))
            R_in, x = R_out, s["draft"]
    print("round step top1 max|dlogit| cosine KL(eng||ref) p_eng p_ref relR")
    for st in stats:
        print("%5d %4d %4s %10.4f %8.6f %10.2e %6.4f %6.4f %.4f" % st)
    m = np.array([s[2:] for s in stats], dtype=np.float64)
    print(f"top-1 agreement {m[:, 0].mean():.4f} over {len(stats)} steps; median cosine {np.median(m[:, 2]):.6f}; "
          f"median KL {np.median(m[:, 3]):.2e}; median rel R {np.median(m[:, 6]):.4f}")
    return 0


def self_check(a) -> int:
    """train.py's MTP (torch) against mtp_ref.py (numpy): real dense, experts and head; synthetic residuals."""
    import torch
    import train
    W, ex = ref.load_dense(a.rt), ref.Experts(a.rt)
    vocab = np.fromfile(a.vocab, dtype="<i4")
    main = ref.MainModel(a.gguf, vocab)
    head = main.head()
    g, u, d = (torch.from_numpy(np.stack(x)) for x in zip(*(ex[e] for e in range(ref.NE))))
    model = train.MTP(W, g, u, d, torch.from_numpy(head))
    rng = np.random.default_rng(1)
    n, steps = 6, 3
    R = rng.standard_normal((n, HCN)).astype(np.float32)
    toks = vocab[rng.integers(0, len(vocab), n)]
    emb = np.stack([main.embed(int(t)) for t in toks])
    with torch.no_grad():
        out = model.chain(torch.from_numpy(R), torch.from_numpy(emb), steps)
    worst = 0.0
    for i in range(n - 1):                                  # base i: numpy chain over the same cells and tokens
        cache = ref.Cache()
        for c in range(i):
            _, _, _, k, v = ref.front(W, R[c], emb[c + 1], c)
            cache.K[c], cache.V[c] = k, v
        R_in = R[i]
        for k in range(1, steps + 1):
            if i + k > n - 1:
                break
            R_in, logits, _, _ = ref.step(W, ex, cache, R_in.reshape(-1), emb[i + k], i + k - 1, head)
            e = ref.rel(out[k - 1][i].numpy(), logits)
            worst = max(worst, e)
            print(f"base {i} step {k}: rel diff torch vs numpy logits {e:.2e}, argmax equal "
                  f"{int(out[k - 1][i].argmax()) == int(np.argmax(logits))}")
    print(f"worst rel diff {worst:.2e}")
    return 0 if worst < 1e-3 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["engine", "self"])
    ap.add_argument("--rt", type=Path, default=ref.DATA / "mtp/rt-basalt-q2b")
    ap.add_argument("--vocab", type=Path, default=ref.DATA / "draft_vocab_agent.bin")
    ap.add_argument("--gguf", type=Path, default=ref.GGUF)
    ap.add_argument("--par", type=Path)
    ap.add_argument("--r", type=Path)
    ap.add_argument("--prompt-ids", type=Path, help="JSON list of the prompt's token ids")
    ap.add_argument("--rounds", type=int, default=20)
    a = ap.parse_args()
    return engine(a) if a.mode == "engine" else self_check(a)


if __name__ == "__main__":
    sys.exit(main())
