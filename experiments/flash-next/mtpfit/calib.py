"""calib.py - a drafter's STRATA_SPEC_CUM_ACC table, measured offline on the held-out split.

    .venv/bin/python -I calib.py --rt DRAFTER_DIR --data data/d1 data/d2 [--n-held 69] [--save ROWS.npz]

The engine's cumulative draft stop (strata/spec/draft_cum.hpp) multiplies acc(p) over a window's drafts, where acc
is the acceptance measured for drafts whose own probability p falls in the bands < 0.3, 0.5, 0.7, 0.8, 0.9, 0.95,
0.99 and above. The draft's p is its top probability when greedy, and under coupled sampling the drawn token's
probability in the final (top_k -> top_p) distribution (kernels/sampler.hpp coupled_draft_sample). Every chain step
here is drafted on the main model's own tokens, as in a window whose earlier drafts held.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_ref as ref  # noqa: E402
import train  # noqa: E402

EDGES = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])


@torch.no_grad()
def rows(t_logp, logits, gen):
    """Per row: greedy (p, held) and coupled (p, held) with one noise draw."""
    lq = torch.log_softmax(logits.float(), -1)
    pg, ig = lq.max(-1)
    greedy = (pg.exp(), ig == t_logp.argmax(-1))
    (tv, ti), (qv, qi) = train.cut(t_logp.float()), train.cut(lq)
    g = -torch.log(-torch.log(torch.rand(lq.shape, device=lq.device, generator=gen).clamp_min(1e-30)))
    jt = (tv + g.gather(-1, ti)).argmax(-1, keepdim=True)
    jq = (qv + g.gather(-1, qi)).argmax(-1, keepdim=True)
    pq = torch.softmax(qv, -1).gather(-1, jq)[:, 0]   # the drawn token's probability after the cut
    coupled = (pq, ti.gather(-1, jt)[:, 0] == qi.gather(-1, jq)[:, 0])
    return greedy, coupled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt", type=Path, required=True)
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--n-held", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--vocab", type=Path, default=ref.DATA / "draft_vocab_agent.bin")
    ap.add_argument("--gguf", type=Path, default=ref.GGUF)
    ap.add_argument("--save", type=Path, help="per transcript [greedy p, held, coupled p, held][step][base] for replay.py")
    a = ap.parse_args()
    dev = torch.device("cuda")
    a.init = a.rt
    model, teacher, emb = train.setup(a, dev)
    held = list(train.transcripts(a.data, "heldout"))[: a.n_held]
    gen = torch.Generator(device=dev).manual_seed(4321)
    stats = {m: [[] for _ in range(a.steps)] for m in ("greedy", "coupled")}
    saved = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for r in held:
            R, ids, q0 = train.chunk(r, a.max_len, dev)
            e = emb[torch.tensor(ids, device=dev)].float()
            t = teacher(R)
            per = np.full((4, a.steps, len(ids) - 1 - q0), np.nan, np.float32)   # step k has bases q0 .. n-1-k
            for k, logits in enumerate(model.chain(R, e, a.steps, q0), start=1):
                tk = t[q0 + k: q0 + k + logits.shape[0]]
                for i, (m, (p, held_)) in enumerate(zip(("greedy", "coupled"), rows(tk, logits, gen))):
                    stats[m][k - 1].append(torch.stack([p.float(), held_.float()], 1).cpu().numpy())
                    per[2 * i, k - 1, :len(p)], per[2 * i + 1, k - 1, :len(p)] = p.float().cpu(), held_.float().cpu()
            saved.append(per)
    if a.save:
        np.savez(a.save, *saved)
    print(f"{a.rt} on {len(held)} held-out transcripts; bands < {' '.join(map(str, EDGES))} and above")
    for m in ("greedy", "coupled"):
        allr = np.concatenate([np.concatenate(s) for s in stats[m]])
        for label, x in [(f"step {k + 1}", np.concatenate(stats[m][k])) for k in range(a.steps)] + [("all", allr)]:
            b = np.searchsorted(EDGES, x[:, 0], side="right")
            acc = [x[b == i, 1].mean() if (b == i).any() else float("nan") for i in range(8)]
            share = [(b == i).mean() for i in range(8)]
            print(f"{m:8} {label:7} held {x[:, 1].mean():.4f}  acc " + ",".join(f"{v:.2f}" for v in acc)
                  + "  share " + " ".join(f"{v:.2f}" for v in share))
        b = np.searchsorted(EDGES, allr[:, 0], side="right")
        tab = [allr[b == i, 1].mean() if (b == i).any() else 0.0 for i in range(8)]
        print(f"STRATA_SPEC_CUM_ACC ({m}, all steps)=" + ",".join(f"{v:.2f}" for v in tab))


if __name__ == "__main__":
    main()
