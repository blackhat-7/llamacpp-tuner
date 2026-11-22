"""ab.py - drafter A/B on the engine's --serve protocol: the same prompts and seeds through each drafter.

    PY=~/Documents/projects/Strata/.venv/bin/python
    $PY -I ab.py --arm base=$DATA/mtp/rt-basalt-q2b --arm "sd=data/rt-sd [K=V ...] [--flag=V ...]" \
        --out data/ab_NAME.jsonl [--heldout data/transcripts.jsonl --limit 24] [--greedy code,prose,edit] [--passes 2]

Each arm is one resident engine started from the user's tuned config with --mtp swapped (plus env overrides K=V and
engine flags --flag=V in place of the config's);
the arms run in the order A B .. then reversed (A B B A for two arms and --passes 2). Held-out transcripts are
resampled with their own seed at the user's sampling (temperature 1.0, top_p 0.95, top_k 20) and length; the greedy
prompts are the measurement texts (P/NAME.txt, 600 tokens, thinking off), as live_check.py sends them.
Under coupled sampling the text is a function of the seed alone (core/coupled_draft.hpp), so the arms should emit
the same ids; a near-tie flip (window composition, rounding level) makes a transcript diverge from there on.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_data as g  # noqa: E402


def jobs(a, tk):
    out = []
    if a.heldout:
        rows = [json.loads(x) for x in open(a.heldout)]
        for t in [t for t in rows if t["split"] == "heldout"][: a.limit]:
            out.append({"name": t["id"], "kind": "temp1", "ids": t["prompt"], "max_new": len(t["completion"]),
                        "keys": f"{g.SAMPLING} seed={t['seed']} ckpt=0", "ref": t["completion"]})
    for name in filter(None, a.greedy.split(",")):
        text = (g.P / f"{name}.txt").read_text()
        ids = tk.encode(g.render([{"role": "user", "content": text}], False), parse_special=True)
        out.append({"name": name, "kind": "greedy", "ids": ids, "max_new": 600, "keys": " temperature=0 seed=7 ckpt=0",
                    "ref": None})
    return out


def first_diff(x, y):
    return next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None if len(x) == len(y) else min(len(x), len(y)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="'NAME=MTP_DIR [K=V ...] [--flag=V ...]'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", help="transcripts.jsonl: resample its held-out split")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--greedy", default="code,prose,edit")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--config", default=str(g.CONFIG))
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    arms = []
    for spec in a.arm:
        name, rest = spec.split("=", 1)
        mtp, *kv = rest.split()   # space-separated: a STRATA_SPEC_CUM_ACC value has commas
        flags = [x.split("=", 1) for x in kv if x.startswith("--")]
        arms.append((name, str(Path(mtp).resolve()), dict(x.split("=", 1) for x in kv if not x.startswith("--")), flags))
    todo = jobs(a, g.tokenizer(cfg))
    order = [arm for p in range(a.passes) for arm in (arms if p % 2 == 0 else arms[::-1])]
    first = {}
    with open(a.out, "a") as f:
        for n_pass, (name, mtp, env, flags) in enumerate(order):
            eng = g.Engine(cfg, Path(a.out).with_suffix(".engine.log"), env=env, drop=("--mtp", *(f for f, _ in flags)),
                           add=("--mtp", mtp, *(x for fv in flags for x in fv)))
            try:
                for j in todo:
                    if (g.P / "STOP").exists():
                        raise SystemExit("STOP file: ending early")
                    t0 = time.time()
                    ids, d = eng.gen(j["max_new"], j["keys"], j["ids"])
                    ref = first.setdefault(j["name"], ids)
                    r = {"arm": name, "pass": n_pass, "name": j["name"], "kind": j["kind"], "n": len(ids),
                         "decode_ms": float(d[4]), "acc": int(d[6]), "off": int(d[7]), "wall": time.time() - t0,
                         "diff_first": first_diff(ids, ref), "diff_ref": first_diff(ids, j["ref"]) if j["ref"] else None}
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    print(f"{name} p{n_pass} {j['name']} {j['kind']}: {r['n']} tok {r['n'] / r['decode_ms'] * 1e3:.1f} t/s "
                          f"drafts {r['acc']}/{r['off']} diff {r['diff_first']}", flush=True)
            finally:
                eng.close()
    summary(a.out)


def summary(path):
    rows = [json.loads(x) for x in open(path)]
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    kinds = sorted({(r["kind"], r["name"] if r["kind"] == "greedy" else "heldout") for r in rows})
    print(f"{'':16}" + "".join(f"{x:>26}" for x in arms))
    for kind, name in kinds:
        cells = []
        for arm in arms:
            rs = [r for r in rows if r["arm"] == arm and r["kind"] == kind and (kind != "greedy" or r["name"] == name)]
            n, ms = sum(r["n"] for r in rs), sum(r["decode_ms"] for r in rs)
            acc, off = sum(r["acc"] for r in rs), sum(r["off"] for r in rs)
            cells.append(f"{n / ms * 1e3 if ms else 0:7.1f} t/s {acc / max(off, 1):5.3f} acc {n / max(n - acc, 1):4.2f}tpw")
        print(f"{name:16}" + "".join(f"{c:>26}" for c in cells))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--summary"]:
        summary(sys.argv[2])
    else:
        main()
