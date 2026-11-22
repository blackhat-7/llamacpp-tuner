"""Compare greedy outputs of two labels in results.jsonl token by token (Inkling tokenizer).
usage: python compare.py results.jsonl LABEL_A LABEL_B [job ...]   (needs the tokenizers package)"""
import json
import os
import pathlib
import sys

from tokenizers import Tokenizer

LCT_HOME = pathlib.Path(os.environ.get("LCT_HOME", pathlib.Path(__file__).resolve().parents[3] / "tmp"))

tok = Tokenizer.from_file(str(LCT_HOME / "loom/Inkling-2-Mini/tokenizer.json"))
rows = [json.loads(l) for l in open(sys.argv[1])]
a_label, b_label = sys.argv[2], sys.argv[3]
jobs = sys.argv[4:] or sorted({r["name"] for r in rows})
for job in jobs:
    ra = [r for r in rows if r["label"] == a_label and r["name"] == job]
    rb = [r for r in rows if r["label"] == b_label and r["name"] == job]
    if not ra or not rb:
        continue
    ta = tok.encode(ra[-1]["reasoning"] + "\x00" + ra[-1]["content"], add_special_tokens=False).ids
    tb = tok.encode(rb[-1]["reasoning"] + "\x00" + rb[-1]["content"], add_special_tokens=False).ids
    n = min(len(ta), len(tb))
    first = next((i for i in range(n) if ta[i] != tb[i]), None)
    if first is None:
        print(f"{job}: identical over {n} tokens")
    else:
        ctx = tok.decode(ta[max(0, first - 12):first])
        print(f"{job}: first divergence at token {first}/{n}; context ...{ctx!r} -> A {tok.decode(ta[first:first+6])!r} | B {tok.decode(tb[first:first+6])!r}")
