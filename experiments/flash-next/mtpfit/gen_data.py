"""Self-distillation data for the MTP draft layer: transcripts sampled by the main model at the user's settings
(temperature 1.0, top_p 0.95, top_k 20), then the main model's final multi-stream residual at every position.

  PY=~/Documents/projects/Strata/.venv/bin/python     (has the tokenizer's `regex` and numpy)
  $PY -I gen_data.py transcripts PROMPTS... --out D/transcripts.jsonl [--max-new 1024] [--limit N] [--dry-run]
  $PY -I gen_data.py dump D/transcripts.jsonl --out D/dump          (GPU-exclusive, like transcripts)
  $PY -I gen_data.py check D/dump

PROMPTS: .jsonl files ({"messages": [{"role", "content"}...]} or {"text": ...} per line) or directories of .txt files
(one user message each). A prompt sharing any 8-word run with the measurement prompts (code/edit/prose.txt and .follow,
corpus/, long/, --exclude) is dropped; a fixed hash of the rendered prompt puts --heldout percent in the held-out split.

Both GPU steps drive the engine's own `--serve` stdin protocol (`GEN <max_new> [keys] <ids>` -> `T <id>` lines, then
`DONE`; serve/server.py's StrataEngine), started from the user's tuned config: the HTTP API returns text, not ids, and
the dump must replay exactly the ids that were sampled.

Dump layout (prefill.cpp:4036-4056): D/dump/r.bin = BF16 rows of 10240 (4 streams x 2560), no header, appended in
position order. index.jsonl, one line per transcript: {id, split, ids (prompt + completion), r_file, r_row0, n, n_prompt};
row r_row0 + i is the residual after the last layer at position i (input ids[i]), i < n = len(ids) - 1: the prompt read
stops at n - 1 and the last id goes through the decode path (generate.cpp:11049 `n - 1`), so it has no row.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent.parent   # tmp/strata-opt: gpu.lock, STOP, the measurement prompts
STRATA = Path.home() / "Documents/projects/Strata"
CONFIG = STRATA / "strata-swift-iq3_xxs-tuned.json"
D = 10240
ROW = 2 * D
SAMPLING = " temperature=1.0 top_p=0.95 top_k=20"   # the engine's own key spelling (server.py sampling_keys)


def tokenizer(cfg):
    sys.path.insert(0, str(STRATA / "tools"))
    from strata_tokenizer import Tokenizer
    d = Path(cfg["tokenizer"])
    vocab = json.loads((d / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = [m for m in (d / "merges.txt").read_text().splitlines() if m and not m.startswith("#")]
    return Tokenizer(tokens, merges, json.loads((d / "token_type.json").read_text()))


def grams(text, n=8):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def measured_grams(tk, extra):
    files = [P / f"{k}.{e}" for k in ("code", "edit", "prose") for e in ("txt", "follow")]
    files += sorted((P / "corpus").glob("*.txt")) + sorted((P / "long").glob("*.txt"))
    files += sorted((P / "long").glob("*.follow"))
    out = set()
    for f in files + [Path(x) for x in extra]:
        text = f.read_text(errors="replace")
        if f.suffix in (".ids", ".follow"):   # comma-separated token ids
            text = tk.decode([int(x) for x in text.replace(",", " ").split()])
        out |= grams(text)
    return out


def read_prompts(paths):
    for p in map(Path, paths):
        if p.is_dir():
            for f in sorted(p.rglob("*.txt")):
                yield [{"role": "user", "content": f.read_text(errors="replace")}]
            continue
        for line in p.open(errors="replace"):
            if line.strip():
                j = json.loads(line)
                yield j["messages"] if "messages" in j else [{"role": "user", "content": j["text"]}]


def render(msgs, think):
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    # serve/chat_template.jinja:176-182, enable_thinking on / off
    return s + "<|im_start|>assistant\n" + ("<think>\n" if think else "<think>\n\n</think>\n\n")


class Engine:
    """One resident `strata --serve` process. Holds gpu.lock for its lifetime, as gen.sh and serve_prof.py do."""

    def __init__(self, cfg, log, env=(), drop=(), add=()):
        self.lock = open(P / "gpu.lock", "w")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        if (P / "STOP").exists():
            raise SystemExit("STOP file present: not starting")
        args, it = [], iter(cfg["args"])
        for x in it:
            if x == "--vision":   # images need strata-vision, which this script does not start
                continue
            if x in drop:
                next(it)
                continue
            args.append(x)
        e = {**os.environ, **cfg.get("env", {}), **dict(env)}
        e["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [os.environ.get("LD_LIBRARY_PATH", "")])
        self.p = subprocess.Popen([cfg["exe"], "--serve", *args, *add], cwd=cfg.get("cwd"), env=e, text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "a"),
                                  start_new_session=True)
        for line in self.p.stdout:
            if line.startswith("READY"):
                return
        raise RuntimeError(f"the engine exited before READY ({self.p.wait()}); see {log}")

    def gen(self, max_new, keys, ids):
        self.p.stdin.write(f"GEN {max_new}{keys} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        out = []
        for line in self.p.stdout:
            if line.startswith("T "):
                out.append(int(line[2:]))
            elif line.startswith("DONE"):
                return out, line.split()
            elif line.startswith("ERR"):   # a refused request leaves the engine idle: no DONE follows (server.py #1059)
                raise ValueError(line.strip())
        raise RuntimeError(f"the engine exited mid-request ({self.p.wait()})")

    def close(self):
        self.p.stdin.close()
        try:
            self.p.wait(90)
        except subprocess.TimeoutExpired:
            os.killpg(self.p.pid, 9)
        self.lock.close()   # releases gpu.lock: a second Engine in this process (ab.py) would wait on it forever


def transcripts(a):
    cfg = json.loads(Path(a.config).read_text())
    tk = tokenizer(cfg)
    banned = measured_grams(tk, a.exclude)
    out = Path(a.out)
    done = {json.loads(x)["id"] for x in out.open()} if out.exists() else set()
    todo, dropped, long_ = [], 0, 0
    for msgs in read_prompts(a.prompts):
        if any(grams(m["content"]) & banned for m in msgs):
            dropped += 1
            continue
        h = hashlib.sha1(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
        if h[:16] in done or any(t["id"] == h[:16] for t in todo):   # the id keys the index: never twice
            continue
        think = int(h[16:18], 16) % 100 < a.think
        ids = tk.encode(render(msgs, think), parse_special=True)
        if len(ids) > a.max_prompt:   # every position is dumped at 20 KB: long contexts cost the most disk
            long_ += 1
            continue
        split = "heldout" if int(h[:8], 16) % 100 < a.heldout else "train"
        todo.append({"id": h[:16], "split": split, "think": think, "seed": int(h[8:16], 16) % 2**31 or 1, "prompt": ids})
    todo = [t for t in todo if t["id"] not in done][:a.limit]
    held = sum(t["split"] == "heldout" for t in todo)
    print(f"{len(todo)} to run ({held} held-out, {sum(len(t['prompt']) for t in todo)} prompt tokens), {len(done)} done, "
          f"{dropped} dropped for overlap, {long_} over --max-prompt", flush=True)
    if a.dry_run or not todo:
        return
    eng = Engine(cfg, out.with_suffix(".engine.log"))
    t_end = time.time() + 60 * a.minutes   # the GPU is shared: hold gpu.lock for one bounded chunk, rerun to resume
    try:
        with out.open("a") as f:
            for i, t in enumerate(todo):
                if (P / "STOP").exists() or time.time() > t_end:
                    print("STOP file or --minutes reached: ending early")
                    break
                try:
                    t["completion"], done_f = eng.gen(a.max_new, f"{SAMPLING} seed={t['seed']}", t["prompt"])
                except ValueError as e:
                    print(f"{t['id']}: {e}")
                    continue
                t["drafts"] = [int(done_f[6]), int(done_f[7])]   # accepted, offered: the installed drafter's baseline
                f.write(json.dumps(t) + "\n")
                f.flush()
                print(f"{i + 1}/{len(todo)} {t['id']} {t['split']} {len(t['prompt'])}+{len(t['completion'])} ids, "
                      f"drafts {done_f[6]}/{done_f[7]}, decode {done_f[4]} ms", flush=True)
    finally:
        eng.close()


def dump(a):
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r, index = out / "r.bin", out / "index.jsonl"
    if r.exists():
        raise SystemExit(f"{r} exists: the engine appends to it, so start from an empty directory")
    # --prompt-cache 0 and no conversation cache: every prompt is read from token 0 (generate.cpp:10267-10275), so
    # every position is dumped; --short-read 0: no part goes through the decode windows, which do not dump
    # (generate.cpp:10540-10541). CPU share 0: the same GPU arithmetic whatever the CPU timing ("auto" measures it).
    eng = Engine(cfg, out / "engine.log", env={"STRATA_PREFILL_DUMP_R_ALL": str(r), "STRATA_PREFILL_CPU_SHARE": "0"},
                 drop=("--conversation-cache-mib", "--conversation-cache-slots", "--prompt-cache-root"),
                 add=("--prompt-cache", "0", "--short-read", "0"))
    row = 0
    try:
        with index.open("w") as f:
            for n_line, line in enumerate(open(a.transcripts)):
                if n_line < a.skip:   # already dumped into an earlier directory
                    continue
                if (P / "STOP").exists():
                    print("STOP file: ending early")
                    break
                t = json.loads(line)
                ids = t["prompt"] + t["completion"]
                eng.gen(1, "", ids)
                rows = (r.stat().st_size if r.exists() else 0) // ROW - row
                split = t["split"]
                if rows != len(ids) - 1:   # rows no longer start at position 0: keep the offsets, never train on it
                    print(f"WARNING {t['id']}: {rows} rows for {len(ids)} ids (expected n - 1): split 'skip'", flush=True)
                    split = "skip"
                f.write(json.dumps({"id": t["id"], "split": split, "ids": ids, "r_file": str(r.resolve()),
                                    "r_row0": row, "n": rows, "n_prompt": len(t["prompt"])}) + "\n")
                f.flush()
                row += rows
                print(f"{t['id']}: rows {rows}, total {row} ({row * ROW / 2**30:.2f} GiB)", flush=True)
    finally:
        eng.close()


def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def check(a):
    d = Path(a.dir)
    idx = [json.loads(x) for x in (d / "index.jsonl").open()]
    r = Path(idx[0]["r_file"])
    size = r.stat().st_size
    total = sum(e["n"] for e in idx)
    assert size == total * ROW, f"{r} holds {size / ROW:.2f} rows, the index {total}"
    row = 0
    for e in idx:
        assert e["r_file"] == str(r) and e["r_row0"] == row, f"index entry out of step: {e['id']}"
        assert e["split"] == "skip" or e["n"] == len(e["ids"]) - 1, f"{e['id']}: rows do not start at position 0"
        assert 0 < e["n_prompt"] <= len(e["ids"]), f"{e['id']}: bad n_prompt"
        row += e["n"]
    mm = np.memmap(r, dtype=np.uint16, mode="r").reshape(-1, D)
    pick = np.unique(np.linspace(0, total - 1, min(total, a.sample)).astype(np.int64))
    x = bf16(np.asarray(mm[pick])).reshape(len(pick), 4, D // 4)
    assert np.isfinite(x).all(), "non-finite values in the dump"
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=(0, 2)))
    split = {s: sum(e["n"] for e in idx if e["split"] == s) for s in ("train", "heldout", "skip")}
    print(f"OK: {len(idx)} transcripts, {total} rows ({size / 2**30:.2f} GiB), rows by split {split}; "
          f"sampled {len(pick)} rows finite, RMS per stream {' '.join(f'{v:.3g}' for v in rms)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("transcripts", help="sample continuations of held-out prompts (GPU-exclusive)")
    t.add_argument("prompts", nargs="+")
    t.add_argument("--out", required=True)
    t.add_argument("--config", default=str(CONFIG))
    t.add_argument("--exclude", nargs="*", default=[], help="more measurement files (.txt text, .ids/.follow ids)")
    t.add_argument("--max-new", type=int, default=1024)
    t.add_argument("--max-prompt", type=int, default=8192)
    t.add_argument("--heldout", type=int, default=10, help="percent of prompts in the held-out split")
    t.add_argument("--think", type=int, default=50, help="percent of prompts with thinking on")
    t.add_argument("--limit", type=int)
    t.add_argument("--dry-run", action="store_true", help="filter, split and count only: no engine")
    t.add_argument("--minutes", type=float, default=18, help="stop starting new transcripts after this long")
    d = sub.add_parser("dump", help="replay each transcript with the residual dump on (GPU-exclusive)")
    d.add_argument("transcripts")
    d.add_argument("--out", required=True)
    d.add_argument("--config", default=str(CONFIG))
    d.add_argument("--skip", type=int, default=0, help="transcript lines already dumped elsewhere")
    c = sub.add_parser("check", help="offline: the dump against its index")
    c.add_argument("dir")
    c.add_argument("--sample", type=int, default=2000, help="rows read for the finiteness check")
    a = ap.parse_args()
    {"transcripts": transcripts, "dump": dump, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
