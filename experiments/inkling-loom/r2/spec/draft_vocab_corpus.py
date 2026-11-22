"""Count token frequencies for a reduced draft vocabulary (CPU only, local files only).

Sources, each normalized to the same total weight so no single one dominates:
  code   tracked source files of the git repos under ~/Documents/projects
  agent  Claude Code transcripts (~/.claude/projects/**/*.jsonl): assistant text, thinking, tool-call JSON, tool results
  notes  markdown notes (~/Documents/Notes)
  model  Inkling's own outputs from the session jsonl files (reasoning + content), first half; the second half is held out

usage: draft_vocab_corpus.py TOKENIZER_JSON OUT_COUNTS.json OUT_HELDOUT.json
"""
import collections
import glob
import json
import os
import pathlib
import random
import subprocess
import sys

from tokenizers import Tokenizer

CAP = 12_000_000  # characters per source
CODE_EXT = {".py", ".ts", ".tsx", ".js", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".sh", ".lua", ".md", ".toml",
            ".yaml", ".yml", ".json", ".sql", ".html", ".css", ".loom"}
HOME = pathlib.Path.home()
rng = random.Random(0)


def take(texts, cap=CAP):
    out, n = [], 0
    for t in texts:
        if n >= cap:
            break
        t = t[: cap - n]
        out.append(t)
        n += len(t)
    return out


def code_texts():
    files = []
    for repo in sorted((HOME / "Documents/projects").iterdir()):
        if not (repo / ".git").exists():
            continue
        try:
            listed = subprocess.run(["git", "-C", str(repo), "ls-files"], capture_output=True, text=True, timeout=30).stdout
        except subprocess.TimeoutExpired:
            continue
        for rel in listed.splitlines():
            p = repo / rel
            if p.suffix in CODE_EXT and "lock" not in p.name and p.is_file() and p.stat().st_size < 200_000:
                files.append(p)
    rng.shuffle(files)
    for p in files:
        try:
            yield p.read_text()
        except (UnicodeDecodeError, OSError):
            pass


def agent_texts():
    files = glob.glob(str(HOME / ".claude/projects/**/*.jsonl"), recursive=True)
    rng.shuffle(files)
    for f in files:
        for line in open(f, errors="replace"):
            try:
                msg = json.loads(line).get("message") or {}
            except json.JSONDecodeError:
                continue
            content = msg.get("content")
            if isinstance(content, str):
                yield content
                continue
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind == "text":
                    yield part["text"]
                elif kind == "thinking":
                    yield part.get("thinking", "")
                elif kind == "tool_use":
                    yield json.dumps(part.get("input", {}), ensure_ascii=False)
                elif kind == "tool_result":
                    c = part.get("content")
                    yield c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)[:4000]


def notes_texts():
    files = sorted(glob.glob(str(HOME / "Documents/Notes/**/*.md"), recursive=True))
    rng.shuffle(files)
    for f in files:
        yield open(f, errors="replace").read()


def model_texts():
    out = []
    for f in sorted(glob.glob(str(HOME / "Documents/projects/llamacpp-tuner/tmp/experiments/inkling-loom/*/*.jsonl"))):
        for line in open(f):
            d = json.loads(line)
            if d.get("reasoning") or d.get("content"):
                out.append((d.get("reasoning") or "") + (d.get("content") or ""))
    seen, uniq = set(), []
    for t in out:  # repeated runs produce the same greedy text
        if t[:200] not in seen:
            seen.add(t[:200])
            uniq.append(t)
    rng.shuffle(uniq)
    return uniq[: len(uniq) // 2], uniq[len(uniq) // 2:]


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    model_train, model_held = model_texts()
    sources = {"code": take(code_texts()), "agent": take(agent_texts()), "notes": take(notes_texts()),
               "model": model_train}
    weights = collections.Counter()
    for name, texts in sources.items():
        counts = collections.Counter()
        for enc in tok.encode_batch(texts):
            counts.update(enc.ids)
        total = sum(counts.values())
        print(name, len(texts), "texts", total, "tokens", len(counts), "distinct", file=sys.stderr)
        for i, c in counts.items():
            weights[i] += c / total
    json.dump({str(k): v for k, v in weights.items()}, open(sys.argv[2], "w"))
    held = [e.ids for e in tok.encode_batch(model_held)]
    json.dump(held, open(sys.argv[3], "w"))
    print("held-out model texts", len(held), sum(map(len, held)), "tokens", file=sys.stderr)


main()
