#!/usr/bin/env python3
"""Speed and recall at depth through any OpenAI-compatible server (streaming).

build: make a code-dump prompt of N tokens (llama-server /tokenize) with three
       "deploy key" facts at 10/50/90% depth.
run:   send code, prose and recall questions after that prompt; print TTFT,
       decode t/s (client-side, from usage) and llama-server timings if present.
"""

import argparse
import os
import json
import pathlib
import time
import urllib.request

LCT_HOME = pathlib.Path(os.environ.get("LCT_HOME", pathlib.Path(__file__).resolve().parents[2] / "tmp"))
SRC = LCT_HOME / "llama.cpp-a3b"
NEEDLES = [("Lark", "amber-falcon-7319"), ("Moss", "quiet-harbor-2264"), ("Fern", "copper-lantern-5081")]
QUESTIONS = {
    "code": "Ignore the dump above for this task. Write a self-contained C++17 program: an LRU cache "
    "class template with get/put/erase and a capacity, plus a main() that tests it with asserts. "
    "Output only the code.",
    "prose": "Ignore the dump above for this task. In plain English, write about 400 words on how the "
    "printing press changed Europe. No headings, no lists.",
    "recall": "Somewhere in the dump above are three comments of the form 'NOTE: the deploy key for "
    "project X is Y.' List each project and its key, one per line as 'X: Y'. Output nothing else.",
}
MAX_TOKENS = {"code": 700, "prose": 700, "recall": int(__import__("os").environ.get("RECALL_MAX", 80))}


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def n_tokens(server: str, text: str) -> int:
    return len(post(f"{server}/tokenize", {"content": text})["tokens"])


def build(args: argparse.Namespace) -> None:
    files = sorted(p for d in ("src", "ggml/src", "common", "tools") for p in (SRC / d).rglob("*")
                   if p.suffix in {".cpp", ".c", ".h"} and p.is_file())
    parts, chars = [], 0
    for p in files:
        parts.append(f"// FILE: {p.relative_to(SRC)}\n{p.read_text(errors='replace')}\n")
        chars += len(parts[-1])
        if chars > args.tokens * 4.5:
            break
    text = "".join(parts)
    lo, hi = 0, len(text)  # longest prefix (cut at a newline) within the token budget
    while hi - lo > 2000:
        mid = text.rfind("\n", 0, (lo + hi) // 2) + 1
        if n_tokens(args.server, text[:mid]) <= args.tokens - 200:
            lo = mid
        else:
            hi = mid
    text = text[:lo]
    for frac, (proj, key) in zip((0.9, 0.5, 0.1), reversed(NEEDLES)):
        at = text.find("\n", int(len(text) * frac)) + 1
        text = text[:at] + f"// NOTE: the deploy key for project {proj} is {key}.\n" + text[at:]
    pathlib.Path(args.out).write_text(text)
    print(f"{args.out}: {n_tokens(args.server, text)} tokens, {len(text)} chars")


def ask(server: str, model: str, context: str, kind: str, extra: dict) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"{context}\n\n{QUESTIONS[kind]}"}],
        "max_tokens": MAX_TOKENS[kind],
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        **extra,
    }
    req = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    usage, timings, out = {}, {}, []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            obj = json.loads(line[5:])
            usage = obj.get("usage") or usage
            timings = obj.get("timings") or timings
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}; piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    last = time.time()
                    first = first or last
                    out.append(piece)
    text = "".join(out)
    n = usage.get("completion_tokens", 0)
    res = {
        "kind": kind,
        "prompt_tokens": usage.get("prompt_tokens"),
        "ttft_s": round((first or time.time()) - t0, 2),
        "gen_tokens": n,
        "decode_tps": round((n - 1) / (last - first), 1) if first and last and last > first and n > 1 else None,
    }
    if timings:
        res["server_tps"] = round(timings.get("predicted_per_second", 0), 1)
        res["prompt_tps"] = round(timings.get("prompt_per_second", 0), 1)
        res["prompt_n"] = timings.get("prompt_n")
        if timings.get("draft_n"):
            res["draft_accept"] = round(timings["draft_n_accepted"] / timings["draft_n"], 2)
    res["text_tail"] = text[-400:]
    if kind == "recall":
        res["recall"] = sum(key in text for _, key in NEEDLES)
        res["answer"] = text.strip()[:200]
    return res


def run(args: argparse.Namespace) -> None:
    context = pathlib.Path(args.prompt).read_text()
    extra = json.loads(args.extra) if args.extra else {}
    for kind in args.kinds.split(","):
        res = {"label": args.label, "prompt_file": args.prompt, **ask(args.server, args.model, context, kind, extra)}
        print(json.dumps(res), flush=True)
        with open(LCT_HOME / "model-hunt/results.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--server", required=True)
    b.add_argument("--tokens", type=int, required=True)
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--server", required=True)
    r.add_argument("--model", default="local")
    r.add_argument("--prompt", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--kinds", default="code,prose,recall")
    r.add_argument("--extra", default="")
    args = ap.parse_args()
    build(args) if args.cmd == "build" else run(args)


if __name__ == "__main__":
    main()
