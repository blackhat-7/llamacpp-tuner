"""Several agents on one lct server, through /v1/messages (Claude Code's API).

Modes: alone (each agent's turns back to back: the chat stays in Loom's row),
turns (round robin: every turn after the first restores a parked chat) and
concurrent (all agents at once, each in its own thread). Greedy, so every mode
must produce the same text per agent and turn. One JSON line per turn.
"""

import argparse
import hashlib
import json
import pathlib
import threading
import time
import urllib.request
import os

LCT_HOME = pathlib.Path(os.environ.get("LCT_HOME", pathlib.Path(__file__).resolve().parents[3] / "tmp"))

PROMPTS = LCT_HOME / "model-hunt"
EXTRA = (LCT_HOME / "model-hunt" / "prompt_inkling_195000.txt").read_text()
QUESTIONS = [
    "In one sentence, what is this text about?",
    "Name one function or term it mentions. One line.",
    "What would you check first if it failed? One sentence.",
    "Give a one-word label for it.",
]


def documents(sizes: list[str]) -> list[str]:
    big = (PROMPTS / "prompt_inkling_195000.txt").read_text()
    mid = (PROMPTS / "prompt_inkling_100000.txt").read_text()
    small = (PROMPTS / "prompt_inkling_64000.txt").read_text()
    pick = {
        "20K": small[:60000],
        "64K": small,
        "100K": mid,
        "150K": big[: int(len(big) * 0.77)],
        "190K": big[: int(len(big) * 0.975)],
    }
    return [pick[s] for s in sizes]


def turn(base: str, history: list, max_tokens: int, temperature: float) -> dict:
    body = {"model": "inkling", "max_tokens": max_tokens, "temperature": temperature,
            "stream": True, "system": "You are a careful assistant. Answer briefly.",
            "messages": history}
    request = urllib.request.Request(f"{base}/v1/messages", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    thinking, text, usage, stop = [], [], {}, None
    with urllib.request.urlopen(request, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line[5:])
            kind = event.get("type")
            if kind == "content_block_delta":
                delta = event["delta"]
                piece = delta.get("thinking") or delta.get("text") or ""
                (thinking if delta.get("type") == "thinking_delta" else text).append(piece)
                last = time.time()
                first = first or last
            elif kind == "message_delta":
                usage = event.get("usage") or {}
                stop = event["delta"].get("stop_reason")
    done = time.time()
    out = usage.get("output_tokens", 0)
    return {"thinking": "".join(thinking), "text": "".join(text), "stop": stop,
            "ttft_s": round(first - t0, 2) if first else None,
            "tps": round((out - 1) / (last - first), 1) if first and last and last > first and out > 1 else None,
            "total_s": round(done - t0, 2), "input": usage.get("input_tokens"),
            "cached": usage.get("cache_read_input_tokens"), "output": out}


def agent(base, index, doc, turns, results, lock, args):
    history: list = []
    for t in range(turns):
        question = QUESTIONS[t % len(QUESTIONS)]
        if t and args.long:  # a long follow-up: restores must read many 512-token chunks
            extra = EXTRA[(index * 7 + t) * 15000:(index * 7 + t + 1) * 15000]
            question = f"More code:\n{extra}\n\n{question}"
        content = f"{doc}\n\n{question}" if t == 0 else question
        history.append({"role": "user", "content": content})
        r = turn(base, history, args.max_tokens, args.temperature)
        blocks = []
        if r["thinking"]:
            blocks.append({"type": "thinking", "thinking": r["thinking"], "signature": ""})
        blocks.append({"type": "text", "text": r["text"] or "(empty)"})
        history.append({"role": "assistant", "content": blocks})
        line = {"mode": args.mode, "agent": index, "turn": t,
                "md5": hashlib.md5((r["thinking"] + "|" + r["text"]).encode()).hexdigest()[:10],
                **{k: v for k, v in r.items() if k not in ("thinking", "text")},
                "text": r["text"][:60]}
        with lock:
            results.append(line)
            print(json.dumps(line), flush=True)
        if args.mode == "turns":
            yield


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--mode", choices=["alone", "turns", "concurrent"], required=True)
    ap.add_argument("--sizes", default="20K,64K,100K,150K")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--long", action="store_true")
    args = ap.parse_args()
    docs = documents(args.sizes.split(","))
    results, lock = [], threading.Lock()
    t0 = time.time()
    if args.mode == "alone":
        for i, doc in enumerate(docs):
            for _ in agent(args.base, i, doc, args.turns, results, lock, args):
                pass
    elif args.mode == "turns":
        runs = [agent(args.base, i, doc, args.turns, results, lock, args) for i, doc in enumerate(docs)]
        for _ in range(args.turns):
            for run in runs:
                next(run, None)
    else:
        threads = [threading.Thread(target=lambda i=i, d=d: [None for _ in agent(args.base, i, d, args.turns, results, lock, args)])
                   for i, d in enumerate(docs)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    print(json.dumps({"mode": args.mode, "wall_s": round(time.time() - t0, 1)}), flush=True)
    with open(args.out, "a") as f:
        for line in results:
            f.write(json.dumps(line) + "\n")


main()
