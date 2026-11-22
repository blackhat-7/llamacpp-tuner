#!/usr/bin/env python3
"""Greedy streamed chat request to a loom-serve; prints one JSON line with decode t/s and the generated text.
usage: e2e_client.py SERVER PROMPT_FILE MAX_TOKENS LABEL [KIND]   KIND: prose (default) or code"""
import json, sys, time, urllib.request

server, prompt, max_tokens, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ctx = open(prompt).read()
QS = {"prose": "Ignore the dump above for this task. In plain English, write about 300 words on how the printing press changed Europe.",
      "code": "Ignore the dump above for this task. Write a self-contained C++17 program: an LRU cache class template with "
              "get/put/erase and a capacity, plus a main() that tests it with asserts. Output only the code."}
kind = sys.argv[5] if len(sys.argv) > 5 else "prose"
q = QS[kind]


def metrics():
    out = {}
    for line in urllib.request.urlopen(f"{server}/metrics", timeout=10).read().decode().splitlines():
        name, _, value = line.rpartition(" ")
        if not line.startswith("#") and any(w in name for w in ("draft", "accept", "propos")):
            try:
                out[name] = float(value)
            except ValueError:
                pass
    return out

for _ in range(900):  # the model loads after the server reports ready
    data = json.load(urllib.request.urlopen(f"{server}/v1/models")).get("data")
    if data:
        break
    time.sleep(1)
model = data[0]["id"]
body = {"model": model, "messages": [{"role": "user", "content": f"{ctx}\n\n{q}"}], "max_tokens": max_tokens,
        "temperature": 0, "logprobs": True, "top_logprobs": 5, "stream": True, "stream_options": {"include_usage": True}}
m0 = metrics()
req = urllib.request.Request(f"{server}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time(); stamps = []; out = []; usage = {}; lps = []
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        line = line.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]":
            continue
        obj = json.loads(line[5:])
        usage = obj.get("usage") or usage
        for ch in obj.get("choices", []):
            d = ch.get("delta") or {}
            lps += ((ch.get("logprobs") or {}).get("content") or [])
            piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
            if piece:
                stamps.append(time.time()); out.append(piece)
n = usage.get("completion_tokens", 0)
m1 = metrics()
res = {"label": label, "kind": kind, "metrics": {k: v - m0.get(k, 0) for k, v in m1.items() if v != m0.get(k, 0)}, "prompt_tokens": usage.get("prompt_tokens"), "ttft_s": round(stamps[0] - t0, 1) if stamps else None,
       "gen_tokens": n, "decode_tps": round((n - 1) / (stamps[-1] - stamps[0]), 1) if len(stamps) > 1 else None,
       "text": "".join(out), "logprobs": [[e.get("token"), [(t["token"], round(t["logprob"], 3)) for t in e.get("top_logprobs", [])]] for e in lps]}
print(json.dumps(res), flush=True)
