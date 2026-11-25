"""Esc mid-reply through lct's /v1/messages (2026-11-25): a pinned 28K chat, a long
streamed reply the client abandons, then a follow-up. lct's log should keep parking on
("1 chats parked") and Loom's model must stay up.

Usage: python esc.py BASE   (BASE = a test lct serve on its own port)
"""

import json
import sys
import time
import urllib.request

BASE = sys.argv[1]
TEXT = "\n".join(
    f"Line {i}: the quick brown fox jumps over the lazy dog {i * 7919 % 10007}."
    for i in range(1400)
)


def request(messages: list[dict], stream: bool) -> urllib.request.Request:
    body = {"model": "x", "max_tokens": 32000, "stream": stream, "messages": messages}
    return urllib.request.Request(
        f"{BASE}/v1/messages",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )


first = [
    {
        "role": "user",
        "content": TEXT + "\nWhich number is on line 777? Answer with the number only.",
    }
]
with urllib.request.urlopen(request(first, False), timeout=600) as r:
    reply = json.load(r)
print("turn 1", reply["stop_reason"], reply["usage"], flush=True)
second = first + [
    {"role": "assistant", "content": reply["content"]},
    {
        "role": "user",
        "content": "Now write an extremely long essay about every line, one paragraph each.",
    },
]
r = urllib.request.urlopen(request(second, True), timeout=600)
events = 0
for line in r:
    events += line.startswith(b"event:")
    if events >= 400:
        break
r.close()  # Esc
print("turn 2 abandoned after", events, "events", flush=True)
time.sleep(2)
third = second + [
    {"role": "user", "content": "Stop. Just tell me the number on line 10."}
]
start = time.monotonic()
with urllib.request.urlopen(request(third, False), timeout=600) as r:
    reply = json.load(r)
print(
    "turn 3",
    reply["stop_reason"],
    reply["usage"],
    f"{time.monotonic() - start:.1f}s",
    flush=True,
)
with urllib.request.urlopen(f"{BASE}/v1/status", timeout=10) as r:
    loom = json.load(r)["data"][0]["loom"]
print("model", loom["status"]["phase"], loom.get("error"))
