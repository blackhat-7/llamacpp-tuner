"""The 2026-11-25 crash path, end to end through lct's /v1/messages, as Claude Code talks.

Turn 1: a long chat (LINES numbered lines; 5,200 is ~108K tokens) ends in a tool call
        whose text has "—" and "→".
Turn 2: the tool result. lct must send the call back spelled as the model wrote it, so
        Loom goes on in place: cached ~ all, about a second.
Turn 3: the client changed the last reply, so Loom must read the whole chat again, with
        Claude Code's max_tokens 32000. lct held the chat's pin through that read until
        2026-11-25, and at ~99K Loom failed the model (RESOURCE_EXHAUSTED); now the pin
        is released first and the turn completes.
Then, PAUSE seconds later, Loom's status: the model must be idle or trimming, not failed.
lct's log should say "live" for turn 2 and "replayed" for turn 3. With PAUSE 0 the status
call races lct's next pin, and Loom drops the pin's connection: lct turns parking off
(an open bug in PLAN.md, older than this fix).

Usage: python reread.py BASE [LINES] [PAUSE]   (BASE = a test lct serve on its own port)
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1]
LINES = int(sys.argv[2]) if len(sys.argv) > 2 else 5200
PAUSE = float(sys.argv[3]) if len(sys.argv) > 3 else 3
TEXT = "\n".join(
    f"Line {i}: the quick brown fox jumps over the lazy dog {i * 7919 % 10007}."
    for i in range(LINES)
)
TOOLS = [
    {
        "name": "post",
        "description": "Post a message.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }
]


def send(messages: list[dict]) -> dict:
    body = {
        "model": "x",
        "max_tokens": 32000,
        "system": "You are an agent. Use tools when asked.",
        "tools": TOOLS,
        "messages": messages,
    }
    request = urllib.request.Request(
        f"{BASE}/v1/messages",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=900) as r:
            reply = json.load(r)
    except urllib.error.HTTPError as error:
        reply = {"error": error.code, "body": error.read().decode()[:200]}
    reply["seconds"] = round(time.monotonic() - start, 1)
    return reply


def show(name: str, reply: dict) -> None:
    usage = reply.get("usage") or {}
    print(
        name,
        json.dumps(
            {
                "seconds": reply["seconds"],
                "stop": reply.get("stop_reason"),
                "read": usage.get("input_tokens"),
                "cached": usage.get("cache_read_input_tokens"),
                **({"error": reply["error"]} if "error" in reply else {}),
            }
        ),
        flush=True,
    )


ask = "\n\nCall the post tool once with exactly this text: Prices — up 10% → café"
messages: list[dict] = [{"role": "user", "content": TEXT + ask}]
reply = send(messages)
show("turn 1", reply)
call = next(b for b in reply["content"] if b["type"] == "tool_use")
print("tool input", json.dumps(call["input"], ensure_ascii=False))
result = {"type": "tool_result", "tool_use_id": call["id"], "content": "posted"}
messages += [
    {"role": "assistant", "content": reply["content"]},
    {"role": "user", "content": [result]},
]
reply = send(messages)
show("turn 2", reply)
edited = [
    {**b, "text": b["text"] + " (edited)"} if b["type"] == "text" else b
    for b in reply["content"]
]
if not any(b["type"] == "text" for b in edited):
    edited.append({"type": "text", "text": "Done (edited)."})
messages += [
    {"role": "assistant", "content": edited},
    {"role": "user", "content": "Now tell me the number on line 10. Answer briefly."},
]
show("turn 3", send(messages))
time.sleep(PAUSE)
with urllib.request.urlopen(f"{BASE}/v1/status", timeout=10) as r:
    status = json.load(r)
loom = status["data"][0]["loom"]
peak = status["server"]["resources"]["pools"][0]["memory"]["retained"]["peak_bytes"]
print("model", loom["status"]["phase"], loom.get("error"), f"peak {peak / 1e9:.2f} GB")
