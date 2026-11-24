"""Worst cases near the context limit, through lct's /v1/messages: a ~190K chat that gets
parked, a small chat in between, the big chat restored, an image at full depth, and a
request over the limit (must fail cleanly). Prints one JSON line per step with peak VRAM."""

import base64
import json
import pathlib
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
import os

LCT_HOME = pathlib.Path(os.environ.get("LCT_HOME", pathlib.Path(__file__).resolve().parents[3] / "tmp"))

BASE = sys.argv[1]
VRAM = pathlib.Path("/sys/class/drm/card2/device/mem_info_vram_used")
DOC = LCT_HOME / "model-hunt/prompt_inkling_195000.txt"
peak = [0]


def sample() -> None:
    while True:
        peak[0] = max(peak[0], int(VRAM.read_text()))
        time.sleep(0.1)


def png() -> str:
    """A 256x256 picture: four coloured squares, base64 PNG."""
    w = h = 256
    colours = [(220, 40, 40), (40, 160, 60), (40, 70, 220), (240, 200, 30)]
    rows = b"".join(
        b"\x00" + b"".join(bytes(colours[(y >= 128) * 2 + (x >= 128)]) for x in range(w))
        for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return base64.b64encode(raw).decode()


def send(messages: list, max_tokens: int) -> dict:
    body = {"model": "inkling", "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "system": "You are a careful assistant. Answer briefly.", "messages": messages}
    request = urllib.request.Request(f"{BASE}/v1/messages", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    peak[0] = 0
    t0 = time.time()
    first, text, thinking, usage, err = None, [], [], {}, None
    try:
        with urllib.request.urlopen(request, timeout=3600) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                event = json.loads(line[5:])
                if event.get("type") == "content_block_delta":
                    first = first or time.time()
                    d = event["delta"]
                    (thinking if d.get("type") == "thinking_delta" else text).append(
                        d.get("thinking") or d.get("text") or "")
                elif event.get("type") == "message_delta":
                    usage = event.get("usage", {})
                elif event.get("type") == "error":
                    err = event
    except urllib.error.HTTPError as e:
        err = {"status": e.code, "body": e.read().decode()[:200]}
    return {"ttft_s": round(first - t0, 2) if first else None, "s": round(time.time() - t0, 1),
            "input": usage.get("input_tokens"), "cached": usage.get("cache_read_input_tokens"),
            "output": usage.get("output_tokens"), "text": "".join(text)[:80], "error": err,
            "peak_vram_gb": round(peak[0] / 1e9, 2), "_thinking": "".join(thinking),
            "_text": "".join(text)}


def reply(r: dict) -> dict:
    blocks = [{"type": "thinking", "thinking": r["_thinking"], "signature": ""}] if r["_thinking"] else []
    return {"role": "assistant", "content": blocks + [{"type": "text", "text": r["_text"] or "(empty)"}]}


def log(step: str, r: dict) -> None:
    print(json.dumps({"step": step, **{k: v for k, v in r.items() if not k.startswith("_")}}), flush=True)


threading.Thread(target=sample, daemon=True).start()
doc = DOC.read_text()
big = [{"role": "user", "content": doc[: int(len(doc) * float(sys.argv[2]))] + "\n\nIn one sentence, what is this?"}]
r = send(big, 8192)
log("big chat, first read", r)
big += [reply(r)]
log("small chat takes the row", send([{"role": "user", "content": "Say hello in Spanish."}], 512))
big += [{"role": "user", "content": "Name one function it defines. One line."}]
r = send(big, 8192)
log("big chat restored", r)
big += [reply(r)]
image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png()}}
big += [{"role": "user", "content": [image, {"type": "text", "text": "What colours are in this picture?"}]}]
r = send(big, 4096)
log("image at full depth", r)
if not r["error"]:
    big += [reply(r)]
big += [{"role": "user", "content": "One more word about it."}]
log("over the limit (max_tokens 32000)", send(big, 32000))
log("small chat after", send([{"role": "user", "content": "Say OK."}], 256))
