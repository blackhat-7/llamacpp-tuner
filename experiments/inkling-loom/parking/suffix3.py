"""Long-suffix continuation after parking steps. Direct to Loom, greedy. Variant in argv[2]:
  eager-live:    pin, suspend, continue in the live row
  eager-rewind:  pin, suspend, another chat takes the row, restore into the same session
  eager-fork:    pin, suspend, another chat takes the row, restore into a new session
  release-live:  pin, release (DELETE), continue in the live row
Prints a readable score for the continued answer and for a fresh full read."""

import json
import re
import sys
import time
import urllib.request
import os
import pathlib

LCT_HOME = pathlib.Path(os.environ.get("LCT_HOME", pathlib.Path(__file__).resolve().parents[3] / "tmp"))

UP, VARIANT = sys.argv[1], sys.argv[2]
RUN = str(int(time.time()) % 100000)
S = f"sx-{RUN}"
small = open(LCT_HOME / "model-hunt" / "prompt_inkling_64000.txt").read()
big = open(LCT_HOME / "model-hunt" / "prompt_inkling_195000.txt").read()


def post(path, body=None, headers=None, method="POST"):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(UP + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "X-Loom-Model": "Inkling-2-Mini", **(headers or {})})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read() or b"{}")


def chat(messages, headers):
    d = post("/v1/chat/completions", {"model": "Inkling-2-Mini", "messages": messages, "max_tokens": 1500, "temperature": 0}, headers)
    return d["choices"][0]["message"], d.get("usage", {})


def score(m):
    text = (m.get("reasoning_content") or "") + " " + (m.get("content") or "")
    words = re.findall(r"\S+", text)
    return round(sum(bool(re.fullmatch(r"[A-Za-z][A-Za-z'’,.:;()-]*", w)) for w in words) / max(1, len(words)), 2), text[:90]


m1 = [{"role": "user", "content": small[:90000] + "\n\nWhat does this code do? Two sentences."}]
r1, _ = chat(m1, {"X-Loom-Session": S})
post(f"/v1/checkpoints/{S}-1", headers={"X-Loom-Session": S})
if VARIANT.startswith("eager"):
    post(f"/v1/checkpoints/{S}-1/suspend")
# pin-live: pinned and left in VRAM while the chat goes on (lct's copy-on-switch)
if VARIANT == "release-live":
    post(f"/v1/checkpoints/{S}-1", method="DELETE")
headers = {"X-Loom-Session": S}
if VARIANT in ("eager-rewind", "eager-fork"):
    chat([{"role": "user", "content": "Say hello in Spanish."}], {"X-Loom-Session": f"other-{RUN}"})
    headers = {"X-Loom-Session": S if VARIANT == "eager-rewind" else f"{S}-fork", "X-Loom-Checkpoint": f"{S}-1"}
a1 = {"role": "assistant", "content": r1.get("content") or ""}
if r1.get("reasoning_content"):
    a1["reasoning_content"] = r1["reasoning_content"]
m2 = m1 + [a1, {"role": "user", "content": "Here is more code:\n" + big[:int(sys.argv[3]) if len(sys.argv) > 3 else 15000] + "\n\nWhat does this second part do? Two sentences."}]
r2, u2 = chat(m2, headers)
cont = score(r2)
r3, _ = chat(m2, {"X-Loom-Session": f"fresh-{RUN}"})
print(json.dumps({"variant": VARIANT, "cached": (u2.get("prompt_tokens_details") or {}).get("cached_tokens"),
                  "continued": cont[0], "fresh": score(r3)[0], "start": cont[1]}), flush=True)
try:
    post(f"/v1/checkpoints/{S}-1", method="DELETE")
except Exception:
    pass
