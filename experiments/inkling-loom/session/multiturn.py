"""Two-turn chat at ~64K to check prefix-cache reuse; prints TTFT and cached tokens per turn. Server must be up on 8093."""
import json
import os
import pathlib
import time
import urllib.request

S = "http://127.0.0.1:8093/v1/chat/completions"
LCT_HOME = pathlib.Path(os.environ.get("LCT_HOME", pathlib.Path(__file__).resolve().parents[3] / "tmp"))
ctx = (LCT_HOME / "model-hunt/prompt_inkling_64000.txt").read_text()


def ask(messages):
    body = {"model": "Inkling-2-Mini", "messages": messages, "max_tokens": 200, "temperature": 0}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(S, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
    return time.time() - t, r


msgs = [{"role": "user", "content": ctx + "\n\nIn one sentence, what language is the code above written in?"}]
dt, r = ask(msgs)
m = r["choices"][0]["message"]
print("turn1", round(dt, 1), r["usage"])
msgs += [{"role": "assistant", "content": m.get("content", ""), "reasoning_content": m.get("reasoning_content", "")},
         {"role": "user", "content": "Now name one file that appears in it."}]
dt, r = ask(msgs)
print("turn2", round(dt, 1), r["usage"])
