"""GPQA Diamond against an OpenAI-compatible server, one question at a time.

Questions run in a fixed seeded order, so two models stopped at the same count saw the
same questions. Results append to a JSONL file; a rerun skips questions already done.
Prompt and grading follow openai/simple-evals (the 'Answer: X' last line).

    uv run --no-project --with pandas --with pyarrow python -I gpqa.py DATA.parquet OUT.jsonl \
        --url http://127.0.0.1:6868 --model swift --sampling '{"temperature":1.0,"top_p":0.95,"top_k":20}'
"""

import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

PROMPT = (
    "Answer the following multiple choice question. The last line of your response should be of the "
    "following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
    "Think step by step before answering.\n\n{question}"
)
ANSWER = re.compile(r"(?i)Answer\s*:\s*\**\s*\(?\s*([ABCD])\b")


def ask(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def cpu_temp() -> float:
    for hw in Path("/sys/class/hwmon").iterdir():
        if (hw / "name").read_text().strip() == "coretemp":
            return int((hw / "temp1_input").read_text()) / 1000
    raise RuntimeError("no coretemp sensor")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--url", default="http://127.0.0.1:6868")
    p.add_argument("--model", required=True)
    p.add_argument("--sampling", default="{}", help="JSON merged into each request")
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument(
        "--limit", type=int, default=198, help="stop after this many questions"
    )
    p.add_argument(
        "--stop-at", type=float, default=0, help="unix time; start no question after it"
    )
    p.add_argument(
        "--cool-below", type=float, default=0, help="wait for CPU package below this °C"
    )
    a = p.parse_args()

    df = pd.read_parquet(a.data)
    order = list(range(len(df)))
    random.Random(0).shuffle(order)
    done = set()
    try:
        with open(a.out) as f:
            done = {json.loads(line)["idx"] for line in f if line.strip()}
    except FileNotFoundError:
        pass

    for rank, idx in enumerate(order[: a.limit]):
        if idx in done:
            continue
        if a.stop_at and time.time() > a.stop_at:
            print("stop time reached", flush=True)
            break
        # This host has crashed under sustained heat; let the CPU cool between questions.
        while a.cool_below and cpu_temp() >= a.cool_below:
            time.sleep(5)
        row = df.iloc[idx]
        body = {
            "model": a.model,
            "max_tokens": a.max_tokens,
            "messages": [
                {"role": "user", "content": PROMPT.format(question=row["question"])}
            ],
            **json.loads(a.sampling),
        }
        t0 = time.time()
        # A server restart or a dropped connection should cost one retry, not the run.
        for attempt in range(3):
            try:
                d = ask(a.url, body, timeout=3600)
                break
            except Exception as e:  # noqa: BLE001
                print(f"q{idx} attempt {attempt}: {e}", flush=True)
                time.sleep(30)
        else:
            print(f"q{idx} failed 3 times; stopping", flush=True)
            break
        msg = d["choices"][0]["message"]
        content = msg.get("content") or ""
        found = ANSWER.findall(content)
        pred = found[-1].upper() if found else None
        rec = {
            "rank": rank,
            "idx": idx,
            "gold": row["answer"],
            "pred": pred,
            "correct": pred == row["answer"],
            "finish": d["choices"][0].get("finish_reason"),
            "completion_tokens": d.get("usage", {}).get("completion_tokens"),
            "seconds": round(time.time() - t0, 1),
            "content": content,
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        }
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(
            f"#{rank} q{idx} gold={rec['gold']} pred={pred} ok={rec['correct']} "
            f"tok={rec['completion_tokens']} {rec['seconds']}s {rec['finish']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
