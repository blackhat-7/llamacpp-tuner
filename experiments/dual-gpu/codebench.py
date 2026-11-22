"""Graded coding test against the live server on :6868: HumanEval+ (164) and MBPP+ (378) with EvalPlus's hidden tests.

Greedy, thinking off, one request at a time. All answers first (OUT/ARM.answers.jsonl, resumable), then each answer's
last python code block runs against its task's own tests in a fresh temporary directory (Strata's venv Python for
numpy; 60 s, 4 GiB of address space), 8 at a time, after the generation so the tests do not slow the model.
Results: OUT/ARM.jsonl and a summary line per set. OUT is $LCT_HOME/experiments/dual-gpu/codebench; its data/ comes
from fetch_data.py. Env: STRATA_SRC (its .venv runs the tests) or CODEBENCH_PYTHON.

usage: codebench.py ARM [LIMIT]
"""
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

OUT = Path(os.environ.get("LCT_HOME") or Path(__file__).resolve().parents[2] / "tmp") / "experiments/dual-gpu/codebench"
PY = os.environ.get("CODEBENCH_PYTHON") or f"{os.environ['STRATA_SRC']}/.venv/bin/python"
URL = "http://127.0.0.1:6868/v1/chat/completions"
FENCE = "`" * 3
BLOCK = re.compile(FENCE + r"(?:python|py)?[ \t]*\n(.*?)" + FENCE, re.S)


def ask(prompt: str) -> tuple[str, int, float]:
    body = {"model": "swift-1.5-iq3_xxs", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1536,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.loads(r.read())
    text = res["choices"][0]["message"].get("content") or ""
    return text, res.get("usage", {}).get("completion_tokens", 0), time.time() - t0


def tasks() -> dict:
    out = {}
    for r in map(json.loads, open(OUT / "data/humanevalplus.jsonl")):
        prompt = ("Complete this Python function. Reply with the whole function, with its imports, in one python code "
                  f"block.\n\n{FENCE}python\n{r['prompt']}{FENCE}")
        out[r["task_id"]] = (prompt, r)
    for r in map(json.loads, open(OUT / "data/mbppplus.jsonl")):
        prompt = (f"{r['prompt']}\nYour code should pass this test:\n{r['test_list'][0]}\n"
                  "Reply with the Python function in one python code block.")
        out[f"Mbpp/{r['task_id']}"] = (prompt, r)
    return out


def program(task_id: str, r: dict, text: str) -> str:
    blocks = BLOCK.findall(text)
    code = blocks[-1] if blocks else text
    if task_id.startswith("HumanEval/"):
        if f"def {r['entry_point']}" not in code:   # a body only: complete the prompt with it
            code = r["prompt"] + code
        header = "\n".join(l for l in r["prompt"].splitlines() if l.startswith(("import ", "from ")))
        return f"{header}\n{code}\n{r['test']}\ncheck({r['entry_point']})\n"
    return "\n".join(r["test_imports"]) + f"\n{code}\n{r['test']}\n"


def run(src: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        Path(d, "t.py").write_text(src)
        try:
            p = subprocess.run(["prlimit", f"--as={4 << 30}", PY, "-I", "t.py"], cwd=d, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False
        return p.returncode == 0


def main() -> None:
    arm = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
    all_tasks = tasks()
    ans_path = OUT / f"{arm}.answers.jsonl"
    answered = {json.loads(l)["task"] for l in open(ans_path)} if ans_path.exists() else set()
    with open(ans_path, "a") as f:
        for i, (task_id, (prompt, _)) in enumerate(all_tasks.items()):
            if i >= limit:
                break
            if task_id not in answered:
                text, ntok, secs = ask(prompt)
                f.write(json.dumps({"task": task_id, "tokens": ntok, "seconds": round(secs, 2), "answer": text}) + "\n")
                f.flush()
    answers = [json.loads(l) for l in open(ans_path)]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        passed = list(pool.map(lambda a: run(program(a["task"], all_tasks[a["task"]][1], a["answer"])), answers))
    with open(OUT / f"{arm}.jsonl", "w") as f:
        for a, ok in zip(answers, passed):
            f.write(json.dumps({"task": a["task"], "pass": ok, "tokens": a["tokens"], "seconds": a["seconds"]}) + "\n")
    for name in ("HumanEval", "Mbpp"):
        rs = [(a, ok) for a, ok in zip(answers, passed) if a["task"].startswith(name + "/")]
        if rs:
            print(f"{arm} {name}+: {sum(ok for _, ok in rs)}/{len(rs)} passed, {sum(a['tokens'] for a, _ in rs)} tokens, "
                  f"{sum(a['seconds'] for a, _ in rs) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
