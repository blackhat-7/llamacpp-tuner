"""Run one loom-serve session and a list of chat jobs; print one JSON line per job.

Run under the GPU lock: flock /tmp/xtx-gpu.lock python3 -I session.py --model DIR --preset P --jobs jobs.json --out out.jsonl
A job: {"name", "prompt_file" (optional), "question", "max_tokens"}.
"""
import argparse
import json
import os
import pathlib
import subprocess
import threading
import time
import urllib.request

LCT_HOME = pathlib.Path(os.environ.get("LCT_HOME", pathlib.Path(__file__).resolve().parents[3] / "tmp"))
LOOM = LCT_HOME / "loom"
OUT = LCT_HOME / "experiments/inkling-loom"
VRAM = pathlib.Path("/sys/class/drm/card2/device/mem_info_vram_used")


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def metrics(server):
    try:
        text = urllib.request.urlopen(f"{server}/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        if "draft" in name or "accept" in name or "propos" in name:
            try:
                out[name] = float(value)
            except ValueError:
                pass
    return out


def chat(server, job):
    context = pathlib.Path(os.path.expandvars(job["prompt_file"])).read_text() if job.get("prompt_file") else ""
    content = f"{context}\n\n{job['question']}" if context else job["question"]
    body = {"model": job["model"], "messages": [{"role": "user", "content": content}],
            "max_tokens": job["max_tokens"], "temperature": job.get("temperature", 0), "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time()
    first = last = None
    usage, reasoning, text, stamps = {}, [], [], []
    with post(f"{server}/v1/chat/completions", body) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            obj = json.loads(line[5:])
            usage = obj.get("usage") or usage
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                if d.get("content"):
                    text.append(d["content"])
                if d.get("reasoning_content") or d.get("content"):
                    last = time.time()
                    first = first or last
    n = usage.get("completion_tokens", 0)
    return {"prompt_tokens": usage.get("prompt_tokens"), "gen_tokens": n,
            "ttft_s": round(first - t0, 2) if first else None,
            "decode_tps": round((n - 1) / (last - first), 1) if first and last and last > first and n > 1 else None,
            "reasoning": "".join(reasoning), "content": "".join(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", default="8093")
    ap.add_argument("--label", default="")
    ap.add_argument("--serve-arg", action="append", default=[])
    args = ap.parse_args()
    server = f"http://127.0.0.1:{args.port}"
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / f"serve-kv8-{args.label or args.preset}.log", "w")
    proc = subprocess.Popen([str(LOOM / "venv/bin/loom-serve"), "--device=amdgpu", f"--model={args.model}",
                             f"--preset={args.preset}", f"--listen=127.0.0.1:{args.port}", *args.serve_arg],
                            stdout=log, stderr=subprocess.STDOUT)
    peak = [0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], int(VRAM.read_text()))
            time.sleep(0.2)

    threading.Thread(target=sample, daemon=True).start()
    try:
        t0 = time.time()
        model = None
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                raise SystemExit(f"loom-serve exited {proc.returncode}; see log")
            try:
                st = json.load(urllib.request.urlopen(f"{server}/v1/models", timeout=5))
                ids = [m["id"] for m in st.get("data", [])]
                if ids:
                    model = ids[0]
                    break
            except Exception:
                pass
            time.sleep(2)
        if model is None:
            raise SystemExit("server not ready")
        load_s = round(time.time() - t0, 1)
        print(json.dumps({"event": "ready", "load_s": load_s, "vram_gb": round(int(VRAM.read_text()) / 1e9, 2)}), flush=True)
        for job in json.load(open(args.jobs)):
            job["model"] = model
            m0 = metrics(server)
            if job.get("repeat"):
                outs = [chat(server, job) for _ in range(job["repeat"])]
                res = {"samples": [o["reasoning"] + "|" + o["content"] for o in outs],
                       "ttft": [o["ttft_s"] for o in outs]}
                m1 = metrics(server)
                res.update(name=job["name"], label=args.label)
                print(json.dumps(res)[:400], flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(res) + "\n")
                continue
            res = chat(server, job)
            m1 = metrics(server)
            res["metrics_delta"] = {k: m1[k] - m0.get(k, 0) for k in m1 if m1[k] != m0.get(k, 0)}
            res.update(name=job["name"], label=args.label, preset=args.preset,
                       peak_vram_gb=round(peak[0] / 1e9, 2))
            print(json.dumps({k: (v[:160] + "..." if isinstance(v, str) and len(v) > 200 else v) for k, v in res.items()}), flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(res) + "\n")
    finally:
        stop.set()
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


main()
