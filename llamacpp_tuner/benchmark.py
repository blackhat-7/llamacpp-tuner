"""Benchmark llama.cpp server performance by measuring TPS."""

import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from llamacpp_tuner.cache import CACHE_DIR
from llamacpp_tuner.llama import get_llama_binary


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    prompt_tps: float
    prompt_time_ms: float
    prompt_tokens: int
    gen_tps: float
    gen_time_ms: float
    gen_tokens: int
    total_time_ms: float
    args: dict[str, Any]

    def summary(self) -> str:
        """Return a formatted summary of the benchmark."""
        return (
            f"Prompt processing: {self.prompt_tps:.2f} TPS "
            f"({self.prompt_tokens} tokens in {self.prompt_time_ms:.0f}ms)\n"
            f"Generation: {self.gen_tps:.2f} TPS "
            f"({self.gen_tokens} tokens in {self.gen_time_ms:.0f}ms)\n"
            f"Total time: {self.total_time_ms:.0f}ms"
        )


def _wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def _send_completion(
    url: str, prompt: str, max_tokens: int = 256
) -> dict[str, Any] | None:
    """Send a completion request and return timing info."""
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def benchmark_server(
    model_path: str,
    server_args: list[str],
    port: int = 8081,
    prompt: str | None = None,
    max_tokens: int = 256,
    warmup: bool = True,
) -> BenchmarkResult | None:
    """Start llama-server with args, benchmark it, and return results.

    Args:
        model_path: Path to GGUF model
        server_args: List of server arguments (excluding -m and --port/--host)
        port: Port to run benchmark server on
        prompt: Custom prompt (uses default if None)
        max_tokens: Max tokens to generate
        warmup: Whether to run a warmup prompt

    Returns:
        BenchmarkResult or None if failed
    """
    binary = get_llama_binary()
    if not binary:
        print("Error: llama-server binary not found.", file=sys.stderr)
        return None

    if not prompt:
        prompt = (
            "Write a detailed analysis of machine learning optimization techniques, "
            "including quantization methods, pruning strategies, and hardware acceleration. "
            "Explain the trade-offs between model size and performance."
        )

    # Use a different port for benchmark to avoid conflicts
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    cmd = [
        str(binary),
        "-m",
        model_path,
        "--port",
        str(port),
        "--host",
        host,
    ] + server_args

    # Capture server output for debugging
    server_logs: list[str] = []

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(CACHE_DIR),
        text=True,
    )

    def _read_logs() -> str:
        """Read any available logs from the server process."""
        if proc.stdout:
            try:
                import select

                ready, _, _ = select.select([proc.stdout], [], [], 0.5)
                if ready:
                    lines = []
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        lines.append(line)
                        if len(lines) > 50:  # Limit log size
                            break
                    return "".join(lines)
            except Exception:
                pass
        return ""

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Starting server...", total=None)
            time.sleep(0.5)

            initial_logs = _read_logs()
            if initial_logs:
                server_logs.append(initial_logs)

            progress.update(task, description="Waiting for server...")
            if not _wait_for_server(url, timeout=60.0):
                remaining_logs = _read_logs()
                if remaining_logs:
                    server_logs.append(remaining_logs)

                print(
                    "\nBenchmark failed: Server failed to start within timeout.",
                    file=sys.stderr,
                )
                if server_logs:
                    print("\nServer logs:", file=sys.stderr)
                    print("-" * 60, file=sys.stderr)
                    print("".join(server_logs)[-2000:], file=sys.stderr)
                    print("-" * 60, file=sys.stderr)
                return None

            if warmup:
                progress.update(task, description="Warming up...")
                _send_completion(url, "Hello, this is a warmup.", max_tokens=32)

            progress.update(task, description="Processing prompt...")
            long_prompt = prompt * 10
            prompt_start = time.time()
            prompt_result = _send_completion(url, long_prompt, max_tokens=8)
            prompt_end = time.time()

            prompt_tokens = 0
            prompt_time_ms = 0.0
            if prompt_result and "usage" in prompt_result:
                usage = prompt_result["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                prompt_time_ms = (prompt_end - prompt_start) * 1000

            progress.update(task, description=f"Generating {max_tokens} tokens...")
            gen_start = time.time()
            gen_result = _send_completion(url, prompt, max_tokens=max_tokens)
            gen_end = time.time()

        gen_tokens = 0
        gen_time_ms = 0.0
        if gen_result:
            if "usage" in gen_result:
                usage = gen_result["usage"]
                gen_tokens = usage.get("completion_tokens", 0)
                gen_time_ms = (gen_end - gen_start) * 1000
            elif "error" in gen_result:
                print(
                    f"\nGeneration error: {gen_result.get('error', 'unknown')}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"\nWarning: unexpected response format: {list(gen_result.keys())}",
                    file=sys.stderr,
                )
        else:
            print("\nWarning: generation request returned no result", file=sys.stderr)

        total_time_ms = prompt_time_ms + gen_time_ms

        prompt_tps = (
            (prompt_tokens / prompt_time_ms * 1000) if prompt_time_ms > 0 else 0
        )
        gen_tps = (gen_tokens / gen_time_ms * 1000) if gen_time_ms > 0 else 0

        args_dict: dict[str, Any] = {}
        i = 0
        while i < len(server_args):
            arg = server_args[i]
            if arg.startswith("-"):
                if i + 1 < len(server_args) and not server_args[i + 1].startswith("-"):
                    args_dict[arg] = server_args[i + 1]
                    i += 2
                else:
                    args_dict[arg] = True
                    i += 1
            else:
                i += 1

        return BenchmarkResult(
            prompt_tps=prompt_tps,
            prompt_time_ms=prompt_time_ms,
            prompt_tokens=prompt_tokens,
            gen_tps=gen_tps,
            gen_time_ms=gen_time_ms,
            gen_tokens=gen_tokens,
            total_time_ms=total_time_ms,
            args=args_dict,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
