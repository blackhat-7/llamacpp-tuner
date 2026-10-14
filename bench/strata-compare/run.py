"""Profile Strata on this machine with fixed inputs, so two machines can be compared stage by stage.

usage: python3 run.py --config PATH/TO/strata-<model>.json [--out results/NAME] [--timeout 900]

Stop the Strata server first (this runs the engine directly and needs the whole GPU). It takes 10-20 minutes:
3 decode runs (code, edit, prose; 600 tokens each, the same continuation as on the other machine), 2 profiled decode
runs, 2 prompt runs (15K and 35K tokens) and 2 decode runs without route-tail skipping. Every run loads the model
again. Standard library only.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"

# name, token file, follow file, max new tokens, extra environment
RUNS = [
    ("speed_code", "code.ids", "code.follow", 600, {}),
    ("speed_edit", "edit.ids", "edit.follow", 600, {}),
    ("speed_prose", "prose.ids", "prose.follow", 600, {}),
    (
        "prof_code",
        "code.ids",
        "code.follow",
        600,
        {"STRATA_VERIFY_PROFILE": "1", "STRATA_DECODE_TIMING": "1"},
    ),
    (
        "prof_prose",
        "prose.ids",
        "prose.follow",
        600,
        {"STRATA_VERIFY_PROFILE": "1", "STRATA_DECODE_TIMING": "1"},
    ),
    ("prompt_15k", "p15k.ids", None, 16, {"STRATA_PREFILL_TIMING": "1"}),
    ("prompt_35k", "long32k.ids", None, 16, {"STRATA_PREFILL_TIMING": "1"}),
    # NVIDIA builds skip low-ranked missed experts by default since 0.1.42 (lossy); these two measure without it
    ("noskip_code", "code.ids", "code.follow", 600, {"STRATA_ROUTE_TAIL_SKIP": "0"}),
    ("noskip_prose", "prose.ids", "prose.follow", 600, {"STRATA_ROUTE_TAIL_SKIP": "0"}),
]

# lines worth reading first, in order
KEY = re.compile(
    r"^(decode|prefill|speculation|verify window|verify GPU stages|pool multi|dispatch|pcie experts|mtp|"
    r"follow|window sizes|accepted per round)\s|R4 expert-cache hits|prefill timing|expert cache|"
    r"pool workers on|STRATA_|is ON|is OFF|strata generate: GPU|engine|probe|GB/s|prompt chunk|borrows"
)


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_busy() -> str:
    """The processes using the GPU, if nvidia-smi can tell."""
    return sh(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )


def system_info(cfg: dict, cfg_path: Path) -> dict:
    cwd = Path(cfg.get("cwd") or cfg_path.parent)
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "os": platform.platform(),
        "cpu": platform.processor()
        or sh(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo"]),
        "lscpu": sh(["lscpu"]),
        "memory": sh(["free", "-g"])
        or sh(["wmic", "computersystem", "get", "TotalPhysicalMemory"]),
        "gpu": sh(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max,"
                "clocks.max.sm,clocks.max.mem,power.limit",
                "--format=csv",
            ]
        ),
        "strata_commit": sh(
            ["git", "-C", str(cwd), "describe", "--tags", "--always", "--dirty"]
        ),
        "strata_head": sh(["git", "-C", str(cwd), "log", "-1", "--format=%H %cd %s"]),
        "config": cfg,
    }


def run_one(
    cfg: dict,
    name: str,
    ids: str,
    follow: str | None,
    max_new: int,
    extra_env: dict,
    out: Path,
    timeout: int,
) -> tuple[int, float]:
    exe = cfg["exe"]
    args = [a for a in cfg.get("args", []) if a != "--serve"]
    run_args = [
        "--tokens-file",
        str(INPUTS / ids),
        "--max-new",
        str(max_new),
        "--stats",
        "--adapt-every",
        "0",
    ]
    if follow:
        run_args += [
            "--spec-follow",
            str(INPUTS / follow),
            "--window-hashes",
            str(out / f"{name}.hashes"),
        ]
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    env.update(extra_env)
    libs = [str(p) for p in cfg.get("lib_dirs") or []]
    if libs:
        key = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[key] = os.pathsep.join(libs + ([env[key]] if env.get(key) else []))
    cwd = cfg.get("cwd") or None
    log = out / f"{name}.log"
    t0 = time.time()
    with open(log, "w") as f:
        f.write("command: " + json.dumps([exe] + args + run_args) + "\n")
        f.write("extra env: " + json.dumps(extra_env) + "\n")
        f.flush()
        try:
            code = subprocess.run(
                [exe] + args + run_args,
                cwd=cwd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            code = -9
    return code, time.time() - t0


def summary(out: Path) -> str:
    lines = []
    for name, *_ in RUNS:
        log = out / f"{name}.log"
        if not log.exists():
            continue
        lines.append(f"===== {name}")
        for line in log.read_text(errors="replace").splitlines():
            if KEY.search(line) and len(line) < 4000:
                lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        required=True,
        help="the Strata run config the server uses (strata-<model>.json)",
    )
    ap.add_argument("--out", help="results folder (default results/<host>-<date>)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per run")
    ap.add_argument("--only", help="comma-separated run names, to repeat some runs")
    a = ap.parse_args()
    cfg_path = Path(a.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    if not Path(cfg["exe"]).exists() and not shutil.which(cfg["exe"]):
        print(f"the engine in the config does not exist: {cfg['exe']}", file=sys.stderr)
        return 2
    busy = gpu_busy()
    if "strata" in busy.lower():
        print(
            "A Strata engine is still using the GPU; stop the server first:\n" + busy,
            file=sys.stderr,
        )
        return 2
    host = platform.node().split(".")[0] or "host"
    # absolute: the engine runs in the config's cwd and writes the window hashes here
    out = (
        Path(a.out)
        if a.out
        else HERE / "results" / f"{host}-{time.strftime('%Y%m%d-%H%M')}"
    ).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "system.json").write_text(json.dumps(system_info(cfg, cfg_path), indent=1))
    only = set(a.only.split(",")) if a.only else None
    for name, ids, follow, max_new, extra in RUNS:
        if only is not None and name not in only:
            continue
        print(f"{name} ...", end=" ", flush=True)
        code, secs = run_one(cfg, name, ids, follow, max_new, extra, out, a.timeout)
        text = (out / f"{name}.log").read_text(errors="replace")
        speed = re.findall(
            r"^(?:decode|prefill)\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s",
            text,
            re.M,
        )
        print(
            f"exit {code}, {secs:.0f} s, tok/s {' / '.join(speed) or '-'}", flush=True
        )
        if code != 0 and "unknown argument" in text:
            print(
                "  this engine does not know one of the flags (see the log); continuing",
                flush=True,
            )
    (out / "summary.txt").write_text(summary(out))
    print(f"\nDone: {out}\nPlease commit that folder and push it to this branch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
