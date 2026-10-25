"""Where a local Strata model's time went in Claude Code sessions.

    python3 bench/agent_time.py ~/Documents/projects/Strata/strata-swift-iq3_xxs-tuned.json

Strata's engine log (the config's "log") has a line per finished request: tokens reused from a cached conversation,
read and generated, with their times. Claude Code's transcripts (~/.claude/projects) keep the usage of every response
from the config's "model_name". A request whose prompt, reused and generated tokens match a response was an agent turn
(main session or subagent); the rest were Claude Code's other calls: auto mode's safety checks and helper calls
(titles, summaries). A turn that reused under half its prompt was read cold; its transcript tells why.
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REQUEST = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+)(?: of \d+)? read in (\d+) ms .*?, (\d+) generated in (\d+) ms"
)
BATCH = re.compile(r"strata batch: (\d+) windows, avg [\d.]+ rows, ([\d.]+) ms/window")
ENGINE_START = "memory guard"


@dataclass
class Request:
    prompt: int
    reused: int
    read: int
    read_ms: int
    generated: int
    gen_ms: int


@dataclass
class Run:
    requests: list[Request]
    batch_ms: float = 0.0


@dataclass
class Turn:
    prompt: int
    reused: int
    generated: int
    why: str


def cold(prompt: int, reused: int) -> bool:
    return reused < prompt / 2


def engine_runs(log: Path) -> list[Run]:
    """One Run per engine start; a two-at-once (batch) request's line counts only its first token."""
    runs: list[Run] = []
    for line in log.open(errors="replace"):
        if ENGINE_START in line:
            runs.append(Run([]))
        elif runs and (m := REQUEST.search(line)):
            runs[-1].requests.append(Request(*map(int, m.groups())))
        elif runs and (m := BATCH.search(line)):
            runs[-1].batch_ms += int(m[1]) * float(m[2])
    return runs


def agent_turns(transcripts: Path, model: str) -> list[Turn]:
    """Every response of `model`, once, with why its conversation was not cached when it was read cold."""
    seen: set[str] = set()
    turns: list[Turn] = []
    for path in sorted(transcripts.glob("*/*.jsonl")) + sorted(
        transcripts.glob("*/*/subagents/*.jsonl")
    ):
        text = path.read_text(errors="replace")
        if model not in text:
            continue
        first, compacted = True, False
        for line in text.splitlines():
            # A live session's last line can be half written.
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                entry.get("isCompactSummary")
                or entry.get("subtype") == "compact_boundary"
            ):
                compacted = True
            message = entry.get("message")
            if not isinstance(message, dict) or message.get("model") != model:
                continue
            # Claude Code writes one line per content block, each with the same usage.
            if message.get("id") in seen:
                continue
            seen.add(message.get("id"))
            usage = message.get("usage") or {}
            reused = usage.get("cache_read_input_tokens", 0)
            if first:
                why = (
                    "subagent's first turn"
                    if "/subagents/" in str(path)
                    else "session start or resume"
                )
            else:
                why = "after compaction" if compacted else "conversation not cached"
            turns.append(
                Turn(
                    usage.get("input_tokens", 0) + reused,
                    reused,
                    usage.get("output_tokens", 0),
                    why,
                )
            )
            first, compacted = False, False
    return turns


def minutes(requests: list[Request], key: str) -> float:
    return sum(getattr(r, key) for r in requests) / 60_000


def rate(tokens: int, ms: float) -> str:
    return f"{tokens / ms * 1000:,.0f}" if ms else "-"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config", type=Path, help="the Strata run config (strata-<model>.json)"
    )
    parser.add_argument(
        "--transcripts", type=Path, default=Path.home() / ".claude" / "projects"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    turns = agent_turns(args.transcripts, config["model_name"])
    exact = Counter((t.prompt, t.reused, t.generated) for t in turns)
    admitted = Counter((t.prompt, t.reused) for t in turns)

    print(
        "Minutes per engine run. Agent turns: writing, reading (cold re-reads of it, turns), read and write t/s."
    )
    print(
        "Other: requests no transcript matches (Claude Code's safety checks and helper calls, other clients)."
    )
    print(
        "Two at once: batch windows; the writing done in them is not in the columns before."
    )
    print(
        f"{'run':>4} {'busy':>6} | {'turns':>5} {'write':>6} {'read':>6} {'cold':>12} {'read/s':>7} {'write/s':>7} |"
        f" {'other':>5} {'write':>6} {'read':>6} | {'2 at once':>9}"
    )
    for number, run in enumerate(engine_runs(Path(config["log"]))):
        if not run.requests:
            continue
        agent, other = [], []
        for r in run.requests:
            # A batch admission's line has 1 generated token; the rest are written in batch windows.
            mine = (
                admitted[(r.prompt, r.reused)]
                if r.generated <= 1
                else exact[(r.prompt, r.reused, r.generated)]
            )
            (agent if mine else other).append(r)
        cold_turns = [r for r in agent if cold(r.prompt, r.reused)]
        busy = (
            minutes(run.requests, "read_ms")
            + minutes(run.requests, "gen_ms")
            + run.batch_ms / 60_000
        )
        print(
            f"{number:>4} {busy:>6.1f} | {len(agent):>5} {minutes(agent, 'gen_ms'):>6.1f}"
            f" {minutes(agent, 'read_ms'):>6.1f} {minutes(cold_turns, 'read_ms'):>6.1f} ({len(cold_turns):>3})"
            f" {rate(sum(r.read for r in agent), sum(r.read_ms for r in agent)):>7}"
            f" {rate(sum(r.generated for r in agent), sum(r.gen_ms for r in agent)):>7} |"
            f" {len(other):>5} {minutes(other, 'gen_ms'):>6.1f} {minutes(other, 'read_ms'):>6.1f} |"
            f" {run.batch_ms / 60_000:>9.1f}"
        )

    reasons: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for t in turns:
        if cold(t.prompt, t.reused):
            reasons[t.why] += 1
            tokens[t.why] += t.prompt - t.reused
    print(
        f"\nCold agent turns in the transcripts ({len(turns)} turns in all), by cause:"
    )
    for why, count in reasons.most_common():
        print(f"  {count:>5}  {why:<26} {tokens[why]:>12,} tokens read")


if __name__ == "__main__":
    main()
