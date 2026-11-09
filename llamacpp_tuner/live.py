"""Requests in flight and served, read from llama-server and Strata log lines."""

import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from itertools import count

HISTORY = 900  # samples kept, one a second: the last 15 minutes

# Strata prints one progress line a second per request, and no slot ids.
READ = re.compile(r"reading the prompt: (?:([\d,]+) of )?([\d,]+) tokens, ([\d.]+) s")
GENERATE = re.compile(
    r"(thinking|answering|writing a tool call: (.+?)|tool call complete): "
    # "of max 32768 tokens" or, since v0.2.9, "of budget 16384 (max 32000) tokens"
    r"([\d,]+) of [^,]*tokens, ([\d.]+) tok/s, ([\d.]+) s"
)
DONE = re.compile(
    r"done: ([\d,]+) tokens in ([\d.]+) s \(([\d.]+) tok/s\) \((\w+), cancel=\w+\)"
    r"(?:, expert cache ([\d.]+)% hit)?"
)
# llama-server logs a slot's start and end; timings arrive only at the end.
SLOTS = re.compile(r"n_slots = (\d+)")
SLOT = re.compile(r"slot +(\w+): id +(\d+) \| task -?\d+ \| (.*)")
PROGRESS = re.compile(r"prompt processing, n_tokens = +(\d+), progress = ([\d.]+)")
TIMING = re.compile(r"\s*(prompt eval|eval|total) time = +([\d.]+) ms / +(\d+) tokens")
DRAFT = re.compile(r"draft acceptance = ([\d.]+)")

_finished = count()


def number(text: str) -> int:
    return int(text.replace(",", ""))


@dataclass
class Request:
    server: str
    lane: int
    started: float
    phase: str = (
        "busy"  # prompt, thinking, answering, tool, or busy when the log does not say
    )
    tool: str = ""
    prompt: int = 0  # prompt tokens; 0 when unknown, such as a fully cached prompt
    read: int = 0  # prompt tokens read so far
    tokens: int = 0
    rate: float = 0.0  # generated tokens per second
    seconds: float = 0.0
    end: str = ""  # why it stopped: stop, length, disconnect
    hit: float | None = (
        None  # Strata's expert cache hit % or llama.cpp's draft acceptance %
    )
    steps: list[tuple[str, str]] = field(default_factory=list)  # (phase, tool) in order
    order: int = 0  # finishing order across all servers

    def enter(self, phase: str, tool: str = "") -> None:
        self.phase, self.tool = phase, tool
        if not self.steps or self.steps[-1] != (phase, tool):
            self.steps.append((phase, tool))


class Monitor:
    """One server's lanes (parallel slots), fed its log line by line."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.backend = ""  # "strata" or "llama.cpp" once a line shows which
        self.lanes = 1
        self.active: dict[int, Request] = {}
        self.done: deque[Request] = deque(maxlen=200)
        self.served = self.prompt_tokens = self.generated = 0
        self.tools: Counter[str] = Counter()
        self.ends: Counter[str] = Counter()
        self.trail: dict[int, deque[str]] = {}  # each lane's phase, one per sample

    def feed(self, line: str, now: float) -> None:
        if line.startswith("[strata] "):
            self.backend = "strata"
            self.strata(line.removeprefix("[strata] "), now)
        elif found := SLOT.search(line):
            self.backend = "llama.cpp"
            self.llama(found[1], int(found[2]), found[3], now)
        elif found := SLOTS.search(line):
            self.lanes = int(found[1])

    def open(self, lane: int, now: float) -> Request:
        if lane not in self.active:
            self.active[lane] = Request(self.name, lane, now)
        return self.active[lane]

    def close(self, lane: int) -> None:
        request = self.active.pop(lane)
        request.order = next(_finished)
        self.done.append(request)
        self.served += 1
        self.prompt_tokens += request.prompt
        self.generated += request.tokens
        self.ends[request.end or "stop"] += 1

    def strata(self, text: str, now: float) -> None:
        if text.startswith("the engine stopped"):
            self.active.clear()
        elif found := READ.match(text):
            request = self.open(0, now)
            request.enter("prompt")
            request.prompt = number(found[2])
            request.read = number(found[1] or found[2])
            request.seconds = float(found[3])
        elif found := GENERATE.match(text):
            request = self.open(0, now)
            phase = found[1] if found[1] in ("thinking", "answering") else "tool"
            # "tool call complete" names no tool: it ends the one being written.
            tool = found[2] or (request.tool if phase == "tool" else "")
            if (
                phase == "tool"
                and tool
                and (request.phase, request.tool) != ("tool", tool)
            ):
                self.tools[tool] += 1
            request.enter(phase, tool)
            request.tokens, request.rate = number(found[3]), float(found[4])
            request.seconds = float(found[5])
        elif found := DONE.match(text):
            request = self.open(0, now)
            request.tokens, request.seconds = number(found[1]), float(found[2])
            request.rate, request.end = float(found[3]), found[4]
            request.hit = float(found[5]) if found[5] else None
            if not request.steps:  # too short for a progress line
                request.enter("answering")
            self.close(0)

    def llama(self, event: str, lane: int, text: str, now: float) -> None:
        if event == "launch_slot_":
            self.active[lane] = Request(self.name, lane, now)
            return
        request = self.active.get(lane)
        if request is None:
            return
        if event == "release":
            if request.prompt:
                request.enter("prompt")
            if request.tokens:
                request.enter("answering")
            if not request.steps:  # embeddings and other prompt-only work
                request.enter("busy")
            self.close(lane)
        elif found := PROGRESS.match(text):
            request.read, progress = int(found[1]), float(found[2])
            request.prompt = round(request.read / progress) if progress else 0
        elif found := TIMING.match(text):
            seconds, tokens = float(found[2]) / 1000, int(found[3])
            if found[1] == "prompt eval":
                request.prompt = request.read = tokens
            elif found[1] == "eval":
                request.tokens = tokens
                request.rate = tokens / seconds if seconds else 0.0
            else:
                request.seconds = seconds
        elif found := DRAFT.search(text):
            request.hit = float(found[1]) * 100

    def elapsed(self, request: Request, now: float) -> float:
        # Strata's lines carry the request's own clock; llama-server's do not.
        return request.seconds if self.backend == "strata" else now - request.started

    def sample(self) -> None:
        for lane in range(self.lanes):
            request = self.active.get(lane)
            trail = self.trail.setdefault(lane, deque(maxlen=HISTORY))
            trail.append(request.phase if request else "")


class Live:
    """Every running server's monitor, plus once-a-second history across them."""

    def __init__(self) -> None:
        self.monitors: dict[str, Monitor] = {}
        # Total generation speed and what the fastest request was doing, per sample.
        self.speed: deque[tuple[float, str]] = deque(maxlen=HISTORY)
        self.busy: deque[int] = deque(maxlen=HISTORY)
        self.peak = 0.0

    def feed(self, server: str, line: str) -> None:
        if server not in self.monitors:
            self.monitors[server] = Monitor(server)
        self.monitors[server].feed(line, time.monotonic())

    def forget(self, server: str) -> None:
        self.monitors.pop(server, None)

    @property
    def serving(self) -> list[Monitor]:
        """Monitors of model servers; others (ComfyUI, Jeeves) log no requests."""
        return [m for m in self.monitors.values() if m.backend]

    @property
    def active(self) -> list[Request]:
        return [r for m in self.serving for r in m.active.values()]

    def sample(self) -> None:
        for monitor in self.serving:
            monitor.sample()
        # Prompt reading has no generation speed yet, so it adds nothing.
        generating = [r for r in self.active if r.phase != "prompt"]
        fastest = max(generating, key=lambda r: r.rate, default=None)
        total = sum(r.rate for r in generating)
        self.peak = max(self.peak, total)
        self.speed.append((total, fastest.phase if fastest else ""))
        self.busy.append(len(self.active))

    def recent(self, limit: int) -> list[Request]:
        done = [r for m in self.serving for r in m.done]
        return sorted(done, key=lambda r: r.order)[-limit:][::-1]
