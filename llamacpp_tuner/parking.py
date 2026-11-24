"""Keep each coding agent's conversation state in host RAM while others take turns.

Loom holds one conversation in its single row. Another agent's turn replaces it, and
reading 100-200K tokens again takes minutes. After a turn ends, Parking pins Loom's
state as a checkpoint, and moves it to host RAM when another chat takes the row (a
restore takes seconds).
A conversation is known by its messages: a request that extends the messages of a
parked turn continues that turn's session.
"""

import contextlib
import hashlib
import http.client
import itertools
import json
import time
import uuid
from dataclasses import dataclass

GIB = 1024**3
FINISHED = ("stop", "tool_calls")  # a turn cut by max_tokens cannot be pinned
TIMEOUT = 120


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _canonical(message: dict) -> dict:
    """The message with tool call arguments in one JSON spelling, so a client that
    re-serializes them still matches."""
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.dumps(json.loads(function["arguments"]), sort_keys=True)
        except (KeyError, TypeError, ValueError):
            calls.append(call)
        else:
            calls.append({**call, "function": {**function, "arguments": arguments}})
    return {**message, "tool_calls": calls} if calls else message


def _digests(request: dict) -> list[str]:
    # Tools and the reasoning level change how Loom renders the prompt, so they
    # are part of the conversation's identity.
    header = {k: request.get(k) for k in ("tools", "reasoning_effort")}
    return [_digest(header)] + [
        _digest(_canonical(m)) for m in request.get("messages", [])
    ]


@dataclass
class Entry:
    session: str
    checkpoint: str
    digests: list[str]
    host_bytes: int  # 0 while the pinned state is still in VRAM
    used: float


@dataclass
class Plan:
    """How one request reaches Loom. An empty session means parking is off."""

    session: str
    how: str  # new, live (still in Loom's row), restored or replayed
    digests: list[str]
    checkpoint: str | None = None  # sent only when the state is not in the row

    def headers(self) -> dict[str, str]:
        if not self.session:
            return {}
        headers = {"X-Loom-Session": self.session}
        if self.checkpoint:
            headers["X-Loom-Checkpoint"] = self.checkpoint
        return headers


def _extends(digests: list[str], prefix: list[str]) -> bool:
    return len(prefix) < len(digests) and digests[: len(prefix)] == prefix


class Parking:
    def __init__(
        self,
        budget: int = 40 * GIB,
        limit: int = 30,
        threshold: int = 4096,
        enabled: bool = True,
    ) -> None:
        self.budget = budget  # host bytes for all parked states
        self.limit = limit  # Loom's checkpoint_capacity, less a little room
        self.threshold = threshold  # below this many tokens a replay is fast
        self.address: tuple[str, int] = ("", 0)  # Loom's, set by make_server
        self.model = ""
        self.entries: dict[str, Entry] = {}  # session -> its latest parked turn
        # The session in Loom's row and the request that put it there: a request
        # extending it needs no restore, parked or not (small chats are not parked).
        self.live: str | None = None
        self.live_digests: list[str] = []
        self.disabled = not enabled
        self._names = itertools.count(1)

    def plan(self, request: dict) -> Plan:
        """Pick the session for a request, and whether Loom must restore a state."""
        if self.disabled:
            return Plan("", "", [])
        self.model = request.get("model", "")
        digests = _digests(request)
        live, live_digests = self.live, self.live_digests
        self.live, self.live_digests = None, []  # this request replaces Loom's row
        parked = [e for e in self.entries.values() if _extends(digests, e.digests)]
        best = max(parked, key=lambda e: (len(e.digests), e.used), default=None)
        in_row = (
            live
            and _extends(digests, live_digests)
            and (not best or len(best.digests) <= len(live_digests))
        )
        # Pinned states stay in VRAM until another chat needs the row: an agent that
        # goes on alone never waits for a copy to host RAM.
        keep = live if in_row else best.session if best else None
        for entry in [e for e in self.entries.values() if not e.host_bytes]:
            if entry.session != keep:
                self._suspend(entry)
        if self.disabled:
            return Plan("", "", [])
        if best and best.session not in self.entries:  # evicted to make room
            best = None
        if live and in_row:
            return Plan(live, "live", digests)
        if not best:
            return Plan(f"lct-{uuid.uuid4().hex[:16]}", "new", digests)
        return Plan(best.session, "restored", digests, best.checkpoint)

    def forget_live(self) -> None:
        """Some other request ran, and it may have replaced Loom's row."""
        self.live = None

    def drop(self, plan: Plan) -> None:
        """Loom refused the checkpoint: forget it; the request goes without it."""
        entry = self.entries.get(plan.session)
        if entry and entry.checkpoint == plan.checkpoint:
            del self.entries[plan.session]
            self._release(entry.checkpoint)
        plan.checkpoint = None
        plan.how = "replayed"

    def finish(self, plan: Plan, finish_reason: str | None, usage: dict | None) -> None:
        """Park the state after a turn. Never raises: the client has its answer."""
        if not plan.session:
            return
        if finish_reason:  # the turn completed, so its session holds Loom's row
            self.live, self.live_digests = plan.session, plan.digests
        try:
            if finish_reason in FINISHED:
                self._park(plan)
        except (OSError, http.client.HTTPException) as error:
            self._disable(f"Loom is not reachable ({error})")
        self._log(plan, usage or {})

    def _park(self, plan: Plan) -> None:
        name = f"{plan.session}-{next(self._names)}"
        path = f"/v1/checkpoints/{name}"
        status, pinned = self._call("POST", path, plan.session)
        if status == 503 and self.entries:  # no checkpoint slot is free
            self._evict_oldest()
            status, pinned = self._call("POST", path, plan.session)
        if status == 405:
            self._disable("Loom has no checkpoint API")
        if status != 201 or self.disabled:
            return
        if pinned.get("position", 0) < self.threshold:
            self._release(name)
            return
        old = self.entries.get(plan.session)
        entry = Entry(plan.session, name, plan.digests, 0, time.monotonic())
        self.entries[plan.session] = entry
        if old:
            self._release(old.checkpoint)
        self._trim()

    def _suspend(self, entry: Entry) -> None:
        """Move a pinned state to host RAM, because another chat takes Loom's row."""
        try:
            status, parked = self._call(
                "POST", f"/v1/checkpoints/{entry.checkpoint}/suspend", entry.session
            )
        except (OSError, http.client.HTTPException) as error:
            self._disable(f"Loom is not reachable ({error})")
            return
        if status == 200:
            entry.host_bytes = parked.get("host_snapshot_bytes") or 1
            self._trim()
            return
        del self.entries[entry.session]
        self._release(entry.checkpoint)
        if status == 400:
            self._disable("Loom's memory is fixed, so states cannot leave VRAM")

    def _trim(self) -> None:
        while self.entries and (
            len(self.entries) > self.limit
            or sum(e.host_bytes for e in self.entries.values()) > self.budget
        ):
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        entry = min(self.entries.values(), key=lambda e: e.used)
        del self.entries[entry.session]
        self._release(entry.checkpoint)

    def _release(self, name: str) -> None:
        # If this fails, Loom frees it at restart; the table no longer lists it.
        with contextlib.suppress(OSError, http.client.HTTPException):
            self._call("DELETE", f"/v1/checkpoints/{name}")

    def _disable(self, reason: str) -> None:
        print(f"lct: chat parking is off for this server: {reason}", flush=True)
        self.disabled = True
        entries, self.entries = list(self.entries.values()), {}
        for entry in entries:
            self._release(entry.checkpoint)

    def _call(
        self, method: str, path: str, session: str | None = None
    ) -> tuple[int, dict]:
        headers = {"X-Loom-Model": self.model}
        if session:
            headers["X-Loom-Session"] = session
        connection = http.client.HTTPConnection(*self.address, timeout=TIMEOUT)
        try:
            connection.request(method, path, headers=headers)
            response = connection.getresponse()
            data = response.read()
        finally:
            connection.close()
        try:
            return response.status, json.loads(data)
        except ValueError:
            return response.status, {}

    def _log(self, plan: Plan, usage: dict) -> None:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        tokens = ""
        if cached is not None and "prompt_tokens" in usage:
            tokens = f", {cached}/{usage['prompt_tokens']} tokens cached"
        size = sum(e.host_bytes for e in self.entries.values()) / 1e9
        print(
            f"lct: chat {plan.session[4:12]} {plan.how}{tokens}; "
            f"{len(self.entries)} chats parked, {size:.1f} GB",
            flush=True,
        )
