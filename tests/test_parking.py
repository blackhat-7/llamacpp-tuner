"""Tests for parking conversation state in host RAM between agents' turns."""

import contextlib
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llamacpp_tuner.messages import make_server
from llamacpp_tuner.parking import Parking, _digests

REPLY = {"role": "assistant", "content": "Hi"}


class CountingParking(Parking):
    """Tells the test when a turn is parked: the client has its answer before that."""

    def __init__(self, **settings) -> None:
        super().__init__(**settings)
        self.turns = threading.Semaphore(0)

    def finish(self, plan, finish_reason, usage) -> None:
        super().finish(plan, finish_reason, usage)
        self.turns.release()


@dataclass
class LoomState:
    reason: str = "stop"
    position: int = 10_000
    host_bytes: int = 1000
    suspend_status: int = 200
    pin_drops: int = 0  # pins closed without an answer, as Loom did on 2026-11-25
    dropped_pins_stay: bool = False  # Loom pinned before it closed the connection
    chats: list[dict] = field(default_factory=list)  # headers, messages, pins held
    calls: list[tuple[str, str]] = field(default_factory=list)  # checkpoint API
    checkpoints: dict[str, list[dict]] = field(default_factory=dict)  # history
    last: list[dict] = field(default_factory=list)
    base: str = ""  # the translator's URL
    turns: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(0))


class FakeLoom(BaseHTTPRequestHandler):
    """Loom's chat and checkpoint API, as verified on the real server."""

    state = LoomState()

    def log_message(self, format, *args):
        pass

    def _json(self, data, status=200):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status, message):
        self._json({"error": {"message": message}}, status)

    def do_GET(self):
        self._json({"data": [{"id": "m", "context_length": 196608}]})

    def do_DELETE(self):
        name = self.path.rsplit("/", 1)[1]
        self.state.calls.append(("delete", name))
        self.state.checkpoints.pop(name, None)
        self._json({"checkpoint": name})

    def do_POST(self):
        state = self.state
        body = json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}"
        )
        if self.path == "/v1/chat/completions":
            self._chat(body)
            return
        name = self.path.split("/")[3]
        if self.path.endswith("/suspend"):
            state.calls.append(("suspend", name))
            if state.suspend_status != 200:
                self._error(state.suspend_status, "memory is fixed")
                return
            self._json({"checkpoint": name, "host_snapshot_bytes": state.host_bytes})
        else:
            state.calls.append(("pin", name))
            if name in state.checkpoints:
                self._error(409, "checkpoint name is already pinned")
                return
            if state.pin_drops:
                state.pin_drops -= 1
                if state.dropped_pins_stay:
                    state.checkpoints[name] = state.last
                self.close_connection = True
                return
            if state.reason == "length":
                self._error(409, "pin requires an idle session with reusable history")
                return
            state.checkpoints[name] = state.last
            self._json({"checkpoint": name, "position": state.position}, 201)

    def _chat(self, body):
        state = self.state
        headers = {k: v for k, v in self.headers.items() if k.startswith("X-Loom")}
        state.chats.append(
            {
                "headers": headers,
                "messages": body["messages"],
                "pinned": set(state.checkpoints),
            }
        )
        name = headers.get("X-Loom-Checkpoint")
        if name:
            history = state.checkpoints.get(name)
            if history is None:
                return self._error(404, "unknown checkpoint")
            if body["messages"][: len(history)] != history:
                return self._error(
                    400, "request history does not extend the named checkpoint"
                )
        state.last = [*body["messages"], REPLY]
        usage = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 90}}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {}, "finish_reason": state.reason}]},
                {"choices": [], "usage": usage},
            ):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            choice = {"finish_reason": state.reason, "message": REPLY}
            self._json({"choices": [choice], "usage": usage})


def handler_for(base: type[FakeLoom], state: LoomState) -> type[FakeLoom]:
    return type("Handler", (base,), {"state": state})


@contextlib.contextmanager
def running(
    parking: CountingParking, loom: type[FakeLoom] = FakeLoom
) -> Iterator[tuple[str, LoomState]]:
    state = LoomState()
    fake = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(loom, state))
    server = make_server("127.0.0.1", 0, f"127.0.0.1:{fake.server_address[1]}", parking)
    state.base = f"http://127.0.0.1:{server.server_address[1]}"
    state.turns = parking.turns
    for s in (fake, server):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        yield state.base, state
    finally:
        server.shutdown()
        fake.shutdown()


def post(url, data) -> bytes:
    request = urllib.request.Request(
        url, json.dumps(data).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read()


def settle(loom: LoomState) -> None:
    assert loom.turns.acquire(timeout=10)


def chat(loom, *texts: str, **extra) -> bytes:
    """A chat with one user message per text, and Loom's reply between them."""
    messages: list[dict] = []
    for text in texts:
        if messages:
            messages.append(REPLY)
        messages.append({"role": "user", "content": text})
    reply = post(
        f"{loom.base}/v1/chat/completions",
        {"model": "local", "messages": messages, **extra},
    )
    settle(loom)
    return reply


def sent(state, index=-1) -> dict:
    return state.chats[index]["headers"]


def test_a_chat_is_new_then_live_then_restored():
    parking = CountingParking()
    with running(parking) as (base, loom):
        chat(loom, "a")
        chat(loom, "a", "more")
        chat(loom, "b")
        chat(loom, "a", "more", "again")

    first, second, other, restored = (c["headers"] for c in loom.chats)
    pins = [name for call, name in loom.calls if call == "pin"]
    assert "X-Loom-Checkpoint" not in first
    assert second["X-Loom-Session"] == first["X-Loom-Session"]
    # Still in Loom's row; its own pin goes along, so Loom refuses a re-read first.
    assert second["X-Loom-Checkpoint"] == pins[0]
    assert other["X-Loom-Session"] != first["X-Loom-Session"]
    assert restored["X-Loom-Session"] == first["X-Loom-Session"]
    assert restored["X-Loom-Checkpoint"] == pins[1]  # the second turn's state


def test_a_chat_going_on_alone_is_pinned_each_turn_and_never_copied():
    parking = CountingParking()
    with running(parking) as (base, loom):
        chat(loom, "a")
        chat(loom, "a", "more")

    first, second = loom.calls[0][1], loom.calls[1][1]
    assert loom.calls == [("pin", first), ("pin", second), ("delete", first)]
    assert [e.checkpoint for e in parking.entries.values()] == [second]


def test_a_chat_moves_to_host_ram_when_another_takes_the_row():
    parking = CountingParking()
    with running(parking) as (base, loom):
        chat(loom, "a")
        chat(loom, "b")

    a, b = loom.calls[0][1], loom.calls[2][1]
    assert loom.calls == [("pin", a), ("suspend", a), ("pin", b)]
    assert [e.host_bytes for e in parking.entries.values()] == [1000, 0]


def test_a_turn_cut_by_length_is_not_pinned_and_the_old_state_stays():
    parking = CountingParking()
    with running(parking) as (base, loom):
        chat(loom, "a")
        loom.reason = "length"
        chat(loom, "a", "cut")
        kept = list(loom.calls)
        loom.reason = "stop"
        chat(loom, "a", "again")

    pinned = loom.calls[0][1]
    assert kept == [("pin", pinned)]  # the cut turn is not pinned at all
    # The edited chat goes back to the first turn's state, still pinned in VRAM.
    assert sent(loom)["X-Loom-Checkpoint"] == pinned
    assert ("suspend", pinned) not in loom.calls


def test_a_checkpoint_loom_refuses_is_dropped_and_the_request_retried():
    parking = CountingParking()
    with running(parking) as (base, loom):
        chat(loom, "a")
        chat(loom, "other")
        # The client changed the reply, so the history does not extend the state.
        reply = json.loads(
            post(
                f"{base}/v1/chat/completions",
                {
                    "model": "local",
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "Changed"},
                        {"role": "user", "content": "more"},
                    ],
                },
            )
        )
        settle(loom)

    refused, retried = loom.chats[-2:]
    assert "X-Loom-Checkpoint" in refused["headers"]
    assert "X-Loom-Checkpoint" not in retried["headers"]
    assert retried["headers"]["X-Loom-Session"] == refused["headers"]["X-Loom-Session"]
    assert reply["choices"][0]["message"] == REPLY
    assert ("delete", refused["headers"]["X-Loom-Checkpoint"]) in loom.calls


def test_a_live_chat_loom_must_read_again_loses_its_pin_before_the_read():
    parking = CountingParking()
    with running(parking) as (base, loom):
        chat(loom, "a")
        # Same messages for lct, but the reply differs from what Loom holds, so Loom
        # would read the whole chat again. A pin held through that read is a second
        # copy of the chat in VRAM (2026-11-25: Loom failed at ~99K tokens).
        changed = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "Changed"},
            {"role": "user", "content": "more"},
        ]
        post(f"{base}/v1/chat/completions", {"model": "local", "messages": changed})
        settle(loom)

    pinned = loom.calls[0][1]
    first, refused, retried = loom.chats
    session = first["headers"]["X-Loom-Session"]
    assert refused["headers"] == {
        "X-Loom-Session": session,
        "X-Loom-Checkpoint": pinned,
    }
    assert retried["headers"] == {"X-Loom-Session": session}
    assert pinned not in retried["pinned"]  # released before Loom reads it again


def test_a_pin_loom_drops_goes_again_and_parking_stays_on(monkeypatch):
    monkeypatch.setattr("llamacpp_tuner.parking.RETRY_PAUSE", 0)
    parking = CountingParking()
    with running(parking) as (base, loom):
        loom.pin_drops = 1
        chat(loom, "a")

    name = loom.calls[0][1]
    assert loom.calls == [("pin", name), ("pin", name)]
    assert [e.checkpoint for e in parking.entries.values()] == [name]
    assert not parking.disabled


def test_a_pin_loom_made_but_dropped_is_released_and_parking_stays_on(monkeypatch):
    monkeypatch.setattr("llamacpp_tuner.parking.RETRY_PAUSE", 0)
    parking = CountingParking()
    with running(parking) as (base, loom):
        loom.pin_drops, loom.dropped_pins_stay = 1, True
        chat(loom, "a")
        kept = list(loom.calls)
        chat(loom, "b")

    # The retry finds the name taken, so the pin it cannot account for goes.
    name = kept[0][1]
    assert kept == [("pin", name), ("pin", name), ("delete", name)]
    assert name not in loom.checkpoints
    assert not parking.disabled
    assert len(parking.entries) == 1  # the next chat is parked as usual


def test_parking_turns_off_when_loom_drops_a_call_twice(monkeypatch):
    monkeypatch.setattr("llamacpp_tuner.parking.RETRY_PAUSE", 0)
    parking = CountingParking()
    with running(parking) as (base, loom):
        loom.pin_drops = 2
        chat(loom, "a")

    assert parking.disabled
    assert parking.entries == {}


def test_small_chats_are_not_kept_but_continue_in_loom_s_row():
    parking = CountingParking(threshold=4096)
    with running(parking) as (base, loom):
        loom.position = 100
        chat(loom, "a")
        chat(loom, "a", "more")
        chat(loom, "b")
        chat(loom, "a", "more", "again")

    pinned = loom.calls[0][1]
    assert loom.calls[:2] == [("pin", pinned), ("delete", pinned)]
    assert parking.entries == {}
    first, second, _, after = (c["headers"] for c in loom.chats)
    assert second == {"X-Loom-Session": first["X-Loom-Session"]}  # still in the row
    # Once another chat took the row, nothing is kept: a new session replays it.
    assert after["X-Loom-Session"] != first["X-Loom-Session"]
    assert "X-Loom-Checkpoint" not in after


def test_the_oldest_chat_goes_when_there_are_too_many():
    parking = CountingParking(limit=2)
    with running(parking) as (base, loom):
        for text in "abc":
            chat(loom, text)

    oldest = loom.calls[0][1]
    assert ("delete", oldest) in loom.calls
    assert len(parking.entries) == 2


def test_the_oldest_chat_goes_when_the_host_bytes_pass_the_budget():
    parking = CountingParking(budget=1500)
    with running(parking) as (base, loom):
        loom.host_bytes = 1000
        chat(loom, "a")
        chat(loom, "b")
        chat(loom, "c")  # b goes to host RAM too: 2000 bytes, so a goes

    assert ("delete", loom.calls[0][1]) in loom.calls
    assert len(parking.entries) == 2


def test_parking_turns_off_when_the_memory_is_fixed():
    parking = CountingParking()
    with running(parking) as (base, loom):
        loom.suspend_status = 400
        chat(loom, "a")
        chat(loom, "b")
        chat(loom, "b", "more")

    pinned = loom.calls[0][1]
    assert loom.calls == [("pin", pinned), ("suspend", pinned), ("delete", pinned)]
    assert parking.disabled
    assert sent(loom, 1) == sent(loom) == {}  # Loom is used as it was before parking


def test_the_chat_passthrough_returns_loom_s_bytes_unchanged():
    parking = CountingParking()
    with running(parking) as (base, loom):
        plain = json.loads(chat(loom, "a"))
        streamed = chat(loom, "b", stream=True)

    assert plain["choices"] == [{"finish_reason": "stop", "message": REPLY}]
    assert streamed.endswith(b"data: [DONE]\n\n")
    assert b'"finish_reason": "stop"' in streamed
    assert len(parking.entries) == 2  # both ends were read for parking


def test_messages_requests_are_parked_too():
    parking = CountingParking()
    with running(parking) as (base, loom):
        message = {"model": "m", "max_tokens": 9, "stream": True}
        user = {"role": "user", "content": "hi"}
        for messages in (
            [user],
            [{**user, "content": "x"}],
            [user, {"role": "assistant", "content": "Hi"}, user],
        ):
            post(f"{base}/v1/messages", {**message, "messages": messages})
            settle(loom)

    assert sent(loom, 2)["X-Loom-Checkpoint"] == loom.calls[0][1]
    assert len(parking.entries) == 2


def test_tool_call_arguments_match_however_the_client_spells_them():
    def request(arguments):
        call = {"id": "1", "function": {"name": "f", "arguments": arguments}}
        return {"messages": [{"role": "assistant", "tool_calls": [call]}]}

    assert _digests(request('{"a": 1, "b": 2}')) == _digests(request('{"b":2,"a":1}'))
    assert _digests(request("{broken")) != _digests(request('{"a": 1}'))


class HangingLoom(FakeLoom):
    """Streams until its client leaves, and says whether the first was gone by the second."""

    gone: list[bool] = []
    sockets: list[socket.socket] = []

    def _chat(self, body):
        if self.sockets:  # the second request: has the first one's connection closed?
            first = self.sockets[0]
            try:
                self.gone.append(
                    first.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
                )
            except BlockingIOError:
                self.gone.append(False)
            except OSError:  # its handler ended and the server closed the socket
                self.gone.append(True)
            return super()._chat(body)
        self.sockets.append(self.connection)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for _ in range(200):
            time.sleep(0.02)
            try:
                if (
                    self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                    == b""
                ):
                    return
            except BlockingIOError:
                pass
            self.wfile.write(b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n')
            self.wfile.flush()


def test_the_next_request_waits_until_loom_sees_the_stopped_one_close():
    with running(CountingParking(), HangingLoom) as (base, loom):
        stopped = socket.create_connection(("127.0.0.1", int(base.rsplit(":", 1)[1])))
        body = json.dumps(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "a"}],
                "stream": True,
            }
        ).encode()
        stopped.sendall(
            b"POST /v1/chat/completions HTTP/1.0\r\nContent-Length: %d\r\n\r\n%s"
            % (len(body), body)
        )
        stopped.recv(1)  # the stream has started
        waiting = threading.Thread(target=chat, args=(loom, "b"))
        waiting.start()
        time.sleep(0.2)  # it holds the lock, so the second request queues
        stopped.close()
        waiting.join()

    assert HangingLoom.gone == [True]
