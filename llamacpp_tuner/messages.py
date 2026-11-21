"""Anthropic's Messages API in front of an OpenAI chat server that lacks it (Loom).

Claude Code (claude-local) speaks only /v1/messages, and reads the context window
from llama.cpp's `meta.n_ctx` in /v1/models. This server translates both and
forwards every other request (the OpenAI API that Pi and opencode use) as is.
"""

import http.client
import json
import threading
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def _text(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block["text"])
        elif block.get("type") == "image":
            parts.append("[image omitted: this model reads text only]")
    return "\n".join(parts)


def _assistant(blocks: list[dict]) -> dict:
    message: dict = {"role": "assistant", "content": _text(blocks)}
    thinking = "".join(b["thinking"] for b in blocks if b.get("type") == "thinking")
    if thinking:
        message["reasoning_content"] = thinking
    calls = [
        {
            "id": b["id"],
            "type": "function",
            "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
        }
        for b in blocks
        if b.get("type") == "tool_use"
    ]
    if calls:
        message["tool_calls"] = calls
    return message


def _user(blocks: list[dict]) -> list[dict]:
    # Tool results answer the previous turn's calls, so they come first.
    messages = [
        {
            "role": "tool",
            "tool_call_id": b["tool_use_id"],
            "content": _text(b["content"]),
        }
        for b in blocks
        if b.get("type") == "tool_result"
    ]
    rest = [b for b in blocks if b.get("type") != "tool_result"]
    if rest:
        messages.append({"role": "user", "content": _text(rest)})
    return messages


def to_openai(body: dict) -> dict:
    """The OpenAI chat request for an Anthropic Messages request."""
    messages = (
        [{"role": "system", "content": _text(body["system"])}]
        if body.get("system")
        else []
    )
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})
        elif message["role"] == "assistant":
            messages.append(_assistant(content))
        else:
            messages.extend(_user(content))
    request: dict = {"model": body["model"], "messages": messages}
    for name in ("max_tokens", "temperature", "top_p", "top_k"):
        if name in body:
            request[name] = body[name]
    if body.get("stop_sequences"):
        request["stop"] = body["stop_sequences"]
    # Claude Code sets output_config.effort (low to max); Loom's reasoning levels
    # use the same names, and "none" for no thinking.
    if (body.get("thinking") or {}).get("type") == "disabled":
        request["reasoning_effort"] = "none"
    elif effort := (body.get("output_config") or {}).get("effort"):
        request["reasoning_effort"] = effort
    # Server tools such as web search have no input schema; the shim answers them.
    tools = [t for t in body.get("tools", []) if "input_schema" in t]
    if tools:
        request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        choice = body.get("tool_choice", {})
        if choice.get("type") == "tool":
            request["tool_choice"] = {
                "type": "function",
                "function": {"name": choice["name"]},
            }
        elif choice.get("type") in ("any", "none"):
            request["tool_choice"] = {"any": "required", "none": "none"}[choice["type"]]
    if body.get("stream"):
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}
    return request


def _usage(usage: dict) -> dict:
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return {
        "input_tokens": usage.get("prompt_tokens", 0) - cached,
        "cache_read_input_tokens": cached,
        "output_tokens": usage.get("completion_tokens", 0),
    }


def _arguments(text: str) -> dict:
    try:
        return json.loads(text or "{}")
    except json.JSONDecodeError:  # the model wrote broken JSON; let the agent see it
        return {"invalid_arguments": text}


def to_anthropic(completion: dict) -> dict:
    """The Anthropic Messages response for an OpenAI chat completion."""
    choice = completion["choices"][0]
    message = choice["message"]
    content: list[dict] = []
    if message.get("reasoning_content"):
        content.append(
            {
                "type": "thinking",
                "thinking": message["reasoning_content"],
                "signature": "",
            }
        )
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call["function"]
        content.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": function["name"],
                "input": _arguments(function["arguments"]),
            }
        )
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": completion.get("model", ""),
        "content": content,
        "stop_reason": STOP_REASONS.get(choice.get("finish_reason"), "end_turn"),
        "stop_sequence": None,
        "usage": _usage(completion.get("usage") or {}),
    }


def _event(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()


class Stream:
    """Turns OpenAI stream chunks into Anthropic stream events, one chunk at a time."""

    def __init__(self, model: str) -> None:
        self.index = -1  # the open content block, or the last one
        self.kind = ""  # "thinking", "text" or "tool_use" while a block is open
        self.tools: dict[int, int] = {}  # OpenAI tool call index -> block index
        self.stop = "end_turn"
        self.usage: dict = {}
        message = {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        self.start = _event(
            "message_start", {"type": "message_start", "message": message}
        )

    def _open(self, block: dict) -> list[bytes]:
        events = self._close()
        self.index += 1
        self.kind = block["type"]
        start = {
            "type": "content_block_start",
            "index": self.index,
            "content_block": block,
        }
        return [*events, _event("content_block_start", start)]

    def _close(self) -> list[bytes]:
        if not self.kind:
            return []
        self.kind = ""
        return [
            _event(
                "content_block_stop",
                {"type": "content_block_stop", "index": self.index},
            )
        ]

    def _delta(self, index: int, delta: dict) -> bytes:
        return _event(
            "content_block_delta",
            {"type": "content_block_delta", "index": index, "delta": delta},
        )

    def feed(self, chunk: dict) -> list[bytes]:
        events: list[bytes] = []
        self.usage = chunk.get("usage") or self.usage
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                if self.kind != "thinking":
                    events += self._open(
                        {"type": "thinking", "thinking": "", "signature": ""}
                    )
                thinking = {
                    "type": "thinking_delta",
                    "thinking": delta["reasoning_content"],
                }
                events.append(self._delta(self.index, thinking))
            if delta.get("content"):
                if self.kind != "text":
                    events += self._open({"type": "text", "text": ""})
                text = {"type": "text_delta", "text": delta["content"]}
                events.append(self._delta(self.index, text))
            for call in delta.get("tool_calls") or []:
                function = call.get("function") or {}
                if call.get("index", 0) not in self.tools:
                    block = {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function["name"],
                    }
                    events += self._open({**block, "input": {}})
                    self.tools[call.get("index", 0)] = self.index
                if function.get("arguments"):
                    json_delta = {
                        "type": "input_json_delta",
                        "partial_json": function["arguments"],
                    }
                    events.append(
                        self._delta(self.tools[call.get("index", 0)], json_delta)
                    )
            if choice.get("finish_reason"):
                self.stop = STOP_REASONS.get(choice["finish_reason"], "end_turn")
        return events

    def finish(self) -> list[bytes]:
        delta = {"stop_reason": self.stop, "stop_sequence": None}
        message_delta = {
            "type": "message_delta",
            "delta": delta,
            "usage": _usage(self.usage),
        }
        return [
            *self._close(),
            _event("message_delta", message_delta),
            _event("message_stop", {"type": "message_stop"}),
        ]


def _chunks(response: http.client.HTTPResponse) -> Iterator[dict]:
    for line in response:
        line = line.strip()
        if line.startswith(b"data:") and line != b"data: [DONE]":
            yield json.loads(line[5:])


def _error(status: int, payload: bytes) -> dict:
    try:
        message = json.loads(payload)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = payload.decode(errors="replace")
    kind = "invalid_request_error" if status < 500 else "api_error"
    return {"type": "error", "error": {"type": kind, "message": message}}


def make_server(host: str, port: int, upstream: str) -> ThreadingHTTPServer:
    """A server on host:port for the OpenAI chat server at upstream (host:port)."""
    upstream_host, upstream_port = upstream.rsplit(":", 1)
    # Loom serves one sequence at a time and does not queue: a second generation that
    # overlaps the first makes it reserve 4 GB more, fail and drop the model until
    # restarted. Claude Code sends side requests (titles) alongside its main one.
    generating = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        upstream_connection: http.client.HTTPConnection | None = None
        answered = False

        def log_message(self, format: str, *args: object) -> None:
            pass  # one line per request would drown the server log

        def finish(self) -> None:
            # Closing tells Loom the client is gone, so it stops generating.
            if self.upstream_connection:
                self.upstream_connection.close()
            super().finish()

        def _upstream(self, method: str, path: str, body: bytes | None = None):
            self.upstream_connection = http.client.HTTPConnection(
                upstream_host, int(upstream_port), timeout=3600
            )
            headers = {"Content-Type": "application/json"} if body is not None else {}
            self.upstream_connection.request(method, path, body=body, headers=headers)
            return self.upstream_connection.getresponse()

        def _start(self, status: int, content_type: str) -> None:
            self.answered = True
            self.send_response(status)
            self.send_header("Content-Type", content_type)

        def _send(self, status: int, data: dict) -> None:
            payload = json.dumps(data).encode()
            self._start(status, "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _forward(self, method: str, body: bytes | None) -> None:
            response = self._upstream(method, self.path, body)
            self._start(
                response.status, response.getheader("Content-Type", "application/json")
            )
            self.end_headers()
            for piece in iter(lambda: response.read1(65536), b""):
                self.wfile.write(piece)
                self.wfile.flush()

        def _models(self) -> None:
            catalog = json.loads(self._upstream("GET", "/v1/models").read())
            for model in catalog.get("data", []):
                if "context_length" in model:
                    model.setdefault("meta", {})["n_ctx"] = model["context_length"]
            self._send(200, catalog)

        def _count(self, body: dict) -> None:
            # Loom counts a chat request's tokens and answers in Anthropic's shape.
            request = {**to_openai(body), "stream": False}
            request.pop("stream_options", None)
            path = "/v1/chat/completions/input_tokens"
            response = self._upstream("POST", path, json.dumps(request).encode())
            self._send(response.status, json.loads(response.read()))

        def _message(self, body: dict) -> None:
            request = to_openai(body)
            response = self._upstream(
                "POST", "/v1/chat/completions", json.dumps(request).encode()
            )
            if response.status != 200:
                self._send(response.status, _error(response.status, response.read()))
                return
            if not request.get("stream"):
                self._send(200, to_anthropic(json.loads(response.read())))
                return
            self._start(200, "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            stream = Stream(body["model"])
            self.wfile.write(stream.start)
            for chunk in _chunks(response):
                for event in stream.feed(chunk):
                    self.wfile.write(event)
                self.wfile.flush()
            self.wfile.write(b"".join(stream.finish()))
            self.wfile.flush()

        def _guarded(self, work) -> None:
            try:
                work()
            except (OSError, http.client.HTTPException) as error:
                if not self.answered:  # mid-stream, the client sees the cut instead
                    self._send(
                        503, _error(503, f"The model server failed: {error}".encode())
                    )

        def do_GET(self) -> None:
            if self.path.split("?")[0] == "/v1/models":
                self._guarded(self._models)
            else:
                self._guarded(lambda: self._forward("GET", None))

        def do_POST(self) -> None:
            path = self.path.split("?")[0]
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if path == "/v1/messages/count_tokens":
                self._guarded(lambda: self._count(json.loads(raw)))
            elif path == "/v1/messages":
                with generating:
                    self._guarded(lambda: self._message(json.loads(raw)))
            elif path == "/v1/chat/completions":
                with generating:
                    self._guarded(lambda: self._forward("POST", raw))
            else:
                self._guarded(lambda: self._forward("POST", raw))

    return ThreadingHTTPServer((host, port), Handler)
