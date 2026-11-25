"""Anthropic's Messages API in front of an OpenAI chat server that lacks it (Loom).

Claude Code (claude-local) speaks only /v1/messages, and reads the context window
from llama.cpp's `meta.n_ctx` in /v1/models. This server translates both and
forwards every other request (the OpenAI API that Pi and opencode use) as is,
except that a chat request without a temperature gets TEMPERATURE.
"""

import contextlib
import http.client
import json
import socket
import threading
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llamacpp_tuner.parking import Parking, Plan

STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
# A request without a temperature means 1 in both APIs, and 1 is what Inkling's GGUF
# recommends. Loom decodes greedily instead, and greedy decoding makes Inkling's
# thinking repeat the same paragraphs until max_tokens.
TEMPERATURE = 1.0
# Past this many prompt tokens images become a note: with elastic memory Loom waits
# forever for room for the image encoder beside a long chat (hung at ~190K, 2026-11-24).
IMAGE_TOKENS = 150_000
DEEP_IMAGE = "[image omitted: the chat is too long for images on this server]"


def _without_images(request: dict) -> dict:
    messages = [
        {
            **m,
            "content": [
                {"type": "text", "text": DEEP_IMAGE}
                if part.get("type") == "image_url"
                else part
                for part in m["content"]
            ],
        }
        if isinstance(m.get("content"), list)
        else m
        for m in request["messages"]
    ]
    return {**request, "messages": messages}


def _has_images(request: dict) -> bool:
    return any(
        isinstance(m.get("content"), list)
        and any(part.get("type") == "image_url" for part in m["content"])
        for m in request["messages"]
    )


def _text(content: str | list[dict], images: bool = False) -> str:
    """The text of the blocks; images become a note unless the model sees them."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block["text"])
        elif block.get("type") == "image" and not images:
            parts.append("[image omitted: this model reads text only]")
    return "\n".join(parts)


def _image(block: dict) -> dict:
    source = block["source"]
    if source.get("type") == "url":
        url = source["url"]
    else:
        url = f"data:{source['media_type']};base64,{source['data']}"
    return {"type": "image_url", "image_url": {"url": url}}


def _parts(blocks: list[dict]) -> list[dict]:
    return [
        _image(b) if b["type"] == "image" else {"type": "text", "text": b["text"]}
        for b in blocks
        if b.get("type") in ("text", "image")
    ]


def _assistant(blocks: list[dict]) -> dict:
    message: dict = {"role": "assistant", "content": _text(blocks)}
    thinking = "".join(b["thinking"] for b in blocks if b.get("type") == "thinking")
    if thinking:
        message["reasoning_content"] = thinking
    # Loom compares the history with the tokens the model wrote, which keep "—".
    # Escaped as "\u2014" it does not match, and Loom reads the whole chat again.
    calls = [
        {
            "id": b["id"],
            "type": "function",
            "function": {
                "name": b["name"],
                "arguments": json.dumps(b["input"], ensure_ascii=False),
            },
        }
        for b in blocks
        if b.get("type") == "tool_use"
    ]
    if calls:
        message["tool_calls"] = calls
    return message


def _user(blocks: list[dict], images: bool, names: dict[str, str]) -> list[dict]:
    # Tool results answer the previous turn's calls, so they come first. Each names its
    # tool: Loom's renderer for image requests cannot look the call up by id.
    results = [b for b in blocks if b.get("type") == "tool_result"]
    messages = [
        {
            "role": "tool",
            "tool_call_id": b["tool_use_id"],
            **({"name": names[b["tool_use_id"]]} if b["tool_use_id"] in names else {}),
            "content": _text(b["content"], images),
        }
        for b in results
    ]
    rest = [b for b in blocks if b.get("type") != "tool_result"]
    if images:
        # OpenAI tool messages carry only text, so a tool's images (a Read of a
        # screenshot) go in the user message that follows.
        shown = [
            block
            for b in results
            if isinstance(b["content"], list)
            for block in b["content"]
            if block.get("type") == "image"
        ]
        rest = shown + rest
        if any(b.get("type") == "image" for b in rest):
            messages.append({"role": "user", "content": _parts(rest)})
            return messages
    if rest:
        messages.append({"role": "user", "content": _text(rest)})
    return messages


def to_openai(body: dict, images: bool = False) -> dict:
    """The OpenAI chat request for an Anthropic Messages request.

    With images, image blocks become image parts; otherwise a note replaces them.
    """
    messages = (
        [{"role": "system", "content": _text(body["system"])}]
        if body.get("system")
        else []
    )
    names: dict[str, str] = {}  # tool call id -> tool name
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})
        elif message["role"] == "assistant":
            messages.append(_assistant(content))
            names.update(
                (b["id"], b["name"]) for b in content if b.get("type") == "tool_use"
            )
        else:
            messages.extend(_user(content, images, names))
    request: dict = {
        "model": body["model"],
        "messages": messages,
        "temperature": TEMPERATURE,
    }
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
        self.reason: str | None = None  # OpenAI's finish_reason
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
                self.reason = choice["finish_reason"]
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


def _chunk(line: bytes) -> dict | None:
    line = line.strip()
    if line.startswith(b"data:") and line != b"data: [DONE]":
        return json.loads(line[5:])
    return None


def _chunks(response: http.client.HTTPResponse) -> Iterator[dict]:
    for line in response:
        if chunk := _chunk(line):
            yield chunk


def _error(status: int, payload: bytes) -> dict:
    try:
        message = json.loads(payload)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = payload.decode(errors="replace")
    kind = "invalid_request_error" if status < 500 else "api_error"
    return {"type": "error", "error": {"type": kind, "message": message}}


def make_server(
    host: str, port: int, upstream: str, parking: Parking | None = None
) -> ThreadingHTTPServer:
    """A server on host:port for the OpenAI chat server at upstream (host:port)."""
    upstream_host, upstream_port = upstream.rsplit(":", 1)
    parking = parking or Parking(enabled=False)
    parking.address = (upstream_host, int(upstream_port))
    # Loom serves one sequence at a time and does not queue: a second generation that
    # overlaps the first makes it reserve 4 GB more, fail and drop the model until
    # restarted. Claude Code sends side requests (titles) alongside its main one.
    generating = threading.Lock()
    served: dict = {}

    def loom_model() -> dict:
        """The model Loom serves, read once: {"id": its id, "images": takes images}."""
        if not served:
            connection = http.client.HTTPConnection(
                upstream_host, int(upstream_port), timeout=60
            )
            try:
                connection.request("GET", "/v1/models")
                models = json.loads(connection.getresponse().read()).get("data")
            finally:
                connection.close()
            if not models:
                raise http.client.HTTPException("Loom lists no model")
            # A text-only package rejects images, so they become a note instead.
            images = "image" in models[0].get("input_modalities", [])
            served.update(id=models[0]["id"], images=images)
        return served

    class Handler(BaseHTTPRequestHandler):
        upstream_connection: http.client.HTTPConnection | None = None
        upstream_response: http.client.HTTPResponse | None = None
        answered = False

        def log_message(self, format: str, *args: object) -> None:
            pass  # one line per request would drown the server log

        def finish(self) -> None:
            self._close_upstream()
            super().finish()

        def _close_upstream(self) -> None:
            # Closing tells Loom the client is gone, so it stops generating.
            # The response holds the socket open after the connection is closed.
            if self.upstream_response:
                self.upstream_response.close()
                self.upstream_response = None
            if self.upstream_connection:
                self.upstream_connection.close()
                self.upstream_connection = None

        def _upstream(
            self,
            method: str,
            path: str,
            body: bytes | None = None,
            headers: dict[str, str] | None = None,
        ):
            self._close_upstream()
            self.upstream_connection = http.client.HTTPConnection(
                upstream_host, int(upstream_port), timeout=3600
            )
            if body is not None:
                headers = {"Content-Type": "application/json", **(headers or {})}
            self.upstream_connection.request(
                method, path, body=body, headers=headers or {}
            )
            self.upstream_response = self.upstream_connection.getresponse()
            return self.upstream_response

        def _generate(self, request: dict) -> tuple[http.client.HTTPResponse, Plan]:
            """Send a chat request to Loom with the headers that find its parked state."""
            if _has_images(request):
                count = {**request, "stream": False}
                count.pop("stream_options", None)
                path = "/v1/chat/completions/input_tokens"
                counted = self._upstream("POST", path, json.dumps(count).encode())
                if json.loads(counted.read()).get("input_tokens", 0) > IMAGE_TOKENS:
                    request = _without_images(request)
            plan = parking.plan(request)
            body = json.dumps(request).encode()
            path = "/v1/chat/completions"
            response = self._upstream("POST", path, body, plan.headers())
            if response.status in (400, 404) and plan.checkpoint:
                # The history does not extend the checkpoint, or Loom lost it.
                response.read()
                parking.drop(plan)
                response = self._upstream("POST", path, body, plan.headers())
            return response, plan

        def _end_stream(self) -> None:
            # The client sees the end of the stream only when the connection closes,
            # which waits for the handler. Parking comes after, so half-close now.
            self.wfile.flush()
            with contextlib.suppress(OSError):
                self.connection.shutdown(socket.SHUT_WR)

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
            model = loom_model()
            request = {
                **to_openai(body, model["images"]),
                "model": model["id"],
                "stream": False,
            }
            request.pop("stream_options", None)
            path = "/v1/chat/completions/input_tokens"
            response = self._upstream("POST", path, json.dumps(request).encode())
            self._send(response.status, json.loads(response.read()))

        def _message(self, body: dict) -> None:
            model = loom_model()
            request = {**to_openai(body, model["images"]), "model": model["id"]}
            response, plan = self._generate(request)
            if response.status != 200:
                self._send(response.status, _error(response.status, response.read()))
                return
            if not request.get("stream"):
                completion = json.loads(response.read())
                self._send(200, to_anthropic(completion))
                parking.finish(
                    plan,
                    completion["choices"][0].get("finish_reason"),
                    completion.get("usage"),
                )
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
            self._end_stream()
            parking.finish(plan, stream.reason, stream.usage)

        def _chat(self, request: dict) -> None:
            # Loom answers only to its own model id; clients made for llama-server name
            # any model ("local"), so every request goes to the one model Loom serves.
            request = {**request, "model": loom_model()["id"]}
            request.setdefault("temperature", TEMPERATURE)
            response, plan = self._generate(request)
            self._start(
                response.status, response.getheader("Content-Type", "application/json")
            )
            self.end_headers()
            reason, usage = None, {}
            pending = b""  # the unfinished last line of a stream, or a JSON reply
            for piece in iter(lambda: response.read1(65536), b""):
                self.wfile.write(piece)
                self.wfile.flush()
                pending += piece
                if request.get("stream"):
                    *lines, pending = pending.split(b"\n")
                    for line in lines:
                        try:
                            chunk = _chunk(line)
                        except ValueError:
                            continue  # the client reads it; parking only wants the end
                        for choice in (chunk or {}).get("choices", []):
                            reason = choice.get("finish_reason") or reason
                        usage = (chunk or {}).get("usage") or usage
            if not request.get("stream") and response.status == 200:
                try:
                    reply = json.loads(pending)
                    reason = reply["choices"][0].get("finish_reason")
                    usage = reply.get("usage")
                except (ValueError, KeyError, IndexError):
                    pass
            self._end_stream()
            if response.status == 200:
                parking.finish(plan, reason, usage)

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
                    try:
                        self._guarded(lambda: self._message(json.loads(raw)))
                    finally:  # Loom must see this one end before the next starts
                        self._close_upstream()
            elif path == "/v1/chat/completions":
                with generating:
                    try:
                        self._guarded(lambda: self._chat(json.loads(raw)))
                    finally:
                        self._close_upstream()
            else:
                parking.forget_live()
                self._guarded(lambda: self._forward("POST", raw))

    return ThreadingHTTPServer((host, port), Handler)
