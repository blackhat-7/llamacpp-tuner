"""Tests for serving Anthropic's Messages API in front of an OpenAI chat server."""

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llamacpp_tuner.messages import Stream, make_server, to_anthropic, to_openai

TOOL = {
    "name": "read",
    "description": "Read a file",
    "input_schema": {"type": "object"},
}


def test_request_becomes_an_openai_chat_request():
    body = {
        "model": "m",
        "max_tokens": 100,
        "system": [{"type": "text", "text": "Be brief.", "cache_control": {}}],
        "messages": [
            {"role": "user", "content": "Read a.txt"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Use the tool.", "signature": ""},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "read",
                        "input": {"p": "a"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "hi"}],
                    },
                    {"type": "text", "text": "Now summarize."},
                ],
            },
        ],
        "tools": [TOOL, {"type": "web_search_20250305", "name": "web_search"}],
        "tool_choice": {"type": "any"},
        "stop_sequences": ["END"],
        "stream": True,
    }

    request = to_openai(body)

    assert request["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Read a.txt"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Use the tool.",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"p": "a"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "hi"},
        {"role": "user", "content": "Now summarize."},
    ]
    assert [t["function"]["name"] for t in request["tools"]] == ["read"]
    assert request["tool_choice"] == "required"
    assert request["stop"] == ["END"]
    assert request["stream_options"] == {"include_usage": True}


def test_completion_becomes_a_message():
    completion = {
        "model": "m",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "reasoning_content": "Need a.txt.",
                    "content": "Reading.",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "read", "arguments": '{"p":"a"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    }

    message = to_anthropic(completion)

    assert message["content"] == [
        {"type": "thinking", "thinking": "Need a.txt.", "signature": ""},
        {"type": "text", "text": "Reading."},
        {"type": "tool_use", "id": "c1", "name": "read", "input": {"p": "a"}},
    ]
    assert message["stop_reason"] == "tool_use"
    assert message["usage"] == {
        "input_tokens": 10,
        "cache_read_input_tokens": 40,
        "output_tokens": 7,
    }


def events(raw: bytes) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in raw.decode().splitlines()
        if line.startswith("data: ")
    ]


def test_stream_opens_and_closes_one_block_per_kind():
    stream = Stream("m")
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"reasoning_content": "Hm"}}]},
        {"choices": [{"delta": {"reasoning_content": "m."}}]},
        {"choices": [{"delta": {"content": "Hi"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "read", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
    ]
    raw = (
        stream.start
        + b"".join(e for chunk in chunks for e in stream.feed(chunk))
        + b"".join(stream.finish())
    )

    sent = events(raw)
    assert [e["type"] for e in sent] == [
        "message_start",
        "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_stop",
        "message_delta",
        "message_stop",
    ]  # fmt: skip
    assert sent[1]["content_block"]["type"] == "thinking"
    assert sent[8]["content_block"] == {
        "type": "tool_use",
        "id": "c1",
        "name": "read",
        "input": {},
    }
    assert sent[-2]["delta"]["stop_reason"] == "tool_use"
    assert sent[-2]["usage"]["output_tokens"] == 4


class FakeLoom(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _json(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._json({"data": [{"id": "m", "context_length": 196608}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions/input_tokens":
            self._json({"input_tokens": len(body["messages"])})
        elif body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"choices": [{"delta": {"content": "Hi"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self._json(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "Hi"}}
                    ],
                    "echo": body,
                }
            )


def post(url, data):
    request = urllib.request.Request(
        url, json.dumps(data).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read()


def test_server_translates_messages_and_forwards_the_rest():
    loom = ThreadingHTTPServer(("127.0.0.1", 0), FakeLoom)
    server = make_server("127.0.0.1", 0, f"127.0.0.1:{loom.server_address[1]}")
    for s in (loom, server):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    message = {
        "model": "m",
        "max_tokens": 9,
        "messages": [{"role": "user", "content": "hi"}],
    }
    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as response:
            catalog = json.load(response)
        reply = json.loads(post(f"{base}/v1/messages", message))
        streamed = events(post(f"{base}/v1/messages", {**message, "stream": True}))
        counted = json.loads(post(f"{base}/v1/messages/count_tokens", message))
        forwarded = json.loads(
            post(f"{base}/v1/chat/completions", {"model": "m", "messages": []})
        )
        greedy = {"model": "m", "messages": [], "temperature": 0}
        forwarded_greedy = json.loads(post(f"{base}/v1/chat/completions", greedy))
    finally:
        server.shutdown()
        loom.shutdown()

    assert catalog["data"][0]["meta"] == {
        "n_ctx": 196608
    }  # claude-local's context window
    assert reply["content"] == [{"type": "text", "text": "Hi"}]
    assert [e["type"] for e in streamed][-2:] == ["message_delta", "message_stop"]
    assert counted == {"input_tokens": 1}
    # Without a temperature Loom decodes greedily, which makes Inkling loop.
    assert forwarded["echo"] == {"model": "m", "messages": [], "temperature": 1.0}
    assert forwarded_greedy["echo"] == greedy


def test_temperature_defaults_to_one_and_keeps_a_given_value():
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    assert to_openai(base)["temperature"] == 1.0
    assert to_openai({**base, "temperature": 0})["temperature"] == 0


def test_effort_becomes_the_reasoning_level():
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    medium = to_openai({**base, "output_config": {"effort": "medium"}})
    disabled = to_openai({**base, "thinking": {"type": "disabled"}})

    assert medium["reasoning_effort"] == "medium"
    assert disabled["reasoning_effort"] == "none"


class SlowLoom(FakeLoom):
    """Counts generations in flight; Loom drops its model when two overlap."""

    in_flight = 0
    most = 0
    lock = threading.Lock()

    def do_POST(self):
        with SlowLoom.lock:
            SlowLoom.in_flight += 1
            SlowLoom.most = max(SlowLoom.most, SlowLoom.in_flight)
        time.sleep(0.3)
        with SlowLoom.lock:
            SlowLoom.in_flight -= 1
        super().do_POST()


def test_generations_reach_the_model_one_at_a_time():
    loom = ThreadingHTTPServer(("127.0.0.1", 0), SlowLoom)
    server = make_server("127.0.0.1", 0, f"127.0.0.1:{loom.server_address[1]}")
    for s in (loom, server):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/messages"
    message = {
        "model": "m",
        "max_tokens": 9,
        "messages": [{"role": "user", "content": "hi"}],
    }
    try:
        threads = [threading.Thread(target=post, args=(url, message)) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        server.shutdown()
        loom.shutdown()

    assert SlowLoom.most == 1


def test_a_dead_model_server_gives_a_clean_error():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead = probe.getsockname()[1]  # nothing listens here once closed
    server = make_server("127.0.0.1", 0, f"127.0.0.1:{dead}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/messages"
    message = {
        "model": "m",
        "max_tokens": 9,
        "messages": [{"role": "user", "content": "hi"}],
    }
    try:
        with pytest.raises(urllib.error.HTTPError) as failure:
            post(url, message)
    finally:
        server.shutdown()

    assert failure.value.code == 503
    assert json.loads(failure.value.read())["error"]["type"] == "api_error"
