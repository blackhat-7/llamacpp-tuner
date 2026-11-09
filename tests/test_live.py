"""Tests for reading llama-server and Strata log lines into requests."""

from collections import Counter

import pytest

from llamacpp_tuner.live import Live, Monitor, Request

# Lines copied from the servers' own logs in tmp/servers, request by request.
STRATA_TOOL_CALL_REQUEST = [
    "[strata] reading the prompt: 21,996 of 21,997 tokens, 1 s so far",
    "[strata] thinking: 54 of max 32768 tokens, 54.5 tok/s, 2 s",
    "[strata] thinking: 129 of max 32768 tokens, 64.0 tok/s, 3 s",
    "[strata] thinking: 193 of max 32768 tokens, 63.8 tok/s, 4 s",
    "[strata] thinking: 248 of max 32768 tokens, 61.3 tok/s, 5 s",
    "[strata] thinking: 309 of max 32768 tokens, 61.1 tok/s, 6 s",
    "[strata] writing a tool call: bash: 376 of max 32768 tokens, 61.7 tok/s, 7 s",
    "[strata] writing a tool call: bash: 437 of max 32768 tokens, 61.4 tok/s, 8 s",
    "[strata] done: 444 tokens in 8 s (61.7 tok/s) (stop, cancel=False), expert cache 86.5% hit",
]
STRATA_CACHED_PROMPT_REQUEST = [
    "[strata] thinking: 24 of budget 16384 (max 32000) tokens, 96.0 tok/s, 1 s",
    "[strata] writing a tool call: Bash: 177 of max 32000 tokens, 153.6 tok/s, 2 s",
    "[strata] done: 312 tokens in 3 s (155.8 tok/s) (stop, cancel=False), expert cache 93.1% hit (+1.8% of the routed experts over PCIe)",
]
STRATA_BARE_DONE = "[strata] done: 60 tokens in 3 s (72.7 tok/s) (stop, cancel=False), expert cache 91.1% hit"
STRATA_DISCONNECT_DONE = "[strata] done: 157 tokens in 4 s (58.3 tok/s) (disconnect, cancel=True), expert cache 88.3% hit"
STRATA_THINKING = "[strata] thinking: 36 of max 32768 tokens, 69.4 tok/s, 1 s"
STRATA_ENGINE_STOPPED = "[strata] the engine stopped unexpectedly (exit code -6). The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | grep -i -E 'killed process|out of memory'); Windows slows down instead. Close other programs or use a smaller model (Q2_0 / IQ2_XS). The next request starts the engine again. Its log: /home/illusion/Documents/projects/Strata/strata-iq3_s.log"

SIDE_SLOTS = "0.41.478.032 I srv    load_model: initializing, n_slots = 2, n_ctx_slot = 32768, kv_unified = 'false'"
SIDE_SLOT_LINES = [
    "29.41.622.768 I slot get_availabl: id  1 | task -1 | selected slot by LRU, t_last = -1",
    "29.41.625.785 I slot launch_slot_: id  1 | task 0 | processing task, is_child = 0",
    "30.05.578.598 I slot get_availabl: id  0 | task -1 | selected slot by LRU, t_last = -1",
    "30.05.578.649 I slot launch_slot_: id  0 | task 3 | processing task, is_child = 0",
]
SIDE_PROMPT_PROGRESS = "30.05.578.688 I slot print_timing: id  1 | task 0 | prompt processing, n_tokens =   2098, progress = 0.24, t =  23.95 s / 87.59 tokens per second"
SWARM_REQUEST_TIMINGS = [
    "0.25.924.510 I slot launch_slot_: id  0 | task 2 | processing task, is_child = 0",
    "0.28.453.148 I slot print_timing: id  0 | task 2 | prompt eval time =    1164.00 ms /   372 tokens (    3.13 ms per token,   319.59 tokens per second)",
    "0.28.453.150 I slot print_timing: id  0 | task 2 |        eval time =    1364.54 ms /    50 tokens (   27.85 ms per token,    35.91 tokens per second)",
    "0.28.453.150 I slot print_timing: id  0 | task 2 |       total time =    2528.55 ms /   422 tokens",
    "0.28.453.150 I slot print_timing: id  0 | task 2 |    graphs reused =          9",
    "0.28.453.152 I slot print_timing: id  0 | task 2 | draft acceptance = 0.92308 (   36 accepted /    39 generated), mean len =  3.77",
    "0.28.453.191 I slot      release: id  0 | task 2 | stop processing: n_tokens = 421, truncated = 0",
]
NON_MODEL_LINE = "(null): No such file or directory"  # jeeves.log: logs no requests


def feed(monitor: Monitor, *lines: str) -> None:
    """Feed a monitor its log lines, one a second apart."""
    for offset, line in enumerate(lines, start=1):
        monitor.feed(line, float(offset))


def one_done(monitor: Monitor) -> Request:
    (request,) = monitor.done
    assert monitor.active == {}
    return request


def test_strata_request_with_repeated_tool_call_lines():
    monitor = Monitor("swift-flash-next-iq3s")
    feed(monitor, *STRATA_TOOL_CALL_REQUEST)

    request = one_done(monitor)
    assert request.server == "swift-flash-next-iq3s" and request.lane == 0
    assert request.prompt == 21997 and request.read == 21996
    assert request.tokens == 444
    assert (request.rate, request.seconds) == pytest.approx((61.7, 8.0))
    assert request.end == "stop" and request.hit == pytest.approx(86.5)
    assert request.steps == [("prompt", ""), ("thinking", ""), ("tool", "bash")]
    assert monitor.served == 1 and monitor.prompt_tokens == 21997
    assert monitor.generated == 444 and monitor.ends == Counter({"stop": 1})
    # The tool call is printed once a second: it counts as one call.
    assert monitor.tools == Counter({"bash": 1})


def test_strata_cached_prompt_starts_on_its_first_thinking_line():
    monitor = Monitor("swift-flash-next")
    feed(monitor, *STRATA_CACHED_PROMPT_REQUEST)

    request = one_done(monitor)
    # A cached prompt logs no reading line: its length stays unknown.
    assert request.prompt == 0 and request.read == 0
    assert request.tokens == 312 and request.hit == pytest.approx(93.1)
    assert request.steps == [("thinking", ""), ("tool", "Bash")]
    assert monitor.tools == Counter({"Bash": 1})

    short = Monitor("swift-flash-next-iq3s")
    feed(short, STRATA_BARE_DONE)  # too short for a progress line
    assert one_done(short).steps == [("answering", "")]


def test_strata_disconnect_end_is_recorded():
    monitor = Monitor("swift-flash-next")
    feed(monitor, STRATA_DISCONNECT_DONE)

    assert one_done(monitor).end == "disconnect"
    assert monitor.ends == Counter({"disconnect": 1})


def test_strata_engine_stop_clears_active_requests():
    monitor = Monitor("swift-flash-next")
    feed(monitor, STRATA_THINKING, STRATA_ENGINE_STOPPED)

    assert monitor.active == {}
    assert list(monitor.done) == []
    assert monitor.served == 0


def test_llama_server_slots_set_lanes_and_run_concurrently():
    monitor = Monitor("side")
    feed(monitor, SIDE_SLOTS, *SIDE_SLOT_LINES)

    assert monitor.backend == "llama.cpp"
    assert monitor.lanes == 2
    assert sorted(monitor.active) == [0, 1]
    assert all(request.phase == "busy" for request in monitor.active.values())


def test_llama_server_timing_and_release_finish_one_slot():
    monitor = Monitor("swarm")
    feed(monitor, *SWARM_REQUEST_TIMINGS)

    request = one_done(monitor)
    assert request.lane == 0
    assert request.prompt == 372 and request.read == 372
    assert request.tokens == 50 and request.rate == pytest.approx(50 / 1.36454)
    assert request.seconds == pytest.approx(2.52855)
    assert request.hit == pytest.approx(92.308)  # draft acceptance as a percentage
    assert request.steps == [("prompt", ""), ("answering", "")]
    assert monitor.served == 1 and monitor.ends == Counter({"stop": 1})


def test_llama_server_prompt_progress_estimates_the_prompt_length():
    monitor = Monitor("side")
    feed(monitor, SIDE_SLOT_LINES[1], SIDE_PROMPT_PROGRESS)

    request = monitor.active[1]
    assert request.read == 2098
    # The log gives no prompt length, only how far reading it has got: 2098 / 0.24.
    assert request.prompt == round(2098 / 0.24)


def test_live_samples_history_and_ranks_finished_requests_newest_first():
    server = Live()
    server.feed("side", SIDE_SLOTS)
    server.feed("side", SIDE_SLOT_LINES[1])  # lane 1 reads its prompt: no speed yet
    server.feed("swift-qwen", STRATA_THINKING)
    server.feed("video", NON_MODEL_LINE)

    # A ComfyUI-style server logs no requests, so it is not serving a model.
    assert [m.name for m in server.serving] == ["side", "swift-qwen"]

    server.sample()
    total, phase = server.speed[-1]
    assert total == pytest.approx(69.4) and phase == "thinking"
    assert server.busy[-1] == 2 and server.peak == pytest.approx(69.4)
    assert server.monitors["side"].trail[1][-1] == "busy"
    assert server.monitors["swift-qwen"].trail[0][-1] == "thinking"

    server.feed("swift-qwen", STRATA_BARE_DONE)
    server.feed("qwen-cpu", STRATA_DISCONNECT_DONE)
    recent = server.recent(5)
    assert [request.tokens for request in recent] == [157, 60]
    orders = [request.order for request in recent]
    assert orders == sorted(orders, reverse=True)  # newest first, across servers
