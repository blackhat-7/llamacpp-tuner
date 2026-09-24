"""Tests for the terminal UI."""

import asyncio
import io
import sys

import pytest
from rich.console import Console
from textual.widgets import Input, OptionList, RichLog, Static

from llamacpp_tuner.cli import load_aliases, write_aliases
from llamacpp_tuner.tui import LctApp, bench_row

FAKE_SERVER = """
import time
print("main: server is listening on http://127.0.0.1:9", flush=True)
try:
    time.sleep(30)
except KeyboardInterrupt:
    pass
"""


async def until(pilot, condition) -> None:
    for _ in range(200):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError("condition never became true")


def text(app: LctApp, selector: str) -> str:
    console = Console(width=200, record=True, file=io.StringIO())
    console.print(app.query_one(selector, Static).content)
    return console.export_text()


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    model = tmp_path / "model-Q4_K_M.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr("llamacpp_tuner.tui.list_downloaded_models", lambda: [model])
    monkeypatch.setattr(
        "llamacpp_tuner.cli.get_aliases_path", lambda: tmp_path / "aliases.toml"
    )
    return model


@pytest.mark.parametrize("width", [40, 80, 120])
def test_renders_every_page_at_common_widths(width, workspace):
    write_aliases({"mine": f"{workspace} --ctx 4096", "gone": "owner/repo --quant Q4"})

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(width, 40)) as pilot:
            assert "4k" in text(app, "#details")
            for key in "231":
                await pilot.press(key)
                await pilot.pause()
            app.query_one("#profiles", OptionList).highlighted = 1
            await pilot.pause()
            assert "missing" in text(app, "#details")

    asyncio.run(run())


def test_enter_starts_and_stops_the_highlighted_profile(monkeypatch, workspace):
    write_aliases({"mine": str(workspace)})
    monkeypatch.setattr("llamacpp_tuner.tui.LCT", [sys.executable, "-c", FAKE_SERVER])

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press("enter")
            await until(pilot, lambda: app.serving == "mine")
            assert "http://127.0.0.1:9" in text(app, "#state")
            await pilot.press("enter")
            await until(pilot, lambda: "idle" in text(app, "#state"))
            assert "server" not in app.procs

    asyncio.run(run())


def test_new_profile_is_saved_and_deleted_only_after_y(workspace):
    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press("n")
            await pilot.pause()
            app.screen.query_one("#name", Input).value = "fresh"
            app.screen.query_one("#ctx", Input).value = "8192"
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert load_aliases() == {"fresh": f"{workspace} --ctx 8192"}

            await pilot.press("d", "x")
            assert "fresh" in load_aliases()
            await pilot.press("d", "y")
            assert load_aliases() == {}

    asyncio.run(run())


def test_search_lists_repo_files_and_enter_downloads(monkeypatch, workspace):
    listing = [("model-Q4_K_M.gguf", 16 * 1024**3), ("model-Q8_0.gguf", 28 * 1024**3)]
    monkeypatch.setattr("llamacpp_tuner.tui.list_repo_models", lambda repo: listing)
    monkeypatch.setattr(
        "llamacpp_tuner.tui.LCT",
        [sys.executable, "-c", "import sys; print('pulled', *sys.argv[1:])"],
    )

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press("2")
            await pilot.pause()
            app.query_one("#repo", Input).value = "owner/repo"
            await pilot.press("enter")
            files = app.query_one("#files", OptionList)
            await until(pilot, lambda: files.option_count == 2)
            assert "downloaded" in str(files.get_option_at_index(0).prompt)

            await pilot.press("p", "down", "enter")
            log = app.query_one("#log", RichLog)
            await until(pilot, lambda: any("pulled" in ln.text for ln in log.lines))
            output = "\n".join(line.text for line in log.lines)
            assert "pull owner/repo --file model-Q8_0.gguf --no-mmproj" in output

    asyncio.run(run())


def test_bench_row_names_prompt_generation_and_depth():
    result = {"model_filename": "/m/a.gguf", "n_depth": 0, "avg_ts": 911.04}
    result["stddev_ts"] = 2.3

    assert bench_row({**result, "n_prompt": 512, "n_gen": 0}) == (
        "a.gguf",
        "pp512",
        "911.0",
        "± 2.3",
    )
    depth = {**result, "n_prompt": 0, "n_gen": 128, "n_depth": 4096}
    assert bench_row(depth)[1] == "tg128 @ d4096"


def test_benchmark_streams_llama_bench_results(monkeypatch, workspace, tmp_path):
    fake = tmp_path / "llama-bench"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "print('loading model', flush=True)\n"
        "print(json.dumps({'model_filename': 'm.gguf', 'n_prompt': 512, 'n_gen': 0,"
        " 'n_depth': 0, 'avg_ts': 900.0, 'stddev_ts': 1.0}))\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr("llamacpp_tuner.tui.get_llama_binary", lambda name: fake)

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press("3")
            await pilot.pause()
            await pilot.press("enter")
            await until(pilot, lambda: app.results and "bench" not in app.procs)
            assert app.results == [("m.gguf", "pp512", "900.0", "± 1.0")]

    asyncio.run(run())
