"""Tests for the terminal UI."""

import asyncio
import io
import sys
from pathlib import Path

import pytest
from huggingface_hub import ModelInfo
from rich.console import Console
from textual.widgets import Input, OptionList, RichLog, Static

from llamacpp_tuner.cli import load_aliases, write_aliases
from llamacpp_tuner.tui import LctApp, bench_row, card_summary

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
            assert app.query_one("#f-ctx", Input).value == "4096"
            for key in "231":
                await pilot.press(key)
                await pilot.pause()
            app.query_one("#profiles", OptionList).highlighted = 1
            await pilot.pause()
            assert app.query_one("#f-model", Input).has_class("-missing")

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


def test_quitting_stops_the_server(monkeypatch, workspace, tmp_path):
    write_aliases({"mine": str(workspace)})
    pid_file = tmp_path / "pid"
    server = f"import os; open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
    monkeypatch.setattr(
        "llamacpp_tuner.tui.LCT", [sys.executable, "-c", server + FAKE_SERVER]
    )

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press("enter")
            await until(pilot, lambda: app.serving == "mine")
            await pilot.press("q")
        pid = int(pid_file.read_text())
        for _ in range(100):
            if (
                not Path(f"/proc/{pid}").exists()
                or " Z " in Path(f"/proc/{pid}/stat").read_text()
            ):
                return
            await asyncio.sleep(0.05)
        raise AssertionError("server survived quitting the UI")

    asyncio.run(run())


def test_settings_save_on_enter_and_rename_on_leave(workspace):
    write_aliases({"mine": f"{workspace} --ctx 4096"})

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            assert app.query_one("#f-model", Input).value == workspace.name
            app.query_one("#f-ctx", Input).focus()
            app.query_one("#f-ctx", Input).value = "8192"
            await pilot.press("enter")
            assert load_aliases() == {"mine": f"{workspace} --ctx 8192"}

            app.query_one("#f-name", Input).focus()
            app.query_one("#f-name", Input).value = "renamed"
            await pilot.press("escape")
            await pilot.pause()
            assert list(load_aliases()) == ["renamed"]

            app.query_one("#f-model", Input).focus()
            app.query_one("#f-model", Input).value = "nope.gguf"
            await pilot.press("enter")
            assert load_aliases() == {"renamed": f"{workspace} --ctx 8192"}

    asyncio.run(run())


def test_new_copies_the_profile_and_delete_needs_y(workspace):
    write_aliases({"mine": f"{workspace} --ctx 4096"})

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press("n")
            await pilot.pause()
            assert load_aliases()["profile-1"] == f"{workspace} --ctx 4096"
            assert app.focused is app.query_one("#f-name")

            await pilot.press("escape", "d", "x")
            assert "profile-1" in load_aliases()
            await pilot.press("d", "y")
            assert list(load_aliases()) == ["mine"]

    asyncio.run(run())


def test_missing_model_is_kept_when_other_settings_change(workspace):
    write_aliases({"swift": "owner/repo --file m.gguf --ctx 4096"})

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            assert app.query_one("#f-model", Input).has_class("-missing")
            app.query_one("#f-ctx", Input).focus()
            app.query_one("#f-ctx", Input).value = "8192"
            await pilot.press("enter")
            assert load_aliases() == {"swift": "owner/repo --file m.gguf --ctx 8192"}

    asyncio.run(run())


def test_search_as_you_type_shows_repo_details_and_downloads(monkeypatch, workspace):
    info = ModelInfo(
        id="owner/repo-GGUF",
        downloads=176256,
        likes=12,
        gguf={
            "architecture": "qwen35",
            "total": 27_000_000_000,
            "context_length": 262144,
        },
        siblings=[
            {"rfilename": "m-Q4_K_M.gguf", "size": 16 * 1024**3},
            {"rfilename": "mmproj-F16.gguf", "size": 1},
        ],
    )
    card = "# Title\n\nThis model is a small and fast fine-tune that answers questions well."
    monkeypatch.setattr("llamacpp_tuner.tui.search_repos", lambda query: [info])
    monkeypatch.setattr("llamacpp_tuner.tui.repo_details", lambda repo: (info, card))
    monkeypatch.setattr(
        "llamacpp_tuner.tui.LCT",
        [sys.executable, "-c", "import sys; print('pulled', *sys.argv[1:])"],
    )

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("2", *"qwen")
            files = app.query_one("#files", OptionList)
            await until(pilot, lambda: files.option_count == 1)
            details = text(app, "#repo-info") + text(app, "#repo-summary")
            assert "176k downloads" in details and "qwen35 · 27B params" in details
            assert "small and fast fine-tune" in details

            await pilot.press("down", "enter", "p", "enter")
            log = app.query_one("#log", RichLog)
            await until(pilot, lambda: any("pulled" in ln.text for ln in log.lines))
            output = "\n".join(line.text for line in log.lines)
            assert "pull owner/repo-GGUF --file m-Q4_K_M.gguf --no-mmproj" in output

    asyncio.run(run())


def test_card_summary_skips_html_headings_and_images():
    card = '<div align="center"><img src="x.png"/></div>\n\n# Name\n\n![b](c.png)\n\n'
    card += "**Name** is a [derivative](https://x) that uses fewer thinking tokens than its base."
    assert card_summary(card) == (
        "Name is a derivative that uses fewer thinking tokens than its base."
    )


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
