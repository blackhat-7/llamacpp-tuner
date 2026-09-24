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


def row(app: LctApp, selector: str, option_id: str) -> str:
    return str(app.query_one(selector, OptionList).get_option(option_id).prompt)


async def edit(pilot, app: LctApp, key: str, value: str) -> None:
    """Open a setting with enter, type a new value and press enter."""
    settings = app.query_one("#settings", OptionList)
    settings.focus()
    settings.highlighted = settings.get_option_index(key)
    await pilot.press("enter")
    app.query_one("#editor-input", Input).value = value
    await pilot.press("enter")
    await pilot.pause()


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
            assert "4096" in row(app, "#settings", "ctx")
            for key in "231":
                await pilot.press(key)
                await pilot.pause()
            assert "model missing" in row(app, "#profiles", "gone")

    asyncio.run(run())


def test_tab_moves_between_panes_and_pages_switch_after_search(workspace):
    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press("tab")
            assert app.focused is app.query_one("#settings")
            await pilot.press("tab")
            assert app.focused is app.query_one("#profiles")

            await pilot.press("2", "slash", "3")
            assert app.query_one("#repo", Input).value == "3"
            await pilot.press("escape", "3")
            assert app.page == "bench"

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
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or " Z " in stat.read_text():
                return
            await asyncio.sleep(0.05)
        raise AssertionError("server survived quitting the UI")

    asyncio.run(run())


def test_settings_edit_in_place_and_reject_unknown_models(workspace):
    write_aliases({"mine": f"{workspace} --ctx 4096"})

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await edit(pilot, app, "ctx", "8192")
            assert load_aliases() == {"mine": f"{workspace} --ctx 8192"}

            await edit(pilot, app, "name", "renamed")
            assert list(load_aliases()) == ["renamed"]

            await edit(pilot, app, "model", "nope.gguf")
            assert load_aliases() == {"renamed": f"{workspace} --ctx 8192"}
            assert app.edit_target is not None
            await pilot.press("escape")
            assert app.edit_target is None
            assert app.focused is app.query_one("#settings")

    asyncio.run(run())


def test_new_copies_the_profile_and_delete_needs_y(workspace):
    write_aliases({"mine": f"{workspace} --ctx 4096"})

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press("n")
            await pilot.pause()
            assert load_aliases()["profile-1"] == f"{workspace} --ctx 4096"
            assert app.focused is app.query_one("#editor-input")

            await pilot.press("escape", "shift+tab", "d", "x")
            assert "profile-1" in load_aliases()
            await pilot.press("d", "y")
            assert list(load_aliases()) == ["mine"]

    asyncio.run(run())


def test_missing_model_is_kept_when_other_settings_change(workspace):
    write_aliases({"swift": "owner/repo --file m.gguf --ctx 4096"})

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await edit(pilot, app, "ctx", "8192")
            assert load_aliases() == {"swift": "owner/repo --file m.gguf --ctx 8192"}

    asyncio.run(run())


def test_search_shows_repo_details_and_downloads(monkeypatch, workspace):
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
            await pilot.press("2", "slash", *"qwen")
            files = app.query_one("#files", OptionList)
            await until(pilot, lambda: files.option_count == 1)
            details = text(app, "#repo-info") + text(app, "#repo-summary")
            assert "176k downloads" in details and "qwen35 · 27B params" in details
            assert "small and fast fine-tune" in details

            await pilot.press("enter", "enter", "p", "enter")
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


def test_benchmark_settings_edit_and_results_stream(monkeypatch, workspace, tmp_path):
    fake = tmp_path / "llama-bench"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "print('args', *sys.argv[1:], flush=True)\n"
        "print(json.dumps({'model_filename': 'm.gguf', 'n_prompt': 1024, 'n_gen': 0,"
        " 'n_depth': 0, 'avg_ts': 900.0, 'stddev_ts': 1.0}))\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr("llamacpp_tuner.tui.get_llama_binary", lambda name: fake)

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.press("3", "tab", "enter")
            app.query_one("#editor-input", Input).value = "1024"
            await pilot.press("enter", "shift+tab", "enter")
            await until(pilot, lambda: app.results and "bench" not in app.procs)
            assert app.results == [("m.gguf", "pp1024", "900.0", "± 1.0")]
            log = "\n".join(line.text for line in app.query_one("#log", RichLog).lines)
            assert "-p 1024" in log

    asyncio.run(run())
