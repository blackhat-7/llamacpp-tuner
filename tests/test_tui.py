"""Tests for the terminal UI."""

import asyncio
import sys

import pytest
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    RichLog,
    Select,
    Switch,
    TabbedContent,
)

from llamacpp_tuner.tui import LctApp


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
def test_renders_all_tabs_at_common_widths(width, workspace):
    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(width, 40)) as pilot:
            tabs = app.query_one(TabbedContent)
            for tab in ("serve-tab", "download-tab", "bench-tab"):
                tabs.active = tab
                await pilot.pause()
                assert app.screen.find_widget(app.query_one(f"#{tab}"))

    asyncio.run(run())


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


def test_serve_starts_and_stops_from_the_form(monkeypatch, workspace):
    monkeypatch.setattr("llamacpp_tuner.tui.LCT", [sys.executable, "-c", FAKE_SERVER])

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            app.query_one("#model", Select).value = str(workspace)
            status = app.query_one("#status", Label)
            await pilot.press("ctrl+s")
            await until(pilot, lambda: "Ready" in str(status.content))
            assert "http://127.0.0.1:9" in str(status.content)
            await pilot.press("ctrl+s")
            await until(pilot, lambda: "Stopped" in str(status.content))
            assert "server" not in app.procs

    asyncio.run(run())


def test_profile_round_trips_through_aliases(workspace, tmp_path):
    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            app.query_one("#model", Select).value = str(workspace)
            app.query_one("#ctx", Input).value = "4096"
            app.query_one("#extra", Input).value = "-fa on"
            app.query_one("#alias-name", Input).value = "mine"
            app.query_one("#save", Button).press()
            await pilot.pause()
            assert "mine" in (tmp_path / "aliases.toml").read_text()

            app.query_one("#ctx", Input).value = ""
            app.query_one("#profile", Select).clear()
            await pilot.pause()
            app.query_one("#profile", Select).value = "mine"
            await pilot.pause()
            assert app.query_one("#ctx", Input).value == "4096"
            assert app.query_one("#extra", Input).value == "-fa on"

    asyncio.run(run())


def test_search_lists_repo_files_and_downloads_the_selected_one(monkeypatch, workspace):
    listing = [("model-Q4_K_M.gguf", 16 * 1024**3), ("model-Q8_0.gguf", 28 * 1024**3)]
    monkeypatch.setattr("llamacpp_tuner.tui.list_repo_models", lambda repo: listing)
    monkeypatch.setattr(
        "llamacpp_tuner.tui.LCT",
        [sys.executable, "-c", "import sys; print('pulled', *sys.argv[1:])"],
    )

    async def run() -> None:
        app = LctApp()
        async with app.run_test(size=(100, 40)) as pilot:
            app.query_one(TabbedContent).active = "download-tab"
            await pilot.pause()
            app.query_one("#repo", Input).value = "owner/repo"
            await pilot.click("#search")
            table = app.query_one("#files", DataTable)
            await until(pilot, lambda: table.row_count == 2)
            assert "downloaded" in str(table.get_row_at(0)[2])
            assert table.get_row_at(1)[1] == "28.0 GiB"

            table.move_cursor(row=1)
            app.query_one("#with-mmproj", Switch).value = False
            await pilot.click("#download")
            log = app.query_one("#log", RichLog)
            await until(
                pilot,
                lambda: any("pulled" in line.text for line in log.lines),
            )
            text = "\n".join(line.text for line in log.lines)
            assert "pull owner/repo --file model-Q8_0.gguf --no-mmproj" in text

    asyncio.run(run())
