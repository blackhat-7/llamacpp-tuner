"""Terminal UI for serving, downloading and benchmarking models."""

import asyncio
import contextlib
import os
import shlex
import signal
import sys
from collections.abc import Callable

import click
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Switch,
    TabbedContent,
    TabPane,
)

from llamacpp_tuner.cli import load_aliases, pick_projector, save_alias, serve
from llamacpp_tuner.downloader import list_downloaded_models, resolve_model

LCT = [sys.executable, "-m", "llamacpp_tuner.cli"]

CSS = """
TabbedContent { height: 1fr; }
.form { height: auto; padding: 1 2 0 2; }
.row { height: auto; }
.row > Label { width: 12; padding: 1 1 0 0; color: $text-muted; text-style: bold; }
.row > .inline { width: auto; padding: 1 1 0 2; background: transparent; }
.row Input, .row Select { width: 1fr; }
.row .narrow { width: 20; }
.actions { height: auto; padding: 1 0 0 12; }
.actions Button { margin-right: 1; }
.actions Label { padding: 1 1 0 1; }
#status.ready { color: $success; }
#status.busy { color: $warning; }
#status.error { color: $error; }
DataTable { height: 1fr; margin: 1 2 0 2; background: $surface; }
#log { height: 12; margin: 0 1; border: round $panel; background: $background; }
"""


async def spawn(cmd: list[str]) -> asyncio.subprocess.Process:
    # A new session lets one SIGINT reach lct and llama-server, like Ctrl-C does.
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        },
        start_new_session=True,
        limit=2**20,
    )


def interrupt(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGINT)


def gib(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.1f} GiB"


def row(label: str, *widgets: Widget) -> Horizontal:
    return Horizontal(Label(label), *widgets, classes="row")


def inline(text: str) -> Label:
    return Label(text, classes="inline")


class LctApp(App[None]):
    TITLE = "lct"
    SUB_TITLE = "llama.cpp control"
    CSS = CSS
    BINDINGS = [
        ("ctrl+s", "toggle_server", "Start/stop server"),
        ("ctrl+l", "clear_log", "Clear log"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.procs: dict[str, asyncio.subprocess.Process] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Serve", id="serve-tab"), Vertical(classes="form"):
                yield row(
                    "Profile",
                    Select([], prompt="Load a saved profile", id="profile"),
                    Input(
                        placeholder="profile name", id="alias-name", classes="narrow"
                    ),
                    Button("Save", id="save"),
                )
                yield row("Model", Select([], prompt="Choose a model", id="model"))
                yield row(
                    "Projector", Select([], prompt="None (text only)", id="mmproj")
                )
                yield row(
                    "Context",
                    Input(placeholder="default", type="integer", id="ctx"),
                    inline("Host"),
                    Input(placeholder="127.0.0.1", id="host"),
                    inline("Port"),
                    Input(
                        placeholder="8080", type="integer", id="port", classes="narrow"
                    ),
                )
                yield row(
                    "Extra args",
                    Input(
                        placeholder="-ngl all -fa on -ctk q8_0 -ctv q8_0", id="extra"
                    ),
                )
                yield Horizontal(
                    Button("▶ Start", variant="success", id="serve"),
                    Label("○ Stopped", id="status"),
                    classes="actions",
                )
            with TabPane("Download", id="download-tab"):
                with Vertical(classes="form"):
                    yield row(
                        "Repository",
                        Input(placeholder="owner/model-GGUF", id="repo"),
                        Button("Search", variant="primary", id="search"),
                    )
                    yield Horizontal(
                        Button("⬇ Download", variant="success", id="download"),
                        Switch(value=True, id="with-mmproj"),
                        Label("with projector"),
                        classes="actions",
                    )
                yield DataTable(id="files", cursor_type="row", zebra_stripes=True)
            with TabPane("Benchmark", id="bench-tab"):
                with Vertical(classes="form"):
                    yield row(
                        "Model", Select([], prompt="Choose a model", id="bench-model")
                    )
                    yield row(
                        "Prompt",
                        Input("512", id="pp"),
                        inline("Generate"),
                        Input("128", id="tg"),
                        inline("Depth"),
                        Input("0", id="depth"),
                        inline("Repeats"),
                        Input("3", type="integer", id="reps", classes="narrow"),
                    )
                    yield row(
                        "Extra args",
                        Input("-fa on -ctk q8_0 -ctv q8_0", id="bench-extra"),
                    )
                    yield Horizontal(
                        Button("▶ Run", variant="primary", id="bench"),
                        classes="actions",
                    )
                yield DataTable(id="results", zebra_stripes=True)
        yield RichLog(id="log", wrap=True, highlight=True, max_lines=5000)
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "tokyo-night"
        self.query_one("#log", RichLog).border_title = "Output"
        self.query_one("#files", DataTable).add_columns("File", "Size", "")
        self.query_one("#results", DataTable).add_columns("Model", "Test", "t/s", "±")
        self.refresh_choices()

    def on_unmount(self) -> None:
        for proc in self.procs.values():
            interrupt(proc)

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    def fail(self, message: str) -> None:
        self.notify(message, severity="error", timeout=8)

    # Shared plumbing

    def refresh_choices(self) -> None:
        """Reload downloaded models and saved profiles, keeping current picks."""
        files = list_downloaded_models()
        models = [(f"{p.name}  {gib(p.stat().st_size)}", str(p)) for p in files]
        choices = {
            "#model": [m for m in models if "mmproj" not in m[1].lower()],
            "#bench-model": [m for m in models if "mmproj" not in m[1].lower()],
            "#mmproj": [m for m in models if "mmproj" in m[1].lower()],
        }
        try:
            choices["#profile"] = [(name, name) for name in load_aliases()]
        except click.ClickException as error:
            self.fail(error.message)
        for selector, options in choices.items():
            select = self.query_one(selector, Select)
            value = select.value
            select.set_options(options)
            if any(value == option[1] for option in options):
                select.value = value

    async def stream(
        self, name: str, cmd: list[str], on_line: Callable[[str], bool] | None = None
    ) -> int:
        """Stream a child's output to the log; on_line returns True to hide a line."""
        log = self.query_one("#log", RichLog)
        log.write(Text(f"$ {shlex.join(cmd)}", style="bold cyan"))
        proc = self.procs[name] = await spawn(cmd)
        try:
            assert proc.stdout
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                if not (on_line and on_line(line)):
                    log.write(line)
            return await proc.wait()
        finally:
            del self.procs[name]

    # Serve tab

    def set_status(self, text: str, state: str = "") -> None:
        status = self.query_one("#status", Label)
        status.update(text)
        status.set_classes(state)
        self.sub_title = text

    def form_args(self) -> list[str]:
        model = self.query_one("#model", Select).value
        if not isinstance(model, str):
            raise ValueError("Choose a model first.")
        args = [model]
        if isinstance(mmproj := self.query_one("#mmproj", Select).value, str):
            args += ["--mmproj", mmproj]
        for flag, selector in (
            ("--ctx", "#ctx"),
            ("--host", "#host"),
            ("--port", "#port"),
            ("--extra-args", "#extra"),
        ):
            if value := self.query_one(selector, Input).value.strip():
                args += [flag, value]
        return args

    @on(Select.Changed, "#profile")
    def load_profile(self, event: Select.Changed) -> None:
        if not isinstance(name := event.value, str):
            return
        try:
            params = serve.make_context(
                "serve", shlex.split(load_aliases()[name])
            ).params
            model = resolve_model(
                params["model"], quant=params["quant"], filename=params["filename"]
            )
            mmproj = pick_projector(
                params["model"], params["mmproj"], params["no_mmproj"]
            )
            self.query_one("#model", Select).value = str(model)
            projector = self.query_one("#mmproj", Select)
            if mmproj:
                projector.value = str(mmproj)
            else:
                projector.clear()
        except (click.ClickException, FileNotFoundError, ValueError) as error:
            self.fail(f"Cannot load {name}: {error}")
            return
        for key in ("ctx", "host", "port"):
            self.query_one(f"#{key}", Input).value = str(params[key] or "")
        self.query_one("#extra", Input).value = params["extra_args"]
        self.query_one("#alias-name", Input).value = name

    @on(Button.Pressed, "#save")
    def save_profile(self) -> None:
        name = self.query_one("#alias-name", Input).value.strip()
        if not name:
            self.fail("Name the profile first.")
            return
        try:
            save_alias(name, self.form_args())
        except (click.ClickException, ValueError) as error:
            self.fail(str(error))
            return
        self.refresh_choices()
        self.query_one("#profile", Select).value = name
        self.notify(f"Saved profile {name}")

    @on(Button.Pressed, "#serve")
    def action_toggle_server(self) -> None:
        if server := self.procs.get("server"):
            self.set_status("◌ Stopping…", "busy")
            interrupt(server)
        else:
            self.start_server()

    @work(group="server")
    async def start_server(self) -> None:
        try:
            args = self.form_args()
        except ValueError as error:
            self.fail(str(error))
            return
        button = self.query_one("#serve", Button)
        button.label, button.variant = "■ Stop", "error"
        self.set_status("◌ Loading model…", "busy")

        def watch(line: str) -> bool:
            if "listening on" in line:
                self.set_status(
                    f"● Ready  {line.split('listening on')[-1].strip()}", "ready"
                )
            return False

        code = await self.stream("server", [*LCT, "serve", *args], watch)
        button.label, button.variant = "▶ Start", "success"
        if code > 0:
            self.set_status(f"✕ Exited with code {code}", "error")
        else:
            self.set_status("○ Stopped")
