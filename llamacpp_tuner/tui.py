"""Terminal UI for serving, downloading and benchmarking models."""

import asyncio
import contextlib
import json
import os
import shlex
import signal
import sys
from collections.abc import Callable
from pathlib import Path

import click
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    ContentSwitcher,
    Input,
    Label,
    OptionList,
    RichLog,
    Static,
    Tab,
    Tabs,
)
from textual.widgets.option_list import Option

from llamacpp_tuner.cli import load_aliases, pick_projector, serve, write_aliases
from llamacpp_tuner.downloader import (
    list_downloaded_models,
    list_repo_models,
    resolve_model,
)
from llamacpp_tuner.llama import get_llama_binary

LCT = [sys.executable, "-m", "llamacpp_tuner.cli"]

ACCENT, MUTED, GOOD, WARN = "#818cf8", "#6b7280", "#7ee2a0", "#f2c572"

THEME = Theme(
    name="lct",
    primary=ACCENT,
    accent=ACCENT,
    foreground="#c9d1d9",
    background="#0b0e14",
    surface="#0b0e14",
    panel="#161b26",
    success=GOOD,
    warning=WARN,
    error="#f38ba8",
    dark=True,
)

CSS = """
Screen { padding: 1 2 0 2; }
#top { height: 1; }
#title { width: 1fr; text-style: bold; }
#state { width: auto; }
Tabs { height: 2; margin-bottom: 1; }
Tabs Tab { padding: 0 3 0 0; color: $text-muted; }
Tabs Tab.-active { color: $foreground; }
ContentSwitcher { height: 1fr; }
.page { height: 1fr; }
.column { width: 1fr; height: auto; padding-right: 4; }
.heading { color: $text-muted; margin-bottom: 1; }
.gap { margin-top: 1; }
OptionList { border: none; background: transparent; padding: 0; height: auto; max-height: 12; text-wrap: nowrap; text-overflow: ellipsis; }
OptionList:focus { border: none; background-tint: transparent; }
OptionList > .option-list--option-highlighted { background: $panel; text-style: none; }
OptionList:focus > .option-list--option-highlighted { background: $primary 25%; color: $foreground; }
Input { border: none; height: 1; padding: 0 1; background: $panel; width: 1fr; }
Input:focus { background: $primary 20%; }
* { scrollbar-size-vertical: 1; scrollbar-background: $background; scrollbar-color: $panel; }
.field { height: 1; margin-bottom: 1; }
.field Label { width: 12; color: $text-muted; }
#log { height: 8; background: transparent; border: none; padding: 0; }
#hints { height: 1; margin-top: 1; }
EditScreen { background: $background; padding: 1 2; }
EditScreen OptionList { max-height: 6; }
"""

HINTS = {
    "serve": "enter start/stop  e edit  n new  d delete",
    "download": "/ search  enter download  p projector",
    "bench": "enter run/stop  tab next field",
    "edit": "tab next field  ctrl+s save  esc cancel",
}
HOME_FOCUS = {"serve": "#profiles", "download": "#repo", "bench": "#bench-model"}


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


def tokens(count: int | None) -> str:
    return (
        f"{count // 1024}k" if count and count % 1024 == 0 else str(count or "default")
    )


def hint_line(keys: str) -> Text:
    """Render 'enter start  e edit' with keys in the accent colour."""
    text = Text()
    for part in keys.split("  "):
        key, _, label = part.partition(" ")
        text.append(f"{key} ", style=ACCENT).append(f"{label}    ", style=MUTED)
    return text


def bench_row(result: dict) -> tuple[str, str, str, str]:
    """Turn one llama-bench jsonl result into a table row."""
    test = f"pp{result['n_prompt']}" if result["n_prompt"] else f"tg{result['n_gen']}"
    if result["n_depth"]:
        test += f" @ d{result['n_depth']}"
    name = os.path.basename(result["model_filename"])
    return name, test, f"{result['avg_ts']:.1f}", f"± {result['stddev_ts']:.1f}"


def profile(args: str) -> dict:
    """Describe a saved profile; model and projector paths are None if missing."""
    ctx = serve.make_context("serve", shlex.split(args), resilient_parsing=True)
    params = ctx.params
    try:
        model = resolve_model(
            params["model"], quant=params["quant"], filename=params["filename"]
        )
        mmproj = pick_projector(params["model"], params["mmproj"], params["no_mmproj"])
    except (FileNotFoundError, ValueError):
        model = mmproj = None
    return {**params, "model_path": model, "mmproj_path": mmproj}


def grid(rows: list[tuple[str, Text | str]]) -> Table:
    table = Table.grid(padding=(0, 3))
    table.add_column(style=MUTED, no_wrap=True)
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return table


def field(label: str, widget: Input) -> Horizontal:
    return Horizontal(Label(label), widget, classes="field")


def model_options(projectors: bool) -> list[Option]:
    return [
        Option(
            Text.assemble(path.name.ljust(52), (gib(path.stat().st_size), MUTED)),
            str(path),
        )
        for path in list_downloaded_models()
        if ("mmproj" in path.name.lower()) == projectors
    ]


class EditScreen(ModalScreen[tuple[str, list[str]] | None]):
    """Create or edit a profile. Returns (name, serve args) or None."""

    BINDINGS = [
        Binding("ctrl+s", "save", priority=True),
        Binding("escape", "dismiss(None)"),
    ]

    def __init__(self, name: str, details: dict | None) -> None:
        super().__init__()
        self.profile_name, self.details = name, details or {}

    def compose(self) -> ComposeResult:
        d = self.details
        title = f"Edit {self.profile_name}" if self.profile_name else "New profile"
        yield Static(Text(title, style="bold"))
        with Vertical(classes="gap"):
            yield field("Name", Input(self.profile_name, id="name"))
            yield Label("Model", classes="heading")
            yield OptionList(*model_options(False), id="model")
            yield Label("Projector", classes="heading gap")
            yield OptionList(Option("none", ""), *model_options(True), id="mmproj")
            with Vertical(classes="gap"):
                yield field("Context", Input(str(d.get("ctx") or ""), id="ctx"))
                yield field("Host", Input(d.get("host") or "", id="host"))
                yield field("Port", Input(str(d.get("port") or ""), id="port"))
                yield field("Extra args", Input(d.get("extra_args") or "", id="extra"))
        yield Static(hint_line(HINTS["edit"]), id="hints")

    def on_mount(self) -> None:
        self.query_one("#ctx", Input).placeholder = "default"
        self.query_one("#host", Input).placeholder = "127.0.0.1"
        self.query_one("#port", Input).placeholder = "8080"
        for selector, key in (("#model", "model_path"), ("#mmproj", "mmproj_path")):
            options = self.query_one(selector, OptionList)
            with contextlib.suppress(Exception):
                options.highlighted = options.get_option_index(str(self.details[key]))
            if options.highlighted is None and options.option_count:
                options.highlighted = 0

    def action_save(self) -> None:
        name = self.query_one("#name", Input).value.strip()
        model = self.query_one("#model", OptionList)
        if not name or model.highlighted is None:
            self.notify("A profile needs a name and a model.", severity="error")
            return
        args = [str(model.get_option_at_index(model.highlighted).id)]
        mmproj = self.query_one("#mmproj", OptionList)
        if mmproj.highlighted and (
            path := mmproj.get_option_at_index(mmproj.highlighted).id
        ):
            args += ["--mmproj", path]
        for flag, selector in (
            ("--ctx", "#ctx"),
            ("--host", "#host"),
            ("--port", "#port"),
            ("--extra-args", "#extra"),
        ):
            if value := self.query_one(selector, Input).value.strip():
                args += [flag, value]
        self.dismiss((name, args))


class LctApp(App[None]):
    TITLE = "lct"
    CSS = CSS
    BINDINGS = [
        Binding("1", "page('serve')"),
        Binding("2", "page('download')"),
        Binding("3", "page('bench')"),
        Binding("e", "edit"),
        Binding("n", "new"),
        Binding("d", "delete"),
        Binding("y", "confirm"),
        Binding("p", "projector"),
        Binding("slash", "focus_search"),
        Binding("escape", "leave"),
        Binding("q", "quit"),
        Binding("ctrl+l", "clear_log"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.procs: dict[str, asyncio.subprocess.Process] = {}
        self.profiles: dict[str, dict] = {}
        self.serving = self.url = ""
        self.pending_delete = ""
        self.with_projector = True
        self.results: list[tuple[str, str, str, str]] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="top"):
            yield Static("lct", id="title")
            yield Static("", id="state")
        yield Tabs(
            Tab("Serve", id="serve"),
            Tab("Download", id="download"),
            Tab("Benchmark", id="bench"),
        )
        with ContentSwitcher(initial="serve-page"):
            with Horizontal(id="serve-page", classes="page"):
                with Vertical(classes="column"):
                    yield Label("Profiles", classes="heading")
                    yield OptionList(id="profiles")
                with Vertical(classes="column"):
                    yield Label("Settings", classes="heading")
                    yield Static(id="details")
            with Vertical(id="download-page", classes="page"):
                yield field("Repository", Input(id="repo"))
                yield Label("", id="files-heading", classes="heading")
                yield OptionList(id="files")
            with Horizontal(id="bench-page", classes="page"):
                with Vertical(classes="column"):
                    yield Label("Model", classes="heading")
                    yield OptionList(id="bench-model")
                with Vertical(classes="column"):
                    yield Label("Settings", classes="heading")
                    yield field("Prompt", Input("512", id="pp"))
                    yield field("Generate", Input("128", id="tg"))
                    yield field("Depth", Input("0", id="depth"))
                    yield field("Repeats", Input("3", id="reps"))
                    yield field(
                        "Extra args",
                        Input("-fa on -ctk q8_0 -ctv q8_0", id="bench-extra"),
                    )
                    yield Static(id="results", classes="gap")
        yield Label("Output", classes="heading")
        yield RichLog(id="log", wrap=True, highlight=True, max_lines=5000)
        yield Static(id="hints")

    def on_mount(self) -> None:
        self.register_theme(THEME)
        self.theme = "lct"
        self.query_one("#repo", Input).placeholder = "owner/model-GGUF, then enter"
        self.refresh_profiles()
        self.refresh_models()
        self.update_state()
        self.query_one("#profiles").focus()

    def on_unmount(self) -> None:
        for proc in self.procs.values():
            interrupt(proc)

    # Shared plumbing

    @property
    def page(self) -> str:
        return str(self.query_one(Tabs).active)

    def fail(self, message: str) -> None:
        self.notify(message, severity="error", timeout=8)

    def update_state(self, text: str = "", style: str = "") -> None:
        """Set the top-right status and redraw the key hints."""
        if not text:
            text, style = ("○ idle", MUTED)
            if self.serving:
                text, style = f"● serving {self.serving} · {self.url}", GOOD
        self.query_one("#state", Static).update(Text(text, style=style))
        keys = HINTS[self.page]
        if self.page == "download":
            state = "on" if self.with_projector else "off"
            keys = keys.replace("p projector", f"p projector-{state}")
        hints = Text.assemble(hint_line(keys), hint_line("1-3 pages  q quit"))
        if self.pending_delete:
            hints = Text(
                f"press y to delete {self.pending_delete}, any other key keeps it",
                style=WARN,
            )
        self.query_one("#hints", Static).update(hints)

    async def stream(
        self, name: str, cmd: list[str], on_line: Callable[[str], bool] | None = None
    ) -> int:
        """Stream a child's output to the log; on_line returns True to hide a line."""
        log = self.query_one("#log", RichLog)
        log.write(Text(f"$ {shlex.join(cmd)}", style=ACCENT))
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

    def on_key(self, event) -> None:
        # Only y confirms a pending delete; any other key abandons it.
        if self.pending_delete and event.key != "y":
            self.pending_delete = ""
            self.update_state()

    def action_page(self, page: str) -> None:
        self.query_one(Tabs).active = page

    @on(Tabs.TabActivated)
    def show_page(self, event: Tabs.TabActivated) -> None:
        page = str(event.tab.id)
        self.query_one(ContentSwitcher).current = f"{page}-page"
        self.query_one(HOME_FOCUS[page]).focus()
        self.update_state()

    def action_leave(self) -> None:
        if isinstance(self.focused, Input):
            target = "#files" if self.page == "download" else HOME_FOCUS[self.page]
            self.query_one(target).focus()

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    def refresh_models(self) -> None:
        options = self.query_one("#bench-model", OptionList)
        options.clear_options().add_options(model_options(False))
        if options.option_count:
            options.highlighted = 0

    # Serve page

    def selected_profile(self) -> str:
        options = self.query_one("#profiles", OptionList)
        if options.highlighted is None:
            return ""
        return str(options.get_option_at_index(options.highlighted).id)

    def refresh_profiles(self, select: str = "") -> None:
        try:
            aliases = load_aliases()
        except click.ClickException as error:
            self.fail(error.message)
            aliases = {}
        self.profiles = {name: profile(args) for name, args in aliases.items()}
        options = self.query_one("#profiles", OptionList)
        current = select or self.selected_profile()
        options.clear_options()
        for name, details in self.profiles.items():
            dot = ("● ", GOOD) if name == self.serving else "  "
            if details["model_path"]:
                meta, style = f"{tokens(details['ctx'])} ctx", MUTED
                if details["mmproj_path"]:
                    meta += " · images"
            else:
                meta, style = "model missing", WARN
            label = Text.assemble(dot, name.ljust(20), (meta, style))
            options.add_option(Option(label, name))
        if names := list(self.profiles):
            options.highlighted = names.index(current) if current in names else 0
        self.show_details()

    @on(OptionList.OptionHighlighted, "#profiles")
    def show_details(self) -> None:
        details = self.profiles.get(self.selected_profile())
        view = self.query_one("#details", Static)
        if not details:
            view.update(Text("No profiles yet. Press n to create one.", style=MUTED))
            return
        model, mmproj = details["model_path"], details["mmproj_path"]
        missing = Text(f"missing · {details['model']}", style=WARN)
        address = f"{details['host'] or '127.0.0.1'}:{details['port'] or 8080}"
        rows: list[tuple[str, Text | str]] = [
            ("Model", model.name if model else missing),
            ("Projector", Path(mmproj).name if mmproj else "none"),
            ("Context", tokens(details["ctx"])),
            ("Address", address),
            ("Extra", Text(details["extra_args"] or "none", overflow="fold")),
        ]
        view.update(grid(rows))

    @on(OptionList.OptionSelected, "#profiles")
    def toggle_server(self) -> None:
        if server := self.procs.get("server"):
            self.update_state("◌ stopping", WARN)
            interrupt(server)
        elif name := self.selected_profile():
            self.start_server(name)

    @work(group="server")
    async def start_server(self, name: str) -> None:
        self.update_state(f"◌ loading {name}", WARN)

        def watch(line: str) -> bool:
            if "listening on" in line:
                self.serving = name
                self.url = line.split("listening on")[-1].strip()
                self.update_state()
                self.refresh_profiles()
            return False

        code = await self.stream("server", [*LCT, "serve", name], watch)
        self.serving = ""
        self.refresh_profiles()
        self.update_state()
        if code > 0:
            self.fail(f"{name} exited with code {code}. See the output log.")

    def action_new(self) -> None:
        if self.page == "serve":
            self.push_screen(EditScreen("", None), self.save_profile)

    def action_edit(self) -> None:
        if self.page == "serve" and (name := self.selected_profile()):
            self.push_screen(
                EditScreen(name, self.profiles[name]),
                lambda result: self.save_profile(result, name),
            )

    def save_profile(
        self, result: tuple[str, list[str]] | None, old_name: str = ""
    ) -> None:
        if not result:
            return
        name, args = result
        aliases = {k: v for k, v in load_aliases().items() if k != old_name}
        write_aliases({**aliases, name: shlex.join(args)})
        self.refresh_profiles(select=name)
        self.notify(f"Saved {name}")

    def action_delete(self) -> None:
        if self.page == "serve" and (name := self.selected_profile()):
            self.pending_delete = name
            self.update_state()

    def action_confirm(self) -> None:
        if name := self.pending_delete:
            self.pending_delete = ""
            write_aliases({k: v for k, v in load_aliases().items() if k != name})
            self.refresh_profiles()
            self.update_state()
            self.notify(f"Deleted {name}")

    # Download page

    def action_focus_search(self) -> None:
        if self.page == "download":
            self.query_one("#repo", Input).focus()

    def action_projector(self) -> None:
        if self.page == "download":
            self.with_projector = not self.with_projector
            self.update_state()

    @on(Input.Submitted, "#repo")
    @work(exclusive=True, group="search")
    async def search(self) -> None:
        repo = self.query_one("#repo", Input).value.strip()
        heading = self.query_one("#files-heading", Label)
        files = self.query_one("#files", OptionList)
        heading.update(f"Searching {repo}…")
        try:
            found = await asyncio.to_thread(list_repo_models, repo)
        except Exception as error:
            heading.update("")
            self.fail(f"Cannot list {repo}: {error}")
            return
        local = {path.name for path in list_downloaded_models()}
        files.clear_options()
        for name, size in sorted(found):
            mark = ("  downloaded", GOOD) if Path(name).name in local else ""
            size_text = (gib(size).rjust(10), MUTED)
            files.add_option(
                Option(Text.assemble(name.ljust(44), size_text, mark), name)
            )
        heading.update(
            f"{len(found)} files in {repo}"
            if found
            else f"No model GGUF files in {repo}"
        )
        if found:
            files.highlighted = 0
            files.focus()

    @on(OptionList.OptionSelected, "#files")
    @work(group="download")
    async def download(self, event: OptionList.OptionSelected) -> None:
        if "download" in self.procs:
            self.fail("A download is already running.")
            return
        name = str(event.option.id)
        repo = self.query_one("#repo", Input).value.strip()
        cmd = [*LCT, "pull", repo, "--file", name]
        if not self.with_projector:
            cmd.append("--no-mmproj")
        self.notify(f"Downloading {name}")
        if await self.stream("download", cmd):
            self.fail(f"Download of {name} failed. See the output log.")
            return
        self.notify(f"Downloaded {name}")
        self.refresh_models()
        self.refresh_profiles()
        self.search()

    # Benchmark page

    @on(OptionList.OptionSelected, "#bench-model")
    @on(Input.Submitted, "#bench-page Input")
    def toggle_bench(self) -> None:
        if bench := self.procs.get("bench"):
            interrupt(bench)
        else:
            self.run_bench()

    def show_results(self, note: str = "") -> None:
        table = Table(box=None, padding=(0, 3, 0, 0), header_style=MUTED)
        for column in ("Test", "t/s", ""):
            table.add_column(column)
        for _, test, speed, spread in self.results:
            table.add_row(test, Text(speed, style="bold"), Text(spread, style=MUTED))
        view = self.query_one("#results", Static)
        view.update(table if self.results else Text(note, style=MUTED))

    @work(group="bench")
    async def run_bench(self) -> None:
        options = self.query_one("#bench-model", OptionList)
        binary = get_llama_binary("llama-bench")
        if options.highlighted is None:
            self.fail("Download a model first.")
            return
        if not binary:
            self.fail("llama-bench not found. Run 'lct setup'.")
            return
        model = str(options.get_option_at_index(options.highlighted).id)
        cmd = [str(binary), "-m", model, "-o", "jsonl"]
        for flag, selector in (
            ("-p", "#pp"),
            ("-n", "#tg"),
            ("-d", "#depth"),
            ("-r", "#reps"),
        ):
            if value := self.query_one(selector, Input).value.strip():
                cmd += [flag, value]
        try:
            cmd += shlex.split(self.query_one("#bench-extra", Input).value)
        except ValueError as error:
            self.fail(f"Extra args: {error}")
            return

        def collect(line: str) -> bool:
            if not line.startswith("{"):
                return False
            try:
                self.results.append(bench_row(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                return False
            self.show_results()
            return True

        self.results = []
        self.show_results(f"running {Path(model).name} · enter stops")
        self.update_state("◌ benchmarking", WARN)
        code = await self.stream("bench", cmd, collect)
        self.update_state()
        if code > 0:
            self.fail(f"llama-bench exited with code {code}. See the output log.")
        self.show_results("stopped")
