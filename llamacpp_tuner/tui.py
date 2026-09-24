"""Terminal UI for serving, downloading and benchmarking models."""

import asyncio
import contextlib
import html
import json
import os
import re
import shlex
import signal
import sys
from collections.abc import Callable
from pathlib import Path

import click
from huggingface_hub import ModelInfo
from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.suggester import SuggestFromList
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
    model_files,
    repo_details,
    resolve_model,
    search_repos,
)
from llamacpp_tuner.llama import get_llama_binary

LCT = [sys.executable, "-m", "llamacpp_tuner.cli"]

ACCENT, MUTED, GOOD, WARN = "#818cf8", "#6b7280", "#7ee2a0", "#f2c572"
NUMBER, MEDIA, SOFT = "#67e8f9", "#f0abfc", "#9aa4b2"

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
* { scrollbar-size-vertical: 1; scrollbar-background: $background; scrollbar-color: $panel; }
#top { height: 1; }
#title { width: 1fr; text-style: bold; }
#state { width: auto; }
Tabs { height: 2; margin-bottom: 1; }
Tabs Tab { padding: 0 3 0 0; color: $text-muted; }
Tabs Tab.-active { color: $accent; text-style: bold; }
ContentSwitcher { height: 1fr; }
.page { height: 1fr; }
.column { width: 1fr; height: 1fr; padding-right: 4; }
.heading { color: $accent; text-style: bold; margin-bottom: 1; }
.gap { margin-top: 1; }
OptionList { border: none; background: transparent; padding: 0; height: auto; max-height: 16; text-wrap: nowrap; text-overflow: ellipsis; }
OptionList:focus { border: none; background-tint: transparent; }
OptionList > .option-list--option-highlighted { background: transparent; text-style: none; }
OptionList:focus > .option-list--option-highlighted { background: $primary 25%; color: $foreground; }
Input { border: none; height: 1; padding: 0 1; background: $panel; width: 1fr; }
Input:focus { background: $primary 20%; }
#search-row { height: 1; margin-bottom: 1; }
#search-row Label { width: 12; color: $text-muted; }
#editor { height: 1; margin-top: 1; display: none; }
#editor.-open { display: block; }
#editor Label { width: auto; color: $accent; text-style: bold; padding-right: 2; }
#log { height: 8; background: transparent; border: none; padding: 0; }
#hints { height: 1; margin-top: 1; }
"""

HINTS = {
    "serve": "enter start/stop  tab settings  n new  d delete",
    "serve-settings": "enter edit  tab profiles",
    "download": "/ search  tab files  enter download  p projector",
    "downloading": "c cancel-download  / search  p projector",
    "bench": "enter run/stop  tab settings",
    "bench-settings": "enter edit  tab models",
    "editor": "enter save  esc cancel",
    "search": "enter results  esc done",
}
PANES = {
    "serve": ["#profiles", "#settings"],
    "download": ["#results", "#files"],
    "bench": ["#bench-model", "#bench-settings"],
}
SETTINGS = {
    "name": "Name",
    "model": "Model",
    "mmproj": "Projector",
    "ctx": "Context",
    "host": "Host",
    "port": "Port",
    "extra": "Extra args",
}
BENCH_SETTINGS = {
    "pp": ("Prompt", "-p", "512"),
    "tg": ("Generate", "-n", "128"),
    "depth": ("Depth", "-d", "0"),
    "reps": ("Repeats", "-r", "3"),
    "extra": ("Extra args", "", "-fa on -ctk q8_0 -ctv q8_0"),
}
DEFAULTS = {"ctx": "model default", "host": "127.0.0.1", "port": "8080"}


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


def compact(count: int | None) -> str:
    count = count or 0
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if count >= size:
            return f"{count / size:.3g}{suffix}"
    return str(count)


def hint_line(keys: str) -> Text:
    """Render 'enter start  e edit' with keys in the accent colour."""
    text = Text()
    for part in keys.split("  "):
        key, _, label = part.partition(" ")
        text.append(f"{key} ", style=ACCENT).append(f"{label}    ", style=MUTED)
    return text


def setting_row(label: str, value: Text | str) -> Text:
    return Text.assemble((label.ljust(12), MUTED), value)


def sized_row(size: int, name: str, mark: str = "") -> Text:
    """A file row with its size first, so long names never run into it."""
    row = Text.assemble((gib(size).rjust(9) + "   ", NUMBER), name)
    return row.append(f"  {mark}", style=GOOD) if mark else row


def card_summary(card: str, limit: int = 420) -> str:
    """First prose paragraph of a model card, without HTML or markdown clutter."""
    text = html.unescape(re.sub(r"<[^>]+>|!\[[^\]]*\]\([^)]*\)", "", card))
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(line.strip() for line in block.splitlines()).strip()
        if len(block) > 60 and not block.startswith(("#", "|", ">", "`", "- ", "* ")):
            block = re.sub(r"[*_`]", "", block)
            return block[:limit] + ("…" if len(block) > limit else "")
    return ""


def bench_row(result: dict) -> tuple[str, str, str, str]:
    """Turn one llama-bench jsonl result into a table row."""
    test = f"pp{result['n_prompt']}" if result["n_prompt"] else f"tg{result['n_gen']}"
    if result["n_depth"]:
        test += f" @ d{result['n_depth']}"
    name = os.path.basename(result["model_filename"])
    return name, test, f"{result['avg_ts']:.1f}", f"± {result['stddev_ts']:.1f}"


def profile(args: str) -> dict:
    """Parse a saved profile into setting values plus the tokens behind them."""
    ctx = serve.make_context("serve", shlex.split(args), resilient_parsing=True)
    p = ctx.params
    try:
        model = resolve_model(p["model"], quant=p["quant"], filename=p["filename"])
        mmproj = pick_projector(p["model"], p["mmproj"], p["no_mmproj"])
    except (FileNotFoundError, ValueError):
        model = mmproj = None
    selector = p["filename"] or p["quant"]
    model_tokens = [p["model"]]
    model_tokens += ["--file", p["filename"]] if p["filename"] else []
    model_tokens += ["--quant", p["quant"]] if p["quant"] else []
    mmproj_tokens = ["--mmproj", str(p["mmproj"])] if p["mmproj"] else []
    mmproj_tokens += ["--no-mmproj"] if p["no_mmproj"] else []
    mmproj_tokens += ["--no-mmproj-offload"] if p["no_mmproj_offload"] else []
    return {
        "missing": model is None,
        "model": model.name if model else f"{p['model']} {selector or ''}".strip(),
        "mmproj": Path(mmproj).name if mmproj else "none",
        "ctx": str(p["ctx"] or ""),
        "host": p["host"] or "",
        "port": str(p["port"] or ""),
        "extra": p["extra_args"] or "",
        "model_tokens": model_tokens,
        "mmproj_tokens": mmproj_tokens,
    }


def grid(rows: list[tuple[str, Text | str]]) -> Table:
    table = Table.grid(padding=(0, 3))
    table.add_column(style=MUTED, no_wrap=True)
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return table


def repo_view(info: ModelInfo) -> Group:
    data = info.card_data.to_dict() if info.card_data else {}
    gguf = info.gguf or {}
    model = " · ".join(
        part
        for part in (
            gguf.get("architecture"),
            gguf.get("total") and f"{compact(gguf['total'])} params",
            gguf.get("context_length") and f"{tokens(gguf['context_length'])} ctx",
        )
        if part
    )
    base = data.get("base_model")
    license = str(data.get("license_name") or data.get("license") or "unknown")
    open_license = license in ("apache-2.0", "mit") or license.startswith(
        ("llama", "gemma")
    )
    popularity = Text.assemble(
        (f"{compact(info.downloads)} ", NUMBER),
        "downloads · ",
        (f"{info.likes or 0} ", MEDIA),
        "likes",
    )
    updated = str(info.last_modified or info.created_at or "—")[:10]
    rows: list[tuple[str, Text | str]] = [
        ("Popularity", popularity),
        ("License", Text(license, style=GOOD if open_license else WARN)),
        ("Base", ", ".join(base) if isinstance(base, list) else str(base or "—")),
        ("Model", Text(model or "—", style=NUMBER)),
        ("Updated", Text(updated, style=SOFT)),
    ]
    return Group(Text(info.id, style="bold"), Text(""), grid(rows))


class PickList(OptionList):
    """An OptionList where a click only highlights; enter or a double-click acts."""

    async def _on_click(self, event: events.Click) -> None:
        event.prevent_default()  # Textual would also run OptionList's click-selects.
        index = event.style.meta.get("option")
        if index is None or self.get_option_at_index(index).disabled:
            return
        self.highlighted = index
        if event.chain > 1:
            self.action_select()


class LctApp(App[None]):
    TITLE = "lct"
    CSS = CSS
    BINDINGS = [
        Binding("1", "page('serve')"),
        Binding("2", "page('download')"),
        Binding("3", "page('bench')"),
        Binding("tab", "pane(1)", priority=True),
        Binding("shift+tab", "pane(-1)", priority=True),
        Binding("right", "pane(1)"),
        Binding("left", "pane(-1)"),
        Binding("n", "new"),
        Binding("d", "delete"),
        Binding("y", "confirm"),
        Binding("c", "cancel_download"),
        Binding("p", "projector"),
        Binding("slash", "search"),
        Binding("escape", "leave"),
        Binding("q", "quit"),
        Binding("ctrl+l", "clear_log"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.procs: dict[str, asyncio.subprocess.Process] = {}
        self.profiles: dict[str, dict] = {}
        self.models: dict[str, Path] = {}
        self.projectors: dict[str, Path] = {}
        self.editing = ""
        self.edit_target: tuple[str, str] | None = None
        self.serving = self.url = ""
        # A question awaiting y, and what y does.
        self.pending: tuple[str, Callable[[], object]] | None = None
        self.with_projector = True
        self.repo = ""
        self.repos: dict[str, tuple[ModelInfo, str]] = {}
        self.bench = {key: default for key, (_, _, default) in BENCH_SETTINGS.items()}
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
                    yield PickList(id="profiles")
                with Vertical(classes="column"):
                    yield Label("Settings", classes="heading")
                    yield PickList(id="settings")
            with Vertical(id="download-page", classes="page"):
                with Horizontal(id="search-row"):
                    yield Label("Search")
                    yield Input(id="repo")
                with Horizontal():
                    with Vertical(classes="column"):
                        yield Label("", id="results-heading", classes="heading")
                        yield PickList(id="results")
                    with VerticalScroll(classes="column"):
                        yield Static(id="repo-info")
                        yield Label("", id="files-heading", classes="heading gap")
                        yield PickList(id="files")
                        yield Static(id="repo-summary", classes="gap")
            with Horizontal(id="bench-page", classes="page"):
                with Vertical(classes="column"):
                    yield Label("Model", classes="heading")
                    yield PickList(id="bench-model")
                with Vertical(classes="column"):
                    yield Label("Settings", classes="heading")
                    yield PickList(id="bench-settings")
                    yield Static(id="results-table", classes="gap")
        with Horizontal(id="editor"):
            yield Label("", id="editor-label")
            yield Input(id="editor-input")
        yield Label("Output", classes="heading gap")
        yield RichLog(id="log", wrap=True, highlight=True, max_lines=5000)
        yield Static(id="hints")

    def on_mount(self) -> None:
        self.register_theme(THEME)
        self.theme = "lct"
        self.query_one("#repo", Input).placeholder = "press / and type a model name"
        self.refresh_models()
        self.refresh_profiles()
        self.show_bench_settings()
        self.query_one("#profiles").focus()

    def stop_children(self) -> None:
        """Stop servers, downloads and benchmarks so none outlive the UI."""
        for proc in self.procs.values():
            interrupt(proc)

    async def action_quit(self) -> None:
        self.stop_children()
        self.exit()

    # Shared plumbing: pages, panes, the edit line, status and hints

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
        self.update_hints()

    def update_hints(self) -> None:
        focused = self.focused
        context = self.page
        if self.edit_target:
            context = "editor"
        elif focused is self.query_one("#repo"):
            context = "search"
        elif (
            self.page != "download"
            and focused is not None
            and f"#{focused.id}" == PANES[self.page][1]
        ):
            context = f"{self.page}-settings"
        if context == "download" and "download" in self.procs:
            context = "downloading"
        keys = HINTS[context]
        if context in ("download", "downloading"):
            state = "on" if self.with_projector else "off"
            keys = keys.replace("p projector", f"p projector-{state}")
        hints = hint_line(keys)
        if context not in ("editor", "search"):
            hints.append_text(hint_line("1-3 pages  q quit"))
        if self.pending:
            hints = Text(self.pending[0], style=WARN)
        self.query_one("#hints", Static).update(hints)

    def on_descendant_focus(self) -> None:
        self.update_hints()

    def on_key(self, event) -> None:
        # Only y confirms a pending question; any other key abandons it.
        if self.pending and event.key != "y":
            self.pending = None
            self.update_hints()

    def ask(self, question: str, action: Callable[[], object]) -> None:
        self.pending = (question, action)
        self.update_hints()

    def action_confirm(self) -> None:
        if self.pending:
            action = self.pending[1]
            self.pending = None
            self.update_hints()
            action()

    def action_page(self, page: str) -> None:
        self.query_one(Tabs).active = page

    @on(Tabs.TabActivated)
    def show_page(self, event: Tabs.TabActivated) -> None:
        page = str(event.tab.id)
        self.close_editor()
        self.query_one(ContentSwitcher).current = f"{page}-page"
        self.query_one(PANES[page][0]).focus()
        self.update_state()

    def action_pane(self, step: int) -> None:
        """Move between the two panes of a page, never through every widget."""
        if self.edit_target:
            return
        panes = PANES[self.page]
        current = self.focused.id if self.focused else None
        index = next((i for i, p in enumerate(panes) if p == f"#{current}"), -1)
        self.query_one(panes[(index + step) % len(panes)]).focus()

    def action_leave(self) -> None:
        if self.edit_target:
            self.close_editor()
        elif self.focused is self.query_one("#repo"):
            self.query_one("#results").focus()

    def open_editor(self, target: tuple[str, str], label: str, value: str) -> None:
        self.edit_target = target
        self.query_one("#editor-label", Label).update(label)
        editor = self.query_one("#editor-input", Input)
        choices = {"model": self.models, "mmproj": self.projectors}.get(target[1])
        suggestions = ["none", *choices] if choices is not None else []
        editor.suggester = SuggestFromList(suggestions, case_sensitive=False)
        editor.value = value
        self.query_one("#editor").add_class("-open")
        editor.focus()

    def close_editor(self) -> None:
        if not self.edit_target:
            return
        page = self.edit_target[0]
        self.edit_target = None
        self.query_one("#editor").remove_class("-open")
        if page == self.page:
            self.query_one(PANES[page][1]).focus()
        self.update_hints()

    @on(Input.Submitted, "#editor-input")
    def apply_edit(self, event: Input.Submitted) -> None:
        if not self.edit_target:
            return
        page, key = self.edit_target
        value = event.value.strip()
        if page == "bench":
            self.bench[key] = value
            self.show_bench_settings()
        elif not self.save_setting(key, value):
            return
        self.close_editor()

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

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
        except asyncio.CancelledError:
            # Textual cancels workers when the app exits or crashes; the child
            # must not outlive it.
            interrupt(proc)
            raise
        finally:
            del self.procs[name]

    def refresh_models(self) -> None:
        files = list_downloaded_models()
        self.models = {p.name: p for p in files if "mmproj" not in p.name.lower()}
        self.projectors = {p.name: p for p in files if "mmproj" in p.name.lower()}
        options = self.query_one("#bench-model", OptionList)
        options.clear_options()
        for name, path in self.models.items():
            options.add_option(Option(sized_row(path.stat().st_size, name), str(path)))
        if options.option_count:
            options.highlighted = 0

    # Serve page: profiles on the left, the highlighted profile's settings on the right

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
            running = name == self.serving
            label = Text.assemble(
                ("● " if running else "  ", GOOD),
                (name[:21].ljust(22), GOOD if running else ""),
            )
            if details["missing"]:
                label.append("model missing", style=WARN)
            else:
                label.append(f"{tokens(int(details['ctx'] or 0))} ctx  ", style=NUMBER)
                if details["mmproj"] != "none":
                    label.append("images", style=MEDIA)
            options.add_option(Option(label, name))
        if names := list(self.profiles):
            options.highlighted = names.index(current) if current in names else 0
        self.show_settings()

    @on(OptionList.OptionHighlighted, "#profiles")
    def show_settings(self) -> None:
        self.editing = self.selected_profile()
        details = self.profiles.get(self.editing)
        options = self.query_one("#settings", OptionList)
        highlighted = options.highlighted
        options.clear_options()
        if not details:
            empty = Text("No profiles yet. Press n to create one.", style=MUTED)
            options.add_option(Option(empty, disabled=True))
            return
        for key, label in SETTINGS.items():
            value = self.editing if key == "name" else details[key]
            if not value:
                shown = Text(DEFAULTS.get(key, "none"), style=MUTED)
            elif key == "model":
                shown = Text(value, style=WARN if details["missing"] else "bold")
            elif key == "mmproj":
                shown = Text(value, style=MUTED if value == "none" else MEDIA)
            elif key in ("ctx", "port"):
                shown = Text(value, style=NUMBER)
            else:
                shown = Text(value, style=SOFT if key == "extra" else "")
            options.add_option(Option(setting_row(label, shown), key))
        options.highlighted = highlighted or 0

    @on(OptionList.OptionSelected, "#settings")
    def edit_setting(self, event: OptionList.OptionSelected) -> None:
        key = str(event.option.id)
        details = self.profiles[self.editing]
        value = self.editing if key == "name" else details[key]
        self.open_editor(("serve", key), SETTINGS[key], value)

    def profile_args(self, details: dict, values: dict[str, str]) -> list[str]:
        """Build serve args, keeping an unchanged model or projector's tokens."""
        if values["model"] == details["model"]:
            args = list(details["model_tokens"])
        elif values["model"] in self.models:
            args = [str(self.models[values["model"]])]
        else:
            raise ValueError(f"{values['model']} is not a downloaded model.")
        if values["mmproj"] == details["mmproj"]:
            args += details["mmproj_tokens"]
        elif values["mmproj"] in self.projectors:
            args += ["--mmproj", str(self.projectors[values["mmproj"]])]
        elif values["mmproj"] in ("", "none"):
            args += [] if args[0].endswith(".gguf") else ["--no-mmproj"]
        else:
            raise ValueError(f"{values['mmproj']} is not a downloaded projector.")
        for flag, key in (("--ctx", "ctx"), ("--host", "host"), ("--port", "port")):
            args += [flag, values[key]] if values[key] else []
        return args + (["--extra-args", values["extra"]] if values["extra"] else [])

    def save_setting(self, key: str, value: str) -> bool:
        old = self.editing
        details = self.profiles[old]
        values = {k: (old if k == "name" else details[k]) for k in SETTINGS}
        values[key] = value
        name = values["name"]
        aliases = load_aliases()
        try:
            if not name:
                raise ValueError("A profile needs a name.")
            if name != old and name in aliases:
                raise ValueError(f"A profile named {name} already exists.")
            if key in ("ctx", "port") and value and not value.isdigit():
                raise ValueError(f"{SETTINGS[key]} must be a number.")
            args = shlex.join(self.profile_args(details, values))
        except ValueError as error:
            self.fail(str(error))
            return False
        write_aliases(
            {
                (name if k == old else k): (args if k == old else v)
                for k, v in aliases.items()
            }
        )
        self.refresh_profiles(select=name)
        self.notify(f"Saved {name}", timeout=2)
        return True

    def action_new(self) -> None:
        if self.page != "serve" or self.edit_target:
            return
        aliases = load_aliases()
        base = aliases.get(self.editing)
        if base is None and not self.models:
            self.fail("Download a model first.")
            return
        name = next(
            f"profile-{i}" for i in range(1, 1000) if f"profile-{i}" not in aliases
        )
        first_model = shlex.quote(str(next(iter(self.models.values()), "")))
        write_aliases({**aliases, name: base or first_model})
        self.refresh_profiles(select=name)
        self.open_editor(("serve", "name"), SETTINGS["name"], name)

    def action_delete(self) -> None:
        if self.page == "serve" and (name := self.editing) and not self.edit_target:
            question = f"press y to delete {name}, any other key keeps it"
            self.ask(question, lambda: self.delete_profile(name))

    def delete_profile(self, name: str) -> None:
        write_aliases({k: v for k, v in load_aliases().items() if k != name})
        self.refresh_profiles()
        self.notify(f"Deleted {name}", timeout=2)

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

    # Download page: search as you type, details of the highlighted repository

    def action_search(self) -> None:
        if self.page == "download" and not self.edit_target:
            self.query_one("#repo", Input).focus()

    def action_projector(self) -> None:
        if self.page == "download":
            self.with_projector = not self.with_projector
            self.update_hints()

    @on(Input.Changed, "#repo")
    @work(exclusive=True, group="search")
    async def search(self) -> None:
        query = self.query_one("#repo", Input).value.strip()
        await asyncio.sleep(0.3)  # Wait for typing to pause before querying.
        heading = self.query_one("#results-heading", Label)
        results = self.query_one("#results", OptionList)
        if len(query) < 2:
            heading.update("")
            results.clear_options()
            return
        heading.update(f"Searching {query}…")
        try:
            found = await asyncio.to_thread(search_repos, query)
        except Exception as error:
            heading.update("")
            self.fail(f"Search failed: {error}")
            return
        results.clear_options()
        for info in found:
            popularity = (f"{compact(info.downloads)} ↓".rjust(7) + "   ", NUMBER)
            results.add_option(Option(Text.assemble(popularity, info.id), info.id))
        count = f"{len(found)} GGUF repositories" if found else "No GGUF repositories"
        heading.update(count)
        if found:
            results.highlighted = 0

    @on(Input.Submitted, "#repo")
    def open_results(self) -> None:
        self.query_one("#results").focus()

    @on(OptionList.OptionHighlighted, "#results")
    @work(exclusive=True, group="details")
    async def show_repo(self, event: OptionList.OptionHighlighted) -> None:
        repo = str(event.option.id)
        await asyncio.sleep(0.2)  # Skip repositories the cursor only passes over.
        if repo not in self.repos:
            loading = Text(f"Loading {repo}…", style=MUTED)
            self.query_one("#repo-info", Static).update(loading)
            try:
                self.repos[repo] = await asyncio.to_thread(repo_details, repo)
            except Exception as error:
                self.fail(f"Cannot load {repo}: {error}")
                return
        self.repo = repo
        info, card = self.repos[repo]
        self.query_one("#repo-info", Static).update(repo_view(info))
        summary = Text(card_summary(card), style=SOFT)
        self.query_one("#repo-summary", Static).update(summary)
        self.show_files(info)

    def show_files(self, info: ModelInfo) -> None:
        files = self.query_one("#files", OptionList)
        found = model_files(info)
        names = [
            Path(re.sub(r"-00001-of-0*(\d+)", r" (\1 parts)", n)).name for n, _ in found
        ]
        # Files share a long model-name prefix; drop it so the quant stays visible.
        prefix = os.path.commonprefix(names) if len(names) > 1 else ""
        prefix = prefix[: max(prefix.rfind("-"), prefix.rfind("_")) + 1]
        heading = f"{len(found)} model files" + (f" · {prefix}…" if prefix else "")
        self.query_one("#files-heading", Label).update(heading)
        files.clear_options()
        for (name, size), label in zip(found, names, strict=True):
            mark = "downloaded" if Path(name).name in self.models else ""
            files.add_option(
                Option(sized_row(size, label.removeprefix(prefix), mark), name)
            )
        if found:
            files.highlighted = 0

    @on(OptionList.OptionSelected, "#results")
    def open_repo(self) -> None:
        self.query_one("#files").focus()

    @on(OptionList.OptionSelected, "#files")
    def confirm_download(self, event: OptionList.OptionSelected) -> None:
        if "download" in self.procs:
            self.fail("A download is already running. Press c to cancel it.")
            return
        name, repo = str(event.option.id), self.repo
        size = dict(model_files(self.repos[repo][0])).get(name, 0)
        extra = " plus its projector" if self.with_projector else ""
        question = f"press y to download {Path(name).name} ({gib(size)}){extra}, any other key cancels"
        self.ask(question, lambda: self.download(repo, name))

    def action_cancel_download(self) -> None:
        if download := self.procs.get("download"):
            interrupt(download)
            self.notify("Cancelling download", timeout=2)

    @work(group="download")
    async def download(self, repo: str, name: str) -> None:
        cmd = [*LCT, "pull", repo, "--file", name]
        if not self.with_projector:
            cmd.append("--no-mmproj")
        self.notify(f"Downloading {name}")
        self.update_hints()
        code = await self.stream("download", cmd)
        self.update_hints()
        if code:
            self.fail(f"Download of {name} stopped. See the output log.")
            return
        self.notify(f"Downloaded {name}")
        self.refresh_models()
        self.refresh_profiles()
        self.show_files(self.repos[repo][0])

    # Benchmark page

    def show_bench_settings(self) -> None:
        options = self.query_one("#bench-settings", OptionList)
        highlighted = options.highlighted
        options.clear_options()
        for key, (label, _, _) in BENCH_SETTINGS.items():
            value = self.bench[key]
            style = SOFT if key == "extra" else NUMBER
            shown = Text(value or "none", style=style if value else MUTED)
            options.add_option(Option(setting_row(label, shown), key))
        options.highlighted = highlighted or 0

    @on(OptionList.OptionSelected, "#bench-settings")
    def edit_bench_setting(self, event: OptionList.OptionSelected) -> None:
        key = str(event.option.id)
        self.open_editor(("bench", key), BENCH_SETTINGS[key][0], self.bench[key])

    @on(OptionList.OptionSelected, "#bench-model")
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
            table.add_row(
                Text(test, style=NUMBER),
                Text(speed, style=f"bold {GOOD}"),
                Text(spread, style=MUTED),
            )
        view = self.query_one("#results-table", Static)
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
        for key, (_, flag, _) in BENCH_SETTINGS.items():
            if flag and self.bench[key]:
                cmd += [flag, self.bench[key]]
        try:
            cmd += shlex.split(self.bench["extra"])
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
