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
from textual import on, work
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
Tabs Tab.-active { color: $foreground; }
ContentSwitcher { height: 1fr; }
.page { height: 1fr; }
.column { width: 1fr; height: 1fr; padding-right: 4; }
.heading { color: $text-muted; margin-bottom: 1; }
.gap { margin-top: 1; }
OptionList { border: none; background: transparent; padding: 0; height: auto; max-height: 16; text-wrap: nowrap; text-overflow: ellipsis; }
OptionList:focus { border: none; background-tint: transparent; }
OptionList > .option-list--option-highlighted { background: $panel; text-style: none; }
OptionList:focus > .option-list--option-highlighted { background: $primary 25%; color: $foreground; }
Input { border: none; height: 1; padding: 0 1; background: $panel; width: 1fr; }
Input:focus { background: $primary 20%; }
Input.-missing { color: $warning; }
.field { height: 1; margin-bottom: 1; }
.field Label { width: 12; color: $text-muted; }
#log { height: 8; background: transparent; border: none; padding: 0; }
#hints { height: 1; margin-top: 1; }
"""

HINTS = {
    "serve": "enter start/stop  tab edit  n new  d delete",
    "download": "↓ results  enter open/download  p projector",
    "bench": "enter run/stop  tab next field",
}
HOME_FOCUS = {"serve": "#profiles", "download": "#results", "bench": "#bench-model"}
SETTINGS = [
    ("Name", "name"),
    ("Model", "model"),
    ("Projector", "mmproj"),
    ("Context", "ctx"),
    ("Host", "host"),
    ("Port", "port"),
    ("Extra args", "extra"),
]


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
    """Parse a saved profile into form values plus the tokens behind them."""
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


def field(label: str, widget: Input) -> Horizontal:
    return Horizontal(Label(label), widget, classes="field")


def repo_view(info: ModelInfo, card: str) -> Group:
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
    rows: list[tuple[str, Text | str]] = [
        (
            "Popularity",
            f"{compact(info.downloads)} downloads · {info.likes or 0} likes",
        ),
        ("License", str(data.get("license_name") or data.get("license") or "unknown")),
        ("Base", ", ".join(base) if isinstance(base, list) else str(base or "—")),
        ("Model", model or "—"),
        ("Updated", str(info.last_modified or info.created_at or "—")[:10]),
    ]
    return Group(Text(info.id, style="bold"), Text(""), grid(rows))


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
        Binding("down", "to_results"),
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
        self.serving = self.url = ""
        self.pending_delete = ""
        self.with_projector = True
        self.repo = ""
        self.repos: dict[str, tuple[ModelInfo, str]] = {}
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
                    for label, key in SETTINGS:
                        yield field(label, Input(id=f"f-{key}", classes="setting"))
            with Vertical(id="download-page", classes="page"):
                yield field("Search", Input(id="repo"))
                with Horizontal():
                    with Vertical(classes="column"):
                        yield Label("", id="results-heading", classes="heading")
                        yield OptionList(id="results")
                    with VerticalScroll(classes="column"):
                        yield Static(id="repo-info")
                        yield Label("", id="files-heading", classes="heading gap")
                        yield OptionList(id="files")
                        yield Static(id="repo-summary", classes="gap")
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
                    yield Static(id="results-table", classes="gap")
        yield Label("Output", classes="heading")
        yield RichLog(id="log", wrap=True, highlight=True, max_lines=5000)
        yield Static(id="hints")

    def on_mount(self) -> None:
        self.register_theme(THEME)
        self.theme = "lct"
        placeholders = {
            "#repo": "type a model name, e.g. qwen 27b",
            "#f-ctx": "default",
            "#f-host": "127.0.0.1",
            "#f-port": "8080",
            "#f-mmproj": "none",
        }
        for selector, text in placeholders.items():
            self.query_one(selector, Input).placeholder = text
        self.refresh_models()
        self.refresh_profiles()
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
        focus = "#repo" if page == "download" else HOME_FOCUS[page]
        self.query_one(focus).focus()
        self.update_state()

    def action_leave(self) -> None:
        if isinstance(self.focused, Input):
            self.query_one(HOME_FOCUS[self.page]).focus()

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()

    def refresh_models(self) -> None:
        files = list_downloaded_models()
        self.models = {p.name: p for p in files if "mmproj" not in p.name.lower()}
        self.projectors = {p.name: p for p in files if "mmproj" in p.name.lower()}
        suggest = [("#f-model", self.models), ("#f-mmproj", self.projectors)]
        for selector, names in suggest:
            suggester = SuggestFromList(["none", *names], case_sensitive=False)
            self.query_one(selector, Input).suggester = suggester
        options = self.query_one("#bench-model", OptionList)
        options.clear_options()
        for name, path in self.models.items():
            size = (gib(path.stat().st_size), MUTED)
            options.add_option(Option(Text.assemble(name.ljust(52), size), str(path)))
        if options.option_count:
            options.highlighted = 0

    # Serve page: the profile list on the left, its settings on the right

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
            meta, style = f"{tokens(int(details['ctx'] or 0))} ctx", MUTED
            if details["missing"]:
                meta, style = "model missing", WARN
            elif details["mmproj"] != "none":
                meta += " · images"
            label = Text.assemble(dot, name.ljust(20), (meta, style))
            options.add_option(Option(label, name))
        if names := list(self.profiles):
            options.highlighted = names.index(current) if current in names else 0
        self.fill_form()

    @on(OptionList.OptionHighlighted, "#profiles")
    def fill_form(self) -> None:
        self.editing = self.selected_profile()
        details = self.profiles.get(self.editing, {})
        for _, key in SETTINGS:
            value = self.editing if key == "name" else details.get(key, "")
            widget = self.query_one(f"#f-{key}", Input)
            widget.value = value
            widget.cursor_position = 0
            widget.disabled = not details
        self.query_one("#f-model").set_class(details.get("missing", False), "-missing")

    def form_args(self, details: dict) -> list[str]:
        """Build serve args from the form, keeping untouched model tokens as-is."""
        value = {
            key: self.query_one(f"#f-{key}", Input).value.strip() for _, key in SETTINGS
        }
        if value["model"] == details["model"]:
            args = list(details["model_tokens"])
        elif value["model"] in self.models:
            args = [str(self.models[value["model"]])]
        else:
            raise ValueError(f"{value['model']} is not a downloaded model.")
        if value["mmproj"] == details["mmproj"]:
            args += details["mmproj_tokens"]
        elif value["mmproj"] in self.projectors:
            args += ["--mmproj", str(self.projectors[value["mmproj"]])]
        elif value["mmproj"] in ("", "none"):
            args += [] if args[0].endswith(".gguf") else ["--no-mmproj"]
        else:
            raise ValueError(f"{value['mmproj']} is not a downloaded projector.")
        for flag, key in (("--ctx", "ctx"), ("--host", "host"), ("--port", "port")):
            args += [flag, value[key]] if value[key] else []
        return args + (["--extra-args", value["extra"]] if value["extra"] else [])

    @on(Input.Submitted, ".setting")
    @on(Input.Blurred, ".setting")
    def save_form(self) -> None:
        old = self.editing
        if not (details := self.profiles.get(old)):
            return
        name = self.query_one("#f-name", Input).value.strip()
        aliases = load_aliases()
        try:
            if not name:
                raise ValueError("A profile needs a name.")
            if name != old and name in aliases:
                raise ValueError(f"A profile named {name} already exists.")
            args = shlex.join(self.form_args(details))
        except ValueError as error:
            self.fail(str(error))
            return
        if name == old and args == aliases.get(old):
            return
        write_aliases(
            {
                (name if k == old else k): (args if k == old else v)
                for k, v in aliases.items()
            }
        )
        self.refresh_profiles(select=name)
        self.notify(f"Saved {name}", timeout=2)

    def action_edit(self) -> None:
        if self.page == "serve" and self.editing:
            self.query_one("#f-name").focus()

    def action_new(self) -> None:
        if self.page != "serve":
            return
        aliases = load_aliases()
        base = aliases.get(self.editing)
        if base is None and not self.models:
            self.fail("Download a model first.")
            return
        name = next(
            f"profile-{i}" for i in range(1, 1000) if f"profile-{i}" not in aliases
        )
        write_aliases(
            {
                **aliases,
                name: base or shlex.quote(str(next(iter(self.models.values())))),
            }
        )
        self.refresh_profiles(select=name)
        self.query_one("#f-name").focus()

    def action_delete(self) -> None:
        if self.page == "serve" and self.editing:
            self.pending_delete = self.editing
            self.update_state()

    def action_confirm(self) -> None:
        if name := self.pending_delete:
            self.pending_delete = ""
            write_aliases({k: v for k, v in load_aliases().items() if k != name})
            self.refresh_profiles()
            self.update_state()
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

    def action_focus_search(self) -> None:
        if self.page == "download":
            self.query_one("#repo", Input).focus()

    def action_to_results(self) -> None:
        if (
            self.focused is self.query_one("#repo")
            and self.query_one("#results", OptionList).option_count
        ):
            self.query_one("#results").focus()

    def action_projector(self) -> None:
        if self.page == "download":
            self.with_projector = not self.with_projector
            self.update_state()

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
            popularity = (f"{compact(info.downloads)} ↓".rjust(7) + "   ", MUTED)
            row = Text.assemble(popularity, info.id)
            results.add_option(Option(row, info.id))
        heading.update(
            f"{len(found)} GGUF repositories" if found else "No GGUF repositories"
        )
        if found:
            results.highlighted = 0

    @on(Input.Submitted, "#repo")
    def open_first_result(self) -> None:
        results = self.query_one("#results", OptionList)
        if results.option_count:
            results.focus()

    @on(OptionList.OptionHighlighted, "#results")
    @work(exclusive=True, group="details")
    async def show_repo(self, event: OptionList.OptionHighlighted) -> None:
        repo = str(event.option.id)
        await asyncio.sleep(0.2)  # Skip repositories the cursor only passes over.
        if repo not in self.repos:
            self.query_one("#repo-info", Static).update(
                Text(f"Loading {repo}…", style=MUTED)
            )
            try:
                self.repos[repo] = await asyncio.to_thread(repo_details, repo)
            except Exception as error:
                self.fail(f"Cannot load {repo}: {error}")
                return
        self.repo = repo
        info, card = self.repos[repo]
        self.query_one("#repo-info", Static).update(repo_view(info, card))
        summary = Text(card_summary(card), style="#9aa4b2")
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
            mark = ("  downloaded", GOOD) if Path(name).name in self.models else ""
            size_text = (gib(size).rjust(9) + "   ", MUTED)
            row = Text.assemble(size_text, label.removeprefix(prefix), mark)
            files.add_option(Option(row, name))
        if found:
            files.highlighted = 0

    @on(OptionList.OptionSelected, "#results")
    def open_repo(self) -> None:
        self.query_one("#files").focus()

    @on(OptionList.OptionSelected, "#files")
    @work(group="download")
    async def download(self, event: OptionList.OptionSelected) -> None:
        if "download" in self.procs:
            self.fail("A download is already running.")
            return
        name, repo = str(event.option.id), self.repo
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
        self.show_files(self.repos[repo][0])

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
