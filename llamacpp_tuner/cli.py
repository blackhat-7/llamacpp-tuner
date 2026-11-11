"""Command-line interface for lct."""

import functools
import json
import random
import shlex
import shutil
import signal
import sqlite3
import subprocess
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

import click

from llamacpp_tuner import __version__, servers
from llamacpp_tuner.cache import get_aliases_path, get_models_dir
from llamacpp_tuner.downloader import (
    download_mmproj,
    download_model,
    download_repository,
    get_mmproj_path,
    list_downloaded_models,
    resolve_model,
    resolve_repository,
)
from llamacpp_tuner.jeeves import install_jeeves, run_jeeves
from llamacpp_tuner.llama import install_llama, list_devices, run_server
from llamacpp_tuner.logs import SOURCES
from llamacpp_tuner.strata import run_strata
from llamacpp_tuner.video import (
    ASPECTS,
    FILES,
    GUIDE_FRAMES,
    MAX_SECONDS,
    MIN_SECONDS,
    QUALITIES,
    STEPS,
    UPSCALE_SHORT_SIDE,
    cancel,
    canvas,
    data_dir,
    download_files,
    frame_count,
    h3_graph,
    has_audio,
    install_comfyui,
    join,
    missing_files,
    missing_for,
    model_path,
    output_dir,
    probe,
    run_comfyui,
    stage,
    stage_tail,
    stage_video,
    submit,
    upscale_graph,
    wait,
)

BACKEND = click.option(
    "--backend",
    type=click.Choice(["llama", "jeeves", "strata", "comfyui"]),
    default="llama",
    help="llama.cpp, Jeeves's own PyTorch server, Strata, or ComfyUI for video",
)
# The profile the Video tab and 'lct video' start; 'lct setup --backend comfyui' adds it.
VIDEO_PROFILE = "video"


def _read_aliases_file() -> dict:
    path = get_aliases_path()
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise click.ClickException(f"Invalid {path}: {error}") from error


def load_aliases() -> dict[str, str]:
    """Read serve aliases, each mapping a name to 'serve' arguments."""
    aliases = _read_aliases_file()
    aliases.pop("stacks", None)
    for name, value in aliases.items():
        if not isinstance(value, str):
            raise click.ClickException(
                f"Alias '{name}' in {get_aliases_path()} must be a string of serve arguments."
            )
    return aliases


def load_stacks() -> dict[str, list[str]]:
    """Read the [stacks] table: a name for a set of profiles started together."""
    stacks = _read_aliases_file().get("stacks", {})
    for name, members in stacks.items():
        if not (isinstance(members, list) and all(isinstance(m, str) for m in members)):
            raise click.ClickException(
                f"Stack '{name}' in {get_aliases_path()} must be a list of profile names."
            )
    return stacks


def write_aliases(aliases: dict[str, str]) -> None:
    """Replace every alias, keeping stacks. Rewriting the file drops its comments."""
    stacks = load_stacks()
    lines = [
        f"{json.dumps(key)} = {json.dumps(value)}\n" for key, value in aliases.items()
    ]
    if stacks:
        lines.append("\n[stacks]\n")
        lines += [
            f"{json.dumps(key)} = {json.dumps(value)}\n"
            for key, value in stacks.items()
        ]
    get_aliases_path().write_text("".join(lines))


def expand_names(names: tuple[str, ...]) -> list[str]:
    """Turn stack and profile names into profile names, in order, without repeats."""
    aliases, stacks = load_aliases(), load_stacks()
    profiles: list[str] = []
    for name in names:
        for profile in stacks.get(name, [name]):
            if profile not in aliases:
                raise click.ClickException(f"Unknown profile or stack: {profile}")
            if profile not in profiles:
                profiles.append(profile)
    return profiles


def pick_projector(model: str, mmproj: Path | None, no_mmproj: bool) -> Path | None:
    """Use an explicit projector, else a repository's downloaded one."""
    if mmproj or no_mmproj or model.lower().endswith(".gguf"):
        return mmproj
    return get_mmproj_path(model)


class _AliasGroup(click.Group):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # Expand before parsing so options typed after an alias override its own.
        if (
            len(args) > 1
            and args[0] == "serve"
            and (alias := load_aliases().get(args[1])) is not None
        ):
            args = ["serve", *shlex.split(alias), *args[2:]]
        return super().parse_args(ctx, args)


@click.group(cls=_AliasGroup)
@click.version_option(version=__version__)
def main() -> None:
    """Download and serve GGUF models with llama.cpp."""


@main.command()
@BACKEND
@click.option("--force", is_flag=True, help="Reinstall the backend")
@click.option(
    "--cmake-arg",
    multiple=True,
    help="Additional CMake argument; use --cmake-arg=-DNAME=VALUE",
)
@click.option(
    "--torch-index",
    help="PyTorch wheel index for Jeeves or ComfyUI, such as "
    "https://download.pytorch.org/whl/rocm7.2; default is PyPI (CUDA)",
)
def setup(
    backend: str, force: bool, cmake_arg: tuple[str, ...], torch_index: str | None
) -> None:
    """Find or download llama-server for this GPU (--cmake-arg builds it), or
    install Jeeves, or ComfyUI with a 'video' profile."""
    try:
        if backend == "strata":
            raise click.UsageError("Strata installs with its own ./setup.sh.")
        if backend == "jeeves":
            click.echo(f"Jeeves: {install_jeeves(torch_index, force=force)}")
        elif backend == "comfyui":
            click.echo(f"ComfyUI: {install_comfyui(torch_index, force=force)}")
            aliases = load_aliases()
            if VIDEO_PROFILE not in aliases:
                data_dir().mkdir(parents=True, exist_ok=True)
                args = [str(data_dir()), "--backend", "comfyui", "--port", "8188"]
                write_aliases({**aliases, VIDEO_PROFILE: shlex.join(args)})
                click.echo(f"Added profile '{VIDEO_PROFILE}' to {get_aliases_path()}")
            if missing := missing_files():
                click.echo(f"Downloading {len(missing)} model files…")
                download_files(missing)
            click.echo("Ready: lct video make 'a description'")
        else:
            binary = install_llama(force=force, extra_cmake_args=cmake_arg)
            click.echo(f"llama-server: {binary}")
            devices = list_devices(binary)
            click.echo(devices)
            if "(none)" in devices:
                click.echo(
                    "No GPU found: install your GPU driver (NVIDIA, or Vulkan for "
                    "AMD/Intel), then run 'lct setup --force'.",
                    err=True,
                )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise click.ClickException(str(error)) from error


@main.command()
@click.argument("repo_id")
@click.option("--quant", "-q", help="Exact quant name, such as Q6_K")
@click.option("--file", "filename", help="Exact GGUF filename instead of --quant")
@click.option("--force", "-f", is_flag=True, help="Re-download existing files")
@click.option("--no-mmproj", is_flag=True, help="Do not download a vision projector")
@click.option("--mmproj-file", help="Exact projector filename")
@click.option(
    "--all", "whole", is_flag=True, help="Every file, for non-GGUF models (Jeeves)"
)
def pull(
    repo_id: str,
    quant: str | None,
    filename: str | None,
    force: bool,
    no_mmproj: bool,
    mmproj_file: str | None,
    whole: bool,
) -> None:
    """Download an exact GGUF artifact, or a whole repository, from Hugging Face."""
    if whole:
        if quant or filename or mmproj_file:
            raise click.UsageError("--all cannot be combined with a file choice.")
        try:
            click.echo(f"Model: {download_repository(repo_id, force=force)}")
        except Exception as error:
            raise click.ClickException(str(error)) from error
        return
    if bool(quant) == bool(filename):
        raise click.UsageError("Provide exactly one of --quant, --file or --all.")
    if no_mmproj and mmproj_file:
        raise click.UsageError("--no-mmproj and --mmproj-file cannot be combined.")

    try:
        model_path = download_model(
            repo_id, quant=quant, filename=filename, force=force
        )
        if not model_path:
            selector = filename or quant
            raise click.ClickException(f"No {selector} GGUF found in {repo_id}.")
        click.echo(f"Model: {model_path}")

        if not no_mmproj:
            mmproj = download_mmproj(repo_id, filename=mmproj_file, force=force)
            if mmproj_file and not mmproj:
                raise click.ClickException(
                    f"Projector not found in {repo_id}: {mmproj_file}"
                )
            if mmproj:
                click.echo(f"Projector: {mmproj}")
    except click.ClickException:
        raise
    except Exception as error:
        raise click.ClickException(str(error)) from error


@main.command()
@click.argument("model")
@click.option("--quant", "-q", help="Exact quant for a downloaded repository")
@click.option("--file", "filename", help="Exact GGUF filename for a repository")
@click.option("--ctx", "-c", type=click.IntRange(min=1))
@click.option("--host", "-h")
@click.option("--port", "-p", type=click.IntRange(min=1, max=65535))
@click.option(
    "--mmproj",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Explicit multimodal projector for a local model",
)
@click.option("--no-mmproj", is_flag=True, help="Disable multimodal support")
@click.option("--no-mmproj-offload", is_flag=True, help="Keep the projector off GPU")
@click.option("--extra-args", "-e", default="", help="Arguments passed to llama-server")
@click.option(
    "--env",
    "env",
    multiple=True,
    help="NAME=VALUE added to the server's environment; repeatable",
)
@click.option(
    "--server",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="llama-server binary to run instead of the installed one",
)
@BACKEND
def serve(
    model: str,
    quant: str | None,
    filename: str | None,
    ctx: int | None,
    host: str | None,
    port: int | None,
    mmproj: Path | None,
    no_mmproj: bool,
    no_mmproj_offload: bool,
    extra_args: str,
    env: tuple[str, ...],
    server: Path | None,
    backend: str,
) -> None:
    """Run llama-server, overriding only explicitly supplied settings.

    MODEL is a GGUF path, a downloaded repository, or an alias from aliases.toml.
    With --backend jeeves, --ctx is Jeeves's --max-len and --extra-args go to
    its server. With --backend strata, MODEL is the strata-<model>.json its setup
    wrote; context and files live in that config. With --backend comfyui, MODEL
    is ComfyUI's base directory (models, input, output).
    """
    if quant and filename:
        raise click.UsageError("--quant and --file cannot be combined.")
    if mmproj and no_mmproj:
        raise click.UsageError("--mmproj and --no-mmproj cannot be combined.")
    if bad := [item for item in env if "=" not in item]:
        raise click.UsageError(f"--env needs NAME=VALUE, got: {', '.join(bad)}")
    if server and backend != "llama":
        raise click.UsageError(
            "--server picks a llama-server binary; use it with llama.cpp."
        )

    environment = dict(item.split("=", 1) for item in env)

    try:
        if backend == "jeeves":
            if quant or filename or mmproj or no_mmproj_offload:
                raise click.UsageError(
                    "Jeeves takes a whole model directory, not a file or projector."
                )
            weights = resolve_repository(model)
            jeeves_args = ["--model", str(weights)]
            jeeves_args += ["--drafter", str(weights / "drafter_k4.safetensors")]
            for flag, value in (("--max-len", ctx), ("--host", host), ("--port", port)):
                if value is not None:
                    jeeves_args.extend([flag, str(value)])
            run_jeeves(jeeves_args + shlex.split(extra_args), env=environment)
            return
        if backend == "strata":
            if quant or filename or ctx or mmproj or no_mmproj or no_mmproj_offload:
                raise click.UsageError(
                    "Strata's model, context and projector are set in its run config."
                )
            strata_args = []
            for flag, value in (("--host", host), ("--port", port)):
                if value is not None:
                    strata_args.extend([flag, str(value)])
            run_strata(
                Path(model).expanduser(),
                strata_args + shlex.split(extra_args),
                env=environment,
            )
            return
        if backend == "comfyui":
            if quant or filename or ctx or mmproj or no_mmproj or no_mmproj_offload:
                raise click.UsageError(
                    "ComfyUI takes its base directory; models live in it."
                )
            comfy_args = []
            for flag, value in (("--listen", host), ("--port", port)):
                if value is not None:
                    comfy_args.extend([flag, str(value)])
            run_comfyui(
                Path(model).expanduser(),
                comfy_args + shlex.split(extra_args),
                env=environment,
            )
            return

        model_path = resolve_model(model, quant=quant, filename=filename)

        selected_mmproj = pick_projector(model, mmproj, no_mmproj)
        if no_mmproj_offload and not selected_mmproj:
            raise click.UsageError(
                "--no-mmproj-offload requires --mmproj or a downloaded repository projector."
            )

        server_args = ["-m", str(model_path)]
        for flag, value in (("-c", ctx), ("--host", host), ("--port", port)):
            if value is not None:
                server_args.extend([flag, str(value)])
        if selected_mmproj:
            server_args.extend(["--mmproj", str(selected_mmproj)])
            if no_mmproj_offload:
                server_args.append("--no-mmproj-offload")
        server_args.extend(shlex.split(extra_args))
        run_server(server_args, env=environment, binary=server)
    except click.ClickException:
        raise
    except (FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    except subprocess.CalledProcessError as error:
        raise click.ClickException(
            f"The server exited with code {error.returncode}."
        ) from error


@main.command(name="models")
def list_models() -> None:
    """List locally downloaded GGUF files."""
    models = list_downloaded_models()
    if not models:
        click.echo("No downloaded models.")
        return

    root = get_models_dir()
    for model in models:
        size_gib = model.stat().st_size / (1024**3)
        tag = " [projector]" if "mmproj" in model.name.lower() else ""
        click.echo(f"{model.relative_to(root)} ({size_gib:.2f} GiB){tag}")


@main.command()
@click.argument("names", nargs=-1, required=True)
def up(names: tuple[str, ...]) -> None:
    """Start profiles or stacks in the background and wait until they listen.

    Servers keep running after this exits; stop them with 'lct down'.
    """
    pending = {}
    for name in expand_names(names):
        pid = servers.running().get(name) or servers.start(name)
        pending[name] = pid
    failed = []
    try:
        while pending:
            for name, pid in list(pending.items()):
                if address := servers.url(name):
                    click.echo(f"{name}: {address}")
                elif not servers.alive(pid):
                    failed.append(name)
                    click.echo(
                        f"{name}: exited, see {servers.log_path(name)}", err=True
                    )
                else:
                    continue
                del pending[name]
            time.sleep(0.5)
    except KeyboardInterrupt:
        click.echo(f"Still loading in the background: {', '.join(pending)}")
        return
    if failed:
        raise click.ClickException(f"Failed to start: {', '.join(failed)}")


@main.command()
@click.argument("names", nargs=-1)
def down(names: tuple[str, ...]) -> None:
    """Stop running profiles or stacks; with no names, stop every server."""
    active = servers.running()
    targets = [n for n in expand_names(names) if n in active] if names else list(active)
    for name in targets:
        servers.interrupt(active[name])
    deadline = time.monotonic() + 60
    while (
        any(servers.alive(active[n]) for n in targets) and time.monotonic() < deadline
    ):
        time.sleep(0.2)
    for name in targets:
        state = "still stopping" if servers.alive(active[name]) else "stopped"
        click.echo(f"{name}: {state}")


@main.command()
def ps() -> None:
    """List running servers with their address, RAM and VRAM."""
    active = servers.running()
    if not active:
        click.echo("No servers running.")
    for name, pid in active.items():
        address = servers.url(name) or "loading"
        used = servers.usage(pid)
        click.echo(
            f"{name:<20} {address:<28} {used.ram / 1024**3:5.1f} GiB RAM"
            f" {used.vram / 1024**3:5.1f} GiB VRAM  pid {pid}"
        )


@main.group()
def logs() -> None:
    """Read local coding-agent logs (Claude Code, Pi, opencode, Codex)."""


@logs.command(name="export")
def export_logs() -> None:
    """Print human prompts and subagent briefs as JSONL, one row per message."""
    for name, read in SOURCES.items():
        # One unreadable store must not hide the others.
        try:
            for row in read(Path.home()):
                click.echo(json.dumps(row))
        except (OSError, sqlite3.Error, ValueError) as error:
            click.echo(f"Skipped the rest of {name}'s logs: {error}", err=True)


# Video: MiniMax H3 on ComfyUI, which runs as the 'video' profile.

# Free VRAM needed to start ComfyUI; its int8 model is 21 GB and partly offloads.
VIDEO_VRAM = 18 * 1024**3
INPUT_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
PROMPT_FILE = click.option(
    "--prompt-file",
    type=INPUT_FILE,
    help="Read the prompt from a file; easier for long, multi-line prompts",
)
QUALITY = click.option(
    "--quality",
    type=click.Choice(QUALITIES),
    default="standard",
    show_default=True,
    help="draft: ~480p, 4 steps, a quick preview; standard: 768p, 8 steps; "
    "high: standard, then upscaled to 1080p",
)
PHOTOS = click.option(
    "--photo",
    "photos",
    multiple=True,
    type=INPUT_FILE,
    help="Reference image of a person, object, place or style; repeat for up to 9. "
    "The prompt calls them <Picture 1>, <Picture 2>, … in this order",
)
AUDIOS = click.option(
    "--audio",
    "audios",
    multiple=True,
    type=INPUT_FILE,
    help="Reference audio, such as a voice or music to reuse; repeat for up to 3. "
    "The prompt calls them <Audio N> after any source video's sound",
)
SEED = click.option(
    "--seed", type=int, help="Seed; the same seed and inputs repeat a clip"
)
OUT = click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Also save the result here; every result stays in the video library",
)


def _prompt_text(prompt: str | None, prompt_file: Path | None) -> str:
    if bool(prompt) == bool(prompt_file):
        raise click.UsageError(
            "Give the prompt as text or with --prompt-file, not both."
        )
    text = prompt_file.read_text() if prompt_file else str(prompt)
    if not text.strip():
        raise click.UsageError("The prompt is empty.")
    return text.strip()


def _check_references(
    photos: tuple[Path, ...], audios: tuple[Path, ...], videos: int
) -> None:
    # MiniMax H3's own limits for Ref2VA.
    if (
        len(photos) > 9
        or len(audios) + videos > 3
        or len(photos) + len(audios) + videos > 12
    ):
        raise click.UsageError(
            "H3 takes up to 9 photos, 3 audio clips and 12 files in all."
        )


def _labels(
    photos: tuple[Path, ...], source: Path | None, audios: tuple[Path, ...]
) -> str:
    """How the prompt refers to each input: photos, then the source video with its
    sound labelled just before it, then the other audio."""
    labels = [f"<Picture {i}> {p.name}" for i, p in enumerate(photos, 1)]
    sounds = 0
    if source:
        if has_audio(source):
            sounds += 1
            labels.append(f"<Audio {sounds}> sound of {source.name}")
        labels.append(f"<Video 1> {source.name}")
    labels += [f"<Audio {sounds + i}> {a.name}" for i, a in enumerate(audios, 1)]
    return ", ".join(labels)


def video_server() -> str:
    """The video profile's address; starts it first when it is not running."""
    if VIDEO_PROFILE not in load_aliases():
        raise click.ClickException(
            f"No '{VIDEO_PROFILE}' profile. Run 'lct setup --backend comfyui' first."
        )
    active = servers.running()
    if VIDEO_PROFILE in active:
        pid = active[VIDEO_PROFILE]
    else:
        gpu = servers.system()
        free = gpu.get("vram_total", 0) - gpu.get("vram_used", 0)
        if "vram_total" in gpu and free < VIDEO_VRAM:
            users = ", ".join(active) or "other programs"
            raise click.ClickException(
                f"The GPU has {free / 1024**3:.1f} GiB free; video needs about "
                f"{VIDEO_VRAM / 1024**3:.0f} GiB. In use by: {users}. "
                "Stop servers with 'lct down NAME'."
            )
        click.echo(f"Starting ComfyUI ('lct down {VIDEO_PROFILE}' stops it)…")
        pid = servers.start(VIDEO_PROFILE)
    while not (address := servers.url(VIDEO_PROFILE)):
        if not servers.alive(pid):
            raise click.ClickException(
                f"ComfyUI exited; see {servers.log_path(VIDEO_PROFILE)}"
            )
        time.sleep(1)
    return address


def _render(graph: dict, what: str) -> Path:
    """Run a workflow on the video server and return the clip it saved."""
    if missing := missing_for(graph):
        raise click.ClickException(
            f"Missing model files ({', '.join(missing)}). "
            "Run 'lct setup --backend comfyui' to download them."
        )
    started = time.monotonic()
    url = video_server()
    prompt_id = submit(url, graph)
    click.echo(f"{what}…")
    try:
        clip = wait(
            url,
            prompt_id,
            servers.log_path(VIDEO_PROFILE),
            lambda message: click.echo(f"  {message}"),
        )
    except KeyboardInterrupt:
        cancel(url, prompt_id)
        raise click.ClickException("Stopped; ComfyUI dropped the job.") from None
    except OSError as error:  # the server stopped answering
        if VIDEO_PROFILE not in servers.running():
            raise click.ClickException(
                f"ComfyUI crashed during the job; see {servers.log_path(VIDEO_PROFILE)}"
            ) from error
        raise
    minutes, seconds = divmod(round(time.monotonic() - started), 60)
    click.echo(f"  done in {minutes}m {seconds:02d}s: {clip}")
    return clip


def _finish(clip: Path, quality: str, out: Path | None, seed: int) -> None:
    """Upscale a 'high' clip, copy the result to --out and print where it is."""
    if quality == "high":
        clip = _render(
            upscale_graph(stage(clip), UPSCALE_SHORT_SIDE, seed, "upscale"),
            "Upscaling to 1080p",
        )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(clip, out)
        clip = out
    width, height, seconds = probe(clip)
    click.echo(f"Saved: {clip} ({width}x{height}, {seconds:.1f} s)")


def _video_errors(command: Callable[..., None]) -> Callable[..., None]:
    """Turn library errors into actionable messages, like every other command."""

    @functools.wraps(command)
    def wrapper(*args: object, **kwargs: object) -> None:
        try:
            command(*args, **kwargs)
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.CalledProcessError,
        ) as error:
            raise click.ClickException(str(error)) from error

    return wrapper


@main.group()
def video() -> None:
    """Make and edit video clips with sound (MiniMax H3 on ComfyUI).

    Clips are 4-15 s at 24 fps; join or extend them for longer videos.
    The first command starts ComfyUI as the 'video' profile; 'lct down video'
    frees the GPU afterwards.
    """


@video.command()
@click.argument("prompt", required=False)
@PROMPT_FILE
@click.option(
    "--seconds",
    type=click.FloatRange(MIN_SECONDS, MAX_SECONDS),
    default=5.0,
    show_default=True,
)
@QUALITY
@click.option(
    "--aspect", type=click.Choice(list(ASPECTS)), default="16:9", show_default=True
)
@PHOTOS
@AUDIOS
@click.option("--first-frame", type=INPUT_FILE, help="Image the clip starts from")
@click.option("--last-frame", type=INPUT_FILE, help="Image the clip ends on")
@SEED
@OUT
@_video_errors
def make(
    prompt: str | None,
    prompt_file: Path | None,
    seconds: float,
    quality: str,
    aspect: str,
    photos: tuple[Path, ...],
    audios: tuple[Path, ...],
    first_frame: Path | None,
    last_frame: Path | None,
    seed: int | None,
    out: Path | None,
) -> None:
    """Make a clip with sound from a description (PROMPT).

    With --photo or --audio, H3 copies who or what is in them (Ref2VA);
    otherwise it works from the text and any first or last frame.
    """
    text = _prompt_text(prompt, prompt_file)
    _check_references(photos, audios, 0)
    seed = random.randrange(2**32) if seed is None else seed
    if photos or audios:
        click.echo(f"References: {_labels(photos, None, audios)}")
    graph = h3_graph(
        text,
        canvas(ASPECTS[aspect], quality),
        frame_count(seconds),
        STEPS[quality],
        seed,
        "make",
        photos=tuple(stage(p) for p in photos),
        audios=tuple(stage(a) for a in audios),
        first_frame=stage(first_frame) if first_frame else None,
        last_frame=stage(last_frame) if last_frame else None,
    )
    _finish(
        _render(graph, f"Making a {seconds:g} s {quality} clip, seed {seed}"),
        quality,
        out,
        seed,
    )


@video.command()
@click.argument("source", type=INPUT_FILE)
@click.argument("prompt", required=False)
@PROMPT_FILE
@QUALITY
@PHOTOS
@AUDIOS
@SEED
@OUT
@_video_errors
def edit(
    source: Path,
    prompt: str | None,
    prompt_file: Path | None,
    quality: str,
    photos: tuple[Path, ...],
    audios: tuple[Path, ...],
    seed: int | None,
    out: Path | None,
) -> None:
    """Remake SOURCE as PROMPT says: new dialogue, a swapped person or object,
    another style or setting. The prompt calls SOURCE <Video 1>.

    Labels: photos are <Picture 1>…; SOURCE's own sound is <Audio 1> and
    --audio files follow it. The command prints them. Only the first 15 s of
    SOURCE are used.
    """
    text = _prompt_text(prompt, prompt_file)
    _check_references(photos, audios, 1)
    width, height, duration = probe(source)
    seed = random.randrange(2**32) if seed is None else seed
    click.echo(f"References: {_labels(photos, source, audios)}")
    graph = h3_graph(
        text,
        canvas(width / height, quality),
        frame_count(min(duration, MAX_SECONDS)),
        STEPS[quality],
        seed,
        "edit",
        photos=tuple(stage(p) for p in photos),
        videos=(stage_video(source),),
        audios=tuple(stage(a) for a in audios),
    )
    _finish(_render(graph, f"Editing {source.name}, seed {seed}"), quality, out, seed)


@video.command()
@click.argument("source", type=INPUT_FILE)
@click.argument("prompt", required=False)
@PROMPT_FILE
@click.option(
    "--seconds",
    type=click.FloatRange(MIN_SECONDS, MAX_SECONDS - 1),
    default=5.0,
    show_default=True,
    help="Length of the new part",
)
@QUALITY
@PHOTOS
@AUDIOS
@SEED
@OUT
@_video_errors
def extend(
    source: Path,
    prompt: str | None,
    prompt_file: Path | None,
    seconds: float,
    quality: str,
    photos: tuple[Path, ...],
    audios: tuple[Path, ...],
    seed: int | None,
    out: Path | None,
) -> None:
    """Continue SOURCE: a new part starts from its last second, so motion and
    sound carry on, and is joined to it. PROMPT describes only what happens next.

    Pass the same --photo files as before to keep people looking the same.
    """
    text = _prompt_text(prompt, prompt_file)
    _check_references(photos, audios, 0)
    width, height, _ = probe(source)
    seed = random.randrange(2**32) if seed is None else seed
    if photos or audios:
        click.echo(f"References: {_labels(photos, None, audios)}")
    graph = h3_graph(
        text,
        canvas(width / height, quality),
        frame_count(seconds + GUIDE_FRAMES / 24),
        STEPS[quality],
        seed,
        "extend-part",
        photos=tuple(stage(p) for p in photos),
        audios=tuple(stage(a) for a in audios),
        guide=stage_tail(source),
    )
    part = _render(
        graph, f"Making the next {seconds:g} s of {source.name}, seed {seed}"
    )
    joined = join(
        [source, part],
        output_dir() / f"extend-{time.strftime('%Y%m%d-%H%M%S')}.mp4",
        skip=GUIDE_FRAMES,
    )
    _finish(joined, quality, out, seed)


@video.command()
@click.argument("source", type=INPUT_FILE)
@click.option(
    "--short-side",
    type=click.IntRange(256, 2160),
    default=UPSCALE_SHORT_SIDE,
    show_default=True,
    help="Pixels on the short side: 1080 for 1080p, 2160 for 4K",
)
@SEED
@OUT
@_video_errors
def upscale(source: Path, short_side: int, seed: int | None, out: Path | None) -> None:
    """Enlarge SOURCE and restore its detail with SeedVR2; the sound is kept."""
    seed = random.randrange(2**32) if seed is None else seed
    graph = upscale_graph(stage(source), short_side, seed, "upscale")
    _finish(
        _render(graph, f"Upscaling {source.name} to {short_side}p"),
        "standard",
        out,
        seed,
    )


@video.command(name="join")
@click.argument("clips", nargs=-1, required=True, type=INPUT_FILE)
@click.option("--out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@_video_errors
def join_clips(clips: tuple[Path, ...], out: Path) -> None:
    """Join CLIPS end to end with hard cuts, at the first clip's size."""
    join(list(clips), out)
    width, height, seconds = probe(out)
    click.echo(f"Saved: {out} ({width}x{height}, {seconds:.1f} s)")


@video.command()
@click.option("--ref", is_flag=True, help="The guide for --photo, --audio and edit")
def guide(ref: bool) -> None:
    """Print MiniMax's official H3 prompt-writing guide."""
    path = model_path("guide_ref" if ref else "guide")
    if not path.is_file():
        raise click.ClickException("Run 'lct setup --backend comfyui' to download it.")
    click.echo(path.read_text())


@video.command(name="status")
def video_status() -> None:
    """Show whether ComfyUI, its model files and the video server are ready."""
    installed = (data_dir().parent / "venv" / "bin" / "python").is_file()
    click.echo(f"ComfyUI    {'installed' if installed else 'not installed'}")
    profile = VIDEO_PROFILE in load_aliases()
    click.echo(f"Profile    {VIDEO_PROFILE if profile else 'missing'}")
    missing = missing_files()
    click.echo(
        f"Models     {len(FILES) - len(missing)}/{len(FILES)} files"
        + (f", missing: {', '.join(missing)}" if missing else "")
    )
    pid = servers.running().get(VIDEO_PROFILE)
    address = servers.url(VIDEO_PROFILE) if pid else ""
    click.echo(f"Server     {address or ('loading' if pid else 'stopped')}")
    gpu = servers.system()
    if "vram_total" in gpu:
        free = (gpu["vram_total"] - gpu["vram_used"]) / 1024**3
        click.echo(f"GPU        {free:.1f} GiB VRAM free")
    click.echo(f"Library    {output_dir()}")
    if not (installed and profile and not missing):
        click.echo("Run 'lct setup --backend comfyui' to finish setting up.")


@main.command()
def tui() -> None:
    """Open the terminal UI."""
    # Imported here so plain CLI commands do not pay for loading Textual.
    from llamacpp_tuner.tui import LctApp

    app = LctApp()
    # A closed terminal (SIGHUP) or kill (SIGTERM) must still stop downloads and
    # benchmarks.
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.exit())
    try:
        app.run()
    finally:
        app.stop_children()


if __name__ == "__main__":
    main()
