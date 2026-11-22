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
from collections.abc import Callable
from pathlib import Path

import click

from llamacpp_tuner import __version__, servers
from llamacpp_tuner.backends import BACKENDS
from llamacpp_tuner.backends.base import ServeOptions, UnsupportedOptionError
from llamacpp_tuner.backends.comfyui import data_dir
from llamacpp_tuner.cache import get_models_dir, get_profiles_path
from llamacpp_tuner.downloader import (
    download_mmproj,
    download_model,
    download_repository,
    list_downloaded_models,
    resolve_repository,
)
from llamacpp_tuner.logs import SOURCES
from llamacpp_tuner.lora import adapter_base, import_lora
from llamacpp_tuner.profiles import (
    ProfileError,
    expand_names,
    load_profiles,
    write_profiles,
)
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
    download_files,
    frame_count,
    h3_graph,
    has_audio,
    join,
    missing_files,
    missing_for,
    model_path,
    output_dir,
    probe,
    stage,
    stage_tail,
    stage_video,
    submit,
    upscale_graph,
    wait,
)

BACKEND = click.option(
    "--backend",
    type=click.Choice(list(BACKENDS)),
    default="llama",
    help="llama.cpp, Jeeves's own PyTorch server, Strata, Loom, or ComfyUI for video",
)
# The profile the Video tab and 'lct video' start; 'lct setup --backend comfyui' adds it.
VIDEO_PROFILE = "video"


class _ProfileGroup(click.Group):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # Expand before parsing so options typed after a profile override its own.
        try:
            if (
                len(args) > 1
                and args[0] == "serve"
                and (saved := load_profiles().get(args[1])) is not None
            ):
                args = ["serve", *shlex.split(saved), *args[2:]]
        except ProfileError as error:
            raise click.ClickException(str(error)) from error
        return super().parse_args(ctx, args)

    def invoke(self, ctx: click.Context) -> object:
        try:
            return super().invoke(ctx)
        except ProfileError as error:
            raise click.ClickException(str(error)) from error


@click.group(cls=_ProfileGroup)
@click.version_option(version=__version__)
def main() -> None:
    """Download models and serve them with llama.cpp, Strata, Loom, Jeeves or
    ComfyUI."""


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
    """Find or download llama-server for this GPU (--cmake-arg builds it), build
    Strata or Loom from engines/, or install Jeeves, or ComfyUI with a 'video'
    profile."""
    try:
        click.echo(BACKENDS[backend].setup(force, cmake_arg, torch_index))
        if backend == "comfyui":
            profiles = load_profiles()
            if VIDEO_PROFILE not in profiles:
                data_dir().mkdir(parents=True, exist_ok=True)
                args = [str(data_dir()), "--backend", "comfyui", "--port", "8188"]
                write_profiles({**profiles, VIDEO_PROFILE: shlex.join(args)})
                click.echo(f"Added profile '{VIDEO_PROFILE}' to {get_profiles_path()}")
            if missing := missing_files():
                click.echo(f"Downloading {len(missing)} model files…")
                download_files(missing)
            click.echo("Ready: lct video make 'a description'")
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
@click.option(
    "--keep-awake/--no-keep-awake",
    default=True,
    help="Keep AMD GPUs powered while the server runs (default); an idle card "
    "with no display otherwise suspends and moves a Vulkan server's model to RAM",
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
    keep_awake: bool,
    backend: str,
) -> None:
    """Run llama-server, overriding only explicitly supplied settings.

    MODEL is a GGUF path, a downloaded repository, or a profile name.
    With --backend jeeves, --ctx is Jeeves's --max-len and --extra-args go to
    its server. With --backend strata, MODEL is the strata-<model>.json its setup
    wrote; context and files live in that config. With --backend loom, MODEL is a
    Loom model package directory; its preset (--extra-args '--preset=NAME') sets
    context and drafting. With --backend comfyui, MODEL is ComfyUI's base
    directory (models, input, output).
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
    if keep_awake:
        # A server that cannot keep its GPU awake still runs; it only slows after idle gaps.
        try:
            if held := servers.keep_awake():
                gpus = "GPU" if held == 1 else "GPUs"
                click.echo(
                    f"Keeping {held} AMD {gpus} powered while the server runs.",
                    err=True,
                )
        except RuntimeError as error:
            click.echo(f"Warning: {error}", err=True)

    options = ServeOptions(
        quant=quant,
        filename=filename,
        ctx=ctx,
        host=host,
        port=port,
        mmproj=mmproj,
        no_mmproj=no_mmproj,
        no_mmproj_offload=no_mmproj_offload,
        extra_args=extra_args,
        server=server,
    )
    try:
        BACKENDS[backend].serve(model, options, environment)
    except UnsupportedOptionError as error:
        raise click.UsageError(str(error)) from error
    except (FileNotFoundError, ValueError, RuntimeError) as error:
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


@main.command(name="import")
@click.argument(
    "adapter", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--name", required=True, help="Profile to save the model as")
@click.option(
    "--quant", "-q", required=True, help="llama-quantize type, such as Q4_K_M"
)
@click.option(
    "--base",
    help="Base model directory or downloaded repository; "
    "default: the adapter's base_model_name_or_path",
)
@click.option("--force", "-f", is_flag=True, help="Replace the model and profile")
def import_adapter(
    adapter: Path, name: str, quant: str, base: str | None, force: bool
) -> None:
    """Merge a PEFT LoRA ADAPTER into its base model, quantize it, save a profile.

    The base model's Hugging Face weights must be local: lct pull REPO --all.
    """
    profiles = load_profiles()
    out = get_models_dir() / "imported" / f"{name}-{quant}.gguf"
    if not force and (name in profiles or out.exists()):
        raise click.ClickException(f"{name} already exists; --force replaces it.")
    try:
        base_dir = resolve_repository(base or adapter_base(adapter))
        model = import_lora(adapter, base_dir, quant, out)
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    except subprocess.CalledProcessError as error:
        raise click.ClickException(
            f"Exit code {error.returncode} from: {shlex.join(error.cmd)}"
        ) from error
    write_profiles({**profiles, name: shlex.join([str(model)])})
    click.echo(f"Model: {model}\nProfile: {name} (lct serve {name})")


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
    if VIDEO_PROFILE not in load_profiles():
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
    profile = VIDEO_PROFILE in load_profiles()
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
