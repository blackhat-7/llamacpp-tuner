"""Command-line interface for lct."""

import json
import shlex
import signal
import subprocess
import time
import tomllib
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
from llamacpp_tuner.llama import install_llama, run_server

BACKEND = click.option(
    "--backend",
    type=click.Choice(["llama", "jeeves"]),
    default="llama",
    help="llama.cpp, or Jeeves's own PyTorch server",
)


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
@click.option("--force", is_flag=True, help="Rebuild or reinstall the backend")
@click.option(
    "--cmake-arg",
    multiple=True,
    help="Additional CMake argument; use --cmake-arg=-DNAME=VALUE",
)
@click.option(
    "--torch-index",
    help="PyTorch wheel index for Jeeves, such as "
    "https://download.pytorch.org/whl/rocm7.2; default is PyPI (CUDA)",
)
def setup(
    backend: str, force: bool, cmake_arg: tuple[str, ...], torch_index: str | None
) -> None:
    """Find llama-server or build llama.cpp from source, or install Jeeves."""
    try:
        if backend == "jeeves":
            click.echo(f"Jeeves: {install_jeeves(torch_index, force=force)}")
        else:
            binary = install_llama(force=force, extra_cmake_args=cmake_arg)
            click.echo(f"llama-server: {binary}")
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
    backend: str,
) -> None:
    """Run llama-server, overriding only explicitly supplied settings.

    MODEL is a GGUF path, a downloaded repository, or an alias from aliases.toml.
    With --backend jeeves, --ctx is Jeeves's --max-len and --extra-args go to
    its server.
    """
    if quant and filename:
        raise click.UsageError("--quant and --file cannot be combined.")
    if mmproj and no_mmproj:
        raise click.UsageError("--mmproj and --no-mmproj cannot be combined.")
    if bad := [item for item in env if "=" not in item]:
        raise click.UsageError(f"--env needs NAME=VALUE, got: {', '.join(bad)}")

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
        run_server(server_args, env=environment)
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
