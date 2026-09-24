"""Command-line interface for lct."""

import json
import shlex
import subprocess
import tomllib
from pathlib import Path

import click

from llamacpp_tuner import __version__
from llamacpp_tuner.cache import get_aliases_path, get_models_dir
from llamacpp_tuner.downloader import (
    download_mmproj,
    download_model,
    get_mmproj_path,
    list_downloaded_models,
    resolve_model,
)
from llamacpp_tuner.llama import install_llama, run_server


def load_aliases() -> dict[str, str]:
    """Read serve aliases, each mapping a name to 'serve' arguments."""
    path = get_aliases_path()
    if not path.is_file():
        return {}
    try:
        aliases = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise click.ClickException(f"Invalid {path}: {error}") from error
    for name, value in aliases.items():
        if not isinstance(value, str):
            raise click.ClickException(
                f"Alias '{name}' in {path} must be a string of serve arguments."
            )
    return aliases


def write_aliases(aliases: dict[str, str]) -> None:
    """Replace every alias. Rewriting the file drops its comments."""
    get_aliases_path().write_text(
        "".join(
            f"{json.dumps(key)} = {json.dumps(value)}\n"
            for key, value in aliases.items()
        )
    )


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
@click.option("--force", is_flag=True, help="Rebuild the managed llama.cpp checkout")
@click.option(
    "--cmake-arg",
    multiple=True,
    help="Additional CMake argument; use --cmake-arg=-DNAME=VALUE",
)
def setup(force: bool, cmake_arg: tuple[str, ...]) -> None:
    """Find llama-server or build llama.cpp from source."""
    try:
        binary = install_llama(force=force, extra_cmake_args=cmake_arg)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"llama-server: {binary}")


@main.command()
@click.argument("repo_id")
@click.option("--quant", "-q", help="Exact quant name, such as Q6_K")
@click.option("--file", "filename", help="Exact GGUF filename instead of --quant")
@click.option("--force", "-f", is_flag=True, help="Re-download existing files")
@click.option("--no-mmproj", is_flag=True, help="Do not download a vision projector")
@click.option("--mmproj-file", help="Exact projector filename")
def pull(
    repo_id: str,
    quant: str | None,
    filename: str | None,
    force: bool,
    no_mmproj: bool,
    mmproj_file: str | None,
) -> None:
    """Download an exact GGUF artifact from Hugging Face."""
    if bool(quant) == bool(filename):
        raise click.UsageError("Provide exactly one of --quant or --file.")
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
) -> None:
    """Run llama-server, overriding only explicitly supplied settings.

    MODEL is a GGUF path, a downloaded repository, or an alias from aliases.toml.
    """
    if quant and filename:
        raise click.UsageError("--quant and --file cannot be combined.")
    if mmproj and no_mmproj:
        raise click.UsageError("--mmproj and --no-mmproj cannot be combined.")

    try:
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
        run_server(server_args)
    except click.ClickException:
        raise
    except (FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    except subprocess.CalledProcessError as error:
        raise click.ClickException(
            f"llama-server exited with code {error.returncode}."
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
def tui() -> None:
    """Open the terminal UI."""
    # Imported here so plain CLI commands do not pay for loading Textual.
    from llamacpp_tuner.tui import LctApp

    LctApp().run()


if __name__ == "__main__":
    main()
