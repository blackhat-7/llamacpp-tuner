"""Run Strata (Niko1221/Strata), an engine built for Qwen3.8-Flash-Next alone.

The repository's engines/strata/build.sh builds it with our changes. A run config
(strata-<model>.json) names the model and engine; lct only starts that config.
"""

import json
import shlex
from pathlib import Path

from llamacpp_tuner.backends.base import (
    ServeOptions,
    UnsupportedOptionError,
    build_engine,
    run_process,
)


def run_strata(
    config: Path, args: list[str], env: dict[str, str] | None = None
) -> None:
    """Run Strata's OpenAI-compatible server for a run config in the foreground."""
    # Strata saves Settings edits by renaming a new file over --config. Given a
    # symlink, that would replace the link with a copy and leave the real config
    # unserved, so pass the file the link points at.
    config = config.resolve()
    if not config.is_file():
        raise FileNotFoundError(
            f"No Strata run config at {config}. Build one with "
            "'lct setup --backend strata' (engines/strata/build.sh)."
        )
    # The Strata checkout; a config's "cwd" lets configs live apart from it.
    cwd = json.loads(config.read_text()).get("cwd")
    folder = Path(cwd).expanduser() if cwd else config.parent
    python = folder / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"Strata's venv is missing in {folder}. Run 'lct setup --backend strata'."
        )
    run_process(
        [
            str(python),
            str(folder / "serve" / "server.py"),
            "--engine",
            "strata",
            "--config",
            str(config),
            *args,
        ],
        env=env,
        cwd=folder,
    )


def serve(model: str, options: ServeOptions, env: dict[str, str]) -> None:
    """Run Strata for a run config; MODEL is the config's path."""
    if options.picks_gguf:
        raise UnsupportedOptionError(
            "Strata's model, context and projector are set in its run config."
        )
    args = []
    for flag, value in (("--host", options.host), ("--port", options.port)):
        if value is not None:
            args.extend([flag, str(value)])
    run_strata(
        Path(model).expanduser(), args + shlex.split(options.extra_args), env=env
    )


def setup(force: bool, cmake_args: tuple[str, ...], torch_index: str | None) -> str:
    return build_engine("strata", force)


def address(line: str) -> str:
    # "ready: http://host:port/v1  (...)"
    if line.startswith("ready: http"):
        return line.split()[1].removesuffix("/v1")
    return ""
