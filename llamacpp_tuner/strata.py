"""Run Strata (Niko1221/Strata), an engine built for Qwen3.8-Flash-Next alone.

Its own ./setup.sh compiles the engine, prepares the model and writes a run
config (strata-<model>.json) in its folder; lct only starts that config.
"""

from pathlib import Path

from llamacpp_tuner.llama import run_process


def run_strata(
    config: Path, args: list[str], env: dict[str, str] | None = None
) -> None:
    """Run Strata's OpenAI-compatible server for a run config in the foreground."""
    if not config.is_file():
        raise FileNotFoundError(
            f"No Strata run config at {config}. Run Strata's ./setup.sh first."
        )
    folder = config.parent
    python = folder / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"Strata's venv is missing in {folder}. Run its ./setup.sh."
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
