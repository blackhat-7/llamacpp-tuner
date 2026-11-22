"""Tests for profiles: ${LCT_HOME} in the file, the cache dir in memory."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from llamacpp_tuner.cli import main
from llamacpp_tuner.profiles import load_profiles, load_stacks, write_profiles

# conftest sets LCT_HOME to tmp_path, so profiles live in tmp_path/aliases.toml.


def test_reading_expands_lct_home(tmp_path):
    (tmp_path / "aliases.toml").write_text(
        '"a" = "${LCT_HOME}/models/m.gguf --ctx 8"\n\n[stacks]\n"s" = ["a"]\n'
    )

    assert load_profiles() == {"a": f"{tmp_path}/models/m.gguf --ctx 8"}
    assert load_stacks() == {"s": ["a"]}


def test_writing_collapses_paths_under_the_cache_dir(tmp_path):
    write_profiles(
        {
            "draft": f"{tmp_path}/models/t.gguf --extra-args "
            f"'-ngl 99 -md {tmp_path}/models/x.gguf'",
            "video": f"{tmp_path} --backend comfyui",
            "other": f"{tmp_path}2/models/m.gguf",
        }
    )

    assert (tmp_path / "aliases.toml").read_text() == (
        '"draft" = "${LCT_HOME}/models/t.gguf --extra-args '
        "'-ngl 99 -md ${LCT_HOME}/models/x.gguf'\"\n"
        '"video" = "${LCT_HOME} --backend comfyui"\n'
        f'"other" = "{tmp_path}2/models/m.gguf"\n'
    )


def test_a_round_trip_keeps_the_file_unchanged(tmp_path):
    text = (
        '"a" = "${LCT_HOME}/models/m.gguf --extra-args \'-md ${LCT_HOME}/d.gguf\'"\n'
        '"b" = "/elsewhere/m.gguf"\n'
        "\n[stacks]\n"
        '"both" = ["a", "b"]\n'
    )
    (tmp_path / "aliases.toml").write_text(text)

    write_profiles(load_profiles())

    assert (tmp_path / "aliases.toml").read_text() == text


def test_serve_gets_the_expanded_arguments(tmp_path):
    (tmp_path / "aliases.toml").write_text(
        '"draft" = "${LCT_HOME}/models/t.gguf --extra-args \'-md ${LCT_HOME}/x.gguf\'"\n'
    )
    with (
        patch(
            "llamacpp_tuner.backends.llama.resolve_model",
            side_effect=lambda model, **_: Path(model),
        ),
        patch("llamacpp_tuner.backends.llama.run_server") as run_server,
    ):
        result = CliRunner().invoke(main, ["serve", "draft"])

    assert result.exit_code == 0, result.output
    assert run_server.call_args.args[0] == [
        "-m",
        f"{tmp_path}/models/t.gguf",
        "-md",
        f"{tmp_path}/x.gguf",
    ]
