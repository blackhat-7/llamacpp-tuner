"""Tests for the lct CLI."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from llamacpp_tuner.cli import main


def test_setup_reports_binary():
    with patch(
        "llamacpp_tuner.cli.install_llama", return_value=Path("/bin/llama-server")
    ):
        result = CliRunner().invoke(main, ["setup"])

    assert result.exit_code == 0
    assert "/bin/llama-server" in result.output


def test_pull_requires_quant_or_file():
    result = CliRunner().invoke(main, ["pull", "owner/repo"])

    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_pull_accepts_any_quant():
    with (
        patch(
            "llamacpp_tuner.cli.download_model",
            return_value=Path("models/owner/repo/model-Q6_K.gguf"),
        ) as download,
        patch("llamacpp_tuner.cli.download_mmproj", return_value=None),
    ):
        result = CliRunner().invoke(main, ["pull", "owner/repo", "--quant", "Q6_K"])

    assert result.exit_code == 0
    download.assert_called_once_with(
        "owner/repo", quant="Q6_K", filename=None, force=False
    )


def test_serve_preserves_llama_defaults():
    with (
        patch(
            "llamacpp_tuner.cli.resolve_model",
            return_value=Path("models/model-Q6_K.gguf"),
        ),
        patch("llamacpp_tuner.cli.run_server") as run_server,
    ):
        result = CliRunner().invoke(
            main,
            [
                "serve",
                "model-Q6_K.gguf",
                "--extra-args",
                '-ngl all -ot "blk.*=CUDA0"',
            ],
        )

    assert result.exit_code == 0
    assert run_server.call_args.args[0] == [
        "-m",
        "models/model-Q6_K.gguf",
        "-ngl",
        "all",
        "-ot",
        "blk.*=CUDA0",
    ]


def test_serve_adds_only_explicit_common_args():
    with (
        patch(
            "llamacpp_tuner.cli.resolve_model",
            return_value=Path("models/model.gguf"),
        ),
        patch("llamacpp_tuner.cli.run_server") as run_server,
    ):
        result = CliRunner().invoke(
            main,
            [
                "serve",
                "model.gguf",
                "--ctx",
                "32768",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
            ],
        )

    assert result.exit_code == 0
    assert run_server.call_args.args[0] == [
        "-m",
        "models/model.gguf",
        "-c",
        "32768",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]


def test_serve_supports_explicit_projector(tmp_path):
    projector = tmp_path / "mmproj.gguf"
    projector.touch()

    with (
        patch(
            "llamacpp_tuner.cli.resolve_model",
            return_value=Path("models/model.gguf"),
        ),
        patch("llamacpp_tuner.cli.run_server") as run_server,
    ):
        result = CliRunner().invoke(
            main,
            [
                "serve",
                "model.gguf",
                "--mmproj",
                str(projector),
                "--no-mmproj-offload",
            ],
        )

    assert result.exit_code == 0
    args = run_server.call_args.args[0]
    assert args[args.index("--mmproj") + 1] == str(projector)
    assert "--no-mmproj-offload" in args


def test_models_lists_namespaced_path(monkeypatch, tmp_path):
    root = tmp_path / "models"
    model = root / "owner" / "repo" / "model.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x" * 1024)
    monkeypatch.setattr("llamacpp_tuner.cli.get_models_dir", lambda: root)
    monkeypatch.setattr("llamacpp_tuner.cli.list_downloaded_models", lambda: [model])

    result = CliRunner().invoke(main, ["models"])

    assert result.exit_code == 0
    assert "owner/repo/model.gguf" in result.output
