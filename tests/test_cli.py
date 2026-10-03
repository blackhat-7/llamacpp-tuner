"""Tests for the lct CLI."""

import sys
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from llamacpp_tuner import servers
from llamacpp_tuner.cli import load_aliases, load_stacks, main, write_aliases


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


def test_serve_expands_alias_and_later_options_override(monkeypatch, tmp_path):
    aliases = tmp_path / "aliases.toml"
    aliases.write_text(
        "qwen = \"owner/repo --file model.gguf --ctx 4096 --extra-args '-ngl all -fa on'\"\n"
    )
    monkeypatch.setattr("llamacpp_tuner.cli.get_aliases_path", lambda: aliases)

    with (
        patch(
            "llamacpp_tuner.cli.resolve_model",
            return_value=Path("models/model.gguf"),
        ) as resolve,
        patch("llamacpp_tuner.cli.get_mmproj_path", return_value=None),
        patch("llamacpp_tuner.cli.run_server") as run_server,
    ):
        result = CliRunner().invoke(main, ["serve", "qwen", "--ctx", "8192"])

    assert result.exit_code == 0
    resolve.assert_called_once_with("owner/repo", quant=None, filename="model.gguf")
    assert run_server.call_args.args[0] == [
        "-m",
        "models/model.gguf",
        "-c",
        "8192",
        "-ngl",
        "all",
        "-fa",
        "on",
    ]


def test_invalid_alias_file_is_a_user_error(monkeypatch, tmp_path):
    aliases = tmp_path / "aliases.toml"
    aliases.write_text("qwen = [\n")
    monkeypatch.setattr("llamacpp_tuner.cli.get_aliases_path", lambda: aliases)

    result = CliRunner().invoke(main, ["serve", "qwen"])

    assert result.exit_code != 0
    assert "Invalid" in result.output


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


FAKE_SERVER = """
import time
print("main: server is listening on http://127.0.0.1:9", flush=True)
try:
    time.sleep(30)
except KeyboardInterrupt:
    pass
"""


def test_up_starts_a_stack_waits_and_down_stops_it(monkeypatch, tmp_path):
    aliases = tmp_path / "aliases.toml"
    aliases.write_text('a = "m.gguf"\nb = "m.gguf"\n\n[stacks]\nboth = ["a", "b"]\n')
    monkeypatch.setenv("LCT_HOME", str(tmp_path))
    monkeypatch.setattr("llamacpp_tuner.cli.get_aliases_path", lambda: aliases)
    monkeypatch.setattr(
        "llamacpp_tuner.servers.LCT", [sys.executable, "-c", FAKE_SERVER]
    )
    try:
        result = CliRunner().invoke(main, ["up", "both"])
        assert result.exit_code == 0, result.output
        assert "a: http://127.0.0.1:9" in result.output
        assert set(servers.running()) == {"a", "b"}
        assert "b " in CliRunner().invoke(main, ["ps"]).output

        result = CliRunner().invoke(main, ["down"])
        assert "a: stopped" in result.output and "b: stopped" in result.output
        assert servers.running() == {}
    finally:
        for pid in servers.running().values():
            servers.interrupt(pid)


def test_up_rejects_unknown_names(monkeypatch, tmp_path):
    aliases = tmp_path / "aliases.toml"
    aliases.write_text('a = "m.gguf"\n[stacks]\nbad = ["a", "nope"]\n')
    monkeypatch.setattr("llamacpp_tuner.cli.get_aliases_path", lambda: aliases)

    result = CliRunner().invoke(main, ["up", "bad"])

    assert result.exit_code != 0
    assert "Unknown profile or stack: nope" in result.output


def test_writing_aliases_keeps_stacks(monkeypatch, tmp_path):
    aliases = tmp_path / "aliases.toml"
    aliases.write_text('a = "m.gguf"\n[stacks]\nall = ["a"]\n')
    monkeypatch.setattr("llamacpp_tuner.cli.get_aliases_path", lambda: aliases)

    write_aliases({"a": "m.gguf", "b": "n.gguf"})

    assert load_aliases() == {"a": "m.gguf", "b": "n.gguf"}
    assert load_stacks() == {"all": ["a"]}


def test_usage_and_system_read_live_numbers():
    import os

    used = servers.usage(os.getpid())
    assert used.cpu_seconds > 0 and used.ram > 0
    system = servers.system()
    assert 0 < system["ram_used"] < system["ram_total"]
    assert system["cpu_busy"] <= system["cpu_total"]


def test_serve_passes_env_to_llama_server(monkeypatch):
    with (
        patch("llamacpp_tuner.cli.resolve_model", return_value=Path("m.gguf")),
        patch("llamacpp_tuner.cli.run_server") as run_server,
    ):
        result = CliRunner().invoke(
            main, ["serve", "m.gguf", "--env", "A=1", "--env", "B=x=y"]
        )
        bad = CliRunner().invoke(main, ["serve", "m.gguf", "--env", "A"])

    assert result.exit_code == 0
    assert run_server.call_args.kwargs["env"] == {"A": "1", "B": "x=y"}
    assert bad.exit_code != 0 and "NAME=VALUE" in bad.output


def test_serve_jeeves_maps_common_args(tmp_path):
    with patch("llamacpp_tuner.cli.run_jeeves") as run_jeeves:
        result = CliRunner().invoke(
            main,
            ["serve", str(tmp_path), "--backend", "jeeves", "--ctx", "2048"]
            + ["--port", "6871", "--no-mmproj", "--extra-args", "--max-rows 2"],
        )
        bad = CliRunner().invoke(
            main, ["serve", str(tmp_path), "--backend", "jeeves", "-q", "Q4_K_M"]
        )

    assert result.exit_code == 0, result.output
    assert run_jeeves.call_args.args[0] == [
        "--model",
        str(tmp_path),
        "--drafter",
        str(tmp_path / "drafter_k4.safetensors"),
        "--max-len",
        "2048",
        "--port",
        "6871",
        "--max-rows",
        "2",
    ]
    assert bad.exit_code != 0 and "whole model directory" in bad.output


def test_serve_strata_runs_its_config(tmp_path):
    config = tmp_path / "strata-iq3_s.json"
    with patch("llamacpp_tuner.cli.run_strata") as run_strata:
        result = CliRunner().invoke(
            main,
            ["serve", str(config), "--backend", "strata", "--host", "100.64.0.1"]
            + ["--port", "6868", "--extra-args", "--api-key k"],
        )
        bad = CliRunner().invoke(
            main, ["serve", str(config), "--backend", "strata", "--ctx", "8192"]
        )

    assert result.exit_code == 0, result.output
    assert run_strata.call_args.args == (
        config,
        ["--host", "100.64.0.1", "--port", "6868", "--api-key", "k"],
    )
    assert bad.exit_code != 0 and "run config" in bad.output


def test_strata_needs_its_own_setup(tmp_path):
    missing = CliRunner().invoke(
        main, ["serve", str(tmp_path / "none.json"), "--backend", "strata"]
    )
    setup_result = CliRunner().invoke(main, ["setup", "--backend", "strata"])

    assert missing.exit_code != 0 and "./setup.sh" in missing.output
    assert setup_result.exit_code != 0 and "./setup.sh" in setup_result.output


def test_pull_all_downloads_the_whole_repository():
    with patch(
        "llamacpp_tuner.cli.download_repository", return_value=Path("models/o%2Fr")
    ) as download:
        result = CliRunner().invoke(main, ["pull", "o/r", "--all"])

    assert result.exit_code == 0
    download.assert_called_once_with("o/r", force=False)


def test_address_reads_llama_jeeves_and_strata_lines():
    assert servers.address("main: server is listening on http://h:1") == "http://h:1"
    assert servers.address('{"serving": "http://h:2", "block": 4}') == "http://h:2"
    strata = (
        "ready: http://h:3/v1  (OpenAI: /v1/chat/completions, context 131072 tokens)"
    )
    assert servers.address(strata) == "http://h:3"
    assert servers.address("loading model") == ""
