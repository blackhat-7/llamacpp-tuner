"""Tests for lct import: a LoRA adapter merged, quantized and saved as a profile."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from llamacpp_tuner.cli import load_aliases, main, write_aliases


def _setup(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    monkeypatch.setenv("LCT_HOME", str(tmp_path))
    source = tmp_path / "llama.cpp"
    source.mkdir()
    (source / "convert_lora_to_gguf.py").touch()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    config = {"base_model_name_or_path": str(base)}
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    return adapter, base


def _fake_run(commands: list[list[str]]):
    def run(command: list[str], check: bool) -> None:
        commands.append(command)
        if "llama-quantize" in command[0]:
            Path(command[2]).write_text("gguf")

    return run


def test_import_merges_quantizes_and_saves_a_profile(tmp_path, monkeypatch):
    adapter, base = _setup(tmp_path, monkeypatch)
    commands: list[list[str]] = []
    with (
        patch("llamacpp_tuner.lora.get_llama_binary", side_effect=Path),
        patch("llamacpp_tuner.lora.subprocess.run", side_effect=_fake_run(commands)),
    ):
        result = CliRunner().invoke(
            main, ["import", str(adapter), "--name", "style", "-q", "IQ4_NL"]
        )

    assert result.exit_code == 0, result.output
    out = tmp_path / "models" / "imported" / "style-IQ4_NL.gguf"
    assert out.read_text() == "gguf"
    assert load_aliases() == {"style": str(out)}
    convert_base, convert_lora, export, quantize = commands
    assert convert_base[-6:-4] == [
        str(tmp_path / "llama.cpp/convert_hf_to_gguf.py"),
        str(base),
    ]
    assert convert_lora[-8:-6] == [
        str(tmp_path / "llama.cpp/convert_lora_to_gguf.py"),
        str(adapter),
    ]
    assert export[0] == "llama-export-lora"
    assert export[2] == convert_base[-1] and export[4] == convert_lora[-1]
    assert quantize[0] == "llama-quantize" and quantize[1] == export[-1]
    assert quantize[-1] == "IQ4_NL"
    # The scratch f16 files are gone.
    assert list(out.parent.iterdir()) == [out]


def test_import_keeps_an_existing_profile_without_force(tmp_path, monkeypatch):
    adapter, _ = _setup(tmp_path, monkeypatch)
    write_aliases({"style": "other.gguf"})
    with patch("llamacpp_tuner.lora.subprocess.run") as run:
        result = CliRunner().invoke(
            main, ["import", str(adapter), "--name", "style", "-q", "Q4_K_M"]
        )

    assert result.exit_code != 0 and "--force" in result.output
    run.assert_not_called()
    assert load_aliases() == {"style": "other.gguf"}


def test_import_failure_saves_nothing(tmp_path, monkeypatch):
    adapter, _ = _setup(tmp_path, monkeypatch)
    failure = subprocess.CalledProcessError(1, ["llama-quantize", "a", "b", "BAD"])
    with (
        patch("llamacpp_tuner.lora.get_llama_binary", side_effect=Path),
        patch("llamacpp_tuner.lora.subprocess.run", side_effect=failure),
    ):
        result = CliRunner().invoke(
            main, ["import", str(adapter), "--name", "style", "-q", "BAD"]
        )

    assert result.exit_code != 0 and "llama-quantize a b BAD" in result.output
    assert load_aliases() == {}
    assert list((tmp_path / "models" / "imported").iterdir()) == []


def test_import_needs_a_downloaded_base(tmp_path, monkeypatch):
    adapter, _ = _setup(tmp_path, monkeypatch)
    config = {"base_model_name_or_path": "Qwen/Qwen3-4B"}
    (adapter / "adapter_config.json").write_text(json.dumps(config))

    result = CliRunner().invoke(
        main, ["import", str(adapter), "--name", "style", "-q", "Q4_K_M"]
    )

    assert result.exit_code != 0
    assert "lct pull Qwen/Qwen3-4B --all" in result.output
