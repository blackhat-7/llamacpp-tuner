"""Integration tests for CLI commands."""

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from llamacpp_tuner.cli import main


class TestSetupCommand:
    def test_setup_without_hardware_cached(self, tmp_path, monkeypatch):
        runner = CliRunner()

        # Mock cache to return no hardware
        monkeypatch.setattr(
            "llamacpp_tuner.cli.load_cache",
            lambda key: None if key == "hardware" else {},
        )

        result = runner.invoke(main, ["setup"])
        assert result.exit_code == 0 or "hardware" in result.output.lower()


class TestArgsCommand:
    @patch("llamacpp_tuner.validators.load_cache")
    @patch("llamacpp_tuner.validators.get_model_path")
    @patch("llamacpp_tuner.cli.llama_has_gpu_support")
    def test_args_shows_optimal_configuration(
        self, mock_gpu, mock_get_model, mock_load_cache
    ):
        runner = CliRunner()

        mock_load_cache.return_value = {
            "gpus": [],
            "cpu_cores": 8,
            "cpu_threads": 16,
            "ram_mb": 16384,
            "backend": "cpu",
        }
        mock_get_model.return_value = Path("/fake/model.gguf")
        mock_gpu.return_value = False

        result = runner.invoke(main, ["args", "test/model", "--quant", "Q4_K_M"])

        assert result.exit_code == 0
        assert "Context size:" in result.output
        assert "GPU layers:" in result.output


class TestBenchmarkCommand:
    @patch("llamacpp_tuner.validators.is_llama_installed")
    def test_benchmark_requires_llama_installed(self, mock_installed):
        runner = CliRunner()
        mock_installed.return_value = False

        result = runner.invoke(main, ["benchmark", "test/model"])

        assert result.exit_code != 0
        assert "not installed" in result.output or "setup" in result.output


class TestCompareCommand:
    @patch("llamacpp_tuner.validators.is_llama_installed")
    def test_compare_requires_llama(self, mock_installed):
        runner = CliRunner()
        mock_installed.return_value = False

        result = runner.invoke(main, ["compare", "test/model"])

        assert result.exit_code != 0


class TestModelsCommand:
    @patch("llamacpp_tuner.cli.list_invalid_models")
    @patch("llamacpp_tuner.cli.list_valid_models")
    @patch("pathlib.Path.stat")
    def test_models_lists_files(self, mock_stat, mock_valid, mock_invalid):
        mock_valid.return_value = [
            Path("/fake/model1.gguf"),
            Path("/fake/model2.gguf"),
        ]
        mock_invalid.return_value = []
        mock_stat.return_value.st_size = 100 * 1024 * 1024

        runner = CliRunner()
        result = runner.invoke(main, ["models"])

        assert result.exit_code == 0
        assert "model1.gguf" in result.output

    @patch("llamacpp_tuner.cli.list_invalid_models")
    @patch("llamacpp_tuner.cli.list_valid_models")
    def test_models_shows_empty_message(self, mock_valid, mock_invalid):
        mock_valid.return_value = []
        mock_invalid.return_value = []

        runner = CliRunner()
        result = runner.invoke(main, ["models"])

        assert result.exit_code == 0
        assert "No models" in result.output or "downloaded" in result.output


class TestCleanCommand:
    @patch("llamacpp_tuner.cli.list_invalid_models")
    def test_clean_no_invalid_models(self, mock_invalid):
        mock_invalid.return_value = []

        runner = CliRunner()
        result = runner.invoke(main, ["clean"])

        assert result.exit_code == 0
        assert "No incomplete" in result.output

    @patch("llamacpp_tuner.cli.remove_model")
    @patch("llamacpp_tuner.cli.list_invalid_models")
    @patch("pathlib.Path.stat")
    def test_clean_removes_invalid_models(self, mock_stat, mock_invalid, mock_remove):
        mock_invalid.return_value = [
            Path("/fake/incomplete.gguf"),
        ]
        mock_stat.return_value.st_size = 100
        mock_remove.return_value = True

        runner = CliRunner()
        result = runner.invoke(main, ["clean", "--force"])

        assert result.exit_code == 0
        assert "Removed" in result.output


class TestStatusCommand:
    @patch("llamacpp_tuner.cli.load_cache")
    def test_status_shows_cached_hardware(self, mock_load_cache):
        runner = CliRunner()
        mock_load_cache.return_value = {
            "gpus": [{"name": "Test GPU", "vram_mb": 8192, "compute": [8, 0]}],
            "cpu_cores": 8,
            "cpu_threads": 16,
            "ram_mb": 16384,
            "backend": "cuda",
        }

        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "Test GPU" in result.output
