"""Tests for benchmark module."""

import json
from unittest.mock import MagicMock, patch

from llamacpp_tuner.benchmark import (
    BenchmarkResult,
    _send_completion,
    _wait_for_server,
    benchmark_server,
)


class TestWaitForServer:
    @patch("urllib.request.urlopen")
    def test_returns_true_when_server_ready(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        assert _wait_for_server("http://localhost:8080", timeout=1.0) is True

    @patch("urllib.request.urlopen")
    def test_returns_false_on_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")

        assert _wait_for_server("http://localhost:8080", timeout=0.1) is False


class TestSendCompletion:
    @patch("urllib.request.urlopen")
    def test_returns_result_on_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                }
            }
        ).encode()
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = _send_completion("http://localhost:8080", "test prompt")
        assert result is not None
        assert result["usage"]["prompt_tokens"] == 10

    @patch("urllib.request.urlopen")
    def test_returns_none_on_error(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection failed")

        result = _send_completion("http://localhost:8080", "test prompt")
        assert result is None


class TestBenchmarkServer:
    @patch("llamacpp_tuner.benchmark.get_llama_binary")
    def test_returns_none_when_binary_not_found(self, mock_get_binary):
        mock_get_binary.return_value = None

        result = benchmark_server("/path/to/model.gguf", [], port=8081)
        assert result is None

    @patch("llamacpp_tuner.benchmark.get_llama_binary")
    @patch("subprocess.Popen")
    @patch("llamacpp_tuner.benchmark._wait_for_server")
    @patch("llamacpp_tuner.benchmark._send_completion")
    def test_successful_benchmark(
        self, mock_send, mock_wait, mock_popen, mock_get_binary
    ):
        mock_get_binary.return_value = "/usr/bin/llama-server"
        mock_wait.return_value = True

        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        # Mock completion responses
        def mock_completion(url, prompt, max_tokens=256):
            if "warmup" in prompt:
                return {"usage": {"prompt_tokens": 5, "completion_tokens": 5}}
            elif len(prompt) > 100:  # long prompt
                return {"usage": {"prompt_tokens": 100, "completion_tokens": 8}}
            else:
                return {"usage": {"prompt_tokens": 10, "completion_tokens": 50}}

        mock_send.side_effect = mock_completion

        result = benchmark_server("/path/to/model.gguf", ["-c", "4096"], port=8081)

        assert result is not None
        assert isinstance(result, BenchmarkResult)
        assert result.prompt_tokens > 0
        assert result.gen_tokens > 0
        assert result.prompt_tps > 0
        assert result.gen_tps > 0

    @patch("llamacpp_tuner.benchmark.get_llama_binary")
    @patch("subprocess.Popen")
    @patch("llamacpp_tuner.benchmark._wait_for_server")
    def test_server_timeout_captures_logs(
        self, mock_wait, mock_popen, mock_get_binary, capsys
    ):
        mock_get_binary.return_value = "/usr/bin/llama-server"
        mock_wait.return_value = False  # Server never ready

        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline.return_value = "error: invalid argument\n"
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        result = benchmark_server("/path/to/model.gguf", ["-bad-arg"], port=8081)

        assert result is None


class TestBenchmarkResult:
    def test_summary_format(self):
        result = BenchmarkResult(
            prompt_tps=100.5,
            prompt_time_ms=50.0,
            prompt_tokens=50,
            gen_tps=50.25,
            gen_time_ms=200.0,
            gen_tokens=100,
            total_time_ms=250.0,
            args={"-c": "4096"},
        )
        summary = result.summary()

        assert "100.50 TPS" in summary
        assert "50 tokens" in summary
        assert "50.25 TPS" in summary
        assert "100 tokens" in summary
        assert "250ms" in summary
