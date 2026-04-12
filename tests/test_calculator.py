"""Tests for calculator module."""

from llamacpp_tuner.calculator import (
    OptimalArgs,
    _estimate_layers,
    _estimate_params,
    _estimate_vram_usage,
    calculate_optimal_args,
)


class TestEstimateParams:
    def test_common_model_sizes(self):
        assert _estimate_params("llama-7b") == 7.0
        assert _estimate_params("model-14b") == 14.0

    def test_uppercase_b(self):
        assert _estimate_params("Model-9B") == 9.0

    def test_decimal_params(self):
        assert _estimate_params("model-1.5b") == 1.5

    def test_unknown_defaults_to_7b(self):
        assert _estimate_params("unknown-model") == 7.0


class TestEstimateLayers:
    def test_common_layer_counts(self):
        assert _estimate_layers("model-7b") == 32
        assert _estimate_layers("model-14b") == 48
        assert _estimate_layers("model-70b") == 80

    def test_unknown_params_defaults_based_on_size(self):
        assert _estimate_layers("some-3b-model") == 28
        assert _estimate_layers("some-9b-model") == 48


class TestEstimateVramUsage:
    def test_small_model(self):
        result = _estimate_vram_usage(0.5, "Q4_K_M", 8192, 24)
        assert 200 < result < 500

    def test_context_size_affects_kv_cache(self):
        small_ctx = _estimate_vram_usage(1.0, "Q4_K_M", 2048, 24)
        large_ctx = _estimate_vram_usage(1.0, "Q4_K_M", 8192, 24)
        assert large_ctx > small_ctx


class TestOptimalArgsToList:
    def test_cpu_only_no_gpu_args(self, cpu_only_hardware):
        args = OptimalArgs(
            ctx_size=4096,
            n_gpu_layers=0,
            tensor_split=None,
            split_mode="layer",
            mlock=False,
            mmap=True,
            numa=False,
            cache_type_k="f16",
            cache_type_v="f16",
        )
        result = args.to_list("/path/to/model.gguf")

        assert "-m" in result
        assert "/path/to/model.gguf" in result
        assert "-c" in result and "4096" in result
        assert "-ngl" not in result
        assert "--split-mode" not in result

    def test_gpu_includes_ngl(self, single_gpu_hardware):
        args, _ = calculate_optimal_args(
            single_gpu_hardware, "test-7b", "Q4_K_M", 4096, llama_has_gpu=True
        )
        result = args.to_list("/path/to/model.gguf")

        assert "-ngl" in result
        assert "all" in result or "auto" in result

    def test_multi_gpu_includes_tensor_split(self, multi_gpu_hardware):
        args, _ = calculate_optimal_args(
            multi_gpu_hardware, "test-7b", "Q4_K_M", 4096, llama_has_gpu=True
        )
        result = args.to_list("/path/to/model.gguf")

        assert "-ts" in result
        ts_index = result.index("-ts")
        assert "0.50" in result[ts_index + 1]
        assert "--split-mode" in result
        assert "row" in result


class TestCalculateOptimalArgs:
    def test_cpu_no_gpu_layers(self, cpu_only_hardware):
        args, warnings = calculate_optimal_args(
            cpu_only_hardware, "test-7b", "Q4_K_M", 4096, llama_has_gpu=True
        )
        assert args.n_gpu_layers == 0

    def test_gpu_has_negative_layers_for_all(self, single_gpu_hardware):
        args, _ = calculate_optimal_args(
            single_gpu_hardware, "test-7b", "Q4_K_M", 4096, llama_has_gpu=True
        )
        assert args.n_gpu_layers == -1

    def test_gpu_disabled_when_llama_has_no_gpu(self, single_gpu_hardware):
        args, _ = calculate_optimal_args(
            single_gpu_hardware, "test-7b", "Q4_K_M", 4096, llama_has_gpu=False
        )
        assert args.n_gpu_layers == 0

    def test_warning_when_model_too_large(self, single_gpu_hardware):
        args, warnings = calculate_optimal_args(
            single_gpu_hardware, "test-70b", "Q8_0", 32768, llama_has_gpu=True
        )
        assert len(warnings) > 0
        assert any("VRAM" in w or "context" in w for w in warnings)

    def test_q8_0_uses_more_vram(self, single_gpu_hardware):
        from llamacpp_tuner.calculator import _estimate_vram_usage

        q4_vram = _estimate_vram_usage(7.0, "Q4_K_M", 4096, 32)
        q8_vram = _estimate_vram_usage(7.0, "Q8_0", 4096, 32)
        assert q8_vram > q4_vram
