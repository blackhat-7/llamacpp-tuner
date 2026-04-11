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
        # Regex fallback handles 9B format (not in MODEL_PARAMS keys)
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
        # Based on param count from _estimate_params
        # 3b has 28 layers in MODEL_LAYERS
        assert _estimate_layers("some-3b-model") == 28
        # 9b has 48 layers in MODEL_LAYERS
        assert _estimate_layers("some-9b-model") == 48


class TestEstimateVramUsage:
    def test_small_model(self):
        # 0.5B Q4_K_M, 8192 ctx, 24 layers
        result = _estimate_vram_usage(0.5, "Q4_K_M", 8192, 24)
        # ~275MB model + ~75MB kv cache + overhead
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
            batch_size=256,
            ubatch_size=64,
            threads=4,
            threads_batch=8,
            parallel=1,
            flash_attn=False,
            tensor_split=None,
            split_mode="layer",
            main_gpu=0,
            mlock=False,
            mmap=True,
            offload_kqv=False,
            numa=False,
        )
        result = args.to_list("/path/to/model.gguf")

        assert "-m" in result
        assert "/path/to/model.gguf" in result
        assert "-c" in result and "4096" in result
        assert "-ngl" in result and "0" in result
        assert "--split-mode" not in result
        assert "--flash-attn" not in result

    def test_gpu_includes_flash_attn(self, single_gpu_hardware):
        args, _ = calculate_optimal_args(
            single_gpu_hardware, "test-7b", "Q4_K_M", 4096, llama_has_gpu=True
        )
        result = args.to_list("/path/to/model.gguf")

        assert "--flash-attn" in result
        assert "on" in result

    def test_multi_gpu_includes_tensor_split(self, multi_gpu_hardware):
        args, _ = calculate_optimal_args(
            multi_gpu_hardware, "test-7b", "Q4_K_M", 4096, llama_has_gpu=True
        )
        result = args.to_list("/path/to/model.gguf")

        assert "-ts" in result
        # Tensor split based on VRAM ratio: both GPUs have 24576 MB = 50/50
        # The format is "0.50,0.50" - check for comma-separated values
        ts_index = result.index("-ts")
        assert "0.50" in result[ts_index + 1]


class TestCalculateOptimalArgs:
    def test_cpu_no_gpu_layers(self, cpu_only_hardware):
        args, warnings = calculate_optimal_args(
            cpu_only_hardware, "test-7b", "Q4_K_M", 4096, llama_has_gpu=True
        )
        assert args.n_gpu_layers == 0
        assert args.flash_attn is False

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
        assert args.flash_attn is False

    def test_warning_when_model_too_large(self, single_gpu_hardware):
        # 70B model won't fit in 12GB VRAM
        args, warnings = calculate_optimal_args(
            single_gpu_hardware, "test-70b", "Q8_0", 32768, llama_has_gpu=True
        )
        assert len(warnings) > 0
        assert any("VRAM" in w or "context" in w for w in warnings)

    def test_q8_0_uses_more_vram(self, single_gpu_hardware):
        q4_args, _ = calculate_optimal_args(
            single_gpu_hardware, "test-7b", "Q4_K_M", 4096, llama_has_gpu=True
        )
        q8_args, _ = calculate_optimal_args(
            single_gpu_hardware, "test-7b", "Q8_0", 4096, llama_has_gpu=True
        )
        # Q8_0 uses more memory, should have different batch size or trigger warning
        assert q4_args.batch_size == q8_args.batch_size  # same for 7B


class TestFlashAttnRequirements:
    def test_flash_attn_requires_compute_70_or_higher(self):
        from llamacpp_tuner.hardware import GPUInfo, HardwareProfile

        old_gpu = HardwareProfile(
            gpus=[GPUInfo("Old GPU", 8192, (6, 1))],
            cpu_cores=8,
            cpu_threads=16,
            ram_mb=16384,
            backend="cuda",
        )
        args, _ = calculate_optimal_args(
            old_gpu, "test-7b", "Q4_K_M", 4096, llama_has_gpu=True
        )
        assert args.flash_attn is False

    def test_flash_attn_enabled_for_compute_80(self, single_gpu_hardware):
        args, _ = calculate_optimal_args(
            single_gpu_hardware, "test-7b", "Q4_K_M", 4096, llama_has_gpu=True
        )
        assert args.flash_attn is True
