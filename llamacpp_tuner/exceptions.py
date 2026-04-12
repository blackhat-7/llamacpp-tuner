"""Custom exception hierarchy for lct."""


class LlamaTunerError(Exception):
    """Base exception for all lct errors."""

    pass


class HardwareNotDetectedError(LlamaTunerError):
    """Hardware has not been detected. Run 'lct setup' first."""

    pass


class LlamaNotInstalledError(LlamaTunerError):
    """llama.cpp is not installed. Run 'lct setup' first."""

    pass


class ModelNotFoundError(LlamaTunerError):
    """Model not found locally or on HuggingFace."""

    def __init__(
        self,
        repo_id: str,
        quant: str | None = None,
        available_models: list[str] | None = None,
    ):
        self.repo_id = repo_id
        self.quant = quant
        message = f"Model not found: {repo_id}"
        if quant:
            message += f" (quantization: {quant})"
        if available_models:
            message += "\n\nAvailable models:"
            for m in available_models[:10]:
                message += f"\n  - {m}"
            if len(available_models) > 10:
                message += f"\n  ... and {len(available_models) - 10} more"
            message += "\n\nUse a model from the list above, or download with 'lct pull <repo_id>'."
        else:
            message += ". No models downloaded yet. Run 'lct pull <repo_id>' first."
        super().__init__(message)


class BenchmarkFailedError(LlamaTunerError):
    """Benchmark execution failed."""

    pass


class BinaryBuildError(LlamaTunerError):
    """Failed to build or download llama.cpp binary."""

    pass
