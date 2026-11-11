"""Turn a fine-tuned LoRA adapter into one quantized GGUF with llama.cpp's own tools."""

import json
import subprocess
import tempfile
from pathlib import Path

from llamacpp_tuner.cache import get_llama_dir
from llamacpp_tuner.llama import get_llama_binary


def adapter_base(adapter: Path) -> str:
    """The base model a PEFT adapter was trained on, as saved in adapter_config.json."""
    config = json.loads((adapter / "adapter_config.json").read_text())
    if not (base := config.get("base_model_name_or_path")):
        raise ValueError(f"{adapter / 'adapter_config.json'} names no base model.")
    return base


def import_lora(adapter: Path, base: Path, quant: str, out: Path) -> Path:
    """Merge a PEFT LoRA adapter into its Hugging Face base model and quantize it.

    quant is any llama-quantize type, such as Q4_K_M.
    """
    source = get_llama_dir()
    if not (source / "convert_lora_to_gguf.py").is_file():
        raise FileNotFoundError(
            f"No llama.cpp source in {source}; its convert scripts are needed. "
            "Build from source: lct setup --force --cmake-arg=-DGGML_VULKAN=ON"
        )
    export = get_llama_binary("llama-export-lora")
    quantize = get_llama_binary("llama-quantize")
    if not (export and quantize):
        raise FileNotFoundError("llama-export-lora or llama-quantize not found.")

    # The convert scripts need torch and transformers; uv builds that env from
    # llama.cpp's own pinned list and caches it, so lct does not depend on them.
    requirements = source / "requirements" / "requirements-convert_lora_to_gguf.txt"
    python = [
        "uv",
        "run",
        "--no-project",
        "--with-requirements",
        str(requirements),
        "python",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    # The f16 middle steps are each twice the size of the final file.
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        base_gguf, lora_gguf, merged, quantized = (
            Path(scratch) / f"{name}.gguf" for name in ("base", "lora", "merged", "out")
        )
        steps = [
            [*python, str(source / "convert_hf_to_gguf.py"), str(base)]
            + ["--outtype", "f16", "--outfile", str(base_gguf)],
            [*python, str(source / "convert_lora_to_gguf.py"), str(adapter)]
            + ["--base", str(base), "--outtype", "f16", "--outfile", str(lora_gguf)],
            [str(export), "-m", str(base_gguf), "--lora", str(lora_gguf)]
            + ["-o", str(merged)],
            [str(quantize), str(merged), str(quantized), quant],
        ]
        for step in steps:
            subprocess.run(step, check=True)
        # Only a finished file replaces an older import.
        quantized.replace(out)
    return out
