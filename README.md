# lct

A small wrapper for finding/building llama.cpp, downloading exact GGUF artifacts, and starting `llama-server`.

Model selection and tuning live in reusable Agent Skills instead of Python heuristics:

- [`skills/select-local-model`](skills/select-local-model/SKILL.md)
- [`skills/tune-local-model`](skills/tune-local-model/SKILL.md)

## Usage

```bash
# Use llama-server from PATH, or build a managed copy.
uv run lct setup

# Quant names are open-ended and exact—there is no allowlist or fallback.
uv run lct pull owner/model-GGUF --quant Q6_K

# If a quant matches multiple artifacts, select the exact repository filename.
uv run lct pull owner/model-GGUF --file model-Q6_K.gguf

# Serve a downloaded repository without contacting Hugging Face.
uv run lct serve owner/model-GGUF --quant Q6_K \
  --ctx 32768 \
  --extra-args "-ngl all -fa on -ctk q8_0 -ctv q8_0"

# Exact local paths work too.
uv run lct serve /models/model.gguf --extra-args "-ngl all"

uv run lct models
```

`serve` changes only options explicitly supplied; all others remain llama.cpp defaults. Use `--mmproj PATH` for a local vision model. Repository pulls download an unambiguous projector automatically unless `--no-mmproj` is given.

A source checkout stores files in `tmp/`; an installed package uses `${XDG_CACHE_HOME:-~/.cache}/lct`. Set `LCT_HOME` to override either location.

For a custom llama.cpp build:

```bash
uv run lct setup --force --cmake-arg=-DGGML_VULKAN=ON
```
