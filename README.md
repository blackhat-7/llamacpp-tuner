# llamacpp-tuner

Auto-optimize llama.cpp parameters for your hardware.

## Quick Start

```bash
# Setup (detect hardware + install llama.cpp)
uv run llamacpp-tuner setup

# Download a model
uv run llamacpp-tuner pull bartowski/Qwen2.5-0.5B-Instruct-GGUF

# Benchmark default args
uv run llamacpp-tuner benchmark bartowski/Qwen2.5-0.5B-Instruct-GGUF

# Benchmark optimized args
uv run llamacpp-tuner benchmark bartowski/Qwen2.5-0.5B-Instruct-GGUF --args optimal

# Run server
uv run llamacpp-tuner serve bartowski/Qwen2.5-0.5B-Instruct-GGUF --ctx 4096
```

## Commands

| Command | Description |
|---------|-------------|
| `setup` | Detect hardware, install llama.cpp |
| `pull <repo>` | Download model from HuggingFace |
| `benchmark <repo>` | Test TPS with args (use `--args optimal`) |
| `compare <repo>` | Compare default vs optimized TPS |
| `args <repo>` | Show optimal arguments (dry run) |
| `serve <repo>` | Run server with optimal args |
| `models` | List downloaded models |
| `status` | Show hardware status |

## Usage Flow

```bash
# 1. Benchmark baseline
uv run llamacpp-tuner benchmark unsloth/Phi-4-GGUF --quant Q4_K_M

# 2. See optimized args
uv run llamacpp-tuner args unsloth/Phi-4-GGUF --quant Q4_K_M --ctx 8192

# 3. Benchmark optimized
uv run llamacpp-tuner benchmark unsloth/Phi-4-GGUF --quant Q4_K_M --args optimal

# 4. Run server
uv run llamacpp-tuner serve unsloth/Phi-4-GGUF --quant Q4_K_M --port 8080
```
# llamacpp-tuner
