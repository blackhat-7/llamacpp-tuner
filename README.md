# lct

A small wrapper for finding/building llama.cpp, downloading exact GGUF artifacts, and starting `llama-server`.

Model selection and tuning live in reusable Agent Skills instead of Python heuristics:

- [`skills/select-local-model`](skills/select-local-model/SKILL.md)
- [`skills/tune-local-model`](skills/tune-local-model/SKILL.md)
- [`skills/jeeves`](skills/jeeves/SKILL.md): classify text with the Jeeves server

## Usage

```bash
# Use llama-server from PATH, or build a managed copy.
# Builds CUDA when nvcc is available, otherwise Vulkan when vulkaninfo is available.
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

Save frequently used `serve` arguments as aliases in `aliases.toml` in the lct cache directory (see below):

```toml
qwen = """owner/model-GGUF --quant Q6_K --ctx 32768 \
  --extra-args '-ngl all -fa on -ctk q8_0 -ctv q8_0'"""
```

Use `--env NAME=VALUE` (repeatable) for backend variables such as `GGML_VK_SUBALLOCATION_BLOCK_SIZE`; they go into llama-server's environment.

`uv run lct serve qwen` then expands to those arguments. Options typed after the alias replace the alias's value for that option; `--extra-args` is replaced as a whole.

### Several servers at once

A `[stacks]` table in `aliases.toml` names profiles that run together, such as a GPU chat model, a CPU side model and a CPU embedding model:

```toml
[stacks]
local = ["qwen", "side", "embed"]
```

```bash
uv run lct up local      # start each profile in the background, wait until all listen
uv run lct ps            # name, address and resident RAM of each running server
uv run lct down side     # stop one; 'lct down' alone stops every server
```

Servers run detached and keep running after `lct up` or the TUI exits. Their logs are in `servers/<name>.log` in the cache directory.

### Jeeves

[Jeeves](https://github.com/PostHog/jeeves) is a reasoning classifier that llama.cpp cannot run. `--backend jeeves` runs it on its own PyTorch server, installed into a separate venv in the cache directory. Profiles, stacks, `lct up/down/ps` and the TUI treat it like any other server.

```bash
uv run lct setup --backend jeeves --torch-index https://download.pytorch.org/whl/rocm7.2  # omit the index on NVIDIA
uv run lct pull PostHog/jeeves --all
uv run lct serve PostHog/jeeves --backend jeeves --ctx 4096 --port 6871 --extra-args '--max-rows 2 --precision fp8'
```

`--ctx` becomes Jeeves's `--max-len`; `--extra-args` go to `python -m inference.serve`. It answers `POST /v1/systemone`, not the OpenAI API.

### Terminal UI

```bash
uv run lct tui
```

It is keyboard-driven and never traps you in a text box: `1`–`3` switch pages, `tab` or `←`/`→` move between the two panes of a page, `esc` backs out, `q` quits. The bottom line always shows the keys for where you are. A mouse click only highlights a row; `enter` or a double-click acts.

- **Serve:** profiles on the left, the highlighted profile's settings on the right. Stacks are listed first as a tree, their profiles nested under them; `enter` on a stack starts all of its profiles, or stops them when all are running, and the stack row sums their RAM and VRAM. `enter` on a profile starts or stops just that one; several can run at once, each marked loading (◌) or ready (●) with its live CPU, RAM and VRAM. The top line shows the whole machine's CPU, RAM, GPU load and VRAM, sampled every 2 s from `/proc` and `/sys` (under 1 ms a sample). `enter` on a setting opens a one-line editor (`enter` saves, `esc` cancels); model and projector autocomplete from downloaded files. `n` copies the highlighted profile, `d` then `y` deletes it.
- **Download:** `/` to search Hugging Face GGUF repositories as you type. The highlighted repository shows popularity, license, base model, architecture, context, its files with sizes and a model-card summary. `enter` opens a repository; `enter` on a file asks for confirmation (`y`) before downloading; `c` cancels a running download; `p` toggles its projector.
- **Benchmark:** `enter` on a model runs `llama-bench` (again to stop). Settings edit like profile settings; prompt, generate and depth accept comma lists such as `0,32768`.

Profiles are the same aliases `lct serve <name>` uses. All output streams into the log at the bottom; server lines start with the profile name. Drag the `Output` rule with the mouse to resize the log.

`serve` changes only options explicitly supplied; all others remain llama.cpp defaults. Use `--mmproj PATH` for a local vision model. Repository pulls download an unambiguous projector automatically unless `--no-mmproj` is given.

A source checkout stores files in `tmp/`; an installed package uses `${XDG_CACHE_HOME:-~/.cache}/lct`. Set `LCT_HOME` to override either location.

After changing GPU vendors, rebuild the managed copy with `uv run lct setup --force`.

For a custom llama.cpp build:

```bash
uv run lct setup --force --cmake-arg=-DGGML_VULKAN=ON
```
