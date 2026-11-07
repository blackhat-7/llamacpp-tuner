# lct

A small wrapper for finding/building llama.cpp, downloading exact GGUF artifacts, and starting `llama-server`.

Model selection and tuning live in reusable Agent Skills instead of Python heuristics:

- [`skills/select-local-model`](skills/select-local-model/SKILL.md)
- [`skills/tune-local-model`](skills/tune-local-model/SKILL.md)

## Install

You need [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`), `git`, and your GPU driver. No compiler, no CUDA toolkit.

Install the `lct` command; its data goes in `~/.cache/lct` (`LCT_HOME` overrides):

```bash
uv tool install git+https://github.com/blackhat-7/llamacpp-tuner
lct setup
```

Or run it from a clone; data goes in the clone's `tmp/`. Run every command below from the clone with `uv run` in front (`uv run lct setup`); elsewhere uv fails with "Failed to spawn: `lct`".

```bash
git clone https://github.com/blackhat-7/llamacpp-tuner
cd llamacpp-tuner
uv run lct setup
```

`lct setup` uses `llama-server` from `PATH` if there is one. Otherwise it downloads the newest prebuilt llama.cpp for your GPU and checks it starts:

| GPU | Build | Needs |
|---|---|---|
| NVIDIA (`nvidia-smi` works) | CUDA 12, runtime included (~765 MB) | NVIDIA driver 570 or newer |
| AMD, Intel | Vulkan (~32 MB) | Vulkan driver: Arch `vulkan-radeon`/`vulkan-intel`, Ubuntu `mesa-vulkan-drivers` |
| Apple silicon | Metal (~12 MB) | nothing |

It ends by listing the GPUs llama.cpp can use, e.g. `Vulkan0: AMD Radeon RX 7900 XTX` or `CUDA0: NVIDIA GeForce RTX 4090`. If it says `(none)`, install the driver above and run `lct setup --force`.

### Building from source

`lct setup --cmake-arg=...` builds from source instead, e.g. `--cmake-arg=-DGGML_VULKAN=ON`; with `nvcc` installed it builds CUDA. That needs a compiler and, for Vulkan, its headers:

```bash
sudo pacman -S git cmake base-devel vulkan-headers shaderc spirv-headers        # Arch
sudo apt install git cmake build-essential libvulkan-dev glslc spirv-headers    # Debian/Ubuntu
```

A missing `SPIRV-Headers` in the CMake output means the last package is missing.

## Usage

```bash
# Find or download llama-server (see Install).
lct setup

# Quant names are open-ended and exact—there is no allowlist or fallback.
lct pull owner/model-GGUF --quant Q6_K

# If a quant matches multiple artifacts, select the exact repository filename.
lct pull owner/model-GGUF --file model-Q6_K.gguf

# Serve a downloaded repository without contacting Hugging Face.
lct serve owner/model-GGUF --quant Q6_K \
  --ctx 32768 \
  --extra-args "-ngl all -fa on -ctk q8_0 -ctv q8_0"

# Exact local paths work too.
lct serve /models/model.gguf --extra-args "-ngl all"

lct models
```

Save frequently used `serve` arguments as aliases in `aliases.toml` in the lct cache directory (see below):

```toml
qwen = """owner/model-GGUF --quant Q6_K --ctx 32768 \
  --extra-args '-ngl all -fa on -ctk q8_0 -ctv q8_0'"""
```

Use `--env NAME=VALUE` (repeatable) for backend variables such as `GGML_VK_SUBALLOCATION_BLOCK_SIZE`; they go into llama-server's environment.
Use `--server PATH` to run a different llama-server build for one profile, for example a pull-request branch that adds a model.

`lct serve qwen` then expands to those arguments. Options typed after the alias replace the alias's value for that option; `--extra-args` is replaced as a whole.

### Several servers at once

A `[stacks]` table in `aliases.toml` names profiles that run together, such as a GPU chat model, a CPU side model and a CPU embedding model:

```toml
[stacks]
local = ["qwen", "side", "embed"]
```

```bash
lct up local      # start each profile in the background, wait until all listen
lct ps            # name, address and resident RAM of each running server
lct down side     # stop one; 'lct down' alone stops every server
```

Servers run detached and keep running after `lct up` or the TUI exits. Their logs are in `servers/<name>.log` in the cache directory.

### Jeeves

[Jeeves](https://github.com/PostHog/jeeves) is a reasoning classifier that llama.cpp cannot run. `--backend jeeves` runs it on its own PyTorch server, installed into a separate venv in the cache directory. Profiles, stacks, `lct up/down/ps` and the TUI treat it like any other server.

```bash
lct setup --backend jeeves --torch-index https://download.pytorch.org/whl/rocm7.2  # omit the index on NVIDIA
lct pull PostHog/jeeves --all
lct serve PostHog/jeeves --backend jeeves --ctx 4096 --port 6871 --extra-args '--max-rows 2 --precision fp8'
```

`--ctx` becomes Jeeves's `--max-len`; `--extra-args` go to `python -m inference.serve`. It answers `POST /v1/systemone`, not the OpenAI API.

### Video

`lct video` makes and edits clips with sound on your GPU with [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), an open audio-video model: dialogue with lip sync, sound effects and music in one pass, 4-15 s at 24 fps. It runs on [ComfyUI](https://github.com/Comfy-Org/ComfyUI), which lct installs in its own venv and runs as the `video` profile, using ComfyUI's own H3 and SeedVR2 workflows; you never edit a node graph. It needs a 24 GB GPU, ffmpeg, and about 80 GB of disk for the models.

```bash
lct setup --backend comfyui --torch-index https://download.pytorch.org/whl/rocm7.2  # AMD; omit the index on NVIDIA
lct video make "a fox runs through snow at dusk" --seconds 5 --quality draft
lct video make --prompt-file shot.txt --photo me.jpg --seconds 10 --aspect 9:16
lct video edit clip.mp4 --prompt-file change.txt     # new dialogue, person, outfit or style
lct video extend clip.mp4 --prompt-file next.txt --seconds 6
lct video upscale clip.mp4                           # SeedVR2, to 1080p
lct video join a.mp4 b.mp4 --out film.mp4
lct down video                                       # free the GPU
```

`--quality` is `draft` (~480p, 4 steps, a quick preview), `standard` (768p, H3's native size, 8 steps) or `high` (standard upscaled to 1080p). The first job starts ComfyUI; jobs refuse to start while other servers hold the GPU. H3 expects prompts in MiniMax's structured format: `lct video guide` prints it, and the [`make-video`](skills/make-video/SKILL.md) skill teaches an agent to plan, write, run and check whole videos from a description, a quality and a length. Every result is kept in `comfyui/data/output/video` in the cache directory. Commercial use of H3 output needs a licence from MiniMax.

On AMD cards (tested on a 7900 XTX), give the `video` profile `--extra-args '--disable-pinned-memory --disable-smart-memory'`: without them ComfyUI crashed with a GPU page fault when it swapped models, or ran out of memory encoding frames. lct also installs a small ComfyUI node that runs Qwen3-VL's image patch embedding through MIOpen, because PyTorch's own ROCm kernel for it segfaults on every job with a photo, frame or source video.

### Terminal UI

```bash
lct tui
```

It is keyboard-driven and never traps you in a text box: `1`–`4` switch pages, `tab` or `←`/`→` move between the two panes of a page, `esc` backs out, `q` quits. The bottom line always shows the keys for where you are. A mouse click only highlights a row; `enter` or a double-click acts.

- **Serve:** profiles on the left, the highlighted profile's settings on the right. Stacks are listed first as a tree, their profiles nested under them; `enter` on a stack starts all of its profiles, or stops them when all are running, and the stack row sums their RAM and VRAM. `enter` on a profile starts or stops just that one; several can run at once, each marked loading (◌) or ready (●) with its live CPU, RAM and VRAM. The top line shows the whole machine's CPU, RAM, GPU load and VRAM, sampled every 2 s from `/proc` and `/sys` (under 1 ms a sample). `enter` on a setting opens a one-line editor (`enter` saves, `esc` cancels); model and projector autocomplete from downloaded files. `n` copies the highlighted profile, `d` then `y` deletes it.
- **Download:** `/` to search Hugging Face GGUF repositories as you type. The highlighted repository shows popularity, license, base model, architecture, context, its files with sizes and a model-card summary. `enter` opens a repository; `enter` on a file asks for confirmation (`y`) before downloading; `c` cancels a running download; `p` toggles its projector.
- **Benchmark:** `enter` on a model runs `llama-bench` (again to stop). Settings edit like profile settings; prompt, generate and depth accept comma lists such as `0,32768`.
- **Video:** the clip library on the left, newest first, with each clip's size and length; `enter` plays it, and `e`, `x` or `u` sets up an edit, extension or upscale of it. On the right, the next job: mode, prompt (`@file.txt` reads a long prompt from a file), photos, seconds, quality, aspect and seed. `▶` runs it with live step progress (again to stop) and shows whether ComfyUI is running.

Profiles are the same aliases `lct serve <name>` uses. All output streams into the log at the bottom; server lines start with the profile name. Drag the `Output` rule with the mouse to resize the log.

`serve` changes only options explicitly supplied; all others remain llama.cpp defaults. Use `--mmproj PATH` for a local vision model. Repository pulls download an unambiguous projector automatically unless `--no-mmproj` is given.

A source checkout stores files in `tmp/`; an installed package uses `${XDG_CACHE_HOME:-~/.cache}/lct`. Set `LCT_HOME` to override either location.

After changing GPU vendors, rebuild the managed copy with `lct setup --force`.

For a custom llama.cpp build:

```bash
lct setup --force --cmake-arg=-DGGML_VULKAN=ON
```
