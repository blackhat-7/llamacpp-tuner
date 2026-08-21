# llama.cpp Optimization Research

> Findings for maximizing optimal vs default TPS gap.

## Benchmark Target

```
uv run lct compare unsloth/gemma-4-26B-A4B-it-GGUF -c 128000 --max-tokens 1000
```

Model: Gemma 4 26B-A4B (MoE, 4B active params per token, 128 experts, 8 active)  
Hardware target: ~24 GB VRAM NVIDIA GPU

---

## Why Round 1 Gap Was Small (34 vs 36 TPS)

After adding `-ngl all -fa on -ctk q8_0 -ctv q8_0`, the gap was only ~6%.

Root cause: **two separate issues.**

### Issue 1: Benchmark prompt too short

`prompt * 10` ≈ 700 tokens against 128k allocated context.  
Flash attention and KV cache quantization only show their benefit at large filled contexts:

| Context filled | KV q8_0 benefit |
|---------------|-----------------|
| 700 tokens    | ~0%             |
| 8k tokens     | ~3%             |
| 32k tokens    | ~15%            |
| 64k+ tokens   | ~20%            |

**Fix:** Use `prompt * 30` ≈ 4k tokens for the prompt TPS bench so flash-attn and KV cache optimizations have meaningful data to operate on.

### Issue 2: GPU batch sizes never set

`batch_size = None, ubatch_size = None` means llama.cpp uses its defaults:
- `-b 2048` — fine for most models
- `-ub 512` — **too small for MoE models** (DocShotgun MoE Offload Guide)

For Gemma 4's architecture (128 experts, 8 active), increasing ubatch allows the GPU to process more tokens per kernel launch, improving GPU utilization during prefill and reducing dispatch overhead.

**Fix:** Set `-b 4096 -ub 4096` for GPU inference (tiered by VRAM).

---

## What llama.cpp Defaults Are Already Good

Do **not** override these:

| Arg | Default | Why It's Fine |
|-----|---------|---------------|
| `-t` threads | all physical cores | llama.cpp auto-tunes well |
| `--mmap` | enabled | best for cold starts |
| `--split-mode` | `layer` | correct for single GPU |
| `--numa` | off | only for 16+ core CPU-only |
| `--rope-scaling` | auto | only touch for out-of-training-range context |
| `--mlock` | off | no benefit unless RAM is being swapped |
| `--no-mmap` | (not set) | slower loads, rarely beneficial |

---

## What We Override and Why

### 1. `-ngl all` — GPU Offload

llama.cpp default: `0` (CPU only).  
**This is the single biggest win.** Without it, inference is CPU-bound.

### 2. `-fa on` — Flash Attention

Default is `auto` — enables itself on CUDA + compatible models. Explicit `on` guarantees it and is required for KV cache quantization to work correctly.

Benefit: 2x–10x prompt processing speedup at large contexts, ~5-15% at typical contexts.

### 3. `-ctk q8_0 -ctv q8_0` — KV Cache Quantization

Saves ~50% KV cache VRAM with no measurable quality loss at q8_0.

For Gemma 4 26B at 128k context:
- f16 KV: ~7.6 GB
- q8_0 KV: ~3.8 GB
- **Saves 3.8 GB** → more VRAM headroom, prevents partial offload at large contexts

### 4. `-b 4096 -ub 4096` — Batch Sizes (GPU, High VRAM)

Default ubatch of 512 is too small for MoE models. Larger ubatch:
- Better GPU utilization during prompt prefill
- Reduces kernel dispatch overhead for MoE expert routing
- Expected: 5–15% prompt TPS improvement, ~2% gen TPS improvement

Tiered by VRAM:
| VRAM | `-b` | `-ub` |
|------|------|-------|
| ≥ 8 GB | 4096 | 4096 |
| 4–8 GB | 2048 | 2048 |
| < 4 GB | 1024 | 512 |
| CPU only | 256 | 64 |

### 5. `--poll 50` — CPU Poll Interval

Default: `0` (no CPU spinning, lowest latency average).  
Setting to 50 reduces CPU-GPU synchronization stalls during the decode loop.  
Expected: ~1–3% gen TPS improvement.

---

## What LM Studio Does Differently

LM Studio wraps llama.cpp with non-default settings:
- Flash attention: **enabled by default**
- KV cache: **q8_0 by default** for large contexts
- GPU offload: auto-calculated
- CUDA Graphs: **enabled by default** (see below)

---

## CUDA Graphs — Biggest Untapped Win (~10–35%)

CUDA Graphs fuses multiple CUDA kernel launches into a single recorded graph, eliminating per-token launch overhead. LM Studio enables this by default. Raw llama.cpp requires building with CUDA graph support.

This is a **compile-time / build configuration** feature, not a runtime flag. If the binary was built with CUDA graphs enabled, it activates automatically for single-token decode steps.

To verify your build has it:
```bash
llama-server --help | grep -i graph
```

If you're building from source:
```bash
cmake -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON ...
```

Expected improvement: **+10–35% generation TPS** on RTX hardware.

---

## Gemma 4 Architecture Notes

- **128 total experts, 8 active per token** (not 2 like Mixtral — 8x more routing overhead)
- Uses interleaved **SWA (512-token sliding window) + global** attention layers
- **GQA** (Grouped Query Attention) — smaller KV cache than standard MHA
- Known bug (Issue #21434): `sliding_window_pattern` type mismatch in some llama.cpp builds may cause incorrect SWA layer identification, wasting KV cache VRAM

---

## Final Optimal Config (24 GB GPU, 128k context)

```
-c 131072 -ngl all -fa on -ctk q8_0 -ctv q8_0 -b 4096 -ub 4096 --poll 50
```

vs default:

```
-c 131072
```

Expected gap (after round 2 fixes): **15–40% gen TPS**, **20–50% prompt TPS**  
(wider gap from longer benchmark prompt + -ub 4096)

---

## Args to Never Touch

- `--rope-scaling` — only if running context beyond model's trained limit
- `--mlock` — no benefit unless RAM is being swapped under load
- `--no-mmap` — slower load times, rarely beneficial
- `-t` manual override — llama.cpp auto-detection is better
- `--swa-full` — forces all Gemma 4 layers to use full 128k KV cache; causes OOM on most 24GB setups
- `-cmoe` / `-ot "exps=CPU"` — only if model doesn't fit in VRAM; adds PCIe transfer overhead per token
