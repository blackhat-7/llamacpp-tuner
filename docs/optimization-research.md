# llama.cpp Optimization Research

> Findings for why `optimal` and `default` produce similar performance, and what to fix.

## Why Default ≈ Optimal Right Now

The `compare` command currently produces:

| Config | Args |
|--------|------|
| `default` | `-c 128000` |
| `optimal` | `-c 128000 -ngl all` |

For most benchmarks the difference is negligible because:
- Flash attention defaults to `auto` in both — llama.cpp enables it automatically when CUDA + compatible model
- If model + KV cache fits in VRAM, both end up fully GPU-accelerated anyway
- Batch sizes and threads are absent from `optimal` — they use llama.cpp defaults, same as `default`

---

## llama.cpp: What Defaults Are Already Good

These defaults are near-optimal — **do not override**:

| Arg | Default | Why It's Fine |
|-----|---------|---------------|
| `-t` threads | all physical cores | llama.cpp auto-tunes this well |
| `-b` batch-size | 2048 | Good for throughput; no need to change unless OOM |
| `-ub` ubatch-size | 512 | Reasonable physical batch; tunable but not critical |
| `-fa` flash-attn | `auto` | Enables itself on CUDA + compatible models |
| `--mmap` | enabled | Best for cold starts; fine to keep |
| `--split-mode` | `layer` | Correct for single GPU |
| `--numa` | off | Only relevant for 16+ core CPU-only setups |

---

## What We Should Override

### 1. KV Cache Quantization — Biggest Win at Large Context

**Args:** `-ctk q8_0 -ctv q8_0`

For `gemma-4-26B-A4B` at 128k context:

| | VRAM |
|-|------|
| Weights (Q4_K_M) | 13.3 GB |
| KV cache f16 (default) | 7.6 GB |
| KV cache q8_0 | 3.8 GB |
| **Savings** | **3.8 GB** |

- **No measurable quality loss** at q8_0 (benchmarks: NVIDIA, community tests)
- Frees ~3.8 GB VRAM, which can enable more layers on GPU or prevent OOM
- Supported since llama.cpp b2000+
- Stacks with flash attention (which enables even q4_0 if needed)

**When to use q4_0 instead:** Only for extreme VRAM pressure (saves ~72% vs f16 but slightly degrades outputs on longer contexts).

### 2. Flash Attention — Explicit `on` for 128k Context

**Arg:** `-fa on`

Default is `auto` which *should* enable it, but explicit `on` guarantees it for:
- Prompt processing: reported 2.9x–10x speedup on long contexts
- Enables KV cache quantization (`q8_0`, `q4_0`) — required dependency
- Reduces VRAM for the attention computation itself

**Note:** `auto` covers most cases. Explicitly setting `on` only matters if auto-detection is unreliable (e.g., older build, non-standard model).

### 3. Batch Sizes — Defined in Constants but Never Used

`constants.py` defines `BATCH_SIZE_CPU`, `UBATCH_SIZE_CPU`, etc., but `calculate_optimal_args()` never emits `-b` or `-ub`.

Recommended by VRAM tier:

| VRAM | `-b` | `-ub` |
|------|------|-------|
| ≥ 8 GB GPU | 2048 (default) | 512 (default) | 
| CPU-only | 256 | 64 |

For GPU inference, llama.cpp defaults are fine. For CPU, halving ubatch-size reduces memory pressure.

### 4. `-ngl all` for GPU — Already Correct

Current code sets `-ngl all` when model fits in VRAM. This is right. The fix is ensuring the VRAM estimate accounts for KV cache quantization (see #1 above), so the threshold calculation is accurate.

---

## What LM Studio Does Differently

LM Studio wraps llama.cpp with these non-default settings:
- Flash attention: **enabled by default**
- KV cache: **q8_0 by default** for large contexts
- GPU offload: auto-calculated based on available VRAM
- Batch size: ~512 (conservative, avoids OOM)

This is why LM Studio often outperforms raw llama.cpp defaults — q8_0 KV + flash attn is their secret sauce.

---

## Recommended Changes to `calculate_optimal_args()`

Priority order for maximum impact:

1. **Add `-ctk q8_0 -ctv q8_0`** when GPU is available — 3.8 GB saved for this benchmark, no quality loss
2. **Add `-fa on`** explicitly when CUDA + model supports it (instead of relying on `auto`)
3. **Add CPU-specific `-b 256 -ub 64`** when `not has_gpu` — use the constants already defined

### Expected Improvement for the Benchmark Target

`uv run lct compare unsloth/gemma-4-26B-A4B-it-GGUF -c 128000 --max-tokens 1000`

With KV cache q8_0 + explicit flash attention:
- 3.8 GB VRAM freed → more layers stay on GPU → higher gen TPS
- Prompt processing TPS improvement: 2x–10x (flash attn on 128k prompt)
- Gen TPS improvement: 10–30% from freed VRAM enabling full GPU offload

Without these: `default` and `optimal` will continue to show near-identical numbers since both miss the KV quantization that actually matters at 128k context.

---

## Args to Never Touch

These will hurt performance or are model-specific:
- `--rope-scaling` — only if running context beyond model's trained limit
- `--mlock` — no benefit unless RAM is being swapped under load
- `--no-mmap` — slower load times, rarely beneficial
- `-t` manual override — llama.cpp auto-detection is better
