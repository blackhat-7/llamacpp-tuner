# Making Swift-Qwen3.8-27B faster on the 7900 XTX

Research of 2026-10-10. Web sources, Strata's code and llama.cpp's source; nothing was run on the GPU.
Numbers marked *est.* are arithmetic, not measurements. "Vendor" means the author's own claim.

## Measured on this box (2026-10-10)

Swift-Qwen3.8-27B Q4_K_S, Vulkan, `swift-qwen` flags, temp 1.0, 5 prompts × 2 seeds. Greedy output was identical
in every config, so all are lossless.

| Config | Prose | Code | Prose, thinking on | Code, thinking on |
|---|---|---|---|---|
| No speculation | 40.3 | 40.1 | — | — |
| MTP n 4 (old profile) | 57.9 | 104.4 | 72.5 | 77.0 |
| MTP n 2 | 61.6 | 84.1 | — | — |
| DFlash2 n 4 | 63.3 | 115.8 | 83.3 | 87.2 |
| **DFlash2 n 5 (adopted)** | 59.8 | 129.0 | 82.2 | 91.9 |
| DFlash2 n 7 | 51.6 | 134.0 | — | — |
| DSpark v2 n 4 | 42.8 | 84.3 | — | — |

- No gain: llama.cpp master `aa94f208` vs `a894dae9`, PR #25666, `--spec-draft-p-min`, n-gram ahead of DFlash2.
- Verify cost (llama-bench, batch of n tokens / 1 token): 2 → 1.04×, 4 → 1.14×, 5 → 1.13×, 8 → 1.31×, 16 → 3.88×.
  So verification is cheap; a DFlash2 cycle spends ~11 ms more on drafting, sampling and state. That overhead,
  not the verify kernel, is what a dedicated engine would have to cut (est. ~1.4–1.7× over today if it did).
- DFlash2 costs ~1.1 GB more VRAM than MTP: `swift-qwen` dropped from 112k to 96k.

## Rule: quality must not drop

Allowed: speculative decoding (the full model checks every token, so output quality is unchanged), faster kernels,
better VRAM accounting. Not allowed: smaller weights than Q4_K_S or KV below q8_0. Lossy options stay below, marked
**lossy**, for reference only.

## Answer

- **2× on prose is not reachable losslessly on this card today.** About 1.2× is realistic (57 → ~65–70 t/s),
  all from speculation. Code edits can pass 2× over plain decode already and reach ~150 t/s with DFlash2.
- **Full 262k does not fit losslessly** with Q4_K_S + q8_0 KV (~26.7 GiB). The lossless ceiling is ~170k
  text-only, ~140k with the image projector on GPU *est.*, once a hidden scratch buffer is budgeted (§4).
  262k only fits by giving up quality (3-bit weights or 4–5-bit KV).
- Lossless levers, in order: retune MTP (n 2–3), try DFlash2, build with Vulkan PR #25666, budget the FA scratch.
- A Strata-style engine for the 27B is not worth it. Strata's big wins come from MoE offload, which the 27B
  does not need. The rest transfers for maybe +10–25%, at weeks to months of kernel work.

## Where we stand

| | Today |
|---|---|
| File | `ukisai/Swift-Qwen3.8-27B-GGUF` Q4_K_S, 16.6 GB |
| Shape | 64 layers: 48 Gated DeltaNet + 16 attention (4 KV heads × 256), 1 MTP layer, vocab 248k |
| Plain decode | 40 t/s = ~660 GB/s, 69% of 960 GB/s |
| MTP n 4 | ~57 t/s prose, ~119 t/s code edits |
| Context | 112k (q8_0 KV); device-lost crashes at 128–150k |

Community check: another 7900 XTX user gets the same 40.2 t/s Vulkan / 33.6 ROCm on Q4_K_S
([r/LocalLLaMA 1w7fkhf](https://reddit.com/r/LocalLLaMA/comments/1w7fkhf/)). The box is not misconfigured.

## Why 2× on prose is hard

- **Bandwidth ceiling.** Plain decode reads every weight once per token. Realistic sustained bandwidth on gfx1100
  is 750–820 GB/s ([hipEngine ROOFLINE.md](https://raw.githubusercontent.com/shisa-ai/hipEngine/main/docs/ROOFLINE.md)).
  At 4.9 bpw that caps plain decode near 48–50 t/s *est.* Kernels alone give ≤1.25×.
- **Every RDNA3 engine lands at the same place.** hipfire 44 t/s (vendor), hipEngine 30.8 t/s on a W7900,
  EXL3 3.0 bpw 44.6 t/s, nasone32's HIP fork +10%. None beats Vulkan plain decode by much.
- **Launch overhead is small here.** hipGraph replay gave +2.6–3.9% on gfx1100 decode (hipEngine). Megakernel
  wins (Hazy 1.5×, Cohere 1.58×) are on 1B models and MoE, where kernels are tiny.
- **Speculation is capped by prose acceptance and verify cost.** Fitted to our numbers *est.*: llama.cpp pays
  ~0.27 of a decode step per extra verified token. At that cost, 2× needs ~4.6 accepted tokens per pass. The best
  drafters reach 3.7–4.3 on MT-Bench (prose-like) — on an H200, with base Qwen
  ([z-lab DFlash2 card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)). A cheaper verifier (~0.08) would make
  2× possible; that is upstream engine work, not a flag.

## Levers, ranked

### 1. Retune MTP: n 2–3, not 4 — minutes

- 3090 study, 25 prompts × 5 classes: prose +48.6% at n 2, +37% at n 3, +7% at n 5
  ([hackmd thc1006](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)).
- 4080: n 2 beat n 3, 55 vs 36 t/s prose ([1x1zhnx](https://reddit.com/r/LocalLLaMA/comments/1x1zhnx/)).
- 7900 XTX Q4_K_M: n 3 ≈ n 4, 69 t/s average across prompt types (1w7fkhf).
- PR [#29679](https://github.com/ggml-org/llama.cpp/pull/29679): RDNA3 slowdown at verify batches of 5–7 tokens.
- Expect 57 → 62–68 t/s prose. Low risk.

### 2. GSQ-RCO 3-bit weights — **lossy**, ruled out

KLD rises from ~0.016 to 0.051 (IQ3_S). Kept here because it is the only big non-speculative speed lever.

GSQ-RCO (ISTA-DASLab, [paper](https://arxiv.org/pdf/2604.18556)) learns the rounding and picks a GGUF type per
tensor under a size budget. The files are plain GGUFs; every type runs on mainline Vulkan.

| File | GB | KLD (wiki) | Task score vs BF16 | 7900 XTX decode |
|---|---|---|---|---|
| Today: Swift v1 Q4_K_S | 16.6 | ~0.016 (base Q4_K_S, bartowski) | — | 40 |
| Swift-1.5 GSQ-RCO **IQ3_S**-mtp | 12.1 | 0.051 (vendor) | 99.4% ([ByteShape](https://byteshape.com/blogs/Qwen3.8-27B/), indep.) | ~46–49 *est.* |
| Swift-1.5 GSQ-RCO IQ3_XXS-mtp | 10.4 | 0.098 (vendor) | 94.4% (ByteShape) | 53 measured ([#27623](https://github.com/ggml-org/llama.cpp/issues/27623)) |

- Repo: [`ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF).
  It is Swift **1.5**, not v1. v1 has only a transplanted layout
  ([`ticeclock/Swift-Qwen3.8-27B-RCO-GGUF`](https://huggingface.co/ticeclock/Swift-Qwen3.8-27B-RCO-GGUF)), KLD ~5–10% worse, unmeasured.
- Pick **IQ3_S**. IQ3_XXS is faster but loses ~5% on tasks.
- i-quant dequant eats about a fifth of the bandwidth gain (1.33× measured vs 1.64× predicted). Use Vulkan:
  on HIP the same file ran 32.6 t/s.
- It frees ~4.5 GB VRAM, which pays for full context (below).

### 3. DFlash2 drafter — low effort, test against MTP

- A ~2B block-diffusion drafter; one pass proposes up to 7 tokens. Merged in llama.cpp
  ([#27342](https://github.com/ggml-org/llama.cpp/pull/27342), 2026-08-27). Our build already has
  `--spec-type draft-dflash`.
- File: `incoai/Qwen3.8-27B-DFlash2-GGUF` Q4_K_M, 1.1 GB, plus its own KV.
- Code: clear win. 2.26× on 100 LiveCodeBench prompts, best width 5
  ([1vvncyh](https://reddit.com/r/LocalLLaMA/comments/1vvncyh/)).
- Prose: mixed. +23% at n 4 vs MTP n 2's +48.6% on a 3090 (hackmd); −11% at n 7. Strix Halo Vulkan: +40% over MTP.
- Trained on base Qwen3.8; acceptance on Swift is unknown.
- Costs 1–2.7 GB VRAM, so it competes with context. Best as a separate code profile.

### 4. Full 262k context — a profile change

**A hidden buffer first.** Vulkan's fast flash-attention path (K and V both q8_0) dequantizes one layer's K+V to
f16 into a scratch buffer (`ggml-vulkan.cpp:8034-8040`, checked in our build `a894dae9`). It is 4 KiB per context
token, grows as the context fills, never shrinks, and is not in llama.cpp's "projected" line: ~450 MiB at 112k,
1 GiB at 262k. This likely explains the spill → slow prefill → device-lost pattern past 110k.

| Config | Weights | KV | FA scratch | Other* | Total *est.* | Quality risk |
|---|---|---|---|---|---|---|
| **A** GSQ IQ3_S-mtp + q8_0/q8_0 | 11.3 | 8.5 | 1.0 | 1.5 | **22.3 GiB** | weights only (above) |
| **B** Q4_K_S (today) + `-ctk q5_0 -ctv q4_1` | 15.4 | 5.3 | 0 | 1.5 | **22.2 GiB** | KV: 92.7% vs 94.6% tail precision ([anbeeld](https://anbeeld.com/articles/kv-cache-quantization-benchmarks-for-long-context)) |
| ✗ Q4_K_S + q8_0/q8_0 | 15.4 | 8.5 | 1.0 | 1.8 | ~26.7 GiB | does not fit |

\* Recurrent state 0.73, compute ~0.5 at `-ub 512`, MTP draft KV at q4_0 0.28. Usable is ~23.5 GiB.

- Shared flags: `-ub 512 -np 1 -ctkd q4_0 -ctvd q4_0 --no-mmproj-offload` (projector on CPU; on GPU it adds ~1.1 GiB).
- **Both A and B are lossy** (3-bit weights; 4–5-bit KV). Lossless: Q4_K_S + q8_0 KV + q8_0 draft KV, sized with
  the scratch counted: ~170k text-only, ~140k with the projector *est.* Confirm with `GGML_VK` buffer lines and
  per-process GTT (`drm-memory-gtt`), not the projected line.
- If quality were negotiable, A would beat B: it keeps the fast q8_0 FA path and q8_0 keys (q4_0 keys made digits drift for one user,
  [ISTA #42](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/discussions/42)).
- Hadamard rotation of quantized KV is already on by default ([#21038](https://github.com/ggml-org/llama.cpp/pull/21038)).
- Precedent: a 7900 XTX ran IQ4_XS + q4_1 KV at 245k in 24.2 GB, 41 t/s at 65–131k, 36 t/s at 131–171k with MTP n 2
  ([1wvtxkx](https://reddit.com/r/LocalLLaMA/comments/1wvtxkx/)).
- **Speed at depth is the real cost.** Decode at 262k ~30–33 t/s plain *est.* (every token reads all KV).
  Cold prefill: 387 t/s at 108k, ~180 t/s at 150k; a cold 262k prompt takes ~15–20 min *est.* Follow-ups are cached.
- Device lost: also raise the watchdog, `amdgpu.lockup_timeout=10000,60000,10000,10000` (kernel default 2000 ms).
  Same crash open upstream: [#30010](https://github.com/ggml-org/llama.cpp/issues/30010) (7900 XTX, Qwen3.8-27B, ~104k).
- Quality past 128k is unverified for this family (no public RULER). Run our own needle test.
- Sparse decode (Strata's QSA) does not transfer: QSA is a trained indexer in Flash-Next. Luce KVFlash does
  training-free sparse KV for the 27B, CUDA only, 15/16 needles at 256k.

### 5. Cheap extras

- `RADV_PERFTEST=nogttspill` (Mesa ≥ 25.2): recommended by the RADV guide
  ([discussion #23295](https://github.com/ggml-org/llama.cpp/discussions/23295)); matters when VRAM is near full.
- Update llama.cpp past 2026-09-24: PR 27952 (int8 coopmat on RDNA3) +6% prefill on a 7900 XTX; decode flat.
- PR [#25666](https://github.com/ggml-org/llama.cpp/pull/25666) (open): stops MMVQ for speculative steps on Vulkan.
  Strix Halo: +12.9% and MTP acceptance 72–75% → 83–85%. Worth a local build.
- `ngram-mod` on top of a drafter: code only. It cost ~30% on prose with DFlash2; we already measured it slower with MTP.

## Rejected, with reasons

| Idea | Why not |
|---|---|
| Strata-style engine for the 27B | Strata's wins are MoE offload; the dense parts are llama.cpp's MMVQ/fattn. Strata hardcodes Flash-Next geometry (`layout.hpp:27-59`); ~30% reusable. |
| hipfire / hipEngine / Luce DFlash (HIP) | Plain decode no better than Vulkan; Luce on gfx1100 silently falls back to slow AR and segfaults at ~56k ([#457](https://github.com/Luce-Org/lucebox/issues/457)). Watch, don't adopt. |
| ROCm/HIP llama.cpp | −15–24% decode, worse on i-quants. |
| EXL3 3.0 bpw | Best size/quality curve (KLD 0.026 at 9.2 GiB), but ROCm support is days old (v1.6.0, 2026-10-07) and 44.6 t/s plain. Revisit in a few months. |
| ik_llama.cpp IQ*_KT | Only CPU and CUDA are supported. |
| Small draft model, EAGLE-3, PARD | −30% on a 3090; no heads for Qwen3.8. |
| Tree verification (STree, Weaver, DDTree) | Not in llama.cpp; chain beat tree on the 7900 XTX. |
| Pruned 20B variants | No benchmarks. |

## What a realistic stack looks like *est.*

| Profile | Changes | Prose | Code | Context |
|---|---|---|---|---|
| Today | Q4_K_S, MTP n 4 | 57 | 119 | 112k |
| Lossless chat | Q4_K_S, MTP n 2–3, PR #25666 build | ~65–70 | ~120 | 112k |
| Lossless long | same, text-only, scratch budgeted | same at shallow depth | — | ~170k |
| Lossless code | Q4_K_S + DFlash2 n 5 | ? | ~150+ | ~96k (drafter VRAM) |
| Lossy, for reference | GSQ IQ3_S, config A | ~70–75 | ~130+ | 262k |

To watch: EXL3 4.0 bpw scored KLD 0.008 at 12.3 GiB on turboderp's trace, at or below Q4_K_S quality while
~3 GB smaller. That could give 262k losslessly once its ROCm support (days old) matures.

Validate each with short runs (PSU caveat): `llama-perplexity --kl-divergence` against the current file, then
`llama-bench` tg128 and a few server replies with `--metrics` for acceptance.
