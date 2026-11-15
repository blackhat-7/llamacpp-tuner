# Two GPUs: what each card should do

Research of 2026-11-15, the day the RTX 4070 went in. No GPU benchmark was run for this page. "est." marks
arithmetic, "reported" marks a third party's number, "vendor" a model maker's own. Sources are at the end.

## Update 2026-11-16: one pool after all (the 4070 as Strata's expert helper)

The "Strata drives one make of card" limit below is gone: engine branch `opt12/nvhelper` (worktree
`~/Documents/projects/Strata-wt/r12-nvhelper`, commit `931a5cea`, local only) lets the HIP engine use the 4070 as its
`--expert-cache-device1` tier. Details and numbers: "The NVIDIA helper" at the end of this page.

## The machine

- RX 7900 XTX 24 GB on CPU PCIe Gen4 x16 (03:00.0). No display since 2026-11-15: Hyprland and Xwayland hold
  0 KiB of its VRAM (`/proc/<pid>/fdinfo`, checked 2026-11-15).
- RTX 4070 12 GB on chipset PCIe Gen4 x4 (08:00.0, root port 00:1d.0). Drives both monitors and CS2. Idle with the
  desktop: 405 MiB used.
- i5-13600K (6P + 8E), 94 GB DDR5-4800, dual channel. The root NVMe (models, Strata's 27 GB n-gram table) also hangs
  off the chipset (00:1a.0) and is 97% full (35 GB free). The other NVMe holds Windows.
- nvidia-open 620.70.06 (driver CUDA 13.5), no CUDA toolkit. ROCm 7.2.4.

## Answer

Use the cards as two lanes, not one pool.

- **XTX: the main model.** Flash-Next on Strata while working; `ornith-vk` while gaming.
- **4070: desktop and games, plus a small fast helper lane** and the side jobs that today force Flash-Next down.

## Why not one pool

- **Strata drives one make of card.** No XTX + 4070 layer split (`Strata/docs/MULTI_GPU.md`); its helper-card expert
  cache is same-make only (`docs/SECOND_GPU.md`); split serving refuses a session saved by CUDA and restored by HIP
  (`docs/SPLIT_SERVE.md`).
- **A weaker second card slows Strata even within one make.** 4080 + 3060 on Flash-Next: decode 55 -> 45 t/s, prompt
  590 -> 250 t/s (reported, r/LocalLLM 1wzyjbx). The 3060's layers run at its own speed and every prompt chunk waits
  on it.
- **Mixed AMD + NVIDIA llama.cpp splits lose to the big card alone when the model fits it.** R9700 + RTX 4500 Ada,
  Vulkan layer split: Qwen3.8 Q6_K 30 vs 40 t/s, Gemma 4 32 vs 50, Flash-Next Q4_K_M 9 vs 14 (reported, r/LocalLLaMA
  1xhq12y). Splits pay only when the model does not fit one card: the same 4080 + 3060 owner could load a dense 27B
  only with both cards (1wzyjbx). One report of a split beating a 5070 Ti alone (38 vs 26 t/s on a 9B Q8, 1qea29t)
  is not evidence: the model fits that card, and 26 t/s is far below its speed alone, so it likely spilled to the CPU.
- **The XTX alone already beats every pooled option.** Flash-Next on Strata: 122-166 t/s. A llama.cpp XTX + 4070
  split of Flash-Next: ~25 t/s (est., project notes).

- **llama.cpp can drive both makes, but even its best split loses to the XTX alone.** An XTX + RTX 2070 pair on
  Vulkan, after the tensor-parallel all-reduce merge (PR #25051, 2026-11-02): Qwen3.6-27B Q4_K_M generated 39.0 t/s on
  the XTX alone, 30.0 with `-sm tensor`, 24.7 with `-sm layer` (reported in that PR, verified 2026-11-15). The drivers
  cannot share sync objects, so every all-reduce goes through the CPU (~330 µs each before the PR).
- **One process with CUDA + ROCm works but is fragile.** `-DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CUDA=ON
  -DGGML_HIP=ON`, then `-dev ROCm0,CUDA0` (discussion #19544, undocumented). Both backends share one backend ID, and a
  bug that duplicated weights across them was closed without a fix (#25270, closed 2026-08-24). Tensor split
  (`-sm tensor`, PR #19378, experimental) is documented as good only on several NVIDIA cards. `-sm row` is gone from
  CUDA/HIP (PR #24216).

## What to run on the 4070, best first

### 1. A helper lane (cheap; the biggest agent-time lever)

Flash-Next answers one request at a time (`"parallel": 1`; two at once gave 92 t/s total vs 133 one after the other,
DECISIONS 2026-10-28). Everything that is not the main agent's turn waits in that queue or goes to Claude:

- Claude Code's Haiku role (titles, WebFetch summaries) runs on Flash-Next in claude-local's default mode.
- `haiku` and `sonnet` subagents run on Flash-Next.
- Auto mode's safety checks took 18-29% of Flash-Next's busy time when they ran locally (DECISIONS 2026-10-31). They
  now go to Claude Haiku, except in `claude-local --offline`.
- `embed` (Qwen3-Embedding-0.6B) runs on 4 CPU threads beside Strata's 13 pool workers. The CPU `side` lane left
  `local-flash` because it fought Strata for CPU and RAM.

A model on the 4070 runs that work in parallel and takes no CPU or DDR5 share from Strata. This is also what users
report as the best use of a second card: a parallel lane for sub-agents and helpers, not a bigger model
(r/LocalLLaMA 1uiuyyp, 88 points; replies in 1xhq12y).

| Model | File | For it | Watch |
|---|---|---|---|
| Gemma 4 12B QAT | `google/gemma-4-12B-it-qat-q4_0-gguf`, `gemma-4-12b-it-qat-q4_0.gguf`, 6.50 GiB | best instruction following of the small models (IFBench 74, tau2 69, vendor); users call its tool calls obedient; its MTP head gave +32-42% on the XTX (79 -> 104-112 t/s, measured 2026-10) | a chat-template fix landed in late July (reported); the official file is dated 2026-07-17, so check the template |
| Qwen3.5-9B | `unsloth/Qwen3.5-9B-GGUF`, `Qwen3.5-9B-Q6_K.gguf`, 6.95 GiB | tau2 79.1, BFCL-V4 66.1 (vendor); 8 of 32 layers attend, so long context is cheap (~190K with q8 KV, est.) | thinks by default; turn it off for the Haiku role |

- Speed on the 4070 (est.): ~50 t/s decode and ~2.8K t/s prompt at 4K, so a 50K-token page reads in ~20-35 s. Basis:
  a 3080 Ti measured 90.0 t/s decode and 3,325 t/s prompt (4K) on the same Gemma file (atomic.chat, llama.cpp b10988,
  CUDA 12.8); scaled by memory bandwidth (504/912 GB/s) and tensor throughput (~0.85x). Qwen3.5-9B Q6_K measured
  93.6 / 3,632 there. VRAM with f16 KV: Gemma 7,680 MiB at 8K, 8,632 at 64K. Measure before relying on any of it.
- Add `Qwen3-Reranker-0.6B` (Q8, ~0.65 GB) beside `embed` if code search wants a reranker.
- Wiring: the claude-local shim knows one local URL (`LOCAL_MODEL_BASE_URL`). It needs a second one for the Haiku role
  and `haiku` subagents (ai-harnesses). Pi's `local-side` provider already points at `pc:6869`: serve the helper there.
- Safety checks stay on Claude Haiku until a replay shows the local judge agrees: log a day of checks with Haiku's
  verdicts in the shim, replay them on the 4070, fail closed on any output that is not a clean verdict. No source
  measures false "safe" rates for these models.
- Quality: a 12B is much weaker than Flash-Next. `haiku` work (lookups, Explore) suits it; edits and judgment stay on
  Flash-Next or Claude.
- Gaming: stop the lane before CS2 (a Paseo window on the 4070 already makes CS2 stutter). Helper calls then fall
  back to the XTX model.
- Gain: not measured. The ceiling is the share of Flash-Next's busy time that is not main-agent turns. Measure with
  `bench/agent_time.py` before and after.

### 2. Jobs that today stop Flash-Next

These share the 4070's ~11.5 GB with the helper lane, so run one at a time.

- **H2O-Lightning-4B bulk classification** (~7 GB VRAM; today "only with Strata down"). Needs llama.cpp PR #30361
  built for CUDA.
- **Images.** SDXL-class models (argus's Illustrious-XL) fit. Qwen-Image 2.1 (15.6 GB in argus) does not without a
  smaller quant.
- **The 4B house-style SFT: borderline.** Unsloth says a 4B bf16 LoRA needs ~10 GB (vendor); the XTX smoke run peaked
  at 10.0 GiB; the 4070 has ~11.8 GB beside the desktop. Unsloth advises against 4-bit QLoRA on Qwen3.5.
  - Transformers takes the fast DeltaNet path only if all four kernels import: causal-conv1d (needs nvcc or a matching
    prebuilt wheel) and flash-linear-attention (`pip install flash-linear-attention`, Triton, no nvcc) (Unsloth PR #5434).
  - Speed vs the XTX: unknown. Time a 50-example run on the 4070 first.
  - Gain: Flash-Next stays up through the ~3 h run. Run it when not gaming: a game starting mid-run can OOM either side.
  - fla also ships a ROCm build (worked on gfx1151, reported); the XTX path still lacks causal-conv1d.

### 3. Free: a smaller Strata VRAM reserve

`--vram-reserve-mib 1744` keeps the old desktop share (~0.86 GB free at the old engine's 2000). The desktop has left
the XTX, so ~0.6-0.7 GB more can go to the expert cache. 4000 -> 1500 gave +10-11% decode (2026-10-12), so expect
~+2-3% (est.). Measure with the usual A/B.

### 4. Safer engine experiments

The XTX no longer drives the desktop, so a GPU hang there should cost only the model server. That lowers the risk
behind the parked persistent pre-expert kernel (+5-6% est.; "Needs the user" in the project notes). Hyprland still
opens the XTX: `AQ_DRM_DEVICES` lists 08:00.0 then 03:00.0 (`~/dotfiles/hypr/hyprland.lua`). List only the 4070 to
decouple it fully. Keep PM4 graph replay off: its page faults kill the server either way.

### 5. Long shot: the 4070 as Strata's expert helper

- Strata's helper caches let a second card compute the cold experts that the CPU pool computes now
  (`docs/SECOND_GPU.md`). That targets this box's bottleneck: the pool is ~33% of a code round and shares DDR5 with the
  DMA (~70 of 76.8 GB/s).
- Ceiling: removing the CPU/PCIe waits outright gives +17% code, +30% prose (measured, round 10/11).
- Same-make evidence: RX 6800 alone 37.5 t/s, with an RX 6600 helper 40.6-44.2 t/s (+8-18%, Strata docs).
- No CUDA helper for a HIP engine exists. One Reddit user reports an LLM-built fork with an MI50 + RTX A4000
  (GLM-5.3 Flash 14 -> 17 t/s, unverified, 1xjyulm).
- Cost: weeks of engine work, and the 4070 is then tied to Strata whenever Flash-Next runs. Realistic +5-15% (est.).

## Don't

- Pool both cards for one model (above).
- Run CPU-offloaded models on the 4070 while Strata decodes: they take the CPU and DDR5 the expert pool needs, and the
  4070 shares the chipset link with the root NVMe.
- Move Flash-Next's prompt reads to the 4070: its x4 link and lower compute make it the slower prefill card.
- Put a draft model on the other card. llama.cpp has `-devd` / `--spec-draft-device`, but MTP, DFlash and EAGLE-3
  drafts crash on another device without open PR #26636, and one cross-device draft was slower than the same draft on
  one card (18.5 vs 26 t/s, issue #31830, reported). Flash-Next drafts inside Strata anyway.

## Gotchas

- **`lct setup` now picks the CUDA release** because `nvidia-smi` exists (`llama.py` `_release_assets`). That build
  cannot see the XTX, so `lct setup --force` would break every profile on the managed build (`swift-qwen`, `side`,
  `embed`, `swarm`). Put CUDA llama.cpp in its own folder and pass `--server`, as `ornith-vk` does.
- **Vulkan numbers both cards:** `GGML_VK_VISIBLE_DEVICES=1` is the XTX, `0` the 4070 (as in `ornith-vk`).
- **Never export `CUDA_VISIBLE_DEVICES` globally.** ROCm honours it too and can hide the XTX from HIP processes
  (Strata, ComfyUI) (ROCm 7.2.4 docs, "GPU isolation"). Set it per process, or use llama.cpp's `-dev CUDA0`.
- **Disk:** 35 GB free on `/`. One helper model (~7 GB) fits; check before more.
- **4070 VRAM:** the desktop takes ~0.4 GB idle, more with browsers. Size the helper's context for ~10.5 GB.

## Next steps (proposed; not in `PLAN.md`)

1. CUDA llama.cpp in its own folder; move `embed` to the 4070. Done when `embed` answers on CUDA0 with no CPU threads.
2. Helper profile on :6869 (Gemma 4 12B QAT, then Qwen3.5-9B); a short `llama-bench` on the 4070. Done when decode
   and prompt t/s are in `PROGRESS.md`.
3. claude-local: a second local URL for the Haiku role and `haiku` subagents. Done when a session's title and an
   Explore subagent hit :6869 while Flash-Next answers the main turn.
4. Safety-check replay against Haiku's verdicts. Done when the disagreement count is known.
5. Strata reserve A/B (1744 vs ~1100). Done when decode is measured both ways.
6. H2O, images or the SFT on the 4070, when needed.

## Sources

- Reddit (read 2026-11-15 through the bestiary Reddit tool; reddit.com/r/<sub>/comments/<id>):
  r/LocalLLaMA 1xhq12y (NVIDIA + AMD setups), 1uiuyyp (second card for parallel agents), 1qea29t (one build with
  CUDA + ROCm + Vulkan via `GGML_BACKEND_DL`; its split number rejected), 1xjyulm (mixed AMD/NVIDIA Strata), 1wm7kws and 1wruh65
  (`--spec-draft-device`); r/LocalLLM 1wzyjbx (4080 + 3060 on Strata).
- Strata (`~/Documents/projects/Strata`, c62d5e3a): `docs/MULTI_GPU.md`, `docs/SECOND_GPU.md`, `docs/SPLIT_SERVE.md`.
- llama.cpp (github.com/ggml-org/llama.cpp): PR #25051 (Vulkan all-reduce, merged 2026-11-02; the XTX + RTX 2070
  table is in its comments), PR #19378 (`-sm tensor`, merged 2026-04-09), issue #25270, discussion #19544,
  PR #24216, PR #26636 (open), issue #31830, `docs/multi-gpu.md`.
- Helper models: Hugging Face API (file sizes, 2026-11-15); model cards for `google/gemma-4-12B-it` and
  `Qwen/Qwen3.5-9B`; atomic.chat "best local LLM 12GB" (2026-09-30, RTX 3080 Ti, llama.cpp b10988).
- Fine-tuning: Unsloth Qwen3.5 docs (unsloth.ai/docs/models/qwen3.5/fine-tune); Unsloth PR #5434 (merged 2026-05-18);
  github.com/fla-org/flash-linear-attention; github.com/Dao-AILab/causal-conv1d releases.
- This project: `DECISIONS.md` (2026-10-28, 2026-10-31), `docs/flash-next-speed-research.md` (rounds 10-11),
  `skills/tune-local-model/references/host-7900xtx.md`.

## The NVIDIA helper (built 2026-11-16)

- **What:** `STRATA_NV_HELPER=<path>/libstrata_nvhelper.so` + `--expert-cache-device1 auto`. The plugin is a CUDA build
  of Strata's own expert kernels (target `strata_nvhelper`, sm_89, CUDA 13.4 in `~/cuda-13.4`, built with the system
  `/usr/bin/g++`: the Nix gcc's loader cannot see `libcuda`). The HIP engine `dlopen`s it; only `snv_api` is exported.
  The 4070 holds the 6,605 next-hottest experts (10.55 GiB) after the XTX's 10,031; the CPU computes the rest.
- **Per layer:** the CPU quantizes the activations to q8_1 (bit-identical to the GPU quantizer), one kernel pulls
  metadata + rows over the x4 link, the grouped kernels run, one kernel writes the rows back in 16-byte stores. The
  launch runs inside the CPU pool's first phase. The 4070 spends ~55 us a layer (stage-in 12, experts 31, rows out 13).
- **Adaptive:** hot CPU misses swap into the 4070 (`STRATA_NV_ADAPT_SWAPS`, default 16 a round) on their own copy
  stream; the expert arena is pinned for CUDA too (double registration works).
- **Measured** (generate mode, follow texts, priced windows, `tmp/dual/price.py`):

| | code | prose | edit |
|---|---|---|---|
| fixed ranking (`--adapt-every 0`), XTX → XTX + 4070 | 153 → 170 (+11%) | 109 → 131 (+20%) | 133 → 187 (+40%) |
| adaptive tiers (`--adapt-every 4`, as the server runs) | 158 → 165 (+4%) | 126 → 130 (+3%) | 150 → 190 (+26%) |

- **Gotchas:** writing results to host memory one float per warp cost 34 us of 50 KB (fixed: stage, then 16-byte
  stores); a metadata `cudaMemcpyAsync` queued behind the adaptive tier's fills on the copy engine (104 us a layer;
  fixed: kernels move the bytes); `pgrep -f`/`pkill -f` patterns that appear in your own command kill your shell.
- **`--spec 6` pays only with the helper** (adaptive, own windows priced, `tmp/dual/price_own.py`): code 4.10 vs 3.26
  tokens a window, +5.7%; prose +3.0%; edit -2% (noise). Without the helper `--spec 6` loses (code 92 vs 135 t/s):
  T5-T6 windows route to more cold experts, which the 4070 now takes. Helper + spec 6 vs today's setup, adaptive:
  code 158 -> 173 (+9%), prose 126 -> 133 (+6%), edit 150 -> ~200 (+32%).
- **Rounding, not quality:** the helper flips the greedy token at 10-11 of 600 code and 18-39 of 601 prose positions
  (teacher-forced, `r6q_flips.py`); a 210-slot change of the XTX cache alone flips 6 and 20, at the same positions.
- **The RDNA3 fork** (`nicejunjie/Strata-rdna3`, `tmp/dual/rdna3-eval.md`) is 7-10% slower than our engine on one XTX;
  its megakernel is for two-XTX tensor parallel and needs Q8_0 attention weights. Nothing to port.
- **4-bit experts (quality route):** today's routed experts are 2-bit (gate/up IQ2_XS/IQ2_XXS, down IQ4_NL/Q2_0).
  Hybrid pack `Strata-data/packs/swift-abl-hybrid-iq4_xs` = the abliterated dense weights + Swift 1.5 IQ4_XS experts
  (`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF`, 91 GiB, sha256 checked; the 96 router tensors are byte-identical, so
  it is one model). Builder: `tmp/dual/hybrid/make_hybrid.py`. With the helper, resident RAM mode, `--spec 6`: code
  113 t/s (-28% vs 158): the CPU computes 3.1 bigger experts a layer. In resident mode the 4070's tier must stay
  static (`STRATA_NV_ADAPT_SWAPS=0`, `STRATA_DISJOINT_ADAPT=1`): its evicted experts are in no RAM copy, so the CPU
  re-read them from the SSD (28 t/s).

### Quality: the 2-bit `down` projections were the damage (`bench/strata-packs/`, Strata's `tools/quant_quality.py`)

Teacher-forced on Strata's fixed set (code, docs, English, multilingual, agent transcripts; 4,995 positions), the full
4-bit hybrid as reference. Perplexity of the text, lower is better. Expert bytes vs today's 40.0 GiB.

| experts | bytes | perplexity | of the 4-bit gain |
|---|---|---|---|
| today: GSQ-RCO "IQ3_XXS" (gate/up IQ2-IQ3, 30 layers Q2_0 down) | 40.0 GiB | 4.793 | 0% |
| A: 8 all-2-bit layers fully 4-bit | 45.1 | 4.500 | 36% |
| C: the 18 IQ4_NL-down layers fully 4-bit | 44.9 | 4.479 | 39% |
| **D: only `down` to IQ4_NL in the 30 Q2_0 layers** | **46.6 (+16.5%)** | **4.153** | **79%** |
| E: D + layers of A fully 4-bit | 50.0 | 4.114 | 84% |
| B: the 30 Q2_0-down layers fully 4-bit | 56.0 | 4.149 | 80% |
| full IQ4_XS experts | 60.9 | 3.986 | 100% |

Today's model picks another top token than the 4-bit one at 21% of positions (KL 0.35). mixD
(`Strata-data/packs/swift-abl-mixD`, `mix_down.py`: v4 per-role shards, gate/up from the abliterated GGUF, down from
the IQ4_XS shards) is the pick: E is 3-8% slower for 1% lower perplexity.

### Speed of mixD + the 4070 (adaptive, `--spec 6`)

| | code | prose | edit | 35K prompt read |
|---|---|---|---|---|
| generate mode: today on the XTX (spec 4) → mixD + 4070 | 158 → 170 (+8%) | 127 → 133 (+4%) | 152 → 173 (+14%) | |
| server, median of 3: today `local-flash` | 163.2 | 125.7 | 141.2 | 2,546 t/s |
| server: today's model + 4070 (`local-flash-dual`) | 171.2 (+5%) | 126.8 (+1%) | 170.5 (+21%) | 2,482 |
| server: mixD + 4070 (`local-flash-mixd`) | 165.7 (+1.5%) | 123.6 (-1.7%) | 143.2 (+1.4%) | 2,153 (-15%) |

The server's learned expert profile already keeps the XTX near 95% hits, so the 4070's decode gain there is smaller
than in generate mode. A 1000 MiB XTX reserve (vs 2000) gave +523 slots and no gain (+1.4% / -0.9%).

### Tail skip beside the helper (engine `e7d42fa9`, both dual configs)

`STRATA_ROUTE_TAIL_SKIP=7` (upstream's CUDA serve default; off on HIP and, before this, off whenever a helper ran)
skips a cold expert that every token of the window routes only at rank 7-9 (the three smallest top-10 weights), and
never one a GPU holds. mixD, generate mode: the CPU's cold experts 2.4 -> 1.5 a layer, code +3.3%, prose +2.8%.
Quality set: perplexity 4.153 -> 4.096 (within noise), KL to the 4-bit reference 0.179 -> 0.181. It changes output
like any rounding move, so it is a choice: `"STRATA_ROUTE_TAIL_SKIP": "0"` in the config turns it off.

| server, median of 3 | code | prose | edit | 35K prompt read | perplexity |
|---|---|---|---|---|---|
| today `local-flash` | 163.2 | 125.7 | 141.2 | 2,546 | 4.79 |
| `local-flash-dual` (+ tail skip) | 175.9 (+8%) | 132.7 (+6%) | 189.8 (+34%) | 2,575 | 4.79 |
| `local-flash-mixd` (+ tail skip) | 173.0 (+6%) | 134.7 (+7%) | 172.5 (+22%) | 2,509 | 4.15 |

mixD's earlier 2,153 t/s prompt read was a cold page cache (the IQ4_XS downs read from the SSD); warm it is 2,509.

`STRATA_HC_Q8=1` (8-bit hyper-connection weights for decode, quantized at load; on by default upstream for RDNA4 and
Strix Halo, off for gfx1100): mixD + tail skip, generate mode, 2 runs each: code +1.0%, prose +1.6%. Now in both dual
configs. Quality noise: the set's "agent" segment swings up to 20% in perplexity between near-identical runs (a few
very unlikely tokens); code / docs / English / multilingual are stable to ~1%. On those, tail skip and HC_Q8 are within
noise, and mixD vs today is -8% to -14% in every one (code 2.41 -> 2.13, docs 11.6 -> 10.3, English 5.57 -> 4.80,
multilingual 2.94 -> 2.72).
