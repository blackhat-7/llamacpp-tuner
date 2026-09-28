---
name: select-local-model
description: Compare and choose the best current local LLM or VLM for a user's workloads and system before a model is fixed. Use for model recommendations or comparisons involving chat, research, documents, coding, agents, tools, vision, multilingual, privacy, or uncensored use. Once a specific model is chosen, use tune-local-model for its artifact and runtime settings.
---

# Select a Local Model

Give an evidence-backed shortlist, not a model dump. Releases change weekly: search current sources, never rely on memory. For this user's host, measured numbers and past verdicts are in `../tune-local-model/references/host-7900xtx.md`; read it first and start from its shortlist.

## 1. Define "best"

Ask only for what is missing:

- weighted use cases with real example tasks (chat, agentic coding in a harness, long documents, vision)
- hard limits: decode t/s floor, context floor, whole-model-in-VRAM or offload allowed, uncensored, license
- accelerator, VRAM, RAM, CPU, OS, backend; other VRAM users (desktop, wallpaper, second server)
- the harness: its system-prompt size (agent harnesses send 10–27k tokens every cold turn), whether it needs tool calls, images, parallel requests

Separate hard requirements from preferences. Quality usually beats speed: users reject a faster model that is visibly dumber.

## 2. Build a current candidate set

Search model hosts, official cards and runtime release notes. For each candidate verify the exact release, instruct/reasoning mode, dense vs MoE (total params = memory, active params = compute), file sizes, context, modalities, chat template and license.

- A quantizer's model card describes the quant. Read the base model's card for capabilities.
- Confirm the runtime supports the architecture on the user's backend (mainline llama.cpp, not a fork; Vulkan/HIP kernels exist). Unsupported arch = reject.
- Names lie: a "Qwen3.8-9B-Distill" was a Qwen3.5-9B fine-tune; a rumored "Qwen3.8-35B-A3B" did not exist.
- Do not transfer evidence from a nearby size or family member.

## 3. Check fit with arithmetic, then measure

Memory = all weights + KV at target context + recurrent state + compute buffer (grows with context × ubatch) + projector (loads lazily on first image) + draft/MTP head + other users. Per-token KV cost comes from the attention-layer count, KV heads and head dim, not total layers: hybrid models (DeltaNet/SSM + few attention layers) are cheap per token.

- Decode speed is bounded by bytes read per token. Fully on GPU: weight size / VRAM bandwidth. Experts in system RAM: dual-channel DDR5 gives ~55–60 GB/s effective, so a 30 t/s floor allows only ~1 GB of RAM reads per token (A3B-class at Q4). An A6B with most experts in RAM ran ~15–18 t/s however it was tuned.
- Prefill cost scales with active params × prompt length. Cold turns in an agent harness are dominated by it.
- Dense models slow with depth (27B: 40 → 36.7 t/s from 0 to 30k); extrapolate before promising a speed at 200k.
- Treat advertised max context as a capability. Reject configs that only fit on paper.

## 4. Compare on evidence the user trusts

Lead with independent evidence, then label vendor numbers as vendor-reported:

1. community reports: r/LocalLLaMA (RSS works with a browser User-Agent, e.g. `https://www.reddit.com/r/LocalLLaMA/search.rss?q=...`; slow down on 429), Hugging Face discussions, Hacker News
2. third-party boards: Artificial Analysis, SWE-rebench, Terminal-Bench, KLD/perplexity studies by quantizers
3. vendor tables last

Report how divided the community is. Vendor tables have hidden looping, knowledge drops, broken MTP heads and tool-call failures ("benchmaxxed").

- Agentic harness work needs tool-call validity and instruction following, not chat scores. Test in the real harness when in doubt.
- Fine-tunes: require provenance and retention evidence. Abliterated/"uncensored" builds cost a little (MMLU −0.2, ARC −1.2 in one case) and have no agentic evals; "fewer thinking tokens" tunes can condense too hard for long tasks. Verify "uncensored" claims: some builds still refuse.
- Quant quality: Q4_K_M ≈ BF16 on agent benchmarks for a 27B; Q2 drops clearly. Prefer KLD studies over perplexity.

## 5. Recommend

Return at most: **default**, **quality alternative** (only if worth its cost), **speed alternative** (only if meaningfully different). For each: exact repo and file, evidence with links, deployment class (full GPU / expert offload / CPU), expected speed marked measured or estimated, the main caveat, confidence. Say plainly when the current model remains best. Keep the answer short; lead with the verdict.

When evidence cannot decide, propose 2–3 real tasks run on each model with the same harness, template, sampling and context. Never present an estimate as a measurement. Hand the choice to `tune-local-model`.

## Rules learned the hard way

- Check the user's real bottleneck first. "Too many compactions" was fixed by more context on the same model, not a new model.
- Spare system RAM rarely makes a VRAM-bound dense model faster. It is best used for the prompt cache, a CPU side model for background jobs, embeddings, and fast model swaps.
- One GPU server slot shared by chat and background jobs (titles, memory reviews, a safety classifier) evicts the chat's prompt cache; plan a second slot or a CPU side model.
- A CPU side model suits only short prompts: CPU prefill is ~90 t/s for an A3B, so a 20k-token prompt takes minutes.
