---
name: tune-local-model
description: Given a specific local LLM or VLM, choose its best quantized artifact and runtime arguments, then prove them with staged benchmarks. Use for GGUF/quant selection, context sizing, offload, KV cache, batching, speculative decoding, multimodal placement, prompt caching, tokens-per-second tuning, or OOM/crash diagnosis. If the model itself is not chosen, use select-local-model first.
---

# Tune a Local Model

Optimize for required quality, speed at realistic depth, context and concurrency — not maximum allocation or one short benchmark. For this user's host, measured baselines, working commands and past failures are in `references/host-7900xtx.md`; read it before measuring anything.

## Safety first

- **No long GPU runs without asking.** This host crashed under sustained load (PSU). Short loads and sub-minute tests are fine; ask before multi-minute prefills or sweeps. Estimate a run's length from a short prefill first.
- Before any fit test: `lct ps`, `ss -ltnp`, and VRAM. A resident server leaves a second model ~50 MiB.
- Keep 10+ GB RAM free when launching from an agent session; an OOM kill can take the session with it.
- Stop llama-server with SIGINT (`lct down`); SIGTERM was ignored for 120 s+. After `ErrorDeviceLost` it can hang holding VRAM: `kill -9`. Killing a wrapper (`uv`, `lct serve`) can orphan its child; check `ps`.
- `setsid` does not escape systemd's OOM group kill: a server OOM can take the agent session down.
- Read the host reference before suggesting any lever. Do not repeat generic community claims it already disproves (f16 KV faster, ROCm 2× prefill, `-b 4096`).
- Run short measurements yourself rather than asking the user to paste output; the user's past loop was paste command → ask for faster → paste result.

## 1. Fix the target

Record exact model, artifact, backend and build (commit), VRAM, RAM, CPU, modalities, slots needed, context floor, typical cold prompt (agent harness system prompts: 10–27k tokens), acceptable cold latency, and decode floor at realistic depth. Read the current server command and the client's context/output limits; the client must compact below the server context. Check the build's `--help` — flags get renamed (`-mmp 0` → `-lm none`) and defaults change.

## 2. Choose the artifact

Inspect real files; never invent filenames or fall back to another quant. Ladder: Q4_K_M (usual winner on 24 GB) → one higher quant only if context and KV survive → one smaller only if forced. A bigger quant that forces lower-precision KV cancels its gain. Importance-matrix and unsloth "UD" quants: compare by KLD, not name. Confirm a speculative head (MTP) is trained and supported by the build.

## 3. Budget memory before loading

Weights + KV (attention layers × KV heads × head dim × 2 × bytes, per token) + recurrent state (fixed) + compute buffer (context × ubatch) + MTP/draft KV (defaults to f16: pass `-ctkd/-ctvd`) + projector (lazy, up to ~1.2 GB on first image) + desktop. Leave ≥1 GB free *after* the projector.

- Trust llama.cpp's `projected to use N MiB` and `buffer size` lines; sysfs VRAM under-reports until memory is touched.
- `-ngl all` disables the auto-fitter; a tight config just OOMs or spills.
- **Check spill per process**: `drm-memory-gtt` in `/proc/<llama-server pid>/fdinfo/*` is GPU memory living in system RAM. A spill of 1+ GB plus a deep prompt caused GPU watchdog resets (`vk::Queue::submit: ErrorDeviceLost`) twice.
- Always pass `-np` explicitly (default is auto, e.g. 4 slots).
- Size weights from the GGUF tensor list, not from model cards or forum posts. Other resident servers count against RAM as well as VRAM: an evicted expert set rereads from SSD.
- Prompt-cache entries are big (~2 GB for 35k tokens on a 27B); `--cache-ram` holds only a few conversations and adds to RSS.

## 4. Baseline, then one change at a time

Baseline: `-fa on`, one slot, target context, q8_0 K and V, default batches, embedded template (`--jinja`). Then vary one thing, recording build, command, mean ± spread and peak memory. Use `llama-bench -o jsonl` (never custom timers):

```bash
llama-bench -m M -p P -n 0 -r 3 [ARGS]          # prefill at a harness-sized P
llama-bench -m M -p 0 -n 128 -d DEPTH -r 3 [ARGS] # decode at realistic depth, one depth per run
```

Order of levers, biggest first (details and numbers in the reference):

1. **Platform**: Resizable BAR on (16 → 67 t/s once); newest llama.cpp build (+5–16% prefill); backend A/B (Vulkan vs HIP differ per model: measure decode *and* prefill at depth).
2. **Speculative decoding**: built-in MTP (`--spec-type draft-mtp --spec-draft-n-max 3–4`) is the largest decode win on models that ship it; verify acceptance in the server log. N-gram drafting helps only copy-heavy edits: it hurt a 27B whose MTP already accepted ~all drafts, but helped a slow offloaded MoE (16 → 20 t/s) — A/B it.
3. **KV type**: backend fast paths can require matching K/V types (Vulkan FA: both q8_0; q5_1 V or f16 was slower at depth).
4. **Batches**: sweep `-b/-ub` pairs separately on a bounded prompt; stop at the plateau. `-ub` is silently capped at `-b` (default 2048), so raise both. Large ubatch helps prefill (MoE offload most) but grows the compute buffer and the longest GPU job — keep `-ub 512–1024` at long context on Vulkan.
5. **Threads (CPU/offload only)**: pin to P-cores (`-t 12 -C 0xfff` on 6P+8E); E-cores hurt decode; `--poll 100` helped offload decode.
6. **MoE offload**: `-ngl 999 -ncmoe N` (each layer ≈ 1.5 GB VRAM ↔ 1.6 GB RAM); `-lm none` needs `--no-host` (pinned host memory is capped at ~half of RAM); `-lm mmap` is much slower. A purpose-built engine (Strata) beat llama.cpp ~3× decode and ~40× prefill on one big MoE; check for one.

The first request after a load is not a measurement: lazy-loaded tensors (`-lzm auto`) stream from disk.

## 5. Deployment checks in the real server

- **Depth**: confirm prefill and decode near the context you will actually reach. Known Vulkan cliff near 131k (llama.cpp #27734): set `GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296` (`lct serve --env`).
- **Prompt cache**: send a cold turn then an append-only follow-up; the log must show `selected slot by LCP similarity` with `f_keep` ≈ 1 and only new tokens processed. `selected slot by LRU` means a miss.
  - Hybrid (recurrent) models cannot trim state: reuse needs an exact prefix up to a checkpoint. `--cache-reuse` is auto-disabled for them and with a projector — drop it. `-cms 2048` bounds rollback after an interrupted reply.
  - `--cache-ram N` keeps prompts in RAM; too small → rereads, too big → OOM with large offloaded models.
  - **Two clients** (chat + classifier/titles): `-np 2 --kv-unified --no-cache-idle-slots`. Without the last flag, every new request clears idle slots and the chat rereads its whole prompt.
  - Client-side prompt changes (dates, tool lists toggled) defeat caching; check the harness. Background requests (titles, autoname) default to the session model in some harnesses and evict its cache.
  - Cache save/restore lines only print at `-lv 4`.
- **Vision**: measure VRAM after the first image, not at startup. Without `--mmproj`, image requests return HTTP 500 and the model learns from its history that it "cannot see"; keep the projector when swapping models behind a port a vision client uses.
- **Client contract** (read the client's config, then check `/props` and `/slots`):
  - Its context and compaction budget must sit below the server's per-slot context (OpenClaw budgets ≈ n_ctx − 20k; Pi compacts at ctx − 16k). An agent compacting every turn loops.
  - Thinking needs `max_tokens` ≈ 32k; at 4k, half the turns ended mid-thought.
  - Claude Code's auto-mode safety checks and helper calls (titles, summaries) send no `thinking` and a small `max_tokens`. A server that thinks anyway gives no verdict, and every tool call waits for retries and a timeout. On Strata set `"anthropic_thinking": "on_request"`.
  - Effort levels are template-specific: read `/props` → `chat_template`. Some templates only have on/off; some ignore `low` and think in full. `--reasoning-budget N` is the only hard cap; `reasoning_effort` just adds a sentence.
  - Clients in containers or on other hosts need `--host 0.0.0.0`. `-a NAME` gives every model the same id.
  - When a harness stalls, repeat the call against a cloud model before blaming the server.
  - Test fallback chains (local → cloud) in both modes; a local-first fallback once broke every cloud session.
- **Soak**: a few real multi-turn sessions; watch GTT spill, RAM, temperature and the kernel log (`journalctl -k | grep amdgpu`).
- **Where agent time goes** (Strata + Claude Code): `python3 bench/agent_time.py <strata config>` splits each engine run into the agent's writing, reading, cold re-reads and the client's other calls. Fix the biggest share first; it was not decode speed.

## 6. Report

Return: exact download command, exact serve command (or `lct` profile), measured cold/cached prefill and decode at stated depths, peak VRAM/GTT/RAM, why each non-default flag won, and one fallback each for OOM and for low speed. Remove flags that repeat defaults unless pinning them matters. Mark every estimate as an estimate. Keep it short: flag diffs, not essays.
