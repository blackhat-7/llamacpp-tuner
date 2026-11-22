# Inkling-2-Mini on the 7900 XTX alone

Research of 2026-11-21 and 2026-11-22. Measured on this box unless marked *est.* (arithmetic) or "reported"
(Reddit and Hacker News, the first two days after the release). Speeds are decode t/s, greedy, over all generated
tokens; the model always thinks.

## Now

Stack `local-inkling` = profile `inkling` (Loom, package `tmp/loom/kv8w4096`, preset `xtx-kv8-196k`) + `embed`.
It is the user's one-GPU stack: best quality and context on the XTX alone, with the 4070 free for the desktop and games.

| depth | code | prose | drafter off |
|---|---|---|---|
| short (1.3K) | 140 | 121 | 110 |
| 64K | 114 | 104 | 93 |
| 195K | 90 | 75 | 73 |

- Context 196,608 tokens. Reading a fresh 195K prompt takes 2.3 min (~1,420 t/s); a growing chat reuses its cache
  (a 64K follow-up starts in 2 s, not 29 s).
- 0.3 GB RAM, 24.0 of 24 GB VRAM, idle CPU 0%. Starting it asks `sudo` once (see "Serving through lct").
- Quality, reported only (no own eval yet): about Qwen3.8-27B level, below Flash-Next on average. Strengths: prose,
  fewer thinking tokens. Weaknesses: refusals and preaching, failed tool calls (92% vs 97-100% in one 37-question
  eval), giant one-shot bash commands. Thinking loops stop with temp 1.0, top-p 1.0, top-k 0, min-p 0.

## The model

- 26.2B total, 3.4B active. The middle 12 of 24 layers run three times: 48 layer visits, 5.4B effective active.
- 8 visits use global attention (physical layers 5, 11 x3, 17 x3, 23); 40 use a 1,024-token sliding window.
  24 query heads, 8 KV heads, head dim 128: the global KV cache is 32 KiB per token in f16, 6.4 GB at 195K.
- `unsloth/Inkling-2-Mini-GGUF` UD-Q4_K_XL (16.4 GiB) keeps 10.7 GiB of experts in NVFP4, the format the model was
  trained in, and the rest (4.0 GiB of experts, embeddings, attention) in Q8_0. Unsloth: KLD 0.0107 vs UD-Q8_K_XL.
- `enable_thinking=false` and `reasoning_effort: none` do not stop its reasoning.

## llama.cpp: bound by small ops at 30-34 t/s

- Own build `tmp/llama.cpp-inkling` (PR #32695 + Unsloth's per-expert scales #377 + `mtp-fixes`, one duplicated
  tensor creation removed after the merge): 23.7 t/s. The SITU activation scaled strided views, which Vulkan rejects,
  so 96 tiny ops a token ran on the CPU (194 graph splits). A `ggml_cont` before it gave 34 t/s Vulkan, 30.5 HIP.
- ~13,200 graph nodes per token (275 per layer visit). UD-IQ2_XXS (7.9 GiB) decodes 36 t/s: bandwidth is not the
  limit, launch overhead is.
- Unsloth's prebuilt b12588 (ROCm, `tmp/llama.cpp-unsloth-b12588`): 30 t/s without a drafter; DSpark n3 code 57 /
  prose 39 greedy, 44 / 31 at temp 1.0; 29 / 22 at 195K. 200K fits with `-ub 64` (23.5 GiB), but reading 195K took
  18 min (180 t/s). Our own build's DSpark made decode slower (15-25 t/s).

## Loom

- AMD's engine: `hrx-loom` 0.2.0 (source ROCm/hrx-system, Apache-2.0) in `tmp/loom/venv` (Python 3.12). Package
  `loom-fun/unsloth-Inkling-2-Mini-loom`, generated on Strix Halo; it runs on gfx1100 unchanged. Its programs are
  `.loom` source compiled at load, so kernels can be edited in place. It bundles its own driver library (no ROCm).
- As shipped on the XTX: 111 t/s short, 44 at 194K (f16 KV). Drafter on (143 / 117 short) fits only up to 32K.

### Tuning (2026-11-22), all rounding-level except the 8-bit cache

1. **Decode attention kernel `g_ink_attn_split_v2`.** Each key is loaded once for all 3 query heads of its KV head;
   waves share 16-key pages; the softmax runs once per page. 541 -> 855 GB/s with f16 at 195K (1.48 -> 0.95 ms a
   layer); 64K end to end 70 -> 77.6 t/s. Partials match the old kernel within 1e-4.
2. **8-bit global KV cache.** int8 codes plus one f16 scale per 32 dims (q8_0 style; `tmp/loom/kv8-format.md`),
   1,088 bytes a position instead of 2,048. The v2 kernel reads it at 750 GB/s (0.565 ms a layer at 195K).
   195K drafter off: 44 -> 73 t/s. Memory left room for the drafter and the prefix cache at 196K (25.17 of 25.75 GB).
   Quality: greedy output differs from f16 at the 2nd token, a near-tie (60 samples at T=1: "pasted" 33% / "asks"
   27% with f16, 27% / 33% with 8 bits); recall 3/3 at 195K; text coherent.
3. **Verify spans through v2.** Drafter-on verification (2 tokens) used the old one-key-a-step span kernel;
   sending it through v2 one token row at a time took 64K drafter-on code from 73 to 102 t/s.
4. **Drafter window of 4,096 tokens.** The drafter's own full-context attention ran at ~61 GB/s: 13.45 ms a pass at
   195K. Attending only to the last 4,096 positions makes it 0.19 ms; output stays exact (the model verifies every
   token) and acceptance barely moved (code 0.80 -> 0.77, prose 0.58 -> 0.62). 64K drafter on: 98.5 / 86.4 ->
   114 / 104; 195K: 90.5 / 75.1. A 16,384 window gave 110 / 99 at 64K.

Not integrated: **span-v2**, a verify kernel that shares K/V loads across the 2 span tokens. It passes 8 tests
against the old span kernel; 0.92-0.96 ms vs 1.13 ms a layer for two v2 rows at 195K (*est.* +8% at 195K). It is
slower than v2 for a single token, so it would serve only verify spans.

### Files

- `engines/loom/build.sh` rebuilds the served package `$LCT_HOME/loom/kv8w4096` (`tmp/loom/kv8w4096` here) from the
  pinned upstream package. `engines/loom/README.md` lists the pins, each change, and how to check a build.
- `experiments/inkling-loom/`: kernel generators (`gen_v2.py`, `gen_span.py`; span-v2 is `kernels/spa_p4.loom`),
  kernel tests and benchmarks, and the end-to-end session tools. Its README says how to run each.
- Raw results: `tmp/model-hunt/results.jsonl`, labels `ink-*`. Prompts: `tmp/model-hunt/prompt_inkling_{1000,64000,195000}.txt`.

## Serving through lct

- `lct serve --backend loom` (`llamacpp_tuner/backends/loom.py`) runs Loom on a private port. `llamacpp_tuner/messages.py`
  serves the public one: Anthropic's `/v1/messages` and `/v1/messages/count_tokens` translated to Loom's chat API
  (streaming, tool calls, thinking; Claude Code's `output_config.effort` becomes `reasoning_effort`), `/v1/models`
  with llama.cpp's `meta.n_ctx` (claude-local's context window), and everything else forwarded. It announces its
  address only after Loom lists the model. Checked with Claude Code: a plain reply and a Read tool call.
- **Two overlapping generations make Loom reserve 4 GB more, fail and drop its model until restart** (reproduced
  with two plain requests). Claude Code sends side requests beside the main one, so the translator queues
  generations. Token counting beside a generation is safe.
- Loom pins 16 MiB through io_uring. Under Linux's 8 MiB default it reports ready and then fails every request.
  lct runs `sudo prlimit` on its own process when the ceiling is below 64 MiB: one approval per start, nothing on
  disk (the user wants no system-file edits). A capability on `loom-serve` does not work: it loads its libraries
  through `$ORIGIN`.
- Loom accepts only its own model id (`Inkling-2-Mini`); it lists it about 5 s after its own ready line.
- Loom's default wait mode is as fast as the busy-wait spin through lct (140 / 121 vs 143 / 121 short) and keeps
  the CPU free for games.

## Open leads

- Wire span-v2 into verification: *est.* +8% at 195K.
- Keep the drafter's KV as a 4K ring: it now stores the whole context but reads only 4K; the ~0.8 GB freed at 196K
  could carry ~256K context with the drafter on (*est.*).
- Prompt reads: ~1,420 t/s at 195K; the prefill kernels are untouched.
- Short-context decode: 8.56 ms of GPU time a token. Top kernels: q8_0 mat-vec at K=2304 (1.46 ms), MoE gate/up in
  NVFP4 (1.35), MoE down (0.68), LM head (0.58), activation quantization to q8 (0.49).
- `lct live` and the TUI's Live page do not parse Loom's log, so they show no request lanes.
- No own quality eval yet (`tmp/ab-eval`, forseti) against the 27B and Flash-Next.
