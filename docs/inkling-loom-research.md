# Inkling-2-Mini on the 7900 XTX alone

Research of 2026-11-21 and 2026-11-22 (round 2: the afternoon of 2026-11-22). Measured on this box unless marked *est.* (arithmetic) or "reported"
(Reddit and Hacker News, the first two days after the release). Speeds are decode t/s, greedy, over all generated
tokens; the model always thinks.

## Now

Stack `local-inkling` = profile `inkling` (Loom, package `tmp/loom/kv8-r3v`, preset `xtx-park`, elastic memory) +
`embed`. It is the user's one-GPU stack: best quality and context on the XTX alone, with the 4070 free for the
desktop and games. Since 2026-11-24 several agents take turns on it: each chat waits in host RAM while the others
run ("Several agents taking turns" below).

| depth | code | prose | edit | fresh prompt read |
|---|---|---|---|---|
| short (1.3K) | 193 (was 141) | 146 (121) | 176 (133) | - |
| 64K | 148 (114) | 108 (104) | - | 26.7 s with 512-token chunks (22.0 at 1,024) |
| 195K | 104 (90) | 82 (75) | - | ~115 s *est.* at 512 (100 s at 1,024) |

- Round 2 numbers are one session each, 2026-11-22 14:09-14:20; "was" is `kv8w4096` (round 1). 195K recall 3/3.
- The table is greedy. Clients that send no temperature get 1 from `messages.py` (greedy loops in thinking).
  Since `kv8-r3` (exact top-k, 2026-11-24) temperature 1 costs far less: seeded code / prose / 66K 171 / 132 /
  125 t/s (122 / 89 / 97 on `kv8-r2v`, byte-identical text).
- Context 204,800 tokens a chat. A growing chat reuses its cache; another chat's turn parks it in host RAM.
- 0.3 GB RAM plus parked chats (~187 MB + 21.5 KB a token each, 40 GiB cap), 20.9 GB VRAM idle, idle CPU 0%.
  Starting it asks `sudo` once only when the shell's memory-lock ceiling is below 64 MiB.
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

### Round 2 (2026-11-22 afternoon): five workstreams

A profile at depth 1 (12.9 ms of device time a verify step) split the work. Five agents each owned one part and
shared the GPU through a lock; each change is a tool in `engines/loom/tools/` (applied in this order), with its
kernel tests (a negative control each) in `experiments/inkling-loom/r2/`.

1. **Draft depth and dynamic spans.** Static depth 3 (verify 4 tokens) beat depth 1 at short context for code
   (+20%) and edits (+12%) but lost prose (-2% short, -15% at 64K, -22% at 195K). A step's cost depends on its
   verify shape: span 2 ~13.2 ms, spans 3-4 ~17.7 ms (shapes are powers of two), span 5 ~26.2 ms. `dynspan.py`
   verifies 2 tokens instead of 4 while a moving average of first-draft misses is above 90/256 (shift 3), so prose
   gets short spans and code long ones. Short average of code/prose/edit jobs: 146.0 vs 138.6 (static d3) vs 130.0
   (d1). It edits only `control.loom` and `prepare.loom`.
2. **Attention.** The SWA verify kernel ran at ~215 GB/s (18.6 us a call at span 2, 28.8 at span 4); one new kernel
   for single tokens and spans loads each key once for the 3 query heads: 8 / 10 / 13 us at spans 1 / 2 / 4. The
   combine went from 11-22 to 1-6 us; global v2 uses one split per 256 keys (was 8), so short-context calls drop
   from 55 to 23 us. Alone: short d1 code / prose / edit +9.5 / +5.3 / +8%, d3 code 187.8; 195K step -3.5%.
   Padding rows of a verify shape exit at once (they were already marked not live).
3. **Dense layers.** All projections of a layer visit in one launch (46 -> 27.6 us, ~825 GB/s), RMS norm fused
   with q8 quantization (three norms in one launch: 10.7 -> 3.4 us), the loop controllers and the mixes fused.
   The program's launch sites drop from 9,122 to 8,465 (with the other tools); output bit-identical; +7-8% on every job.
4. **MoE.** The expert kernels reach 86-88% of a plain read of the same bytes, and the infinity cache already
   removes repeated reads across span tokens (consecutive tokens share ~4 of 8 experts), so only small gains: a
   one-wave router (3.6 -> 2.7 us) and 2-row gate/up (~1%). Padding tokens of a verify shape now skip the experts
   (span 5 ran as an 8-token shape: MoE went from 5.1 to 9.9 ms); this pays only once odd spans are used.
5. **Prefill.** `prepare.loom` built drafter proposal stages for up to 1,024 spans although `rows` = 1 lets one span
   propose; they reserved ~1.5 GiB of workspace, which kept the prompt chunk at 512. Built only up to `rows`, a
   1,024-token chunk fits at 196K. Split-K global flash attention (4 key ranges a tile) balances the GPU's last
   round. TTFT 8K / 64K / 187K: 2.57 / 28.75 / 128.0 s -> 2.06 / 22.06 / 94.7 s. Global attention is ~28% of a
   64K read and ~53% of a 187K one, so a faster attention kernel is the next prefill lever.

Quality, `experiments/inkling-loom/r2/quality/qa.py` (18 tasks with hidden checks: 6 Python functions run against
asserts, a C++ LRU compiled and run, the code edit, 10 short reasoning questions; greedy, same session): round 1
(`kv8w4096`) 17/18, round 2 (`kv8-r2`) 17/18. Both fail the same task the same way (the C++ answer omits `typename`
before a dependent type), so no sign of degradation; the round 2 run took 87 s for 16,452 tokens vs 96 s for 13,131.
Dense and MoE changes are bit-identical, attention and prefill change rounding only (kernel tests within one q8
step / 2e-4), and the drafter cannot change output: the model verifies every token.

Measured, not adopted:
- **Reduced draft vocabulary** (32,768 most frequent tokens, 99.67% coverage on Inkling's own output): the draft
  head read drops 259 -> 42.5 MB a draft step (*est.* ~0.85 ms of a d3 step). Its kernel test passes, but the
  first package failed registration ("stage does not match the token runner bindings": the one-span proposal stage
  bound different parameter roots than the others). A fix is built, not yet loaded.
- `--sconv` (fused short-conv kernels): kernel tests pass, registers; no end-to-end run yet.
- Mix fusion alone: +0.5-1.2% (noise). An O-projection rewrite: 4% on its kernel, left out.
- n-gram drafting for edits: needs a token-history plane in Loom's draft cache; a simulation of the edit job
  predicts at most +7% (only ~600 of 4,000 output tokens are copied code; the rest is reasoning).
- MoE dedup across span tokens: the cache already does it; rewrites of the NVFP4 decode lost.

### Files

- `engines/loom/build.sh` rebuilds the served package `$LCT_HOME/loom/kv8-r2` (`tmp/loom/kv8-r2` here) from the
  pinned upstream package. `engines/loom/README.md` lists the pins, each change, and how to check a build.
- `experiments/inkling-loom/`: kernel generators (`gen_v2.py`, `gen_span.py`; span-v2 is `kernels/spa_p4.loom`),
  kernel tests and benchmarks, and the end-to-end session tools. Its README says how to run each.
- Raw results: `tmp/model-hunt/results.jsonl`, labels `ink-*`. Prompts: `tmp/model-hunt/prompt_inkling_{1000,64000,195000}.txt`.

## Image input (2026-11-22 evening)

Inkling-2-Mini is image-text-to-text (Unsloth ships `mmproj-*.gguf`, projector type `inkling2v`), but Loom's vision
generator had no `inkling2v` family, so the upstream package was text-only. `kv8-r2v` = `kv8-r2` + an image component
(`engines/loom/README.md` item 12):

- Encoder: 23-layer ViT (hidden 1024, 16 heads) with a CLS row, learned positions (bilinear) plus 2-D RoPE,
  LayerScale, 2x2 merge, projector mm.1 -> GELU -> mm.2 to 2304. Built as a new family in loom-skills' generator,
  following llama.cpp's `tools/mtmd/models/inkling2v.cpp`; patches are padded, not resized, below the token limit.
- Q8_0 projector (343 MB): the F16 one (0.64 GB) does not fit beside 196K context. Encoder rows vs llama.cpp's F16
  encoder: per-row cosine min 0.998-0.9995, mean >= 0.9998 on a chart, a text image and a 1600x1200 photo.
- Same request to Loom and llama.cpp (`--image-max-tokens 1024`): equal prompt token counts; the answers (chart
  values, the text "BLUE TIGER 4729", shapes and colours) are right and nearly identical. Through lct's
  `/v1/messages`: a text image read in 0.8 s; a screenshot returned by a Read tool call read correctly.
- Cost: image rows only. Text output byte-identical to `kv8-r2`, short decode 192.8 / 144.8 / 175.1 vs 193.2 /
  145.5 / 175.4 t/s; VRAM 25.21 GB peak text-only, 25.56 GB once an image loaded the projector; a 195K prompt after
  an image still serves (102.6 s read, recall 3/3). ~1.8 s for a 117-row image, ~6 s for 972 rows.
- Gotcha: an image as the very first request after start activates the image and text workspaces together and
  fails (`RESOURCE_EXHAUSTED`); lct's Loom backend sends a 1-token text request before it reports ready.
- Gap: in an image request, Loom's per-message renderer names a tool result only from the message's `name`, so
  `messages.py` sets it (text requests render the same with or without it: 109 = 109 tokens).

## Serving through lct

- `lct serve --backend loom` (`llamacpp_tuner/backends/loom.py`) runs Loom on a private port. `llamacpp_tuner/messages.py`
  serves the public one: Anthropic's `/v1/messages` and `/v1/messages/count_tokens` translated to Loom's chat API
  (streaming, tool calls, thinking; Claude Code's `output_config.effort` becomes `reasoning_effort`), `/v1/models`
  with llama.cpp's `meta.n_ctx` (claude-local's context window), and everything else forwarded. It announces its
  address only after Loom lists the model. Checked with Claude Code: a plain reply and a Read tool call.
- **Two overlapping generations make Loom reserve 4 GB more, fail and drop its model until restart** (reproduced
  with two plain requests). Claude Code sends side requests beside the main one, so the translator queues
  generations. Token counting beside a generation is safe.
- Loom's AMD driver pins 16 MiB of file-staging buffers through io_uring (`file_staging_allocate(..., 8, 2 MiB)` in
  hrx-system 0.2.0's `gpu/device.c`, no flag). Under Linux's 8 MiB default the pin fails with ENOMEM on the first
  request and Loom fails the model (`RESOURCE_EXHAUSTED`). Since 2026-11-25 lct starts Loom under a seccomp filter
  that fails `io_uring_setup` with ENOSYS; IREE then falls back to plain I/O, pins nothing (VmPin 0) and runs at
  the same speed. No sudo, no system-file edits (the user's rule). Until then lct ran `sudo prlimit` at each start.
  A capability on `loom-serve` does not work: it loads its libraries through `$ORIGIN`.
- Loom accepts only its own model id (`Inkling-2-Mini`) and lists it about 5 s after its own ready line. Clients made
  for llama-server name any model (birbot's Hermes sent `local` and got 404), so the translator sends every request to
  the one model Loom serves.
- Loom's default wait mode is as fast as the busy-wait spin through lct (140 / 121 vs 143 / 121 short) and keeps
  the CPU free for games.

## Open leads

- Load the fixed reduced draft head (`experiments/inkling-loom/r2/spec/draft_head.py`): *est.* +5% decode.
- Dynamic spans over more lengths (`dynspan.py --lengths 2,3,4,6 ...` on a depth-7 preset), now that padding
  tokens skip the experts; measure static step costs at depths 1-7 first.
- Prefill: the global flash-attention kernel runs at ~27 of ~123 TFLOP/s. A prefetch variant (`prefill.py`
  `PREFILL_ATTN=pf`) compiles without spills and passes its test; not timed.
- Wire span-v2 into verification: the round 2 kernels changed the baseline; re-measure before building it.
- Keep the drafter's KV as a 4K ring: it now stores the whole context but reads only 4K; the ~0.8 GB freed at 196K
  could carry ~256K context with the drafter on (*est.*).
  It would also let the prompt chunk grow to 1,536-2,048 (64K read 23.4 s at 1,024 vs 21.0 s at 2,048).
- `lct live` and the TUI's Live page do not parse Loom's log, so they show no request lanes.
- No own quality eval yet (`tmp/ab-eval`, forseti) against the 27B and Flash-Next.

## Several agents taking turns (2026-11-24)

Loom serves one chat at a time; another agent's turn used to replace it, and the next turn re-read the whole
context (100 s at 195K). Now lct keeps each chat as a Loom checkpoint and parks it in host RAM, like Strata's
conversation cache.

- **How.** `llamacpp_tuner/parking.py`: a chat is known by its messages (one digest each); a request extending a
  parked turn continues that session. After each finished turn lct pins the state (`POST /v1/checkpoints/NAME`,
  10-60 ms) and releases the previous pin; when another chat takes the row it moves the pin to host RAM
  (`.../suspend`); the next turn sends `X-Loom-Checkpoint` and Loom copies it back. A turn that goes on in the row
  sends its own pin too (2026-11-25). A history that no longer extends the checkpoint gets 400 before any work:
  lct releases it and re-reads. Small chats (< 4,096 tokens) are not
  kept; 30 chats or 40 GiB at most, oldest out first. `backends/loom.py` turns it on only for presets with
  checkpoint slots and elastic memory.
- **Loom side.** Preset `xtx-park`: 204,800 tokens a chat, `max_sessions` 1 (Loom itself queues a second
  request instead of dropping the model), 32 checkpoint slots, no automatic caches, 512-token prompt chunks.
  Profile flags `--pool_backing=elastic --weight_slab_bytes=32mib` (suspend needs elastic memory; 32 MiB slabs
  save ~1 GB against the 1 GiB default: 20.9 vs 21.9 GB idle).
- **Exact.** Greedy, 4 agents at 20K-151K, 3 turns each: alone, round robin and all at once give the same text
  in every turn (36/36, and 24/24 with ~5K-token follow-ups at 512-token chunks). Elastic memory: same speed
  and same tokens as fixed (short / 20K / 66K 208.8 / 147.6 / 125.3 vs 209.4 / 148.2 / 126.0).
- **Speed.** Restore 0.7-1.2 s at 20K-151K (plus the new tokens); a 189K chat in 3.7 s vs a 96 s re-read. Round
  robin of 4 agents: 166 s vs 161 s alone (re-reads would take ~430 s). A lone agent's next turn starts in
  0.1-0.8 s: its pin stays in VRAM until another chat needs the row.
- **Limits.** Prompt + `max_tokens` over 204,800 gets a clean 400. An image request at ~190K gets a clean 503 (the
  image component needs ~0.5 GB more). Physical limit ~227-232K tokens (20.78 GB idle of a 25.75 GB pool, 21.4 KB a
  token; VRAM full, then Loom fails the model). Loom reserves prompt + `max_tokens` before the first token (30K of
  output: +0.67 GB at once), so Claude Code's 32,000 counts in full.
- **Two copies of one chat (fixed 2026-11-25).** A pin held while Loom reads the same chat again keeps the old
  state: +0.74 GB at 28K. An agora chat at ~99K (pin ~99K + re-read ~99K + 32K reserved) failed the model, the
  503 "model is unavailable" until a restart; reproduced at 107K with byte-identical pool numbers (committed /
  cached / peak 25.501 / 25.166 / 25.549 GB). The live turn's pin now goes with it, so Loom refuses first and lct
  releases the pin: the re-read peaks at one copy. Check: `experiments/inkling-loom/parking/reread.py`.
- **Gotchas.**
  - With 1,024-token prompt chunks a restored chat (or one that goes on while its pin is in VRAM) corrupts on a
    follow-up longer than one chunk: the answer turns to symbols. 512 is clean on every path; the stock package at
    512 is clean; at 1,024 not settled (our package or Loom). `experiments/inkling-loom/parking/suffix3.py`
    reproduces it (`eager-rewind 6000`). The current fixed preset `xtx-r2` (no pins) is not affected.
  - `--memory_bytes` makes Loom wait forever for memory near the budget (0% GPU, no error) instead of refusing;
    without it Loom fails the model past the physical limit. Neither is used: the 204,800 limit keeps one chat
    below the measured ~227K.
  - Suspend copies run behind the HTTP answer; a pin or restore right after a big suspend can wait 1-4 s. A suspend
    still copying does not add to the next chat's peak (measured at 28K).
  - Loom never pins the host copy (VmPin was 16 MiB with io_uring even with unlimited memlock, 0 without), so
    copies run at ~1-2 GB/s.
  - Loom's row only moves forward (no automatic caches in `xtx-park`): any change in the rendered history, even one
    escaped character in a tool call, means a read from token 0 (~50 s at 108K). Tool arguments must stay as the
    model wrote them: `\u2014` for "—" reused 0 of 5,117 tokens; spaces after `:` do not matter.
  - A restore of a chat whose row is past its pin (a cut or stopped turn) makes Loom keep that newer state beside
    the new reservation, under any session name. Claude Code retries a cut turn with max_tokens 128,000: at 20:27
    85K + 181K failed the model. lct releases the pin of a turn cut by max_tokens (the retry reads afresh). Esc
    mid-reply is still open (`PLAN.md`): releasing the pin while that stream ends made Loom hang the next pin.
  - Loom can close a checkpoint call without an answer: 4 pins right after a re-read, each with a `/v1/status`
    beside it, in one 20-minute window on 2026-11-25 (io_uring off), and the DELETE sent right after was reset.
    Not seen since: 0 of 16 such flows, 0 of 80 concurrent calls after re-reads. Cause in Loom unknown. lct used to
    turn parking off for good on the first drop; now `parking._call` sends a dropped call again once (0.5 s
    later), and a pin that does not come back 201 is released by name. Loom answers 503 when two checkpoint calls
    overlap; lct never sends two at once (they run under its generation lock).
- **Tests.** `experiments/inkling-loom/parking/`: `agents.py` (modes alone / turns / concurrent, `--long`),
  `suffix3.py` (restore + long follow-up vs a fresh read), `worst.py` (near the limit), `claude.sh` (real
  `claude-local --offline -p` sessions at once), `capture.py` (records `/v1/messages` traffic), `reread.py` (the
  2026-11-25 crash path: a non-ASCII tool call, then a forced re-read at ~108K).

## Temperature 1 sampling (2026-11-24)

At T > 0 `g_head_select` found its 64 candidates with 64 passes over the 200,058 logits: 5.5 ms on every verify
step (greedy: 2 us). `engines/loom/tools/spec/topk.py` ranks the 64 best chunks of 128 ids by their maxima (the
head kernels' partials), then only those ids: 0.125 ms, the same candidates in the same order. Seeded T=1 text is
byte-identical; code / prose / 66K 122 / 89 / 97 -> 171 / 132 / 125 t/s. Coupling the drafter to the sampler's
Gumbel noise was exact but gained -0.6% to +2.1% (`experiments/inkling-loom/sampling/`).

