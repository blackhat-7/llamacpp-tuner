# Host reference: 7900 XTX box (measured 2026-09-19 → 10-06)

Facts measured on this machine unless marked *reported* (third party) or *est.* Re-check anything older than a few weeks; builds and models move fast.

## Hardware and platform

- RX 7900 XTX 24 GB (RDNA3, gfx1100), usable ~23.5 GB after the desktop (~0.5 GB; a wallpaper engine took up to 1.35 GB). Power cap 291 W default, adjustable 261–350 W (LACT, `amdgpu.ppfeaturemask=0xffffffff`); prefill is power-bound (GPU pinned at the cap, 2465 MHz). Runs before ~09-20 were capped at 261 W. 350 W is an untested lever; the user wants ≥50% gains before bothering.
- i5-13600K: CPUs 0–11 = 6 P-cores with SMT, 12–19 = E-cores. 94 GB DDR5 (2×48, dual channel) running 4800 MT/s, rated 5200 (XMP off). Measured CPU decode bandwidth ~55–58 GB/s.
- **Hard power-offs under load** came from a 750 W PSU (journal stops mid-line, no amdgpu lines); now 1000 W. GPU ring timeouts still happen (a HIP sweep rebooted the host at 261 W; CS2 hung the GPU at stock). Still no long runs without asking.
- **Sustained CPU offload overheats the CPU**: Strata on ~12 cores for 3 h hit 100 °C package and throttled. Use fewer threads (decode is bandwidth-bound) and stop idle servers.
- **Resizable BAR** must be on (`lspci -vv`: 32 GB BAR). With a 256 MiB BAR, Q4_K_M 27B decoded 16.4 t/s; `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` gave 38; BIOS F23 + ReBAR + MTP gave 67.
- Arch Linux, fish shell; user wants plain terminal launches (no systemd units). Servers run via `lct` profiles in `tmp/aliases.toml`, stack `local` (`lct up local`).
- amdgpu GPU watchdog resets a GPU job that runs too long: log `ring comp_… timeout` → `ErrorDeviceLost` in llama.cpp. Seen twice at 110k+ depth with memory spilled to GTT. Afterwards llama-server can hang holding VRAM: `kill -9`.
- Resident servers' RAM matters too: `swift-qwen` with `--cache-ram 32768` reached 37.8 GB RSS, `side` 22.9 GB, `birbot-a3b` 22 GB; together 91% of RAM. A stray 2B server (7 GB RAM) pushed Flash-Next experts to SSD: 2–5 t/s.

## Backends

- **Vulkan (RADV) is the default.** Qwen3.8-27B Q4_K_M, commit 17197474, q8/q8 KV:

  | test | Vulkan | HIP |
  |---|---|---|
  | pp4096 | 854 | 955 |
  | tg128 | 40.0 | 33.9 |
  | pp4096 @30k | 558 | 575 |
  | tg128 @30k | 36.7 | 27.9 |

  HIP wins prefill 3–12%, loses decode 15–24%. HIP rejects mixed K/V types and q5_1 in FA, has no WMMA FA for head dim 256, and its mmap load crawls under RAM pressure (use `--load-mode none`). HIP build: `--cmake-arg -DGGML_VULKAN=OFF --cmake-arg -DGGML_HIP=ON --cmake-arg -DGPU_TARGETS=gfx1100`.
- Updating llama.cpp (a894dae9, 2026-09-20) gave pp4096 911 (+5%) and pp@30k ~630 (+16%); tg unchanged. Newer builds are the cheapest win.
- Vulkan FA fast path needs K **and** V q8_0. f16 KV @30k: 452 vs q8 629. q8/q5_1: 543.
- No effect: `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`, `GGML_VK_MAX_NODES_PER_SUBMIT`. Profile ops with `GGML_VK_PERF_LOGGER=1` (at 30k: matmul 55%, FA 33%).
- `GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296`: fix for the decode/prefill cliff near 131k (llama.cpp #27734, reported on this exact card and model).

## Qwen3.8-27B dense (daily driver)

- Arch qwen35: 65 layers, hybrid Gated DeltaNet, 16 attention layers, 4 KV heads, head dim 256, built-in MTP layer. Q4_K_M 15.65 GiB (JonathanColetti Uncensored); Swift Q4_K_S 16.6 GB (ukisai).
- Memory at q8 KV: ~34–37 KB/token (98k 3.26 GB, 160k 5.44 GB); recurrent state 748 MiB fixed; MTP draft KV f16 640 MiB @160k → 340 with `-ctkd q8_0 -ctvd q8_0`; compute buffer @160k: ub2048 1.26 GB, ub1024 0.58 GB; projector ~0.9 GB, up to 1.16 GB on the first image.
- Fits (one slot, ub1024): 128k + projector ≈ 23.1 GB; 160k text-only 23.4 GB; 180k too tight. With two unified slots add ~0.5 GB.
- **Current server (2026-09-28)**: `--ctx 114688 --env GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296 -ngl all -np 2 --kv-unified --no-cache-idle-slots -cms 2048 -fa on -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -b 2048 -ub 512 --cache-ram 32768 --reasoning-effort medium --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --spec-type draft-mtp --spec-draft-n-max 4 --image-min-tokens 1024 --jinja`. 22.3 GiB VRAM, 1.7 GB free, GTT 0.3 GB.
  - 150k failed: 1.6–2.7 GB spilled to GTT, prefill fell to 57 t/s at 110k, GPU reset. 128k + ub1024 failed the same way at 125k.
  - Two slots: chat and the auto-mode classifier each keep a cached slot (classifier ~2.3 s a pass with low reasoning).
- Speed: bench tg 40 (depth 0), 36.8 (@30k). Served with MTP: code edits ~114–120 t/s (acceptance ~1.0), prose ~54–60 (acceptance ~0.35). Live sessions (10-02): 46–60 t/s short, 28–35 t/s at 90k+, mean acceptance ~45%. Prefill ~930 t/s cold → ~690 by 28k → ~340 at 85k (best case); with ~1 GB GTT spill 264 t/s at 95k, then device lost.
- Thinking volume: at xhigh it writes ~12.7k output tokens per agent task (Claude: 2–5k). Wall-clock-limited evals then measure GPU speed, not ability.
- MTP draft length (sweep): n-max 3 = 67.9, 4 = 66.1, 6 = 56.6, none = 38 t/s.
- Batch sweep pp8192: ub256 745, ub512 788, ub1024 814, ub2048 820 (+1 GB VRAM), ub4096 slower; `-b` above 2048 no gain.
- N-gram drafting on top of MTP: rejected (code edit 120 → 105, prose unchanged).
- Thinking: template accepts `reasoning_effort` low/medium/xhigh only (other values throw); default xhigh; `enable_thinking=false` disables. xhigh cost 8× tokens for no gain once. Thinking sampling per Qwen: temp 0.6 (we run 1.0). Classifier call: `none` 0.5 s / 2 tokens, `low` ~2.3 s.
- One prompt-cache entry at 34.6k tokens is ~2.0 GB, so `--cache-ram 8192` holds ~4 conversations.

## Qwen3.6-35B-A3B on GPU (`swarm`, `birbot-a3b`)

- KV ~10.9 KB/token (measured from fits). Memory is set by total context, not slot count. Without `--kv-unified` each slot gets a fixed ctx/N; one agent cannot starve the rest.
- Files: UD-Q4_K_S 21.4 GB (only 5×26k fits), UD-IQ4_NL 18.54 GB (5×64k: 23.0 GB VRAM, 171 MB GTT, one reply 183 t/s; 5-way untested), IQ4_XS 18.21, Q3_K_XL 17.23, IQ3_S 15.35.
- With mmproj-F16 (0.9 GB), after an image: 3×96k 23.3 GB (1.2 GB free), 2×144k 23.1 GB (1.4 GB free), 3×64k 2.4 GB free.
- **Ignores `reasoning_effort=low`**: 405 thinking tokens on a yes/no check vs 2 with `none`. A 512-token-capped classifier hit `length` and failed closed. Map low → thinking off for it.
- Quality: 48% vs 63% for the 27B in forseti's agent eval; hard tasks 7% vs 43%. A ~55k-token session compacting every turn looped ~25 failing tool calls; a fresh session fixed it.

## Qwen3.8-Flash-Next (125B-A6B MoE, qwen4exp) — overnight long-context runs on Strata

- **Strata engine** (`~/Documents/projects/Strata`, `--backend strata`) beats llama.cpp for this model by far:
  - GSQ-RCO IQ3_S: 56 t/s prose, 67 code, ~50 at 117k, prefill 831–847 t/s; 46.8 GB experts in RAM, 22.5 GB VRAM. Since deleted.
  - Swift 1.5 IQ3_S vs IQ3_XXS: 83.7 vs 76.0 GB on disk (+10%); Pi replay of 8 past tasks tied 5/8 but took 104 vs 66 min. Stay on IQ3_XXS; IQ3_S deleted 2026-10-12.
  - Swift 1.5 GSQ-RCO IQ3_XXS (current, profile `strata-swift-flash-next` → `strata-swift-iq3_xxs-tuned.json` since 2026-10-12, :6868, 262k): median decode 67 t/s, MTP acceptance 77%, 83–90% of expert lookups on GPU; prefill only ~334 t/s on small agent requests (hipBLASLt untuned). 45.4 GiB RAM, 20.1 GiB VRAM. Forseti score 65% vs 63% for the 27B and 58% for IQ3_S.
  - User verdict: the 27B stays the daily driver; Swift Flash is for overnight long-context runs.
  - **Tuning (2026-10-12, engine 0.1.42, `docs/flash-next-speed-research.md`):** `--pcie-frac 0.35` (the probe's 0.55 is too high for this AVX2 CPU vs a 27 GB/s link; calibrate.py +9%), `--pcie-mode dma` (copies overlap GPU work; with 0.35 a warm server 93.5 vs ~76 t/s), env `STRATA_HIPBLASLT_TUNING=<Strata>/tools/hip/gfx1100-hipblaslt-100202.txt` (must match hipBLASLt 1.2.2 = ROCm 7.2.4; 35k prompt 856 → 1,546 t/s) + `STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_SWITCH_MIN_T=4096` (1,722), `--ple-io ram` (SSD stalls otherwise cost 10–15% of decode). Reserve 1500 instead of 4000: +12% at temp 1.0. 205k prompt: 1,554 t/s, decode 76 t/s at depth.
  - Dead ends here: gfx1151's exact switch table (flat; `MMVF_ROWS`, `ATTN_LANECELL`, `ONE_TOKEN_COMMIT` slower), ROCm 7.14.1 SDK build (prompt −21%), HIP launch env knobs, fewer pool workers, faster adaptive tier. Screen kernel switches in generate mode (`--stats`, ±1.5%); server A/Bs move ±10% per start.
  - **Round 2 (2026-10-13, engine 0.1.43):** a Swift-1.5 expert profile (`Strata-data/profiles/swift-1.5-iq3_xxs.bin`, from 18 routing traces with `tools/make_profile_mix.py`; the shipped one is the base model's and held 51% of Swift's experts), 13 pool workers under the 75 W cap, `STRATA_ADAPT_LAG=2` (the tier's copy stall cost prose 15 t/s) and the env node cuts `STRATA_FUSE_WAIT/FUSE_HITS/QFUSE_HC/GR_NORM_FAST=1`. Live greedy: code 92 -> 110, edit 78 -> 96, prose 76 -> 81 t/s.
  - **Rounds 3-4 (2026-10-14):** a local engine build (`Strata-data/engines/strata-0.1.43-gfx1100opt-6ecfbdb4`, branch `opt/combined` in `~/Documents/projects/Strata-opt`): bit-exact gfx1100 decode kernels, and a prompt chunk's PLE rows read on 16 threads with pread (the unlocked 27 GB table gets evicted; one thread took 15.8 s on a 15K prompt). Config: `STRATA_HIP_WMMA/SELECT_WMMA/HC_UPMIX=1` + the exact GDN/`PF_PAD` switches + `--prefill auto:16384` (prompts +45%; the gfx11 WMMA kernels ship in 0.1.43 although its docs say gfx12 only), `--vram-reserve-mib 2000` (desktop keeps ~0.86 GB), `--mtp-draft-vocab Strata-data/draft_vocab_agent.bin`, `--mtp-window 8192`. Live greedy: code 138, prose 101, edit 122 t/s; 35K prompt ~2,730 t/s. With `side` up the table cannot stay in RAM: the engine prefetches each token's rows (`MADV_WILLNEED`, Linux analog of Strata's Windows `PrefetchVirtualMemory`) and the config uses `--ple-io mmap` (stack up: code 131, prose 96, edit 118; was 108 / 60 / 111). Not `MADV_RANDOM` on the table: its pages get reclaimed first.
  - **Rounds 5-6 (2026-10-15):** engine `strata-0.1.43-gfx1100opt-ce0f1ae8` (branch `opt6/integrate2`): i-quant signs with `v_perm_b32` and upstream's batched commit (both bit-exact) plus a draft stop on cumulative acceptance (`STRATA_SPEC_CUM=0.35` + `STRATA_SPEC_CUM_ACC`, drafts only); env `STRATA_PREFILL_CPU_SHARE=auto` + `_MAX=4096` lets the idle CPU pool read rarely routed experts of short prompt chunks (rounding-level; 0.8-3.6K prompts 19-34% faster). Decode +3-5% on code and edits, +1-3% on prose (A/B); live with the stack up code 145, prose 107, edit 123 t/s. **Never use the PM4 graph replay port** (`~/rocm/clr-pm4-snap2`, `DEBUG_HIP_GRAPH_PM4`): +10% but it page-faulted the GPU 3 times in ~14 runs, each a full GPU reset that can take the desktop down. Faster CPU expert kernels don't pay at `--pcie-frac 0.35`: the pool and the PCIe DMA share DDR5 (~70 of 76.8 GB/s).
  - Bigger models don't pay: GLM-5.3-Flash (320B-A18B) is ~120 GB at 3 bits (more than VRAM + RAM), moves 3.6× more expert bytes per token and has no Strata port; estimate 1–5 t/s for a ~2-point benchmark lead.
  - Profile: ~2,000 GPU kernels per verify window, 3.2 µs median gap, gaps 28% of decode; `wait_flag_ge_kernel` 17% of kernel time (rocprofv3 from TheRock 7.14.1 in `~/rocm`).
  - The crash (~1 in 90 requests, `std::vector<unsigned long>::operator[]` assertion) is a **ROCm bug**, not Strata: `BlitSdma::PendingBytes()` in hsa-rocr 7.2.4 indexes one past its copy-ring table when the copy engine's read pointer is in the ring's last 7 bytes; Arch's `-D_GLIBCXX_ASSERTIONS` turns the over-read into an abort. Reproducer and one-line patch: `tmp/strata-tuning/rocr-fix/`; patched lib in `~/rocm/rocr-7.2.4-fix/lib` (first in the run config's `lib_dirs`; rebuild after a ROCm update). After an abort, systemd-coredump dumps the ~50 GB engine for minutes. Stop the server with SIGTERM (it ignores SIGINT while busy).
  - **Heat:** 13 pool workers spin through decode (14 busy cores); with an agent's test suite beside it the i5-13600K hit 99 °C in ~2 min. `--pool-workers 6` costs 3% decode and halves the busy cores; still ~97–99 °C under pytest load. Fix chosen: RAPL PL1 75 W / PL2 125 W (was 95/181) via the system-manager service `cpuPowerLimit` in `~/dotfiles/nix/linux/default.nix` (`just switch`); speed cost not yet measured.
  - Gotchas: one request at a time (parallel callers evict each other's cache); crashed ~1 in 90; after a crash `/v1/models` is empty until a chat request; rerunning `setup.sh` resets `--vram-reserve-mib` (4000 → 700, VRAM hit 25.6 GB), `sampling` and `fit_max_tokens`; no `sampling` block = greedy; `high` maps to `xhigh`, which writes 4.6–9.8k-token turns of second-guessing; CPU vision ~300 tokens/image. StrataGP fork: no gain (THP already `always`).

### On llama.cpp (rejected)

- UD-Q4_K_XL 111 GB: experts 77 GB, PLE/n-gram table 28.8 GB, attention 3.9 GB. KLD: IQ3_XXS 0.165, IQ4_XS 0.084, Q4_K_XL 0.047.
- Best: `-ngl 999 -ncmoe 42 -lm none --no-host -t 12 -C 0xfff --poll 100 -fa on -ctk q8_0 -ctv q8_0 -b 1024 -ub 1024 --cache-ram 4096`, ctx 153600: VRAM 23.6 GB, GTT 1.4 GB, 7 GB RAM free. Decode 15–20 t/s (MTP via unmerged PR #28243 + `-md mtp-…-shared-Q8_0.gguf -ngld 99`: prose 19.7, code 26.5); cold 60k prefill 69 t/s (14.5 min). In Pi a "hi" took ~4 min.
- Cost model: ~27 ms/token fixed GPU time (thousands of tiny kernels) + ~0.8 ms per CPU expert layer. Even IQ3 caps ~23 t/s.
- OOM causes: fit estimate ignores the MTP head (~3.2 GB); compute buffer grows with ctx×ub (200k/ub2048 → 13 GB GTT spill); default `--cache-ram 8192` pushed RSS 59 → 63 GB; Docker VM (8 GB) running.
- No gain: HIP (17.4 vs 17.3), expert-cache PR #27861, ik_llama.cpp (no Vulkan kernels), unpinned `-t 12`.
- Prefill: `-b 4096 -ub 4096 -t 6` gave 295 t/s on 8k vs 127 at batch 2048. The first request after load lazy-loads the 27.5 GiB PLE table from SSD (~190 MB/s): 45 t/s, not a benchmark. `--cache-ram 0` rereads the whole prompt each turn. `-lm mmap` at 262k: 7.4 prose, 12.4 code.
- N-gram drafting **helped** here (unlike the 27B): code edit 16.4 → 19.8 t/s; with MTP code 26.5 → 28.0, prose 19.7 → 17.8.
- The AI misjudged sizes twice: it said the lookup table was 46 GB (28.8) and that context "barely touches RAM" (compute buffer + MTP head spilled 13 GB, OOM). Size from the GGUF tensor list and llama.cpp's projection.

## CPU side models (while the GPU serves the 27B)

- `side`: Qwen3.6-35B-A3B on CPU (`-dev none -np 2 -t 12 -C 0xfff -lm none -rea off`), now UD-IQ4_NL (Q4_K_S deleted). Measured on Q4_K_S: 20.5 GiB RAM, decode ~15 t/s, prefill ~90 t/s (as low as ~68 on short uncached prompts), title < 1 s, 3k prompt 33 s cold. Used for titles only. Memory reviews moved to the session model; the classifier never got a cache hit there (15–17 s per tool call). It competes for CPU and RAM bandwidth, not the GPU.
- Gemma-4-12B on GPU (argus side use): HauhauCS Gemma4-12B-QAT-Uncensored-Balanced Q4_K_M did not refuse; with its MTP head (`-md mtp-gemma-4-12B-it.gguf --spec-type draft-mtp`) 79 → 104–112 t/s. Template has thinking on/off only; thinking ate the whole budget until `--reasoning off`.
- Qwen3-Embedding-0.6B Q8_0 (`--embedding --pooling last`): ~2 GB RAM, powers code-review-graph semantic search.
- 12B dense Q4 on CPU: 7.9 t/s decode, 44 t/s prefill.

## Model verdicts (with reasons)

- **Keep Qwen3.8-27B dense.** Beats every MoE that fits: AA index 34 vs Gemma-4-31B 19 and Qwen3.6-35B-A3B 18; SWE-rebench 31.2% vs 24.7%. The user rejected MoE swaps twice on quality. Q4_K_M ≈ BF16 on Terminal-Bench (~77%).
- Swift fine-tune: fewer thinking tokens claimed; reviews mixed (condenses too hard, one A/B used more tokens). Uncensored abliteration: small benchmark cost, no agentic evals.
- Fast lane if needed: Qwen3.6-35B-A3B (reported ~120 tg, 2700 pp on this card; 10 attention layers, 2 KV heads → ~10 KB/token, 200k ≈ 2.2 GB KV); KAT-Coder-V2.5-Dev (never fails a tool call, reported).
- Rejected: Ornith-1.5 (IFBench 54 → 39, loops, untrained MTP, 6 fix iterations vs 0 in a Pi test); Nemotron-3.5-Lightning (poor output); Gemma-4-26B-A4B (weak agentic, SWE-Pro 13.8) and Gemma-4-31B (tool-call loops); Xing4.0, K2-Horizon (no mainline arch); Bonsai-2 (CUDA/Metal fork); GLM-4.7-Flash (below Qwen); Qwen3-Coder-Next (2-bit only); gemma-4-12b-heretic (still refused).

## Jeeves (PostHog, `--backend jeeves`)

- ROCm needed three fixes (GQA K/V repeat, 2-stage fp8 GEMM for 64 KB LDS, NVIDIA-only Triton kernels off). First start ~20 min (Triton compile), then 16–20 s. fp8 accuracy 99.38% (bar 99.5%, misses 1 ULP). Not usable as Pi's classifier while `swift-qwen` holds the GPU.

## Harness (Pi) facts

- System prompt 15–27k tokens (82% tool schemas); every cold turn pays it: ~20–35 s on the 27B.
- `pi --local` needs `lct up local`; it switches main, side and classifier to local servers and exits if one is down.
- Image input works only if the provider marks the model multimodal (read from `/v1/models` capabilities).
- Pi compacts at context − 16384 tokens; keep the server context below where speed collapses.
- Toggling tools (e.g. web) changes the prompt prefix and defeats the cache (after `web_enable` 42% matched; the hybrid model reused none).
- Pi's local provider `maxTokens` is 32768 (was 8192: xhigh thinking hit `length`). Pi overrides server sampling and effort defaults. Effort reaches the template only via `chat_template_kwargs` (`reasoning_effort: {$var:'thinking.effort', omitWhenOff:true}` in pi-ai); without it the UI offers on/off only.
- `pi-autoname` defaults to the session model and evicts the chat cache: point it at a cloud model. Requests timing out at exactly 10 s: Node's 4-thread libuv pool; `UV_THREADPOOL_SIZE=64` in the `pi` function. Debug stalls by repeating the call on a cloud model first (one was a third-party extension).
- Servers bind `--host 0.0.0.0`; `/etc/hosts` maps `pc` → 127.0.0.1 so `pi --local` works without Tailscale.

## Other clients

- **birbot / OpenClaw** (`http://<host>:6868/v1`, model id `local`): cold prompt only ~1.6k tokens; prompt budget ≈ n_ctx − 20k (45,536 on a 64k slot), so size per-slot context for that; client-declared context must not exceed the server's; Docker clients need `--host 0.0.0.0`; a dead upstream shows as HTTP 408 or "network connection error".
- **Images without `--mmproj` return HTTP 500** and the model then believes it cannot see images. Keep the projector when swapping models behind a port, or update the client's `input` list.
- `-a NAME` makes every model report one id; clients keying results on the id then mix models. `/props` (`model_path`, `total_slots`, `chat_template`, `modalities`) and `/slots` show what a client really hits.
- Thinking clients need `max_tokens` ≈ 32k: at 4096, 28 of 64 forseti tries ended mid-thinking.

## Downloads

Check for a VPN first (1–2 MB/s). `HF_XET_FIXED_DOWNLOAD_CONCURRENCY=16` reached 13 MB/s on Wi-Fi, 25 MB/s on 10-02; root disk was 98% full on 10-04, so check space before large pulls; a restarted xet pull starts from zero; aria2c does not help.
