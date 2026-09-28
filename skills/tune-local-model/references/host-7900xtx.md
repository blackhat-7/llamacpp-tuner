# Host reference: 7900 XTX box (measured 2026-09-19 → 09-28)

Facts measured on this machine unless marked *reported* (third party) or *est.* Re-check anything older than a few weeks; builds and models move fast.

## Hardware and platform

- RX 7900 XTX 24 GB (RDNA3, gfx1100), usable ~23.5 GB after the desktop (~0.5 GB; a wallpaper engine took up to 1.35 GB). Power cap ~291 W; prefill is power-bound (GPU pinned at the cap, 2465 MHz).
- i5-13600K: CPUs 0–11 = 6 P-cores with SMT, 12–19 = E-cores. 94 GB DDR5 (2×48, dual channel) running 4800 MT/s, rated 5200 (XMP off). Measured CPU decode bandwidth ~55–58 GB/s.
- **PSU has crashed under sustained GPU load.** No long runs without asking.
- **Resizable BAR** must be on (`lspci -vv`: 32 GB BAR). With a 256 MiB BAR, Q4_K_M 27B decoded 16.4 t/s; `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` gave 38; BIOS F23 + ReBAR + MTP gave 67.
- Arch Linux, fish shell; user wants plain terminal launches (no systemd units). Servers run via `lct` profiles in `tmp/aliases.toml`, stack `local` (`lct up local`).
- amdgpu GPU watchdog resets a GPU job that runs too long: log `ring comp_… timeout` → `ErrorDeviceLost` in llama.cpp. Seen twice at 110k+ depth with memory spilled to GTT.

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
- Speed: bench tg 40 (depth 0), 36.8 (@30k). Served with MTP: code edits ~114–120 t/s (acceptance ~1.0), prose ~54–60 (acceptance ~0.35), typical 60–90. Prefill 550–1000 t/s, falling with depth (~340 t/s at 85k).
- MTP draft length (sweep): n-max 3 = 67.9, 4 = 66.1, 6 = 56.6, none = 38 t/s.
- Batch sweep pp8192: ub256 745, ub512 788, ub1024 814, ub2048 820 (+1 GB VRAM), ub4096 slower; `-b` above 2048 no gain.
- N-gram drafting on top of MTP: rejected (code edit 120 → 105, prose unchanged).
- Thinking: template accepts `reasoning_effort` low/medium/xhigh only (other values throw); default xhigh; `enable_thinking=false` disables. xhigh cost 8× tokens for no gain once. Thinking sampling per Qwen: temp 0.6 (we run 1.0).

## Qwen3.8-Flash-Next (125B-A6B MoE, qwen4exp) — rejected as daily driver

- UD-Q4_K_XL 111 GB: experts 77 GB, PLE/n-gram table 28.8 GB, attention 3.9 GB. KLD: IQ3_XXS 0.165, IQ4_XS 0.084, Q4_K_XL 0.047.
- Best: `-ngl 999 -ncmoe 42 -lm none --no-host -t 12 -C 0xfff --poll 100 -fa on -ctk q8_0 -ctv q8_0 -b 1024 -ub 1024 --cache-ram 4096`, ctx 153600: VRAM 23.6 GB, GTT 1.4 GB, 7 GB RAM free. Decode 15–20 t/s (MTP via unmerged PR #28243 + `-md mtp-…-shared-Q8_0.gguf -ngld 99`: prose 19.7, code 26.5); cold 60k prefill 69 t/s (14.5 min). In Pi a "hi" took ~4 min.
- Cost model: ~27 ms/token fixed GPU time (thousands of tiny kernels) + ~0.8 ms per CPU expert layer. Even IQ3 caps ~23 t/s.
- OOM causes: fit estimate ignores the MTP head (~3.2 GB); compute buffer grows with ctx×ub (200k/ub2048 → 13 GB GTT spill); default `--cache-ram 8192` pushed RSS 59 → 63 GB; Docker VM (8 GB) running.
- No gain: HIP (17.4 vs 17.3), expert-cache PR #27861, ik_llama.cpp (no Vulkan kernels), unpinned `-t 12`.

## CPU side models (while the GPU serves the 27B)

- Qwen3.6-35B-A3B UD-Q4_K_S on CPU (`-dev none -np 2 -t 12 -C 0xfff -lm none -rea off`): 20.5 GiB RAM, decode ~15 t/s, prefill ~90 t/s, title < 1 s, 3k prompt 33 s cold. Fine for titles/memory reviews; too slow for the classifier (timed out on parallel tool calls).
- Qwen3-Embedding-0.6B Q8_0 (`--embedding --pooling last`): ~2 GB RAM, powers code-review-graph semantic search.
- 12B dense Q4 on CPU: 7.9 t/s decode, 44 t/s prefill.

## Model verdicts (with reasons)

- **Keep Qwen3.8-27B dense.** Beats every MoE that fits: AA index 34 vs Gemma-4-31B 19 and Qwen3.6-35B-A3B 18; SWE-rebench 31.2% vs 24.7%. The user rejected MoE swaps twice on quality. Q4_K_M ≈ BF16 on Terminal-Bench (~77%).
- Swift fine-tune: fewer thinking tokens claimed; reviews mixed (condenses too hard, one A/B used more tokens). Uncensored abliteration: small benchmark cost, no agentic evals.
- Fast lane if needed: Qwen3.6-35B-A3B (reported ~120 tg, 2700 pp on this card; 10 attention layers, 2 KV heads → ~10 KB/token, 200k ≈ 2.2 GB KV); KAT-Coder-V2.5-Dev (never fails a tool call, reported).
- Rejected: Ornith-1.5 (IFBench 54 → 39, loops, untrained MTP, 6 fix iterations vs 0 in a Pi test); Nemotron-3.5-Lightning (poor output); Gemma-4-26B-A4B (weak agentic, SWE-Pro 13.8) and Gemma-4-31B (tool-call loops); Xing4.0, K2-Horizon (no mainline arch); Bonsai-2 (CUDA/Metal fork); GLM-4.7-Flash (below Qwen); Qwen3-Coder-Next (2-bit only); gemma-4-12b-heretic (still refused).

## Harness (Pi) facts

- System prompt 15–27k tokens (82% tool schemas); every cold turn pays it: ~20–35 s on the 27B.
- `pi --local` needs `lct up local`; it switches main, side and classifier to local servers and exits if one is down.
- Image input works only if the provider marks the model multimodal (read from `/v1/models` capabilities).
- Pi compacts at context − 16384 tokens; keep the server context below where speed collapses.
- Toggling tools (e.g. web) changes the prompt prefix and defeats the cache.

## Downloads

Check for a VPN first (1–2 MB/s). `HF_XET_FIXED_DOWNLOAD_CONCURRENCY=16` reached 13 MB/s; a restarted xet pull starts from zero; aria2c does not help.
