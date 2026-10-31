# Flash-Next on the 7900 XTX: a better model, and how fast it can go

Research of 2026-10-12. Measured on this box unless marked: RX 7900 XTX 24 GB, i5-13600K, 94 GB DDR5-4800,
system ROCm 7.2.4, Strata 0.1.42 (local HIP build, `~/Documents/projects/Strata`), model Swift 1.5
Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS. Web claims come from a 105-agent research run with 3-vote verification;
"vendor" marks a model maker's own number, *est.* marks arithmetic. Scripts: `tmp/strata-tuning/` (gitignored).

## Answer

- **No better model fits.** Nothing verified beats Flash-Next for long agentic coding and fits 24 GB VRAM +
  ~80 GB RAM at 3 bits or more. The closest, GLM-5.3-Flash, needs ~120 GB and leads by only ~2 points.
- **Flash-Next on Strata is already ~4× llama.cpp** (22 t/s plain on the same file). Config changes found
  today make prompts **2.2× faster** (35k: 854 → 1,848 t/s) and decode **+3–15%** in fresh starts (+23% on
  a warm server), all lossless or rounding-only. Config: `~/Documents/projects/Strata/strata-swift-iq3_xxs-tuned.json`.
- **200k+ works.** A 205k-token prompt was read in 2.2 min (1,554 t/s); decode at that depth 76 t/s; the
  needle-style answer was right.
- **The next 2× needs engine code.** A decode step runs ~2,000 GPU kernels. Idle gaps between them take
  28% of decode time; spinning on CPU handshakes takes another 17% of kernel time.
- **Reliability:** the long-session crash is a ROCm bug (copy-engine ring bookkeeping), reproduced without
  Strata and fixed with a one-line runtime patch. Unattended runs also need `--pool-workers 6` for heat.

## Models

| Model | Size | Fits at ≥3 bits? | Evidence |
|---|---|---|---|
| Qwen3.8-Flash-Next (today) | 125B-A6B + 51B n-gram table on SSD | yes (IQ3_XXS 43 GB experts) | the community's preferred step up from the 27B |
| GLM-5.3-Flash | 320B-A18B, KDA + sparse attention | no: UD-IQ3_XXS 120 GB | AA index 41.8 vs 39.8; Terminal-Bench 4.0 32.8% vs 25.3% (z ≈ 1.7, weak); 3.6× more expert bytes per token |
| DeepSeek-V4-Flash | 291B, FP4 experts, compressed attention | no: ~277B params in experts, ~147 GB | config only |
| Flash-Next Coder (pruned to 256 experts) | 58 GB | yes | SWE-bench Verified 75.6 vs 82.0 unpruned (vendor); loop reports |
| Ling-3.0-flash | 127B, MLA + linear hybrid, 512×top-8 | likely | no independent agentic evals found; Strata port = weeks |
| Laguna-S-2.1 (poolside) | 118B, SWA 3:1, 256×top-10 | likely | no independent agentic evals found; Strata port = weeks |

Higher-bit Flash-Next is the only "better quality" lever that fits. KLD vs BF16 (Unsloth, base model):
UD-IQ3_XXS 0.165, UD-Q3_K_XL 0.107, UD-IQ4_XS 0.084, UD-Q4_K_XL 0.047. GSQ-RCO "IQ3_XXS" keeps ~61% of expert
weight at 2-bit types. No task-level gain is shown yet: our own replay found IQ3_S = IQ3_XXS (5/8 each).
UD-IQ4_XS (~60 GB experts) fits with Strata's resident low-RAM mode *est.* and would cut expert slots in VRAM
by ~37%, so expect slower decode.

## What we changed

All on top of `strata-swift-iq3_xxs.json`. Lossless = same weights; "rounding" = same math, other kernels.

| Change | Measured effect | Kind |
|---|---|---|
| `--pcie-frac 0.35` (probe default 0.55) | calibrate.py: 82.2 vs 75.2 t/s (+9%) | rounding (CPU vs GPU) |
| `--pcie-mode dma` | copies of missed experts overlap GPU work; with 0.35, warm server 93.5 vs ~76 t/s | lossless |
| `STRATA_HIPBLASLT_TUNING=<Strata>/tools/hip/gfx1100-hipblaslt-100202.txt` | 35k prompt 856 → 1,546 t/s | rounding |
| `STRATA_PF_FUSED=1 STRATA_PF_GEMM=1 STRATA_PF_SWITCH_MIN_T=4096` | 35k prompt 1,546 → 1,722 t/s | rounding (maintainers gate it by KL) |
| `--ple-io ram` | n-gram table in page cache (not locked); decode no longer drops 10–15% when the SSD is busy | lossless |
| `--vram-reserve-mib 1500` (user's 4000) | 10,228 vs 8,732 expert slots; temp 1.0 decode 89.7 vs 80.3 | lossless; user's call (desktop on this GPU) |
| `STRATA_SPEC_COUPLED=1 STRATA_SPEC_GUMBEL=1` | +2% at temp 1.0 (noisy) | same distribution |
| `--pool-workers 6` (default 13) | −3% decode, half the busy CPU cores; keeps unattended runs out of 100 °C | rounding (row split) |

Before/after on one interleaved A/B: see "Headline" below.

## Where the time goes

One verify window, prose, tuned config: ~28 ms for ~2.4 tokens.

- GPU compute ~21 ms: hyper-connection reads + router 3.7, VRAM-hit experts 4.0, hc0 2.7, dense GEMVs ~6.5,
  head 1.3, attention bits ~1.5. The big GEMVs and expert hits already run at ~650 GB/s.
- GPU waits 4–5 ms (11 ms on file edits): PCIe copies 1.8–2.1, CPU experts 1.5–1.8, host plan ~0.8.
- MTP drafting 2.7–3.3 ms, serial after verify.
- rocprofv3 trace (400 tokens): **~1,981 kernels per window**, median kernel 4.2 µs, median gap 3.2 µs,
  **gaps 1,269 of 4,545 ms**. `wait_flag_ge_kernel` (spin on host flags) is 17% of kernel time.
  ~700 kernels per window run under 2.5 µs (quantize, swiglu, sigmoid/combine, mapped copies, norms).

## What did not help

| Idea | Result |
|---|---|
| Strix Halo's bit-exact switch table (`arch_defaults.cpp`) | flat on gfx1100; `MMVF_ROWS` −5%, `ATTN_LANECELL` −7%, `ONE_TOKEN_COMMIT` −7%; `SH_STREAM` fails |
| ROCm 7.14.1 SDK build (TheRock, user space) | decode +2% (noise), prompt −21% even with its 100401 table |
| HIP runtime knobs (`HIP_FORCE_DEV_KERNARG`, `ROC_ACTIVE_WAIT_TIMEOUT`, graph batching/segments) | flat; `ROC_SKIP_KERNEL_ARG_COPY=1` crashes |
| Pool workers 9 / 6 / 5 / 3 vs 13, for speed | flat or worse (6 is still worth it for heat, see Reliability) |
| Adaptive tier every window (80/160 swaps) | slower |
| `--prefill auto:32768` | worse at 35k (1,425 vs 1,546) |
| Bigger `--ple-row-cache` | same hit rate on a fresh run (14–21%) |
| Foresight prefetch (`STRATA_FS_SLOTS`) | lost on every box measured upstream |

## Engine work for the next 2×

Ranked by expected gain *est.*; each needs bitwise parity checks.

1. **Fewer kernels per window.** Fold activation quantize into producers, swiglu into the gate/up epilogue,
   sigmoid/combine/add-hits/mapped copies into one combine, the hc norm into the hc down kernel. ~10 fewer
   kernels per layer ≈ 2.3 ms per window (+8%). A persistent per-layer kernel could recover most of the 28%.
2. **Fewer CPU handshakes.** Plan VRAM hits on the device (no host round trip before the hits); fuse the plan
   copy into the hits kernel. +5–10%.
3. **Long-context top-k on HIP.** With `--max-context 262144`, decode always uses the serial reference top-k
   (`qsa_select.cu:1311`); a register kernel for 270k cells exists but decode passes no active count.
4. **Overlap the MTP draft with the verify tail** (2.7–3.3 ms per window, serial today).
5. ~~Fix the long-session crash~~: done, it was ROCm (below).

## Other engines checked

- llama.cpp (aa94f208 Vulkan, same IQ3_XXS, `-fit on`): 21.9 / 21.8 / 22.2 t/s plain. MTP needs PR #28243's
  build (earlier: prose 19.7, code 26.5 on UD-Q4_K_XL).
- llama.cpp `--moe-cache-mib` (PR #29887, merged 2026-10-07): 1.6–2.2× on NVIDIA; one 7900 XTX report ~35 t/s.
- exllamav3-rocm-moe fork: 53 t/s code without drafts, 83 with MTP, on a 7900 XTX with an AVX-512 CPU (one
  developer). Its AVX2 path is untested; this CPU has no AVX-512.
- KTransformers / DALI / AcceptMoE: past 2× only when lossy, batched, or against weak baselines.

## Reliability

- **The "~1 in 90" Strata crash is a ROCm runtime bug, fixed here.** `BlitSdma::PendingBytes()` in ROCr
  (`hsa-rocr` 7.2.4) reads its copy-ring table at the copy engine's read pointer. The table rounds byte
  offsets up to 8-byte slots and has `kQueueSize / 8` of them, so a read pointer in the ring's last 7 bytes
  (inside the end-of-ring padding) indexes one past the end. Elsewhere that is a silent over-read; Arch builds
  ROCm with `-D_GLIBCXX_ASSERTIONS`, so it aborts: `std::vector<unsigned long>::operator[] ... Assertion
  '__n < this->size()'`. HIP calls it on every copy (`DmaCopyStatus` → `DmaEngineIsFree` → `PendingBytes`).
  - Reproduced without Strata: `tmp/strata-tuning/rocr-fix/sdma_stress2.cpp` (varied async copies) aborted the
    stock runtime twice (after ~16.5 M and ~8 M copies); the backtrace frames resolve to exactly that chain.
  - Fix: one line, `data_.resize(convert(size) + 1)` (`rocr-7.2.4-blit-sdma-oob.patch`). Built with Arch's
    own flags into `~/rocm/rocr-7.2.4-fix` (`build.sh`); the run config puts that `lib` first in `lib_dirs`.
    The patched runtime survived 40 M copies and Strata ran normally on it.
  - Rebuild it after any ROCm update (or drop it from `lib_dirs`): a 7.2.4 runtime beside a newer HIP may
    not load. Also unfixed upstream: `PadRingToEnd` fills to `WrapIntoRing(new_index)`, which is 0, so the
    padding's entries stay stale.
  - After the abort, systemd-coredump dumps the ~50 GB process for minutes (it looks like a freeze), and
    `/v1/models` stays empty until a chat request.
- **SSD stalls hit decode.** The n-gram table is read from the SSD every token by default. A weekly `fstrim`
  halved decode for minutes; p99 reads swing 7 ↔ 54–72 ms with background I/O. `--ple-io ram` avoids it.
- Strata's server ignores SIGINT while busy: stop it with SIGTERM.
- **Heat.** The CPU expert workers spin 20 ms before sleeping, and a decode window comes every ~30 ms, so all
  13 workers and the host thread never sleep while decoding (14.2 busy cores). With an agent's tests running
  beside it, the CPU package reached 95–100 °C within minutes. `--pool-workers 6`: 116.8 vs 120.3 t/s (−3%)
  on 7.3 busy cores; `--pool-workers 9`: 119.2 on 10.1. A shorter spin (`STRATA_POOL_SPIN_US=200`) costs −22%.
  Use 6 workers for unattended runs.

## Round 2 (2026-10-13): decode

Four lossless changes, now in `strata-swift-iq3_xxs-tuned.json` (backup of the earlier file:
`~/Documents/projects/Strata-data/strata-swift-iq3_xxs-tuned.json.before-2026-10-13`):

| Change | Why | Measured |
|---|---|---|
| Strata 0.1.43 (`./setup.py --update` after a fast-forward to `7695859a`) | adds gfx12 node cuts that gfx1100 can force on; fixes an RX 7000 crash on a text prompt after an image (#1712) | same tokens as 0.1.42 |
| env `STRATA_FUSE_WAIT=1 STRATA_FUSE_HITS=1 STRATA_QFUSE_HC=1 STRATA_GR_NORM_FAST=1` | fewer graph nodes per layer | code +4%, bit-exact |
| `--expert-profile ~/Documents/projects/Strata-data/profiles/swift-1.5-iq3_xxs.bin` | the shipped ranking is the base model's: it held 51% of Swift's routed experts | 18 Swift routing traces (`tmp/strata-opt/corpus`), `tools/make_profile_mix.py`: held-out hits 51 -> 68% (code); code 75.9 -> 103.9 t/s cold |
| default 13 pool workers (was 6) | the 75/125 W CPU cap now limits heat; 6 workers left the CPU tier slow | code +10%, edit +18%; CPU max 83 C over a minute of decode |
| env `STRATA_ADAPT_LAG=2` | the adaptive tier's swap copies stalled every 4th window; with the new profile prose fell to 65 t/s | prose 67 -> 81 t/s greedy; code and edit up too |

Live server after a fresh start (`tmp/strata-tuning/bench.py`, 400-token replies, after one warm-up round), against
the same prompts on the original config:

| Prompt | Original: greedy / temp 1.0 | Now: greedy / temp 1.0 |
|---|---|---|
| code | 92.0 / 85.2 | 109.6 / 99.2 (+19% / +16%) |
| edit | 78.4 / 79.3 | 96.1 / 92.8 (+23% / +17%) |
| prose | 76.0 / 71.6 | 81.1 / 78.4 (+7% / +9%) |

Engine work toward 2x is paused mid-way: six levers (dense GEMVs, small serial kernels, hyper-connection reads,
CPU pool parking, MTP draft chain, long-context top-k) in worktrees `~/Documents/projects/Strata-wt/*`; the router top-10
rewrite is done and bit-exact. Harness, references and resume steps: `tmp/strata-opt/RESUME.md`; analysis:
`tmp/strata-opt/audits/` (both gitignored).

## Round 3 (2026-10-14): engine changes, all bit-exact

Six levers were built in parallel worktrees (`~/Documents/projects/Strata-wt/*`), merged into branch `opt/combined` of
`~/Documents/projects/Strata-opt` (commit `caac96fe`, on top of upstream 0.1.43) and installed as
`~/Documents/projects/Strata-data/engines/strata-0.1.43-gfx1100opt-caac96fe` (the run config's `exe`; the stock engine
in `~/Documents/projects/Strata/engine/` is untouched). Every change is on by default on gfx1100 only, has an env switch
to turn it off, and gives bit-identical verify windows (`--spec-follow` + `--window-hashes`, `tmp/strata-opt/`).

| Lever | What changed | Measured alone |
|---|---|---|
| small kernels | router top-10 one wave per token (DPP), router GEMV + top-10 + doorbell in one launch, GDN a/b loads first, multi-block argmax | code +2.5%, edit +1.3% |
| dense GEMVs | gfx1100 kernels for 1-4 columns (row loads first, plain q8_1 columns), HIP port of the interleaved GEMV | GPU wait -0.4 to -0.6 ms/round (-4%) |
| HC read | down projection with pipelined loads and LDS-only barriers | code +6.5%, edit +8.9% |
| CPU pool | parking in C0.2 (`umwait`) instead of spinning; guided task split | same speed, ~20 W less, max 82 C (was 86) |
| MTP draft | fewer draft kernels, fused norm+rope, skip the draft before lookup windows | draft -0.15 to -1.1 ms/round |
| long context | register top-k and DPP scorer for QSA decode | 200k: 78.6 -> 82.2 t/s (+4.6%) |

Combined vs stock 0.1.43 (generate, greedy, 600 tokens, 2 interleaved rounds): code 106.9 -> 124.3 t/s (+16%), prose
75.8 -> 86.9 (+15%), edit 99.0 -> 104.7 (+6%). Live server, warm (greedy): code ~123, prose ~85, edit ~106 t/s; the first
requests after a start run ~10% lower while the adaptive tier settles.
At a 205k-token context (generate, MTP drafts, 600 tokens): 78.7 -> 92.9 t/s (+18%), GPU time per round 16.4 -> 12.9 ms,
windows bit-identical. Against the original config of 2026-10-12
(code 92.0, prose 76.0, edit 78.4): +34%, +11%, +35%.

## Round 4 (2026-10-14 evening): prompts +45%, decode +12-19%, the slow prompts fixed

Prompted by an RTX 5090 result (190 t/s code decode, 6,090 t/s on a 15K prompt, same model and quant; Strata's
published 5090 reports agree). The 5090 leads mostly on hardware (a faster chip per kernel, 2.5 vs 0.55 us per graph
node, 7x the int8 matrix rate), but four of its advantages were config or code we could take. Scripts and logs:
`tmp/strata-opt/` (screen5.sh, confirm5.sh, quality5.sh, pcie_sweep.sh, spec_sweep.sh, test6.sh, apply_round4.py).

**Installed:** engine `Strata-data/engines/strata-0.1.43-gfx1100opt-eb17930a` (round 3 + the PLE fix) and the settings
below in both run configs (backups `Strata-data/*.before-round4-2026-10-14`). Server mode, greedy, 600 tokens, right after
a start: **code 138.3, prose 100.8, edit 121.5 t/s** (were ~123 / ~85 / ~106). 10,021 experts cached (was 8,711).

| Change | Where | Measured (generate, greedy, 2 interleaved rounds unless noted) |
|---|---|---|
| Prompt attention and block scorer on matrix cores (`STRATA_HIP_WMMA=1`, `STRATA_SELECT_WMMA=1`; the gfx11 kernels ship in 0.1.43, the docs say gfx12 only), `STRATA_HC_UPMIX=1`, the exact GDN / padding switches, `--prefill auto:16384` | prompt | 15K 1,961 -> 2,847 t/s, 35K 1,870 -> 2,732 (+45%); decode after it -1% |
| `--vram-reserve-mib 2000` (was 4000) | decode | code +4%, edit +6%, prose +5%; at 35K context +13% |
| Agent draft vocabulary (`--mtp-draft-vocab Strata-data/draft_vocab_agent.bin`) | decode | +3% more on all three (draft 2.8 -> 2.46 ms a round) |
| `--mtp-window 8192` | decode | 35K context 97.2 -> 99.5 t/s (one run; draft 3.03 -> 2.18 ms a round) |
| A prompt chunk's PLE table rows read on 16 threads with pread (engine) | prompt | table evicted: 15K prompt 634 -> 1,639 t/s; resident: unchanged, bit-identical |

- **The slow prompts** (about half of the fresh 10-13K prompts read at ~830 t/s): the 27 GB PLE table is not locked
  (`ulimit -l` is 8 MB), so the page cache can drop it under memory pressure (the `side` model, other files). One
  thread then read a chunk's 131k rows one SSD read at a time (15.8 s for a 15K prompt). Decode pays too: 5-7 ms a
  round while its rows are evicted. A clean cold start keeps the table: the expert load's read-once pages go first.
- **Quality:** a 256-token continuation teacher-forced after the 35K prompt: the prompt stack flips 1 of 95 window
  predictions and the outputs are identical, the same as the reserve change alone (1 of 94). The exact switches are
  bit-identical. All decode changes are lossless (cache size and drafts only).
- **Measured and dropped:** `--pcie-frac` 0.25 / 0.30 / 0.45 (0.35 stays best), `--spec-min-p` 0.4-0.7 (flat),
  `--lookup-chain 3`, `--mtp-q4` (noise), `STRATA_WMMA_GEMM` (-3%), `STRATA_PA_FAST` (+3%, FP16 attention: not taken),
  `STRATA_HC_Q8` (this GGUF has no Q8_0 HC weights), `MADV_RANDOM` on the table (pages reclaimed first, decode slower),
  64 table threads (no gain over 16), dropping the expert load's page-cache copy (no gain once `MADV_RANDOM` was
  gone), reserve 1500 (leaves the desktop 0.3 GB).
- **VRAM:** Strata allocates ~1.2 GB after sizing its cache, so the desktop's headroom is the reserve minus 1.2 GB:
  0.86 GB measured at 2000. The desktop on both monitors used 0.9 GB steady (Hyprland 325 MiB, quickshell 151,
  Docker 124). Games or heavy GPU apps beside Strata need the reserve back up (`--vram-reserve-mib` in the config).
- **The CPU expert pool** is still 35-53% of each decode round. It already uses AVX-VNNI and is compute-bound under the
  75 W cap (it reads 42 GB/s; the same 14 cores stream 67 GB/s).
- **With the whole stack up (`side` holds 18 GB), the PLE table does not fit in RAM** and decode lost up to 80% on
  fresh text (prose 21-52 t/s on first passes). Engine `6ecfbdb4` adds the Linux analog of Strata's Windows
  `PrefetchVirtualMemory`: `MADV_WILLNEED` on each token's rows when the verify window and the draft layer stage them.
  With `--ple-io mmap` (no 27 GB touch at every start), server mode, 4 samples each: code 107.7 -> 131.1 t/s, prose
  60.0 -> 95.9, edit 111.4 -> 117.6; the worst first pass went 21 -> 80 t/s (prose). `--ple-io direct` (Strata's
  default) was no better: SSD reads p50 434 us, p99 66 ms.
- **The learned expert profile is on** (`"expert_profile_save"` in both configs ->
  `Strata-data/profiles/swift-1.5-iq3_xxs-learned.bin`, saved at the first request 5+ minutes after the last save). A session's hit rate
  climbs from ~73-76% at the first requests to ~87% as the adaptive tier learns (a repeated prompt: 88.6 -> 95.5%,
  72.8 -> 111 t/s); each start now begins from what the last one learned. `lct down` signals the whole process group,
  so the engine never runs its own save-on-exit; the 5-minute save covers it.

## RTX 5090 side by side (2026-10-14 night)

A friend ran `bench/strata-compare/run.py` (same inputs, the engine's own profilers) on an RTX 5090 + i9-14900KS,
64 GB DDR5-4200, Strata 0.1.44.1, `--pcie-frac 0.90`, the shipped expert profile. Results: lct branch
`strata-5090-compare`. Generate mode, greedy, 600-token fixed continuations:

| | RTX 5090 (its config) | RTX 5090 + our profile/vocab/flags | RX 7900 XTX (ours) |
|---|---|---|---|
| Code | 116-126 t/s | 160-168 | 133 |
| Edit | 81-85 | 113-122 | 117 |
| Prose | 138-140 | 108-114 | 94 |
| 15K / 35K prompt | 4,372-4,580 / 5,048 | 4,076 / 4,820 | ~2,850 / ~2,730 warm |

- Its "R4 expert-cache hits" of 90-95% count the PCIe-fetched experts as hits (90% of misses go over PCIe 5 there);
  `fetch_blobs_kernel` is 38% of its GPU time. At `--pcie-frac 0.35` it decodes code at 99 t/s: its CPU pool reads
  only ~18 GB/s (23 workers, DDR5-4200); ours reads 42 GB/s.
- Per round, its GPU work is 7.3 ms against our 12.8 (about the 1.87x bandwidth ratio), and its draft 1.1 against
  2.3 ms. Its prompts are 1.6x ours. The 190 t/s in the friend's screenshot came from another harness (a warm server).
- Our Swift profile gave the 5090 +37-45% code and +33-40% edit but -17% prose: the profile trades code against prose
  (here too: shipped code 92 / prose 85, Swift 131 / 92 t/s). A code/prose-balanced profile is being tested.

## Rounds 5-6 (2026-10-15): decode +3-5%, the idle CPU pool helps short prompts, PM4 resets the GPU

**Installed:** engine `Strata-data/engines/strata-0.1.43-gfx1100opt-ce0f1ae8` (branch `opt6/integrate2` in
`~/Documents/projects/Strata-wt/r5-iq3-sign`, local only) and four env lines in both run configs (backups
`Strata-data/*.before-r6-2026-10-15`, written by `tmp/strata-opt/apply_round6.py`). Notes and logs:
`tmp/strata-opt/progress/r6-*.md`, `r6b-*.md`. Live after the install, whole stack up (`side` + `embed`), greedy,
600 tokens, 2 each: **code 144.8, prose 107.1, edit 122.9 t/s** (round 4 with the stack up: 131 / 96 / 118; not an
interleaved A/B). Image requests work (a rendered invoice was read exactly); 0 amdgpu kernel lines.

| Change | Kind | Measured |
|---|---|---|
| i-quant signs with `v_perm_b32` (IQ2/IQ3 signs, IQ4 table lookups) | engine, bit-exact | expert kernels -12 to -16% (real weights, 10 layers, T=1-4); GPU rings -0.3 to -0.6 ms a round |
| Draft stop on the cumulative acceptance estimate: `STRATA_SPEC_CUM=0.35`, `STRATA_SPEC_CUM_ACC=0.12,0.26,0.43,0.51,0.59,0.66,0.79,0.96` (with `--spec 4`) | engine + env, drafts only | with the sign change: generate code +3.8-4.6%, edit +3.7%, prose +1-2%; server at temp 1.0, medians of 4: code +3.7%, edit +5.3%, prose +2.1% |
| Batched commit graph (upstream 6f3e3810; `STRATA_COMMIT_BATCH=0` turns it off) | engine, bit-exact | draft -0.35 ms a round: code +1.3%, edit +1.6%, prose +2.5% (one pair each) |
| `STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_PREFILL_CPU_SHARE_MAX=4096` | env, rounding-level | fresh 0.8-3.6K prompts -19 to -34%; agent turns with 0.9-4K new tokens -11 to -19% by median (8 vs 8 sessions); 12K+ reads unchanged. Live, stack up, on/off/on, medians of 3 fresh prompts: -27% at 0.9K tokens, -20% at 1.8K, -13% at 2.5K, -4% at 3.4K |

- **Exactness.** The two engine changes keep every verify window bit-identical (code, edit, prose; gen.sh's env and
  the production env), and `hip_iq_signs`, `gdn_parity` and `qsa_parity` pass. The draft stop changes window shapes,
  so greedy picks flip at near-ties (code 0-1 of 602, prose 10-11 of 600), the same class as a `--spec-min-p`
  change. The CPU share computes rarely routed experts in the CPU's activation format (flips: code 3/603, edit
  0/602, prose 11/601; upstream reports first-token KL 0.006 nats), and with `auto` a reply can differ from run to
  run. Set `STRATA_PREFILL_CPU_SHARE=0` for exactness checks.
- **Soak:** 25 min (210 mixed requests) on the candidate, plus 7 min (60) with the batched commit: 0 failures, 0
  amdgpu kernel lines, CPU max 83 C.
- **PM4 graph replay** (a ROCm CLR port that replays HIP graphs as PM4 command buffers, `~/rocm/clr-pm4-snap2` +
  `DEBUG_HIP_GRAPH_PM4=1`): code +10%, edit +6%, bit-identical, but **3 GPU page faults in ~14 runs** (address 0x0,
  scalar data). Each one hung the MES and ended in a full GPU reset with VRAM lost; the second crashed Hyprland.
  Suspect: the encoder skips SET_SH_REG writes it thinks are unchanged inside a command buffer. Not usable without a
  fix and a soak, and any test can reset the desktop GPU.
- **Rejected:** faster CPU expert kernels (pool -8-11% a round, decode -1%: the pool and the PCIe DMA read DDR5
  together, ~70 of 76.8 GB/s, so a faster pool makes the DMA slower; with `--pcie-frac 0` they give edit +8.7%);
  reading the next layer's CPU experts into L3 during the GPU pre path (+1% even with an oracle);
  `STRATA_SPEC_CUM_ADAPT`; `--spec 5`/`6`; a two-stage verify head (+0.7%, approximates the non-top logits);
  `--prompt-cache-tail` (a no-op with 16K chunks); `STRATA_CACHE_MESSAGE_BOUNDARY` (+0.4-0.6 s per short tool result
  past 8K in an appending agent loop); the upstream 0.1.44.1 merge (exact, speed-neutral).
- **Where a code round goes now** (stock libs, 24.4 ms; `STRATA_TIMELINE` on branch `opt6/timeline`): GPU pre path
  9.7 ms (hc reads 3.5, GDN 3.3, QSA 1.9, router 0.8; 2.5-4x their bandwidth floors), expert phase 8.9 ms (CPU pool,
  cache hits and PCIe DMA in parallel), head and tail 1.8, draft 2.1. The GPU is busy ~85% and loses ~2.3 ms to
  dispatch gaps (what PM4 removed). Next levers: fewer dispatches by fusion, faster pre-path kernels, and the draft
  step (0.56 ms against a ~0.17 ms floor; drafts only, so its rounding is free).

## Round 7 (2026-10-16): decode +10-18%, the main model still bit-exact

**Installed:** engine `Strata-data/engines/strata-0.1.43-gfx1100opt-210ebbdd` (branch `opt7/integrate` in
`~/Documents/projects/Strata-wt/r5-upstream`, local only, built with `build.sh`'s default options), 12 env lines and the
drafter directory `Strata-data/mtp/rt-basalt-q2b` in both run configs (backups `Strata-data/*.before-r7-2026-10-16`,
written by `tmp/strata-opt/apply_round7.py`). Notes and logs: `tmp/strata-opt/progress/r7-*.md`. Live after the
install, whole stack up, greedy, 600 tokens, 2 each: **code 161.3 / 165.5, prose 126.8 / 123.1, edit 136.7 / 136.5
t/s** (round 6 live: 145 / 107 / 123; not an interleaved A/B). Image answers are word for word round 6's; the agent
replay runs with `side` busy; 0 amdgpu kernel lines. Re-measured later the same morning after a restart: code 158 / 165,
prose 124 / 114, edit 140 / 143 t/s. Fresh prompt reads (live, second run of each size): 0.85K tokens 0.87 s, 1.6K
1.20 s, 3.2K 1.83 s, 12K 4.1 s, 34K 11.6 s (~2,900 t/s); about round 6's (0.86 / 1.26 / 1.93 / 4.6 / 11.8-12.8 s). First
reads of each size are 10-40% slower while the PLE table's pages cache.

| Change | Kind | Measured |
|---|---|---|
| Hyper-connection read as one kernel ("gr_one", on by default; `STRATA_GR_ONE=0` = three kernels) | engine, bit-exact | read 1.16-1.40x faster at 1-6 tokens; GPU rings -0.5 to -0.6 ms a round; decode +1.5-2.5% |
| Fewer graph nodes per verify window: `STRATA_SH_GATE_TAIL`, `_COMBINE_MAPPED`, `_QSA_GATE_Q8`, `_VERIFY_QDEDUP`, `_QFUSE`, `_DF_PLE`, `_MAPPED_FAST`, `_T1_FUSE`, `_QSA_FOLD`, `_ROUTER_TAIL_ROWS=4` | engine + env, bit-exact | 4-token window 1,674 -> 1,193 nodes; rings -0.9 ms a round; code +5% |
| QSA decode attention: the chunk kernel of upstream PR #540 on gfx1100 too, merge over live chunks only (on for HIP) | engine, bit-exact | attention 0.59 -> 0.37 ms a window; code +3.4% (ABBA) |
| gfx1100 dense GEMVs, four rows a wave, at 5-6 tokens (`STRATA_MMVQ_R4=0` turns them off) | engine, bit-exact | 5-6 token qkv 76-87 -> 29-34 us; edit +6% in one binary (half from more 6-token windows) |
| Draft step: no residual copy (`STRATA_MTP_NO_RCOPY`), final mixer as the fused read (`STRATA_MTP_MIX_FUSED`) | engine + env; bit-exact / drafts only | draft 1.92 -> 1.78 ms a round |
| Drafter: Basalt's self-distilled MTP dense + our Q2_0 MTP experts re-rounded with a signed least-squares scale | config, drafts only | prose tokens a round 2.02 -> 2.17, code 3.15 -> 3.24; 35K acceptance 0.80 -> 0.87 |
| QSA prompt attention v2 (WMMA; `STRATA_PA_V2=0` = the old kernel) | engine, bit-exact | 603 -> 466 ms per 15K prompt (-2.7% of its GPU time) |

- **Combined A/B** (generate in follow mode, so both arms read the same text; A C C A, 600 tokens): code 136.7 / 140.4
  -> 152.8 / 153.4 t/s (+10.5%), edit 120.7 / 118.7 -> 136.1 / 136.3 (+14%), prose 98.4 -> 114.9 / 114.7 (+17%).
- **Exactness.** With fixed drafts (`--suffix-draft 0`) every verify window and every emitted argmax equals the
  installed engine on code, edit and prose; 200 free-running tokens after a 35K prompt are identical; the kernel parity
  tests pass. The new drafter changes which tokens are verified together, so greedy picks flip at near-ties: code 2,
  edit 0, prose 12 of 600 (round 6's draft stop: prose 9-11). Rounding-level, like round 6.
- **Soak:** 15 min, 139 mixed requests (32K reads, cancelled streams, two at once, thinking on, temp 1.0): 0 failures,
  0 amdgpu lines, CPU max 79 C, GPU 80 C.
- **gr_one blocks wait on each other inside one kernel**, only on blocks dispatched before them. A wait gives up after
  ~1M polls (~0.5 s); the engine then prints ERR and exits after that request. Never seen. `STRATA_GR_ONE=0` removes
  the wait for ~2% of decode.
- **Left out:** `STRATA_HEAD_MIX_MULTI` and `STRATA_HEAD_Q8` (they move the main model's head mix onto the fused read,
  not bitwise on HIP, for 0.04 ms a round). `STRATA_VERIFY_PROFILE`, `_VERIFY_DEVICE_PLAN`, `_DOORBELL_STORE` and
  `_LFUSE` turn the new paths off; keep them out of the config.
- **Rejected:** prefetching predicted expert misses to VRAM (look-ahead recall 0.58; decode -2 to -4.5%), expert
  kernels with weights staged in LDS (1.3-1.45x slower), prompt attention v3 (slower than v2), `STRATA_GDN_SPLIT`
  (recurrence 0.48 -> 0.79 ms), a 4-wide GDN state kernel (the 2 MB state is cold each round), `STRATA_MTP_NORM_Q8`
  (no gain), `STRATA_SPEC_COUPLED_DT=0.7` (sampled prose 99 -> 89 t/s).
- **Next:** fit the MTP dense to Swift IQ3_XXS ourselves (Basalt fitted it to base Qwen). Server transcripts at temp
  1.0, the main model's hidden states from `STRATA_PREFILL_DUMP_R_ALL`, a PyTorch copy of the MTP layer checked against
  the engine's draft logits, KL training on the dense part (~108M params). Needs the GPU alone for 4-9 h. Drafts only.

**Round 8 (same morning, 1 hour): nothing installed.**
- Three more folds on the expert path (branch `opt8/launch` in `~/Documents/projects/Strata-wt/r8-launch`; env
  `STRATA_GU_SWIGLU`, `STRATA_DOWN_WAITB`, `STRATA_DOWN_WAITA`, all off by default): 4-token window 1,193 -> 1,001 nodes,
  windows and argmaxes bit-identical, but code +1%, prose -2.3 to +0.3%, edit -2.9 to +0.5%. The folded nodes run while
  the GPU waits for the CPU experts anyway. Node cuts pay only on the critical path: 9 per GDN layer (combine, two HC
  reads, qkv, conv, z, step norm, out, router tail) and 21 per QSA layer; those need gr_one-style in-kernel waits.
- A leaner gr_one (fewer up-role registers, deeper weight prefetch, split streams): no variant beat round 7's.
- `STRATA_MMVQ_IL_ROWS` (a round-7 suggestion) does nothing on HIP: the interleaved GEMV path is CUDA-only, and
  gfx1100's own dispatch already picks the faster kernel per shape (`r4_from` in `native_mmvq.cu`).

## Round 9 (2026-10-17/18): a self-distilled drafter, temp-1 decode ~+3%

- **Launch cuts** on the critical path (`STRATA_QSA_NR3`, `STRATA_Z_CONVAB`, branch `opt9/critical`): bit-exact,
  GPU -0.17-0.20 ms a round (~0.8%), within noise. Not installed.
- **Drafter fit** (`tmp/strata-opt/mtpfit/`): 714 transcripts sampled at the user's settings (temp 1.0, top_p 0.95,
  top_k 20); the main model's final residual dumped at every position (16 GiB); a PyTorch copy of the draft layer
  (vs the engine: top-1 0.93, logit cosine 0.998); KL + total-variation training of the dense part (88M params,
  experts frozen) toward Swift. Grouped expert matmuls made a training run 2× faster (~8 min).
- **Offline, 69 held-out transcripts:** drafts held per round greedy 2.223 -> 2.303, temp-1 2.188 -> 2.254 (+3%).
  A higher learning rate, trainable expert scales, or TV weight 2 with 3 epochs added at most 0.4%.
- **The draft stop had to be refit.** The trained drafter is less overconfident, so Basalt's table with min-p 0.5
  offered it fewer drafts. Its own table (mean of its greedy and coupled calibrations) with `--spec-min-p 0.3`: on
  the follow texts, priced at one ms-by-window-size table, code -0.4%, prose +1.7%, edit -0.1%; temp-1 serve A/B
  (tokens and drafts per round priced the same way) +3%. Installed as `Strata-data/mtp/rt-sd2-swift`.
- **Ceiling:** replaying the stop rule, even a perfect stop (knowing which drafts hold) gives only +9.5% with the old
  drafter and +11-13% with the new one; five-token windows never pay. The drafter is spent. The next lever is the
  dense layers' GPU time, ~2.5× their bandwidth floor.
- **Live after install** (whole stack): code 163-170, prose 125-128, edit 138-143 t/s; prompt reads 12.5K tokens
  in 4.3-4.8 s. That is inside the earlier band: server t/s swings 8-16% between identical runs on this box.

## Upstream, Basalt and r/LocalLLaMA review (2026-10-26): only 1-6% items left

Read-only review (code, logs, notes; nothing measured on the GPU). Gains are bounded by the round-6 decode profile
and the prompt-stage costs above.

- **Upstream Strata** is at v0.2.6 (4b9e05a2), 2,337 commits past our base v0.1.43; 37 of the 56 files we patch
  changed, so moving is a port, not a rebase. No 7900 XTX result in upstream `bench/results` beats ours (best 62-97
  t/s decode, up to 2,190 t/s prompt). Reported risks: #4081 (0.1.49.2 on gfx1100 parks but never restores), a 0.2.0
  beta at 155 -> 95 t/s on two 4090s.
- **Basalt** (github.com/jesdga95/basalt) is a hard fork of Strata from 2026-10-04, NVIDIA Blackwell only. Its HF
  drafter is the file we already use. The 99.5% cache hits in the r/LocalLLaMA RTX 5080 post come from opencode's
  append-only prompts; Basalt's cache code is Strata's.
- **Why a 5080 reads prompts at 5,000+ t/s:** ~3.7x our int8 rate for the expert GEMMs (30% of our prompt time),
  PCIe 5.0, and our hipBLASLt at ~40% of peak under the 291 W cap.

| Candidate | Source | Gain for us (estimate) | Work | Exactness |
|---|---|---|---|---|
| `--spec 8 --mtp-max-t 4`: 8-wide windows for appends of 64 tokens or fewer | Basalt default | -25 to -40% read time on those appends | config, recheck decode | window-composition class |
| `STRATA_PF_F16=1` | upstream edf61f18 and 5 more | prompt +3-6% | applies cleanly | first-token KL 0.0085 vs 0.0068 now |
| Top-k scan, `TOPK_REG_MIN_BLOCKS=4096` | upstream 6ef23b89, b6ccf873 | prompt +1-3% at 16-150K | applies cleanly | exact |
| Chunked GDN prompt recurrence in FP32 on HIP (`STRATA_GDN_CHUNKED` does not build for HIP; Basalt's is BF16) | Basalt 1d1159aa | prompt +3-5% (ceiling 7.7%) | 3-5 days | rounding |
| GDN layer middle as one persistent launch | Basalt mid chain | decode +2.5-5% (ceiling 7%) | 7-12 days | exact |
| Next layer's weights read into the 96 MB Infinity Cache while the combine waits | Basalt abccf9af | decode +1.5-5% | 2-4 days, microbench first | exact |
| One launch per window for two requests; `--adapt-async 1` | upstream f629fb04 and 3 more, 1aa0abf6 | +1.5-6%, only with two at once | cherry-picks | exact / rounding |

- **Correctness:** upstream 3f2db5e2 fixes CPU experts computed from a stale activation when `STRATA_ADAPT_LAG` > 1
  (we run 2). Our gfx1100 router tail (`STRATA_ROUTER_TAIL`, on by default, with `STRATA_T1_FUSE=1`) publishes
  every window's activations, so the skipped-publish path should not run. Unproven: an exact LAG 1 vs 2 A/B settles it.
- **Lossy, left out:** int8 hyper-connection reads (Basalt HC8, `STRATA_HC_Q8`), `STRATA_ROUTE_TAIL_SKIP`, FP16
  prompt attention, relaxed drafting.
- **`"parallel": 2` slows a lone request.** The engine runs `--batch 2`, which arms no prefill CPU share
  (`share_pool = o.batch <= 0`, generate.cpp:6925; the share measured -11 to -24% read time on 0.7-4K appends in
  round 6). Batch paths are not lagged (generate.cpp:1336). The slots take 1.9 GiB from the expert cache, and decode
  is -3%. Two at once run ~50 t/s each, 92 total, against 133 one after the other.
- **Agent cache misses:** on 10-23 the 4 GiB / 3-slot cache evicted 236 times and skipped 135 parks; the 16 GiB /
  6 slots of 10-25 have skipped none so far. Another 49 mid-session re-reads (6.3M tokens) came from claude-local's
  shim moving Claude Code 2.1.309's in-conversation system messages to the top of the prompt. Fixed in ai-harnesses
  0b83eef: the turn after a `cd` reads 58 tokens instead of 8,077.

## Round 10 (2026-10-31): DFlash2, upstream v0.2.9, Basalt, Reddit, a fresh GPU profile

Trigger: r/LocalLLaMA 1xjlgm4 ("upgrade Flash Next to DFlash2, +17%", an M5 Max in llama.cpp). The user's bar: only
levers worth 10-20%, counted as agent work per hour (parallel agents, long chats kept cached), not only one request's
t/s. Measured on the GPU: DFlash2's acceptance on our model, the PCIe split at 32K and 200K, a rocprofv3 profile. The
rest is read from code, logs and posts. Notes: `tmp/strata-opt/progress/r10-*.md`.

**No lossless engine change worth 10% is ready to install.** The levers that can reach it are multi-day work:

| Lever | Ceiling | Basis | Work |
|---|---|---|---|
| Matrix-core (WMMA) math for 3-6 token verify windows: dense GEMVs, HC reads | code +11% (T4 at T1-2 speed) to +16% (memory floor), prose +5-7%; deeper drafts would then pay | dense GEMVs 1.09x their bandwidth floor at T1-2, 1.64x at T4, 2.03x at T6; HC reads 1.36-2.36x; the extra is per-token decode math, not memory | 1-2 weeks; an integer version stays bit-exact |
| Next layer's weights into the 96 MB Infinity Cache while the GPU waits on CPU/PCIe | ~+8% code, ~+10% prose | waits cover ~65% of a GDN layer's 68 MB pre-path weights; round 7 saw qkv 37% faster from cache, assumed for all kernels | 1 h microbench, then 2-4 days |
| Basalt's disk tier for the conversation cache (`--conversation-cache-disk-mib`) | 1x -> 2x+ work rate once 4-5 long chats overflow RAM; 0-3% with one agent | a 200K chat is 4.4 GB, 3 fit in 16 GiB; 60 GB of disk holds 13-18 (shared prefixes stored once); restore ~3-4 s vs 60-80 s re-read | 5-8 days, ~2,300 lines; `/home` 88% full, DRAM-less QLC NVMe shared with the PLE table |
| Upstream sibling borrow (c9e571e0, 2b296987): a new subagent reuses a parked chat's prefix without evicting it | +2-8% (est.) | sibling subagents share system prompt and tools | ~1 day; first check that their first ~50K tokens match |

- **DFlash2** (incoai/Qwen3.8-Flash-Next-DFlash2 Q8_0, llama.cpp v0.8.0 HIP, greedy, thinking off). Acceptance by
  position 1-7, given the earlier ones held: code .87 .94 .82 .74 .77 .73 .83 (llama.cpp's MTP .91 .91 .90 .87 .77),
  prose .67 .69 .55 .62 .68 .57 .69 (MTP .69 .65 .68 .65 .65), edit .94 .85 .84 .82 .76 .73 .70 (MTP ~.76), 35K
  .66 .68 .55 .59. Draft step 5.2 ms in llama.cpp (our MTP 2.7-3.2). Priced with our round costs at a 2.5 ms step:
  code +2%, prose +4%, edit -9% (against MTP plus suffix drafts), 32K -6%. A window above 4 tokens costs +5.5 ms per
  token here, so deep drafts cannot pay. Upstream has it opt-in (7f97c35a, CLI only) and measured it slower on an
  R9700. Not ported (8-12 days).
- **PCIe split for missing experts** (`--pcie-frac`; upstream 8d56d417 moved gfx12 to 0.15). Interleaved, greedy,
  600 tokens: 32K 0.35 106.4, 0.25 106.6, 0.15 103.9 t/s; 200K 0.35 110.4, 0.15 98.8 (-10.5%; the CPU pool grows
  9.1 -> 11 ms a round). 0.35 stays.
- **Profile** (TheRock's rocprofv3 1.3.2 attaches to the 7.2.4 runtime, +1.45 us per dispatch, subtracted). A code T4
  window: HC reads 2.98 ms, dense GEMVs 3.47, router 0.69, GDN 0.56, QSA 0.72, head 0.81, waits ~3.0, idle gaps 2.8
  (13%; 28% before round 7). Removing the CPU/PCIe waits outright is +17% code, +30% prose, but that needs the experts
  in VRAM. A persistent critical-path kernel would recover ~1.3 ms (+6.5%). The CPU experts already use AVX-VNNI.
- **Upstream v0.2.7-v0.2.9** (733 commits): gfx12 and Strix Halo work only. Best bundle for us: `STRATA_PF_F16` +
  `STRATA_GDN_CHUNKED` (the gfx11 WMMA version exists upstream; the 10-26 row was wrong), ~+8% prompt reading, ~3% of
  agent time. The other 10-26 rows were estimates and shrink: `--spec 8` <=0.2%, top-k scan <=1%, GDN mid chain ~1.5%,
  MALL read-ahead as a wait filler <2%, two-at-once items 0 at `parallel 1`. Basalt's 124 new commits: nothing portable.
- **Batch:** two requests at once lose here (92 t/s total vs 133 one after the other; 122 with `--batch-mtp`).
  Upstream's batch wins need all experts resident or two GPUs.
- **Reddit (10-15 to 10-31):** this box is the fastest single-7900 XTX Flash-Next result found. Radiance is RDNA4-only
  and lossy (W4A8), QEngine CUDA-only and lossy, Betelgeuse a low-RAM llama.cpp fork; `--adapt-async 1` measured
  0.99-1.00x in round 5.
- **Agent work rate:** auto mode's safety checks took 18-29% of the local model's busy time. With `LOCAL_HELPER_MODEL`
  claude-local now sends them to Claude Haiku (ai-harnesses 5dbb263), checked end to end against a stand-in server.
- **Thinking effort** (4 prompts x 2 rounds, `output_config.effort` as Claude Code sends it; Strata maps max and high
  to xhigh): max 551 s in total, medium 96 s (5.7x). Max hit the 16K thinking cap twice on a two-tailnet ssh/curl
  question (137-140 s) and was right as often as medium there (1 of 2). Answers were equally correct on code, debugging
  and the ssh question; max was better on an open-ended scheduling plan and wrote slightly cleaner code. Notes:
  `tmp/strata-opt/progress/r10-effort.md`.

## Headline

Interleaved A/B, 2 rounds, a fresh server per arm (`tmp/strata-tuning/final_ab.sh`): 12 replies at temp 1.0,
6 greedy replies, then a fresh 35k-token prompt.

| Config | Decode, temp 1.0 (mean / median) | Decode, greedy | Decode after 35k | 35k prompt |
|---|---|---|---|---|
| `strata-swift-iq3_xxs.json` (before) | 78.7 / 79.1 t/s | 82.1 | 78 | 854 t/s |
| `strata-swift-iq3_xxs-tuned.json` (reserve 4000) | 81.2 / 79.1 | 86.1 | 85 | 1,848 |
| same with `--vram-reserve-mib 1500` | 89.2 / 88.7 | 92.5 | 90 | 1,859 |

- Prompts are 2.2× faster. Decode gains are +3–8% at the 4000 reserve (6 workers cost ~3% for heat) and
  +13–15% at 1500. On a warm server (adaptive tier settled) DMA + `--pcie-frac 0.35` measured +23%.
- Fresh starts understate warm sessions: the expert cache starts at 72–85% hits and reaches 95–98%.
- vs llama.cpp on the same file (22 t/s decode, a few hundred t/s prompt): ~4× decode, 5×+ prompt.
