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
