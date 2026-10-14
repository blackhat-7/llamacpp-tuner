# Progress

Handoff note. Rewritten at the end of every session, never appended to. Cap 40 lines.
Next task: `PLAN.md`. Rules: `AGENTS.md`.

**Last session:** 2026-10-14 evening (Strata round 4: matrix-core prompt path + 16K chunks, a PLE table fix in the engine `strata-0.1.43-gfx1100opt-eb17930a`, reserve 2000, agent draft vocabulary, `--mtp-window 8192`; live greedy code 138, prose 101, edit 122 t/s)

## State

- `lct up/down/ps` run several servers detached. Stacks are a `[stacks]` table in `aliases.toml`. State is `servers/<name>.json` + `.log` in the cache dir.
- Stack `local` in `tmp/aliases.toml` = `swift-qwen` (GPU, :6868) + `side` (Qwen3.6-35B-A3B UD-IQ4_NL on CPU, :6869) + `embed` (Qwen3-Embedding-0.6B, :6870). `local-swarm` = `swarm` + `side` + `embed`; only one of the two can run.
- `swift-qwen`: Swift-Qwen3.8-27B Q4_K_S + DFlash2 n5, 96k, two slots: 82 t/s prose / 92 code (thinking on). Managed llama.cpp `aa94f208`.
- Profile `strata-swift-flash-next` (`--backend strata`, stack `local-flash`, running): Swift 1.5 Flash-Next IQ3_XXS on Strata 0.1.43, 262k, :6868, config `strata-swift-iq3_xxs-tuned.json` (reserve 2000). Strata's own `strata-swift-iq3_xxs.json` (what `./setup.sh` starts) has the same settings (backups `Strata-data/*.before-round4-2026-10-14`).
- The config adds `--pcie-frac 0.35 --pcie-mode dma --ple-io ram --prefill auto:16384 --mtp-window 8192`, the Swift expert profile, the agent draft vocabulary (`Strata-data/draft_vocab_agent.bin`), the hipBLASLt 100202 table, fused and matrix-core prompt kernels (`STRATA_HIP_WMMA/SELECT_WMMA/HC_UPMIX` + exact GDN switches), coupled drafts, `STRATA_ADAPT_LAG=2`, the env node cuts and the patched ROCr. 35k prompt: 854 -> ~2,730 t/s.
- Patched ROCr: `~/rocm/rocr-7.2.4-fix` (one line in `amd_blit_sdma.h`); patch, reproducer, rebuild script and an upstream issue draft in `tmp/strata-tuning/rocr-fix/`. Not filed yet.
- ROCm 7.14.1 SDK (TheRock) in `~/rocm/therock-7.14.1` with `rocprofv3`; Strata build `build-rocm714` (prompt −21%, unused).
- Profiles `swarm`, `birbot-a3b`, `jeeves`, `side` unchanged; see `skills/tune-local-model/references/host-7900xtx.md`.
- Decode now (server, greedy, 600 tokens): code 138, prose 101, edit 122 t/s (2026-10-12 original: 92 / 76 / 78). Engine source: branch `opt/combined` in `~/Documents/projects/Strata-opt` (local only). Round 4 notes: `docs/flash-next-speed-research.md`.
- Checks: `pytest` 74/74, ruff clean, pyright clean.

## Gotchas

- **The custom Strata engine (`Strata-data/engines/strata-0.1.43-gfx1100opt-*`):** roll back by pointing `exe` in `strata-swift-iq3_xxs-tuned.json` at `Strata/engine/strata`; `STRATA_GFX1100_DEFAULTS=0` turns off only 7 of its 14 changes. After any ROCm upgrade rerun its parity tests (`tmp/strata-opt/RESUME.md`) before trusting bit-exactness.
- **VRAM reserve 2000 leaves the desktop ~0.86 GB:** Strata allocates ~1.2 GB more after sizing its cache. Raise `--vram-reserve-mib` before games or heavy GPU apps run beside it.
- **hsa-rocr 7.2.4 aborts copy-heavy apps:** `BlitSdma::PendingBytes` over-reads by one slot; Arch's `-D_GLIBCXX_ASSERTIONS` turns it into `vector<unsigned long>::operator[]` aborts. **Rebuild `tmp/strata-tuning/rocr-fix/build.sh` after any ROCm update**, or drop the path from `lib_dirs`.
- **CPU heat:** Strata's pool workers spin through decode (13 workers = 14 busy cores). With an agent's test suite beside it the CPU hit 99 °C in ~2 min, even at 6 workers. Long unattended runs need a power limit or better cooling.
- **Strata's 27 GB n-gram (PLE) table is unlocked** (`ulimit -l` 8 MB) and gets evicted under memory pressure; then decode pays 5-7 ms a round. `--ple-io ram` keeps it in the page cache; never `MADV_RANDOM` it (pages go first).
- **Strata noise:** ±10% start to start. Screen kernel switches in generate mode (`--stats`, ±1.5%), not server A/Bs.
- **`pkill -f PATTERN` kills the calling shell** when the pattern is in its own command line (exit 144). Anchor patterns with `^`.
- **Strata's server ignores SIGINT while busy;** `lct down` still works (lct's wrapper sends SIGTERM).
- **A 5-token verify pass costs only 1.13× one decode step** on the 27B; DFlash2's overhead (~11 ms/cycle) is elsewhere.
- **Vulkan FA hides a scratch buffer** (4 KiB per context token with q8_0 K and V, not in the projected line).
- **TUI tests flake under load** (load average 8–20). Rerun before blaming a change.
- **Jeeves upstream does not run on ROCm;** `tmp/jeeves/src` is on `blackhat-7/jeeves` branch `rocm-attention`.
- **With `--kv-unified`, use `--no-cache-idle-slots`** or two clients evict each other's cache.
- **Check per-process GTT spill** (`drm-memory-gtt` in `/proc/<pid>/fdinfo`); spill + deep prompt = GPU reset.
- **One GPU slot: background requests evict the chat's prompt cache.** Keep them on `side`.
- **RAM runs at DDR5-4800** (rated 5200, XMP off). 14 cores stream 67 GB/s; Strata's CPU expert pool reads 42 GB/s (compute-bound under the 75 W cap), so XMP would add little.
- **`lct setup --force` without `--cmake-arg` replaces a source build** with the prebuilt one.
- **Textual:** `OptionList` needs `text-wrap: nowrap`; never name an `App` method `run`.
