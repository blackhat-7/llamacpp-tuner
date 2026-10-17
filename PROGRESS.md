# Progress

Handoff note. Rewritten at the end of every session, never appended to. Cap 40 lines.
Next task: `PLAN.md`. Rules: `AGENTS.md`.

**Last session:** 2026-10-17 (TUI profile list fills its panel; `serve --server PATH`; H2O-Lightning-4B profiles + stack `local-flash-h2o`; Strata round 9 launch cuts bit-exact but ~0.8% of a round, not installed; self-distilled drafter prepared, to run overnight)

## State

- `lct up/down/ps` run several servers detached. Stacks are a `[stacks]` table in `aliases.toml`. State is `servers/<name>.json` + `.log` in the cache dir.
- Stack `local` in `tmp/aliases.toml` = `swift-qwen` (GPU, :6868) + `side` (Qwen3.6-35B-A3B UD-IQ4_NL on CPU, :6869) + `embed` (Qwen3-Embedding-0.6B, :6870). `local-swarm` = `swarm` + `side` + `embed`; only one of the two can run.
- `swift-qwen`: Swift-Qwen3.8-27B Q4_K_S + DFlash2 n5, 96k, two slots: 82 t/s prose / 92 code (thinking on). Managed llama.cpp `aa94f208`.
- Profile `strata-swift-flash-next` (`--backend strata`, stacks `local-flash` and `local-flash-h2o`; the latter running): Swift 1.5 Flash-Next IQ3_XXS on Strata 0.1.43, 262k, :6868, config `strata-swift-iq3_xxs-tuned.json` (reserve 2000). Strata's own `strata-swift-iq3_xxs.json` (what `./setup.sh` starts) has the same settings (backups `Strata-data/*.before-r7-2026-10-16`, older `*.before-r6-2026-10-15`).
- The config adds `--pcie-frac 0.35 --pcie-mode dma --ple-io mmap --prefill auto:16384 --mtp-window 8192`, the Swift expert profile, the agent draft vocabulary (`Strata-data/draft_vocab_agent.bin`), the hipBLASLt 100202 table, fused and matrix-core prompt kernels (`STRATA_HIP_WMMA/SELECT_WMMA/HC_UPMIX` + exact GDN switches), coupled drafts, `STRATA_ADAPT_LAG=2`, the env node cuts, the draft stop (`STRATA_SPEC_CUM=0.35` + `STRATA_SPEC_CUM_ACC`), `STRATA_PREFILL_CPU_SHARE=auto` (`_MAX=4096`), round 7's 12 node-cut / draft-step switches, the drafter `--mtp Strata-data/mtp/rt-basalt-q2b` and the patched ROCr. The learned expert profile is saved (at the first request 5+ min after the last save) to `Strata-data/profiles/swift-1.5-iq3_xxs-learned.bin` and loaded at start.
- Patched ROCr: `~/rocm/rocr-7.2.4-fix` (one line in `amd_blit_sdma.h`); patch, reproducer, rebuild script and an upstream issue draft in `tmp/strata-tuning/rocr-fix/`. Not filed yet.
- ROCm 7.14.1 SDK (TheRock) in `~/rocm/therock-7.14.1` with `rocprofv3`; Strata build `build-rocm714` (prompt −21%, unused).
- H2O-Lightning-4B (decision model, `/v1/systemone`, :6872) needs llama.cpp PR #30361: `tmp/llama.cpp-h2o/build-hip` (HIP), profiles pass it with `--server`. `h2o-lightning` (GPU, Q8_0 + projector, 40k ctx, ~7 GB VRAM): ~0.1 s for 3 questions on 321 tokens, prompt ~5,600 t/s; cannot run beside Strata (VRAM). `h2o-lightning-cpu` (`-dev none -t 6`, text only): ~4 s idle (~80 t/s prompt), 5.7-7.4 s while Strata decodes. Stack `local-flash-h2o` = `local-flash` + `h2o-lightning-cpu`; constant decisions cost Strata ~5% code/prose, ~15% edit.
- Profiles `swarm`, `birbot-a3b`, `jeeves`, `side` unchanged; see `skills/tune-local-model/references/host-7900xtx.md`.
- Speed now (server, whole stack up, 2026-10-16). Decode, greedy, 600 tokens: code 158-166, prose 114-127, edit 137-143 t/s (round 6: 145 / 107 / 123; 2026-10-13 start: 92 / 76 / 78). Fresh prompt reads, warm run: 0.85K 0.87 s, 1.6K 1.20 s, 3.2K 1.83 s, 12K 4.1 s, 34K 11.6 s (~2,900 t/s); first reads are 10-40% slower while PLE pages cache. Engine source: branch `opt7/integrate` in `~/Documents/projects/Strata-wt/r5-upstream` (local only); rollback = the `before-r7` backups. Notes: `docs/flash-next-speed-research.md`, `tmp/strata-opt/RESUME.md`.
- Checks: `pytest` 76/76, ruff clean, pyright clean.

## Gotchas

- **The custom Strata engine (`Strata-data/engines/strata-0.1.43-gfx1100opt-*`):** roll back by pointing `exe` in `strata-swift-iq3_xxs-tuned.json` at `Strata/engine/strata`; `STRATA_GFX1100_DEFAULTS=0` turns off only 7 of its 14 changes. After any ROCm upgrade rerun its parity tests (`tmp/strata-opt/RESUME.md`) and check the startup line still ends `1-6 tokens in one kernel`. `STRATA_VERIFY_PROFILE`, `_VERIFY_DEVICE_PLAN`, `_DOORBELL_STORE` and `_LFUSE` turn round 7's paths off.
- **VRAM reserve 2000 leaves the desktop ~0.86 GB:** Strata allocates ~1.2 GB more after sizing its cache. Raise `--vram-reserve-mib` before games or heavy GPU apps run beside it.
- **hsa-rocr 7.2.4 aborts copy-heavy apps:** `BlitSdma::PendingBytes` over-reads by one slot; Arch's `-D_GLIBCXX_ASSERTIONS` turns it into `vector<unsigned long>::operator[]` aborts. **Rebuild `tmp/strata-tuning/rocr-fix/build.sh` after any ROCm update**, or drop the path from `lib_dirs`.
- **CPU heat:** Strata's pool workers spin through decode (13 workers = 14 busy cores). With an agent's test suite beside it the CPU hit 99 °C in ~2 min, even at 6 workers. Long unattended runs need a power limit or better cooling.
- **Strata's 27 GB n-gram (PLE) table does not fit beside `side`** (unlocked, `ulimit -l` 8 MB): engine `6ecfbdb4` prefetches each token's rows (`MADV_WILLNEED`) and the config uses `--ple-io mmap`. Never `MADV_RANDOM` it (pages go first).
- **Strata noise:** ±10% start to start. Screen kernel switches in generate mode (`--stats`, ±1.5%), not server A/Bs.
- **`pkill -f PATTERN` kills the calling shell** when the pattern is in its own command line (exit 144). Anchor patterns with `^`.
- **Never run Strata with the PM4 graph replay libs** (`~/rocm/clr-pm4-snap2`, `DEBUG_HIP_GRAPH_PM4`): 3 GPU page faults in ~14 runs, each a full GPU reset; one crashed Hyprland (now in safe mode until the user restarts it).
- **`STRATA_PREFILL_CPU_SHARE=auto` makes Strata replies differ run to run** (timing-based split). Set it to 0 for any exactness check.
- **Vulkan FA hides a scratch buffer** (4 KiB per context token with q8_0 K and V, not in the projected line).
- **Jeeves upstream does not run on ROCm;** `tmp/jeeves/src` is on `blackhat-7/jeeves` branch `rocm-attention`.
- **With `--kv-unified`, use `--no-cache-idle-slots`** or two clients evict each other's cache.
- **Check per-process GTT spill** (`drm-memory-gtt` in `/proc/<pid>/fdinfo`); spill + deep prompt = GPU reset.
- **One GPU slot: background requests evict the chat's prompt cache.** Keep them on `side`.
- **RAM runs at DDR5-4800** (rated 5200, XMP off). 14 cores stream 67 GB/s; Strata's CPU expert pool reads 42 GB/s (compute-bound under the 75 W cap), so XMP would add little.
- **`lct setup --force` without `--cmake-arg` replaces a source build** with the prebuilt one.
- **Textual:** `OptionList` needs `text-wrap: nowrap`; never name an `App` method `run`. The global `max-height: 16` is lifted for `#profiles`/`#settings` only.
