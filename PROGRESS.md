# Progress

Handoff note. Rewritten at the end of every session, never appended to. Cap 40 lines.
Next task: `PLAN.md`. Rules: `AGENTS.md`.

**Last session:** 2026-10-12 (better model? none fits; Strata tuned for this box; the "~1 in 90" Strata crash traced to a ROCm bug and patched; `docs/flash-next-speed-research.md`)

## State

- `lct up/down/ps` run several servers detached. Stacks are a `[stacks]` table in `aliases.toml`. State is `servers/<name>.json` + `.log` in the cache dir.
- Stack `local` in `tmp/aliases.toml` = `swift-qwen` (GPU, :6868) + `side` (Qwen3.6-35B-A3B UD-IQ4_NL on CPU, :6869) + `embed` (Qwen3-Embedding-0.6B, :6870). `local-swarm` = `swarm` + `side` + `embed`; only one of the two can run.
- `swift-qwen`: Swift-Qwen3.8-27B Q4_K_S + DFlash2 n5, 96k, two slots: 82 t/s prose / 92 code (thinking on). Managed llama.cpp `aa94f208`.
- Profile `strata-swift-flash-next` (`--backend strata`): Swift 1.5 Flash-Next IQ3_XXS on Strata, 262k, :6868, in no stack. **Now points at the tuned config** (2026-10-12, user approved); reserve still 4000.
- **Tuned config:** `~/Documents/projects/Strata/strata-swift-iq3_xxs-tuned.json` adds `--pcie-frac 0.35 --pcie-mode dma --ple-io ram --pool-workers 6`, the hipBLASLt 100202 table, fused prompt kernels, coupled drafts, and the patched ROCr in `lib_dirs`. A/B: 35k prompt 854 → 1,848 t/s; decode +3–8% (+13–15% with `--vram-reserve-mib 1500`, the user's call: the desktop runs on this GPU).
- Patched ROCr: `~/rocm/rocr-7.2.4-fix` (one line in `amd_blit_sdma.h`); patch, reproducer, rebuild script and an upstream issue draft in `tmp/strata-tuning/rocr-fix/`. Not filed yet.
- ROCm 7.14.1 SDK (TheRock) in `~/rocm/therock-7.14.1` with `rocprofv3`; Strata build `build-rocm714` (prompt −21%, unused).
- Profiles `swarm`, `birbot-a3b`, `jeeves`, `side` unchanged; see `skills/tune-local-model/references/host-7900xtx.md`.
- The 8-task replay was stopped by its 97 °C guard (CPU 99 °C). User chose a lower CPU power limit; `tmp/strata-tuning/cpu-power-limit.conf` (PL1 75 W, PL2 125 W) waits for `sudo` to install.
- Checks: `pytest` 74/74, ruff clean, pyright clean.

## Gotchas

- **hsa-rocr 7.2.4 aborts copy-heavy apps:** `BlitSdma::PendingBytes` over-reads by one slot; Arch's `-D_GLIBCXX_ASSERTIONS` turns it into `vector<unsigned long>::operator[]` aborts. **Rebuild `tmp/strata-tuning/rocr-fix/build.sh` after any ROCm update**, or drop the path from `lib_dirs`.
- **CPU heat:** Strata's pool workers spin through decode (13 workers = 14 busy cores). With an agent's test suite beside it the CPU hit 99 °C in ~2 min, even at 6 workers. Long unattended runs need a power limit or better cooling.
- **Strata reads the n-gram table from the SSD every token** unless `--ple-io ram`: weekly `fstrim` or builds can halve decode.
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
- **RAM runs at DDR5-4800** (rated 5200, XMP off); CPU decode is bandwidth-bound (~57 GB/s).
- **`lct setup --force` without `--cmake-arg` replaces a source build** with the prebuilt one.
- **Textual:** `OptionList` needs `text-wrap: nowrap`; never name an `App` method `run`.
