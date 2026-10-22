# Progress

Handoff note. Rewritten at the end of every session, never appended to. Cap 40 lines.
Next task: `PLAN.md`. Rules: `AGENTS.md`.

**Last session:** 2026-10-22 (Flash-Next serves two requests at once: `"parallel": 2` in its Strata config)

## State

- `lct up/down/ps` run several servers detached. Stacks are a `[stacks]` table in `aliases.toml`. State is `servers/<name>.json` + `.log` in the cache dir.
- Stack `local` in `tmp/aliases.toml` = `swift-qwen` (GPU, :6868) + `side` (Qwen3.6-35B-A3B UD-IQ4_NL on CPU, :6869) + `embed` (Qwen3-Embedding-0.6B, :6870). `local-swarm` = `swarm` + `side` + `embed`; only one of the two can run.
- `swift-qwen`: Swift-Qwen3.8-27B Q4_K_S + DFlash2 n5, 96k, two slots: 82 t/s prose / 92 code (thinking on). Managed llama.cpp `aa94f208`.
- Profile `strata-swift-flash-next` (`--backend strata`, stack `local-flash`): **uncensored** Swift 1.5 Flash-Next IQ3_XXS (huihui-ai abliterated v2 `swift-IQ3_XXS`, pack `Strata-data/packs/swift-abl-iq3_xxs`) on Strata 0.1.43, 262k, :6868, config `strata-swift-iq3_xxs-tuned.json` (reserve 2000, `"parallel": 2`; rollback `Strata-data/*.before-parallel-2026-10-22`). Strata's own `strata-swift-iq3_xxs.json` has the same settings. Rollback to censored Swift: `Strata-data/*.before-abl-2026-10-18` (older `*.before-r9-2026-10-18`); the old GGUF and pack are still on disk.
- The config adds `--pcie-frac 0.35 --pcie-mode dma --ple-io mmap --prefill auto:16384 --mtp-window 8192`, the Swift expert profile, the agent draft vocabulary (`Strata-data/draft_vocab_agent.bin`), the hipBLASLt 100202 table, fused and matrix-core prompt kernels (`STRATA_HIP_WMMA/SELECT_WMMA/HC_UPMIX` + exact GDN switches), coupled drafts, `STRATA_ADAPT_LAG=2`, the env node cuts, the draft stop (`STRATA_SPEC_CUM=0.35` + `STRATA_SPEC_CUM_ACC` measured for the drafter, `--spec-min-p 0.3`), `STRATA_PREFILL_CPU_SHARE=auto` (`_MAX=4096`), round 7's 12 node-cut / draft-step switches, the drafter `--mtp Strata-data/mtp/rt-sd2-swift` (Basalt's dense self-distilled to Swift, `tmp/strata-opt/mtpfit/`) and the patched ROCr. The learned expert profile is saved (at the first request 5+ min after the last save) to `Strata-data/profiles/swift-1.5-iq3_xxs-learned.bin` and loaded at start.
- Patched ROCr: `~/rocm/rocr-7.2.4-fix` (one line in `amd_blit_sdma.h`); patch, reproducer, rebuild script and an upstream issue draft in `tmp/strata-tuning/rocr-fix/`. Not filed yet.
- ROCm 7.14.1 SDK (TheRock) in `~/rocm/therock-7.14.1` with `rocprofv3`; Strata build `build-rocm714` (prompt −21%, unused).
- H2O-Lightning-4B (decision model, `/v1/systemone`, :6872) needs llama.cpp PR #30361: `tmp/llama.cpp-h2o/build-hip`, passed with `--server`. Profile `h2o-lightning` (GPU, Q8_0 + projector, 40k, ~7 GB VRAM): ~0.1 s for 3 questions on 321 tokens; only with Strata down. A CPU profile was dropped: ~4 s idle, 6-7 s beside Strata, slower than asking Flash-Next itself (~0.5-1 s, thinking off). The Claude skill `classify` (dotfiles) uses Flash-Next by default, H2O on the GPU for big bulk jobs.
- Profiles `swarm`, `birbot-a3b`, `side` unchanged (`jeeves` retired 2026-10-17; its venv and weights are still in the cache); see `skills/tune-local-model/references/host-7900xtx.md`.
- Speed now (server, whole stack up, 2026-10-18). Decode, greedy, 600 tokens: code 163-170, prose 125-128, edit 138-143 t/s (2026-10-16: 158-166 / 114-127 / 137-143; 2026-10-13 start: 92 / 76 / 78). Prompt reads: 0.95K 1.0-1.3 s, 4.1K 2.1-3.2 s, 12.5K 4.3-4.8 s (~2,700 t/s); 34K 11.6 s on 10-16. Engine source: branch `opt7/integrate` in `~/Documents/projects/Strata-wt/r5-upstream` (local only); rollback = the `before-r9` backups. Notes: `docs/flash-next-speed-research.md`, `tmp/strata-opt/RESUME.md`.
- Checks: `pytest` 76/76, ruff clean, pyright clean.

## Gotchas

- **The custom Strata engine (`Strata-data/engines/strata-0.1.43-gfx1100opt-*`):** roll back by pointing `exe` in `strata-swift-iq3_xxs-tuned.json` at `Strata/engine/strata`; `STRATA_GFX1100_DEFAULTS=0` turns off only 7 of its 14 changes. After any ROCm upgrade rerun its parity tests (`tmp/strata-opt/RESUME.md`) and check the startup line still ends `1-6 tokens in one kernel`. `STRATA_VERIFY_PROFILE`, `_VERIFY_DEVICE_PLAN`, `_DOORBELL_STORE` and `_LFUSE` turn round 7's paths off.
- **VRAM reserve 2000 leaves the desktop ~0.86 GB:** Strata allocates ~1.2 GB more after sizing its cache. Raise `--vram-reserve-mib` before games or heavy GPU apps run beside it.
- **hsa-rocr 7.2.4 aborts copy-heavy apps:** `BlitSdma::PendingBytes` over-reads by one slot; Arch's `-D_GLIBCXX_ASSERTIONS` turns it into `vector<unsigned long>::operator[]` aborts. **Rebuild `tmp/strata-tuning/rocr-fix/build.sh` after any ROCm update**, or drop the path from `lib_dirs`.
- **CPU heat:** Strata's pool workers spin through decode (13 workers = 14 busy cores). With an agent's test suite beside it the CPU hit 99 °C in ~2 min, even at 6 workers. Long unattended runs need a power limit or better cooling.
- **Strata's 27 GB n-gram (PLE) table does not fit beside `side`** (unlocked, `ulimit -l` 8 MB): engine `6ecfbdb4` prefetches each token's rows (`MADV_WILLNEED`) and the config uses `--ple-io mmap`. Never `MADV_RANDOM` it (pages go first).
- **Strata noise:** server t/s swings 8-16% between identical runs. Screen kernel switches in generate mode (`--stats`, ±1.5%); compare drafters or draft stops on the follow texts priced at one ms-by-window-size table (`tmp/strata-opt/r7d_std.py`, `r9d.sh`), not by t/s.
- **Downloads cap at ~19 MiB/s (~160 Mbit/s), the internet plan:** the PC is on Wi-Fi (`wlan0`, 1170 Mbit/s link); `enp7s0` is a shared link (10.42.0.1) to another machine, not an uplink. More connections or another server add nothing.
- **Two requests at once on Flash-Next run ~50 t/s each (92 total, vs 133 one after the other)**; a request alone loses ~3% (the slots take 1.9 GB of expert cache). `--batch-mtp`: 61 each, but alone -10%.
- **An empty `servers/<name>.json` (left by a freeze) crashes `lct ps/up`**: delete it.
- **`pkill -f PATTERN` kills the calling shell** when the pattern is in its own command line (exit 144). Anchor patterns with `^`.
- **Never run Strata with the PM4 graph replay libs** (`~/rocm/clr-pm4-snap2`, `DEBUG_HIP_GRAPH_PM4`): 3 GPU page faults in ~14 runs, each a full GPU reset; one crashed Hyprland (now in safe mode until the user restarts it).
- **`STRATA_PREFILL_CPU_SHARE=auto` makes Strata replies differ run to run** (timing-based split). Set it to 0 for any exactness check.
- **With `--kv-unified`, use `--no-cache-idle-slots`** or two clients evict each other's cache.
- **Check per-process GTT spill** (`drm-memory-gtt` in `/proc/<pid>/fdinfo`); spill + deep prompt = GPU reset.
- **One GPU slot: background requests evict the chat's prompt cache.** Keep them on `side`.
- **`lct setup --force` without `--cmake-arg` replaces a source build** with the prebuilt one.
- **Textual:** `OptionList` needs `text-wrap: nowrap`; never name an `App` method `run`. The global `max-height: 16` is lifted for `#profiles`/`#settings` only.
