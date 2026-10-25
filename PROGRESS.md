# Progress

Handoff note. Rewritten at the end of every session, never appended to. Cap 40 lines.
Next task: `PLAN.md`. Rules: `AGENTS.md`.

**Last session:** 2026-10-25 (agent sessions on Flash-Next: no thinking on requests that don't ask, 16 GiB conversation cache, `side` out of `local-flash`; claude-local fully local or with Claude Haiku helpers; `bench/agent_time.py`)

## State

- `lct up/down/ps` run several servers detached. Stacks are a `[stacks]` table in `aliases.toml`. State is `servers/<name>.json` + `.log` in the cache dir.
- Stack `local` in `tmp/aliases.toml` = `swift-qwen` (GPU, :6868) + `side` (Qwen3.6-35B-A3B UD-IQ4_NL on CPU, :6869) + `embed` (Qwen3-Embedding-0.6B, :6870). `local-swarm` = `swarm` + `side` + `embed`; `local-flash` = `strata-swift-flash-next` + `embed`. Only one GPU stack at a time.
- `swift-qwen`: Swift-Qwen3.8-27B Q4_K_S + DFlash2 n5, 96k, two slots: 82 t/s prose / 92 code (thinking on). Managed llama.cpp `aa94f208`.
- Profile `strata-swift-flash-next` (`--backend strata`): **uncensored** Swift 1.5 Flash-Next IQ3_XXS (huihui-ai abliterated v2 `swift-IQ3_XXS`, pack `Strata-data/packs/swift-abl-iq3_xxs`) on Strata 0.1.43, 262k, :6868, config `strata-swift-iq3_xxs-tuned.json`: reserve 2000, `"parallel": 2`, `"reasoning_budget_tokens": 16384`, `"anthropic_thinking": "on_request"`, `--conversation-cache-mib 16384 --conversation-cache-slots 6`. Rollbacks: `Strata-data/*.before-agent-2026-10-25`, `*.before-budget-2026-10-25`, `*.before-parallel-2026-10-22`; censored Swift `*.before-abl-2026-10-18` (old GGUF and pack still on disk).
- The config adds `--pcie-frac 0.35 --pcie-mode dma --ple-io mmap --prefill auto:16384 --mtp-window 8192`, the Swift expert profile, the agent draft vocabulary (`Strata-data/draft_vocab_agent.bin`), the hipBLASLt 100202 table, fused and matrix-core prompt kernels (`STRATA_HIP_WMMA/SELECT_WMMA/HC_UPMIX` + exact GDN switches), coupled drafts, `STRATA_ADAPT_LAG=2`, the env node cuts, the draft stop (`STRATA_SPEC_CUM=0.35` + `STRATA_SPEC_CUM_ACC`, `--spec-min-p 0.3`), `STRATA_PREFILL_CPU_SHARE=auto` (`_MAX=4096`), round 7's switches, the drafter `--mtp Strata-data/mtp/rt-sd2-swift` and the patched ROCr. The learned expert profile is saved to `Strata-data/profiles/swift-1.5-iq3_xxs-learned.bin` and loaded at start.
- claude-local (ai-harnesses `552d49e`, deployed): fully local by default (Claude Code's Haiku role on the main model, WebSearch removed). `LOCAL_HELPER_MODEL=claude-haiku-5-5` sends the Haiku role and WebSearch to Claude. Paseo provider "Claude (local + Haiku)" is committed in dotfiles (`aa38468`, not pushed) and appears at the next `just home`, which restarts Paseo.
- Agent sessions, 2026-10-25 short runs: a safety check reuses ~31K of its ~32K tokens and answers in 1.3-4 s (was 8-80 s, 31 timeouts); titles 0.2-0.6 s. Where an agent's time goes: `python3 bench/agent_time.py <strata config>`.
- Patched ROCr: `~/rocm/rocr-7.2.4-fix` (one line in `amd_blit_sdma.h`); patch, reproducer, rebuild script and an upstream issue draft in `tmp/strata-tuning/rocr-fix/`. Not filed yet. ROCm 7.14.1 SDK with `rocprofv3`: `~/rocm/therock-7.14.1` (Strata build `build-rocm714`, prompt −21%, unused).
- H2O-Lightning-4B (decision model, `/v1/systemone`, :6872) needs llama.cpp PR #30361: `tmp/llama.cpp-h2o/build-hip`, passed with `--server`. Profile `h2o-lightning` (GPU, ~7 GB VRAM), only with Strata down. The Claude skill `classify` (dotfiles) uses Flash-Next by default, H2O on the GPU for big bulk jobs.
- Profiles `swarm`, `birbot-a3b`, `side` unchanged (`jeeves` retired 2026-10-17); see `skills/tune-local-model/references/host-7900xtx.md`.
- Speed now (server, 2026-10-18). Decode, greedy, 600 tokens: code 163-170, prose 125-128, edit 138-143 t/s (2026-10-13 start: 92 / 76 / 78). Prompt reads ~2,700 t/s at 12.5K; agent sessions average ~2,300. Engine source: branch `opt7/integrate` in `~/Documents/projects/Strata-wt/r5-upstream` (local only). Notes: `docs/flash-next-speed-research.md`, `tmp/strata-opt/RESUME.md`.
- Checks: `pytest` 76/76, ruff clean, pyright clean.

## Gotchas

- **The custom Strata engine (`Strata-data/engines/strata-0.1.43-gfx1100opt-*`):** roll back by pointing `exe` in `strata-swift-iq3_xxs-tuned.json` at `Strata/engine/strata`. After any ROCm upgrade rerun its parity tests (`tmp/strata-opt/RESUME.md`) and check the startup line still ends `1-6 tokens in one kernel`.
- **VRAM reserve 2000 leaves the desktop ~0.86 GB:** Strata allocates ~1.2 GB more after sizing its cache. Raise `--vram-reserve-mib` before games or heavy GPU apps run beside it.
- **hsa-rocr 7.2.4 aborts copy-heavy apps:** `BlitSdma::PendingBytes` over-reads by one slot; Arch's `-D_GLIBCXX_ASSERTIONS` makes it fatal. **Rebuild `tmp/strata-tuning/rocr-fix/build.sh` after any ROCm update**, or drop the path from `lib_dirs`.
- **CPU heat:** Strata's pool workers spin through decode (13 workers = 14 busy cores). With an agent's test suite beside it the CPU hit 99 °C in ~2 min. Long unattended runs need a power limit or better cooling.
- **Strata's 27 GB n-gram (PLE) table shares RAM with the conversation cache** (up to 16 GiB; ~28 GB free with Flash-Next up): `--ple-io mmap` + engine `6ecfbdb4`'s row prefetch (`MADV_WILLNEED`). Never `MADV_RANDOM` it (pages go first).
- **Strata noise:** server t/s swings 8-16% between identical runs. Screen kernel switches in generate mode (`--stats`, ±1.5%); compare drafters on the follow texts priced at one ms-by-window-size table (`tmp/strata-opt/r7d_std.py`, `r9d.sh`), not by t/s.
- **Downloads cap at ~19 MiB/s, the internet plan** (Wi-Fi `wlan0`; `enp7s0` is a shared link to another machine, not an uplink).
- **Two requests at once on Flash-Next run ~50 t/s each (92 total, vs 133 one after the other)**; a request alone loses ~3%. `--batch-mtp`: 61 each, but alone -10%.
- **An empty `servers/<name>.json` (left by a freeze) crashes `lct ps/up`**: delete it.
- **`pkill -f PATTERN` kills the calling shell** when the pattern is in its own command line (exit 144). Use `pgrep -f "name[.]py"`.
- **Never run Strata with the PM4 graph replay libs** (`~/rocm/clr-pm4-snap2`, `DEBUG_HIP_GRAPH_PM4`): each GPU page fault reset the whole GPU; one crashed Hyprland.
- **`STRATA_PREFILL_CPU_SHARE=auto` makes Strata replies differ run to run.** Set it to 0 for any exactness check.
- **llama.cpp: with `--kv-unified`, use `--no-cache-idle-slots`**, and check per-process GTT spill (`drm-memory-gtt` in `/proc/<pid>/fdinfo`); spill + deep prompt = GPU reset.
- **`home-manager` (`just home`) needs `nix` on PATH**: `PATH=/nix/var/nix/profiles/default/bin:$PATH`. A Paseo config change restarts the daemon and kills its agents, this session's too.
- **`lct setup --force` without `--cmake-arg` replaces a source build** with the prebuilt one.
- **Textual:** `OptionList` needs `text-wrap: nowrap`; never name an `App` method `run`. The global `max-height: 16` is lifted for `#profiles`/`#settings` only.
