# Plan

One line per task, ordered. Top unchecked line is next. `[ ]` open · `[~]` in progress · `[x]` done.
Each line carries its own done-check. A task too big for one session gets split before it is started.

## Now

- [x] **Serve aliases.** `lct serve <name>` expands arguments saved in `aliases.toml` (cache dir); later options override. Done: commit `9c53d08`, two CLI tests.
- [x] **Adopt the long-task framework.** `AGENTS.md` routine, `PLAN.md`, `PROGRESS.md`, `DECISIONS.md`, `CLAUDE.md` symlink. Done when all four exist within their caps and are committed.
- [x] **TUI shell.** `lct tui` opens a Textual app with Serve / Download / Benchmark tabs, a shared output log and footer shortcuts. Done when a `run_test` smoke test renders at widths 40, 80 and 120.
- [x] **Serve tab.** Load a profile (alias) into a form, edit model / projector / ctx / host / port / extra args, start and stop the server, save the form as a profile. Done when a test starts a fake server through the form and stops it.
- [x] **Download tab.** Search a repository, list its GGUF files with sizes and a downloaded mark, download the selected one with an optional projector. Done when a test fills the table from a mocked listing.
- [x] **Benchmark tab.** Run `llama-bench -o jsonl` on a downloaded model with prompt / gen / depth / repeats / extra args; show results in a table; stop mid-run. Done when a test parses jsonl lines into rows.
- [x] **Redesign the TUI in forseti's style.** Keyboard-driven, no boxes or buttons, near-black theme with one indigo accent, profiles as the unit on Serve. Done: screenshots at 80/120, 8 TUI tests.
- [x] **Inline profile editing and live Hugging Face search.** Settings edit in place and autosave; Download searches as you type and shows repository details. Done: 47 tests, screenshots at 90/120.
- [~] **Validate Swift aliases.** `swift-qwen` (128k + projector) projects 21408 MiB, fits. Still needed: `swift-qwen-no-img` at 160k, run only while no other server holds the GPU. Done when the number is in `PROGRESS.md`.
- [x] **Merge to main.** Merged as PR #1.
- [x] **Kanagawa Dragon theme.** TUI colours match the user's nvim and alacritty. Done: screenshots at 120, 52 tests.
- [x] **Several servers at once.** `[stacks]` in `aliases.toml`; `lct up/down/ps`; the TUI runs and tracks many servers (state, address, RAM). Done: 55 tests, live `lct up local`.
- [x] **CPU side lane and embeddings.** Profiles `side` (Qwen3.6-35B-A3B on CPU, port 6869, model name `side`) and `embed` (Qwen3-Embedding-0.6B, port 6870), stack `local`. Done when both answer after `lct up local`.
- [x] **N-gram drafting on the 27B.** Rejected: `draft-mtp,ngram-mod` made a code edit slower (120 → 105 t/s), prose unchanged (~60). MTP alone stays.
- [x] **Live usage and a resizable log in the TUI.** Per-server CPU/RAM/VRAM, machine CPU/RAM/GPU/VRAM, draggable `Output` rule; `lct ps` shows VRAM. Done: 58 tests, screenshots at 90/120.
- [x] **Jeeves backend.** `--backend jeeves` runs PostHog/jeeves on its own PyTorch venv; profiles, `lct up/down/ps` and the TUI treat it like llama-server. Done: 64 tests, live fp8 run on the 7900 XTX.
- [x] **Strata backend.** `--backend strata` starts a Strata run config (`strata-<model>.json`); profiles, `lct up/down/ps` and the TUI treat it like llama-server. Done: 68 tests, live `lct up/down strata-flash-next`.
- [x] **Setup without a compiler.** `lct setup` downloads a prebuilt llama.cpp release (Vulkan; Metal on macOS) and checks it runs; source build only for `nvcc` or `--cmake-arg`. Done: 73 tests, fresh `LCT_HOME` install in 3 s finds the GPU, README install steps.
- [x] **Pick the build by GPU.** `nvidia-smi` → prebuilt CUDA 12.8 + bundled runtime; else Vulkan; macOS Metal; `lct setup` lists the devices it found. Done: 74 tests, live Vulkan, simulated CUDA. A real NVIDIA run is still owed.
- [x] **MTP n-max sweep on prose.** `swift-qwen` with `--spec-draft-n-max` 2, 3, 4; short replies, `--metrics` for acceptance. Done when the best n is in the profile and `PROGRESS.md`. Done: n2 best on prose (61.6), n4 on code; DFlash2 beat both.
- [x] **Vulkan PR #25666 build.** Build it beside the current one; compare MTP acceptance and prose t/s. Done: no gain on the 7900 XTX.
- [ ] **Longest lossless context.** Q4_K_S + q8_0 KV, text-only, FA scratch budgeted (~170k est., `docs/qwen27b-speed-research.md` §4). Done when a deep prompt runs with no GTT spill and the ctx is in the profile.
- [x] **DFlash2 code profile.** `incoai/Qwen3.8-27B-DFlash2-GGUF` with `--spec-type draft-dflash`, n 4–5, vs MTP on code and prose. Done: n5 adopted in `swift-qwen` at 96k.
- [ ] **Profile the speculation cycle.** A 5-token verify costs 1.13× one step, but a DFlash2 cycle adds ~11 ms more (drafter, sampling, DeltaNet state). Done when that 11 ms is broken down per part, to decide if an engine or llama.cpp patch is worth it.
- [x] **Better model + Flash-Next speed research.** Done: `docs/flash-next-speed-research.md` (no better model fits; tuned config `strata-swift-iq3_xxs-tuned.json`: prompts 2.2×, decode +3–15%; the Strata crash is a ROCm bug, patched).
- [~] **Adopt the tuned Strata config.** Point profile `strata-swift-flash-next` at `strata-swift-iq3_xxs-tuned.json` and pick the VRAM reserve with the user (4000 vs 1500). Done when `lct up` serves it and the 206k prompt test reproduces ~1,550 t/s.
- [ ] **Report the ROCr BlitSdma over-read** (ROCm/rocm-systems and Arch `hsa-rocr`) from `tmp/strata-tuning/rocr-fix/ISSUE.md`, with the user's OK. Done when both are filed.
- [ ] **Keep the CPU under ~90 °C in long runs** (user chose a power limit: service `cpuPowerLimit` in `~/dotfiles/nix/linux/default.nix`, PL1 75 W / PL2 125 W), then replay the 8 past tasks on the tuned config. Done when pass rate and minutes compare to the 66 min / 5-of-8 baseline.
- [ ] **Higher-bit Flash-Next (UD-IQ4_XS) on Strata.** 94 GB download, resident low-RAM mode. Done when the 8-task replay compares it to IQ3_XXS.
- [ ] **Fewer kernels per Strata verify window on gfx1100.** Fold quantize/swiglu/combine (~10 kernels per layer of ~41). Done when generate-mode decode gains ≥5% with identical greedy tokens.
