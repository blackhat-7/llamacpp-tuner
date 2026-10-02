# Progress

Handoff note. Rewritten at the end of every session, never appended to. Cap 40 lines.
Next task: `PLAN.md`. Rules: `AGENTS.md`.

**Last session:** 2026-10-02

## State

- `lct up/down/ps` run several servers detached. Stacks are a `[stacks]` table in `aliases.toml`. State is `servers/<name>.json` + `.log` in the cache dir; the old `server.json` is gone.
- The TUI tracks many servers: stacks first (`enter` toggles the whole stack), each profile marked ◌ loading / ● ready with its RAM. Server log lines start with the profile name.
- Stack `local-swarm` = `swarm` + `side` + `embed`; only one of the two stacks can run (both use :6868). Stack `local` in `tmp/aliases.toml` = `swift-qwen` (GPU, :6868) + `side` (Qwen3.6-35B-A3B UD-Q4_K_S on CPU, :6869, served as model `side`) + `embed` (Qwen3-Embedding-0.6B Q8_0 on CPU, :6870).
- Stack `birbot` = profile `birbot-a3b` (for `../birbot`): Qwen3.6-35B-A3B UD-IQ4_NL + mmproj-F16, `-np 3`, `--ctx 294912` = 96k each, :6868, model id `local`. After an image: 23.3 GB VRAM, 250 MB GTT, ~1.2 GB card free. 3 × 64k left 2.4 GB free; drop to that if deep prompts spill.
- Measured `side` on CPU: loads in 10 s, 20.5 GiB RAM, decode ~15 t/s, prefill ~91 t/s (3k prompt 33 s cold, 4 tokens when cached), a title in under 1 s.
- `swift-qwen`: 112k, two slots, `-ub 512`, 4 GiB Vulkan suballocation blocks (`--env`). 1.7 GB VRAM free at start; decode ~119 t/s code, ~57 prose. Deep-prompt (>100k) prefill not re-measured yet.
- Pi's auto-mode classifier runs on `swift-qwen`'s second slot (~2.3 s a pass); on `side` it took 10–25 s and timed out on parallel tool calls.
- Consumers live in the `ai-harnesses` repo: `pi --local`, `claude-local` Haiku → side, code-review-graph embeddings → :6870.
- Profile `jeeves` (PostHog/jeeves, `--backend jeeves`, fp8, 4096 ctx, 2 rows, :6871): 15 GB VRAM. 3 questions: 70 s thinking, 0.9 s without (Jeeves master + ROCm fix; the pre-#7 commit took 32 s). Cannot share the GPU with `swift-qwen`.
- Profile `swarm` (Qwen3.6-35B-A3B UD-IQ4_NL on GPU, `-np 5`, `--ctx 327680` = 64k per agent, MTP, :6868 like `swift-qwen` so `pi --local` finds it) for parallel Pi agents. Loads: 23.0 GB VRAM, 171 MB GTT. One short reply: 183 t/s. 5-way speed unmeasured. Cannot share the GPU with `swift-qwen`.
- The TUI profile list shows `N slots` (stopped or running) when the extra args set `-np`/`--parallel`.
- Checks: `pytest` 66/66, ruff clean, pyright clean.

## Gotchas

- **`test_enter_on_a_stack_starts_and_stops_all_of_it` is flaky** (~1 in 6 runs, also before the slots change). Rerun before blaming a change.

- **Jeeves upstream does not run on ROCm.** `tmp/jeeves/src` is on branch `rocm-attention` of `blackhat-7/jeeves` (commit `267cff8`). Upstream takes PRs from collaborators only. `lct setup --backend jeeves --force` re-clones upstream and loses the fix.

- **With `--kv-unified`, llama-server's default `--cache-idle-slots` clears idle slots on every new request.** Use `--no-cache-idle-slots` or two clients evict each other's cache.
- **Check per-process spill, not just card VRAM.** `drm-memory-gtt` in `/proc/<llama-server pid>/fdinfo/*` is GPU memory living in system RAM. Large spill + deep prompt = GPU watchdog reset (`ErrorDeviceLost`).
- **One GPU slot: any background request evicts the chat's prompt cache.** A 957-token title request forced a 22 s re-read of an 18k prompt. Keep background jobs on `side`.
- **RAM is BIOS-limited to DDR5-4800;** the kit is rated 5200 (XMP off). CPU decode is bandwidth-bound (~57 GB/s measured).
- **sysfs `mem_info_vram_used` under-reports at startup** (lazy commit). Trust llama.cpp's `projected to use N MiB` line or the sum of its `buffer size` lines.
- **MTP draft KV cache defaults to f16** unless `-ctkd q8_0 -ctvd q8_0` is passed.
- **A plain SIGTERM to `llama-server` did not stop it within 120 s; SIGINT did.** `lct down` sends SIGINT to the group.
- **The projector loads lazily.** It adds up to ~1.16 GB when the first image arrives, not at startup.
- **Slow downloads: first check for a VPN.** `HF_XET_FIXED_DOWNLOAD_CONCURRENCY=16` measured 13 MB/s on this Wi-Fi.
- **Check `lct ps` and VRAM before any GPU fit test.** A second GPU model sees ~50 MiB free.
- **Textual:** `OptionList` needs `text-wrap: nowrap`; never name an `App` method `run`; `_on_click` runs for every class in the MRO; workers are cancelled before `lct tui`'s `finally`.
- **Quantizer model cards describe quantization, not the model.** `repo_details` loads the base model's card first.
