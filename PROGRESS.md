# Progress

Handoff note. Rewritten at the end of every session, never appended to. Cap 40 lines.
Next task: `PLAN.md`. Rules: `AGENTS.md`.

**Last session:** 2026-09-28

## State

- `lct up/down/ps` run several servers detached. Stacks are a `[stacks]` table in `aliases.toml`. State is `servers/<name>.json` + `.log` in the cache dir; the old `server.json` is gone.
- The TUI tracks many servers: stacks first (`enter` toggles the whole stack), each profile marked ◌ loading / ● ready with its RAM. Server log lines start with the profile name.
- Stack `local` in `tmp/aliases.toml` = `swift-qwen` (GPU, :6868) + `side` (Qwen3.6-35B-A3B UD-Q4_K_S on CPU, :6869, served as model `side`) + `embed` (Qwen3-Embedding-0.6B Q8_0 on CPU, :6870).
- Measured `side` on CPU: loads in 10 s, 20.5 GiB RAM, decode ~15 t/s, prefill ~91 t/s (3k prompt 33 s cold, 4 tokens when cached), a title in under 1 s.
- `swift-qwen`: 128k, two slots sharing one KV pool, `-ub 512`. VRAM 22.3 GiB for the server, 894 MiB free on the card; the projector needs up to 1158 MiB, so an image spills ~0.26 GB.
- Pi's auto-mode classifier runs on `swift-qwen`'s second slot (~2.3 s a pass); on `side` it took 10–25 s and timed out on parallel tool calls.
- Consumers live in the `ai-harnesses` repo: `pi --local`, `claude-local` Haiku → side, code-review-graph embeddings → :6870.
- Checks: `pytest` 55/55, ruff clean, pyright clean.

## Gotchas

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
