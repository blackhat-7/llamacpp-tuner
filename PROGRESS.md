# Progress

Handoff note. Rewritten at the end of every session, never appended to. Cap 40 lines.
Next task: `PLAN.md`. Rules: `AGENTS.md`.

**Last session:** 2026-09-24

## State

- Branch `feat/tui` (from `feat/serve-aliases`). Neither is merged to `main`. Both track `origin`.
- `lct serve <alias>` works. Local aliases in `tmp/aliases.toml`: `qwen`, `qwen-no-img`, `swift-qwen`, `swift-qwen-no-img`, all on `100.64.0.1:6868`.
- `ukisai/Swift-Qwen3.8-27B-GGUF` Q4_K_S (16.6 GB) was downloading at ~3 MB/s. The `swift-*` aliases fail with "Model not found" until it lands.
- `textual` is a new dependency. `llama.get_llama_binary(name)` now finds `llama-bench` as well as `llama-server`.
- `lct tui` exists: tabs, output log, Serve tab (profiles, start/stop with ctrl+s, save). Download and Benchmark tabs are layout only.
- Checks: `pytest` 39/39, ruff clean, pyright clean.

## Gotchas

- **sysfs `mem_info_vram_used` under-reports at startup** (lazy commit). Trust llama.cpp's `projected to use N MiB` line or the sum of its `buffer size` lines.
- **MTP draft KV cache defaults to f16** unless `-ctkd q8_0 -ctvd q8_0` is passed.
- **A plain SIGTERM to `llama-server` did not stop it within 120 s; SIGINT did.**
- **The projector loads lazily.** It adds up to ~1.16 GB when the first image arrives, not at startup.
- `hf_xet` downloads do not grow the `.incomplete` file steadily; check `~/.cache/huggingface/xet/logs/` to see progress.
