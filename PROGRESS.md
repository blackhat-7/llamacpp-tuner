# Progress

Handoff note. Rewritten at the end of every session, never appended to. Cap 40 lines.
Next task: `PLAN.md`. Rules: `AGENTS.md`.

**Last session:** 2026-09-24

## State

- Branch `feat/tui` (from `feat/serve-aliases`). Neither is merged to `main`. Both track `origin`.
- `lct serve <alias>` works. Local aliases in `tmp/aliases.toml`: `qwen`, `qwen-no-img`, `swift-qwen`, `swift-qwen-no-img`, all on `100.64.0.1:6868`.
- `ukisai/Swift-Qwen3.8-27B-GGUF` Q4_K_S (16.6 GB) was downloading at ~3 MB/s. The `swift-*` aliases fail with "Model not found" until it lands.
- `textual` is a new dependency. `llama.get_llama_binary(name)` now finds `llama-bench` as well as `llama-server`.
- `lct tui` is complete: Serve (profiles, start/stop with `ctrl+s`, save), Download (search with sizes, download), Benchmark (`llama-bench -o jsonl` into a table, stop mid-run). Screenshot-checked at 70 and 120 columns.
- Checks: `pytest` 43/43, ruff clean, pyright clean.

## Gotchas

- **sysfs `mem_info_vram_used` under-reports at startup** (lazy commit). Trust llama.cpp's `projected to use N MiB` line or the sum of its `buffer size` lines.
- **MTP draft KV cache defaults to f16** unless `-ctkd q8_0 -ctvd q8_0` is passed.
- **A plain SIGTERM to `llama-server` did not stop it within 120 s; SIGINT did.**
- **The projector loads lazily.** It adds up to ~1.16 GB when the first image arrives, not at startup.
- **Slow Hugging Face downloads are the ISP route, not xet.** Measured 2026-09-25: xet at concurrency 1 and forced to 16 both gave ~1.4 MB/s; OVH also ~1 MB/s. Watch progress via the `.incomplete` file size under the repo's `.cache/huggingface/download/`.
- **Textual `Label` has built-in `success` / `warning` / `error` classes** that paint a badge background. The status pill uses them on purpose.
- **Never name an `App` method `run`.** It overrides Textual's `App.run` and `lct tui` breaks; the helper is `stream`.
