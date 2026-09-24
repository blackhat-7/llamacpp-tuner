# Progress

Handoff note. Rewritten at the end of every session, never appended to. Cap 40 lines.
Next task: `PLAN.md`. Rules: `AGENTS.md`.

**Last session:** 2026-09-24

## State

- Branch `feat/tui` (from `feat/serve-aliases`). Neither is merged to `main`. Both track `origin`.
- `lct serve <alias>` works. Local aliases in `tmp/aliases.toml`: `qwen`, `qwen-no-img`, `swift-qwen`, `swift-qwen-no-img`, all on `100.64.0.1:6868`.
- `ukisai/Swift-Qwen3.8-27B-GGUF` Q4_K_S and its projector are downloaded. `swift-qwen` at 128k projects 21408 MiB before the projector's ~1158 MiB; fits.
- `textual` is a new dependency. `llama.get_llama_binary(name)` now finds `llama-bench` as well as `llama-server`.
- `lct tui` is keyboard-driven in forseti's style. Serve edits profiles in place (autosave on enter or leaving a field; `n` copies, `d`+`y` deletes). Download searches Hugging Face as you type and shows details, files (shards grouped, smallest first) and a card summary from the base model.
- Checks: `pytest` 48/48, ruff clean, pyright clean.

## Gotchas

- **sysfs `mem_info_vram_used` under-reports at startup** (lazy commit). Trust llama.cpp's `projected to use N MiB` line or the sum of its `buffer size` lines.
- **MTP draft KV cache defaults to f16** unless `-ctkd q8_0 -ctvd q8_0` is passed.
- **A plain SIGTERM to `llama-server` did not stop it within 120 s; SIGINT did.**
- **The projector loads lazily.** It adds up to ~1.16 GB when the first image arrives, not at startup.
- **Slow downloads: first check for a VPN.** With one on, everything ran at ~1.5 MB/s. Without it, xet's adaptive concurrency still stalls at 1–5 connections on this Wi-Fi; `HF_XET_FIXED_DOWNLOAD_CONCURRENCY=16` measured 13 MB/s. Restarting a pull does not resume a partial xet file.
- **`OptionList` wraps long rows unless CSS sets `text-wrap: nowrap`;** Rich's `no_wrap` on the prompt is ignored.
- **Textual's SVG screenshots drop a leading space in a styled span.** Put separators inside the preceding span.
- **Never name an `App` method `run`.** It overrides Textual's `App.run` and `lct tui` breaks; the helper is `stream`.
- **Quantizer model cards describe quantization, not the model.** `repo_details` loads the base model's card first.
- **Check `ss -ltnp` and VRAM before any GPU fit test.** The user often has a server on `100.64.0.1:6868`; a second model then sees ~50 MiB free.
- **Textual's `App.on_unmount` did not run on quit**, so servers outlived the UI. `action_quit` and `lct tui`'s `finally` now call `stop_children()`; a test pins it.
