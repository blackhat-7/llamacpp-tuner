# Decisions

Why things are the way they are. Append only. One entry, one to three lines. Add one when you choose something a later reader might otherwise undo.

- **Aliases are plain `lct serve` argument strings, expanded before Click parses.** (2026-09-24) One concept, no schema. Options typed after an alias win because Click keeps the last value; `--extra-args` is replaced whole, not merged.
- **Aliases live in `aliases.toml` in the cache dir (`tmp/` in a checkout).** (2026-09-24) It is user data, so it follows the `cache.py` rule and is gitignored. Saving from the TUI rewrites the file and drops comments.
- **The TUI is Python + Textual, not Go/Rust/TypeScript.** (2026-09-24) It calls lct's own functions and CLI, so no logic is duplicated and there is one toolchain. Textual gives tabs, tables, selects and async logs out of the box.
- **The TUI runs `lct serve` / `lct pull` as child processes and `llama-bench` directly.** (2026-09-24) Reuses CLI resolution and signal handling. Children get their own session and are stopped with SIGINT to the group, which is what Ctrl-C does.
- **Benchmarks use `llama-bench`, never custom timing code.** (2026-09-24) A custom benchmark was removed in `816ef60`; llama.cpp's tool is the reference.
- **Default context limits come from measured VRAM on the 7900 XTX (24 GB).** (2026-09-24) Qwen3.8-27B Q4_K_M: 128k with the projector, 160k text-only, using `-ub 1024` and q8_0 KV for the MTP draft (`-ctkd/-ctvd`). 180k text-only leaves ~0.35 GB and is too tight.
- **The TUI copies forseti's look: text and keys, not widgets.** (2026-09-25) The first version used Textual's boxed inputs, selects and buttons and read as generic. Now: near-black theme, one indigo accent, colour only for status, a key-hint line, OptionLists and one-line inputs. Serve acts on saved profiles, so a server always starts from a named, reproducible command.
