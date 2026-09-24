# lct — agent guide

**Every session, read these four files first, in order. They are short on purpose.**
`AGENTS.md` (this file) · `PROGRESS.md` (where things stand) · `PLAN.md` (what to do next) · `DECISIONS.md` (why things are the way they are).

## What this is

`lct` is a small Python wrapper around llama.cpp. It does five things:

1. find or build `llama-server`
2. download an exact GGUF quantization and optional projector
3. start `llama-server` with explicit pass-through arguments, directly or from a saved alias
4. list downloaded GGUF files
5. offer a terminal UI (`lct tui`) for the above plus `llama-bench` runs

Model recommendations and tuning methodology belong in `skills/`, not Python heuristics.

## Commands

```bash
uv run pytest          # tests
uv run ruff check .    # lint
uv run ruff format .   # format
uv run pyright         # types
uv run lct tui         # the terminal UI
```

## Session routine

1. **Orient.** Read the four files. Run `git log --oneline -5`, then the four checks above. Confirm reality matches `PROGRESS.md`. Say so if it does not.
2. **Pick one task** — the top unchecked line in `PLAN.md`. Mark it `[~]`.
3. **Do only that task.** Read the code before changing it.
4. **Verify** with the task's own done-check plus the four checks.
5. **Leave the trail.** Tick `[x]` in `PLAN.md`. Rewrite `PROGRESS.md`. Append to `DECISIONS.md` if you chose something a later reader might undo. Commit saying what and why, then `git push`.
6. **Stop.** Checks green, nothing half-done.

**A task is not finished until step 5 is finished.** Report honestly at step 4: a failing check gets written down, never worked around.

## Project rules

- Prefer the simplest correct solution. Keep code short, clear and maintainable.
- Keep quantization names open-ended strings. Never add a fixed quant allowlist.
- Verify exact Hugging Face filenames; never silently fall back to another quant.
- Prefer `huggingface_hub` and llama.cpp behavior (`llama-bench`, `llama-server`) over custom download, cache or benchmark code.
- Keep runtime flags explicit. Do not guess hardware-specific "optimal" arguments.
- Library code raises exceptions; Click commands and the TUI turn them into actionable user errors.
- Public functions have type hints. Code comments say **why**, never what.
- Keep user-writable paths in `cache.py`; package directories may be read-only.
- Do not add dependencies unless the standard library or an existing dependency cannot do the job.
- **Never run long GPU benchmarks unprompted.** The host has crashed under sustained load. Short model loads are fine.
- Never weaken a test, type or check to get green.

## Writing rules

These keep the four files usable by a fresh reader with no memory of this session.

- **Write for someone who knows nothing.** Name files, flags and commands.
- **Plain words, short sentences, one idea each.**
- **Rewrite `PROGRESS.md`, never append to it.** It is a handoff note, not a log.
- **Size caps:** `AGENTS.md` ≤ 80 lines · `PROGRESS.md` ≤ 40 lines · `PLAN.md` one line per task · `DECISIONS.md` ≤ 3 lines per entry.
- Past a cap, cut. Detail belongs in `docs/`, code comments or the commit message.
- **Record gotchas, not narrative.** One line each, in `PROGRESS.md`.
