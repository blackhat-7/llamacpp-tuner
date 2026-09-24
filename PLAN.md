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
- [ ] **Validate Swift aliases.** After `ukisai/Swift-Qwen3.8-27B-GGUF` Q4_K_S finishes downloading, load `swift-qwen-no-img` at 160k and read llama.cpp's projected device memory. Done when the number is in `PROGRESS.md`.
- [ ] **Merge to main.** Branches `feat/serve-aliases` and `feat/tui` are pushed but not merged. Done when the user says how (fast-forward or PR).
