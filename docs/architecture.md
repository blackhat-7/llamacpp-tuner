# Architecture

lct serves local models through several engines. Every engine change we adopt must be rebuildable from this repo:
a fresh clone, the build scripts and the model downloads give back what the profiles serve. `tmp/` is scratch,
never the only copy of anything we rely on.

## Layout

| Path | Holds | In git |
|---|---|---|
| `llamacpp_tuner/` | lct: CLI (`cli.py`), TUI (`tui.py`), library modules | yes |
| `llamacpp_tuner/backends/` | one module per engine lct runs, and the registry `BACKENDS` | yes |
| `profiles.toml` | the profiles and stacks lct serves; paths written as `${LCT_HOME}/...` | yes |
| `engines/<engine>/` | how to rebuild an engine with our changes: pin, changes, `build.sh`, `README.md` | yes |
| `experiments/<study>/` | research tools and studies, any method | yes, small files only |
| `bench/` | measurement tools reused across studies | yes |
| `docs/` | research write-ups with the measured numbers; this file | yes |
| `skills/` | agent skills: model choice, tuning method, video | yes |
| `$LCT_HOME` | downloads, engine builds and outputs, server state and logs, scratch | no |

`$LCT_HOME` defaults to `tmp/` in a checkout and to `~/.cache/lct` for an installed lct.

## Profiles

- A profile is a name for `lct serve` arguments; a stack is a list of profiles (`[stacks]`).
- In a checkout, lct reads and writes `profiles.toml` at the repo root. `LCT_HOME` moves only the cache dir, so a
  worktree can share the main checkout's `tmp/` and serve its own branch's profiles. An installed lct uses
  `aliases.toml` in its cache dir. The format is the same.
- `${LCT_HOME}` in a profile becomes the cache dir when lct reads it. lct writes paths under the cache dir back as
  `${LCT_HOME}/...` (TUI edits, `lct import`, `lct setup`), so the file stays portable.
- The profiles in git are what this machine serves. Another machine edits them; a fork keeps its own.

## Backends (Python)

- `llamacpp_tuner/backends/<name>.py` runs one engine: `serve(model, options, env)` in the foreground,
  `setup(...)` to install it, `address(line)` to find the URL in its ready log line.
- `backends/__init__.py` lists them in `BACKENDS`. The CLI's `--backend` choices, `lct serve`, `lct setup` and the
  server address lookup all go through it. A new engine is one module plus one registry line.
- `messages.py` (Anthropic Messages API over an OpenAI-style server) is shared by any backend that needs it.

## Engines: the pattern

```
engines/<engine>/
  README.md   upstream and exact pin; our changes, one line each (what, measured effect, research doc);
              toolchain; outputs; how to check a build
  build.sh    idempotent: fetch upstream at the pin, apply our changes, build, install under $LCT_HOME
  ...         the changes in the engine's own form: patches/ (git format-patch), kernels, transform
              tools, config templates, presets
```

- Pins are exact: a tag or commit, a package revision, a release build number.
- Outputs go only under `$LCT_HOME`. A script never overwrites an output that differs without `--force`.
- A changed build gets a new name (Strata names builds by source tree hash; Loom takes `--output`), so old and new
  sit side by side and a profile rolls back by pointing at the old one.
- An artifact too big for git or too slow to remake gets a recipe and a stated source. If it has neither, the
  README says so.

Engines now: `strata/` (33 patches on v0.2.9, HIP engine + CUDA helper, run configs, model packs), `loom/`
(Inkling-2-Mini package: 8-bit KV, attention kernel v2, drafter window), `llama.cpp/` (named builds the profiles
use), `rocr/` (ROCm runtime fix). Jeeves and ComfyUI install through `lct setup --backend NAME`.

## Making an optimization

1. **Experiment any way.** Branch the engine checkout that `build.sh` made, edit kernels, transform models, sweep
   configs, try a new engine. Keep scratch in `$LCT_HOME`. Scripts worth keeping go to `experiments/<study>/`.
2. **Measure.** Use `bench/` or your own tools. Record numbers and the method in `docs/<topic>-research.md`.
3. **Adopt by making it reproducible.** Put the change into `engines/<engine>/` (export patches, add the kernel or
   build step), point the profile in `profiles.toml` at the new output, and add a `DECISIONS.md` line.
   A change that lives only in `tmp/` or in an unpushed branch is not adopted.
