# Experiments

Research tools and studies. Any method is fine: kernel generators, A/B drivers, sweeps, evals, one-off probes.
Adopted results do not live here; they go to `engines/` and `profiles.toml` (`docs/architecture.md`).

- One folder per study or tool set, with a short `README.md`: the question, how to run each tool, the env vars
  it needs, and the research doc that holds the numbers.
- Paths come from env vars (`LCT_HOME`, `STRATA_SRC`, `STRATA_DATA`, …) or arguments, never `/home/<user>/…`.
  Scripts copied as they ran may keep their old `~/…` paths; their README names them.
- Commit scripts and small inputs. Outputs, logs and big files go to `$LCT_HOME/experiments/<study>/`.
- This folder is not linted; keep scripts runnable rather than polished.

| Folder | Study |
|---|---|
| `flash-next/` | Strata decode harness for Qwen3.8 Flash-Next (`docs/flash-next-speed-research.md`) |
| `dual-gpu/` | the 4070 as the XTX engine's expert helper: agent replay and coding tests (`docs/dual-gpu-research.md`) |
| `inkling-loom/` | Loom kernels and benchmarks for Inkling-2-Mini (`docs/inkling-loom-research.md`) |
