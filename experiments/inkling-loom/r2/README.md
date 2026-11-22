# Round 2 (2026-11-22): five workstreams in parallel

Research scripts from round 2 of the Inkling-on-Loom work. Results: `docs/inkling-loom-research.md` "Round 2".
The adopted tools live in `engines/loom/tools/{attn,prefill,dense,moe,spec}`. These scripts hard-code this PC's
paths (`/home/illusion/...`, `$LCT_HOME=<repo>/tmp`); edit them before running elsewhere.

- `window/`: how the GPU was shared. `BRIEF.md` is the brief all five agents got. `gpu-run CMD` waits while
  `/tmp/xtx-gpu.hold` exists (the user's own server owns the GPU), then takes `flock /tmp/xtx-gpu.lock`.
  `regcheck.sh PKG PRESET` loads a package and prints "ready" or the registration error (~20 s); run it before any
  long session. `window-a-tests.sh` runs every new kernel test; `window-b2.sh` with `jobs_base.json` and
  `jobs_long.json` is the end-to-end comparison. `wait-idle.sh N` exits when the user's server is idle N minutes.
- `attn/`: SWA and combine kernel generators (`gen_swa.py`, `gen_combine.py`), kernel tests and benchmarks.
- `prefill/`: split-K flash-attention test and benchmark generators; `pending.sh` lists the runs not yet done
  (the prefetch variant `PREFILL_ATTN=pf`).
- `dense/`: kernel tests for the fused projections, norms, controllers and mixes; `gpu-plan.sh` lists the
  `--sconv` runs not yet done.
- `moe/`: MoE microbenchmark, router and padding tests, float64 reference.
- `spec/`: draft-vocabulary corpus and frequency tools, the reduced draft head (`make_draft_head.py`,
  `draft_head.py`, `draft_vocab_32768.json`; not adopted: its first version failed registration, the fix is
  untested), n-gram simulation, summaries, and the run scripts.
