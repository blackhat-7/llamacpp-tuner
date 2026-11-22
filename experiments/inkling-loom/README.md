# Inkling-2-Mini on Loom: kernels and benchmarks

The question: how fast can Loom decode Inkling-2-Mini on the 7900 XTX alone? Numbers and conclusions are in
`docs/inkling-loom-research.md`. Adopted changes live in `engines/loom/`.

## Paths

- `LCT_HOME` (default `<repo>/tmp`). The scripts use `$LCT_HOME/loom/venv` (from `engines/loom/build.sh`) and
  packages under `$LCT_HOME/loom/`.
- Logs and results go to `$LCT_HOME/experiments/inkling-loom/`. Benchmark directories are arguments.
- Long prompts are `$LCT_HOME/model-hunt/prompt_inkling_{64000,195000}.txt`. They are not committed (190 and
  577 KB). `probe_loom.py build` makes them: a code dump of N tokens with three "deploy key" facts.
- Everything that touches the GPU takes `flock /tmp/xtx-gpu.lock`. Do not run benchmarks unprompted.

## Kernel generators (print or write `.loom`; no GPU)

- `gen_v2.py NW > k.loom` (`FMT=f16|kv8`, `U`): decode attention v2. `FMT=kv8 U=2 gen_v2.py 8` is the kernel in
  `engines/loom/kernels/`; `gen_v2.py 4` is the f16 one.
- `gen_span.py > k.loom` (`FMT`, `U`, `P`, `TG`, `ALLH`, `DL`): span-v2, the verify kernel that shares K/V loads
  across span tokens. `kernels/spa_p4.loom` = `FMT=kv8 ALLH=1 P=4 U=2 TG=2 DL=8`. The other `kernels/sp*.loom` are
  earlier sweep points.
- `dsp_window.py LLM_LOOM W`: prints the drafter attention kernel limited to the last W positions.
  `engines/loom/tools/dsp_window.py` is the same edit, applied in place.
- `kv8-format.md`: the 8-bit KV layout.

## Kernel tests and benchmarks (`loom-benchmark` on the GPU)

- `gen.py OUTDIR SPLIT_KERNEL SOURCE.loom...`: `OUTDIR/bench.loom` (one decode step at 64K and 195K) and
  `OUTDIR/test.loom` (the kernel's partials against the original's).
- `gen_spanbench.py OUTDIR KERNEL.loom SOURCE.loom...`: the same for span kernels (spans of 1 to 8 tokens).
- `gen_dsp.py OUTDIR SOURCE.loom...`: drafter, verify and decode attention costs at each depth.
- `run.sh OUTDIR`: runs `OUTDIR/bench.loom`; prints time per kernel and GB/s. `run2.sh OUTDIR`: time per kernel
  for each named benchmark.

## End to end (`loom-serve` on the GPU)

- `e2e.sh PKG LABEL MAX_TOKENS PROMPT...` with `e2e_client.py`: serves a package, sends each prompt greedy,
  appends decode t/s and text to `e2e.jsonl`.
- `headline.sh`: serves `kv8w4096` with `xtx-kv8-196k` and runs code, prose and recall at 195K with
  `probe_loom.py`, with draft counters and VRAM.
- `probe_loom.py build|run`: makes depth prompts; runs code, prose and recall questions through any
  OpenAI-style server.
- `session/session.py --model DIR --preset P --jobs session/jobs_*.json --out OUT.jsonl`: one server, many jobs,
  one JSON line each (TTFT, t/s, draft metrics, peak VRAM). `$LCT_HOME` in a job's `prompt_file` is expanded.
- `session/summ.py < OUT.jsonl`: one line per job. `session/compare.py OUT.jsonl A B`: first token where two labels'
  greedy outputs differ.
- `session/fit.sh MODEL PRESET LABEL`: VRAM before and after one short request. `session/probe_start.sh`: start and
  one "Say hi". `session/mt.sh` with `multiturn.py`: two turns at 64K, to check prefix-cache reuse.
- `prompts/`: the 1K depth prompt and a one-line prompt.
