# Strata, stage by stage on two machines

We run the same model (Qwen3.8-Flash-Next, Swift 1.5 GSQ-RCO IQ3_XXS) on Strata on an RX 7900 XTX and want to know
exactly where an RTX 5090 is faster. `run.py` runs the engine directly on fixed token inputs, with the engine's own
per-stage profilers, so the two machines can be compared line by line. Our results are in
`results/7900xtx-13600k-20261014/` (no `noskip_*` runs there: the skip is off on AMD anyway).

## For the agent on the RTX 5090 machine

Please do this, then push. Change nothing else in the repository.

1. Check out this branch and pull.
2. Find the Strata run config the server uses: a `strata-<model>.json` in the Strata folder (the file passed to
   `serve/server.py --config`, or the one setup wrote). Note which model and quant it loads (`--native`), the Strata
   version (`git describe` in the Strata folder) and any non-default `env` entries (for example calibrated flags).
3. Stop the Strata server: the runs load the engine themselves and need the whole GPU. Close other GPU programs if
   you can. On Linux, an `nvidia-smi` that lists no compute process means the GPU is free.
4. Run, from the repository root (any Python 3.10+; no packages needed):

   ```
   python3 bench/strata-compare/run.py --config /path/to/strata-<model>.json \
     --out bench/strata-compare/results/5090-<cpu>-<yyyymmdd>
   ```

   To repeat some runs: add `--only speed_code,prof_code` (same `--out`).

   It takes about 10-20 minutes: 9 runs (10 with `nsys`), each loading the model again. Each prints its speed when
   done. **If NVIDIA Nsight Systems (`nsys`) can be installed, please install it first**: its per-kernel table is the
   most useful part for us.
   - A run that fails with `unknown argument` means the engine is older than the script expects. Note which flag in
     `notes.md`, and keep the other runs.
   - On Windows use `python` and backslashes; the script handles `lib_dirs` and `.exe` engines.
5. Write `notes.md` in the results folder with:
   - the Strata version or commit and how it was built;
   - the model file and quant;
   - OS, CPU, RAM size and speed, GPU power limit;
   - anything else running on the GPU or CPU during the runs;
   - the server's startup log, if you have it (the lines from `engine started` to `session is up`).
6. Commit the results folder and push to this branch.
7. Start the Strata server again.

## What it measures

| Run | What it shows |
|---|---|
| `speed_code`, `speed_edit`, `speed_prose` | Decode speed on a fixed 600-token continuation (`--spec-follow`), greedy, adaptive tier off |
| `prof_code`, `prof_prose` | The same with `STRATA_VERIFY_PROFILE=1` and `STRATA_DECODE_TIMING=1`: GPU time per stage, the waits, CPU pool time, hit rates, tokens per round (slower: the profiler adds time) |
| `prompt_15k`, `prompt_35k` | Prompt speed with `STRATA_PREFILL_TIMING=1`: time per prompt stage and expert streaming |
| `noskip_code`, `noskip_prose` | Decode with `STRATA_ROUTE_TAIL_SKIP=0`, NVIDIA's lossy default turned off |
| `nsys_code` (only with Nsight Systems installed) | One code decode traced per kernel (`nsys --cuda-graph-trace=node`); the per-kernel table is kept, the trace file deleted |

`summary.txt` collects the key lines of every log; `system.json` has the hardware, software and the config used.
