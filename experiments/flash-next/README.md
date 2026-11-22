# Flash-Next decode harness (Strata)

Measures Strata's decode speed and exactness for Qwen3.8 Flash-Next (Swift) in generate mode: greedy, 600 tokens,
fixed prompts. Numbers, rounds and conclusions: `docs/flash-next-speed-research.md`.

## Tools

| Tool | Measures |
|---|---|
| `gen.sh ENGINE NAME IDS MAX_NEW [args]` | one run: `decode ... tok/s`, `--stats` timings; with a `.follow` file the window hashes |
| `r7_run.sh` / `r8_run.sh NAME IDS ENGINE [K=V] -- [args]` | `gen.sh` with the round-6 / round-7 installed flags and env; also writes the per-window `.spec` log |
| `cmp.py BASE.log OTHER.log ...` | speed, tokens per round, and whether ids and window hashes match BASE |
| `ab.sh TAG ROUNDS "ENG_A [K=V]" "ENG_B [K=V]"` | interleaved A/B on code/edit/prose; exactness per pair, mean t/s per prompt |
| `r9d.sh TAG ENGINE DRAFTER` | drafter and draft-stop arms on the follow texts, priced by `r7d_std.py` |
| `r7d_std.py TAG [ARM ...]` | prices every arm's windows at one shared ms-by-window-size table: drafter changes without the 8-16% run noise |
| `prof.sh ENGINE NAME IDS MAX_NEW` | `gen.sh` under rocprofv3 kernel trace (`ROCPROF` = path; Arch's ROCm 7.2.4 has none) |
| `gpu_run.sh CMD ...` | any GPU program (parity test, microbenchmark) under the lock, 10-minute timeout |

Inputs: `code`, `edit`, `prose` as `.txt` (the prompt), `.ids` (its tokens) and `.follow` (a fixed continuation, so
two builds verify the same windows).

## Exactness

Run with `--suffix-draft 0` and compare to a reference made the same way: `cmp.py` must print `windows bit-identical`.
With the lookup drafter on, windows depend on timing.

## Env and outputs

- `STRATA_DATA` (packs, drafters, profiles, engines) and `STRATA_SRC` (the Strata checkout) are required.
  `LCT_HOME` defaults to `<repo>/tmp`. `GGUF`, `PACK`, `MTP`, `PROFILE` override `gen.sh`'s model files.
- Logs: `$LCT_HOME/experiments/flash-next/logs/NAME.{log,hashes,spec}`.
- ROCr: runs put `$LCT_HOME/rocr-7.2.4-fix/lib` first (`engines/rocr/`). Without it, long runs can abort.

## Lock convention

All GPU jobs of all studies take `flock $LCT_HOME/experiments/gpu.lock`; builds and dual-GPU runs also take
`build.lock` (a build steals the CPU the expert pool needs). `touch $LCT_HOME/experiments/STOP` makes queued jobs exit
instead of starting. Runs wait until the CPU is below 80 C. The old copies in `tmp/strata-opt/` use their own lock.

Not copied: one-off round scripts, logs, binaries, the routing-trace corpus (134 MB), the long-context prompts
(`long/`, 3 MB) and `swift-profile-mix.bin` (now `gen.sh` defaults to `$STRATA_DATA/profiles/swift-1.5-iq3_xxs.bin`).

`mtpfit/`: the scripts that refit the MTP drafter `rt-sd2-swift` to Swift (round 9b); see its README.
