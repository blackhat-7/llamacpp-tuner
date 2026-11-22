# Refit Strata's MTP drafter to Swift (round 9b)

These scripts made `mtp/rt-sd2-swift`, the drafter the Flash-Next run configs load (`--mtp`). They refit the dense
part of Basalt's drafter to Swift by self-distillation; the experts stay at q2b. Result: drafts held per round +3%
offline, temp-1 decode ~+3% (`docs/flash-next-speed-research.md`, round 9). Copied from `tmp/strata-opt/mtpfit/`
as they ran; they still name the old paths below, so set those to your own before a run.

## Inputs

- Strata checkout with its venv (`~/Documents/projects/Strata`, i.e. `$STRATA_SRC`) and `$STRATA_DATA`.
- The starting drafter `$STRATA_DATA/mtp/rt-basalt-q2b`: Basalt's drafter in Strata's format, made with
  `tools/mtp_q2b.py` (in our patch series, `engines/strata/patches/`).
- A parity engine: an upstream Strata worktree at engine 210ebbdd (`Strata-wt/r5-upstream`) with `parity_commit.diff`
  (or `mtp_parity.patch`) applied, built with the flags in `engines/strata/build.sh`.
- The Swift GGUF the configs load, and a GPU. GPU steps take the shared `$LCT_HOME/experiments/gpu.lock` (`gpu_hold.sh`, `../gpu_run.sh`).

## Steps

1. `prompts.py OUT.jsonl`: held-out code, edit and prose prompts (from the Python stdlib and `/usr/include`).
2. `gen_data.py transcripts PROMPTS --out D/transcripts.jsonl`: the main model's own replies at the user's
   sampling; `gen_data.py dump` then `check`: its residual at every position (GPU).
3. `train.py --data D/d1 D/d2 --init $STRATA_DATA/mtp/rt-basalt-q2b --out rt-sd2 --tv 1.0`: the fit
   (`chainC.sh` ran it; `sd2` = KL + TV weight 1, lr 2e-5, the one adopted). `train.py --dry` checks shapes on CPU.
4. `calib.py --rt rt-sd2 --data ...`: the drafter's `STRATA_SPEC_CUM_ACC` table for the run config.
5. `parity.py`: the numpy reference step (`mtp_ref.py`) against the engine's own draft logits.
6. `ab.py --arm base=... --arm sd=...`: drafters A/B on the engine's serve protocol; `replay.py` prices the draft
   stop offline.

`chain*.sh` are the orchestration as run on 2026-10-18. Transcripts, dumps and checkpoints are not committed:
step 2 remakes them. The fitted drafter itself is published separately (see `engines/strata/README.md`).
