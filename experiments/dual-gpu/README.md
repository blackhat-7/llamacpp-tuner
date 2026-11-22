# Dual GPU: the 4070 as the XTX engine's expert helper

Tools that measured Strata with the RTX 4070 holding experts for the 7900 XTX: decode speed, agent replays and a
graded coding test. Numbers and conclusions: `docs/dual-gpu-research.md`.

## Tools

| Tool | Measures |
|---|---|
| `run.sh NAME PROMPT [K=V] -- [args]` | one greedy generate run with the deployed mixB + 4070 setup (`ENG`, `PLUG`, `NOHELPER=1`, `GGUF`, `PACK`, `PROFILE`, `RPROF`) |
| `ab.sh TAG REPS "code prose edit" ARMS...` | interleaved arms through `run.sh`: `base`, `nv`, `nvN`, `nvd`, `arm%K=V`, `arm:--flag:value` (see its header) |
| `engine_price.py TAG PROMPT ARMS...` | per-round engine cost: every arm's ms-by-window-size table prices the same windows, so drafting noise drops out |
| `eval_chain.sh MODEL_ID ARM=STACK ...` | agent replay A/B: per arm `lct up STACK`, tasks 1-5 then 1-3 again through `run_eval.py`, `lct down` |
| `codebench.py ARM [LIMIT]` | HumanEval+ (164) and MBPP+ (378) pass counts against the server on `:6868`; greedy, thinking off |
| `fetch_data.py` | downloads codebench's test sets at pinned Hub revisions (`uv run --with pyarrow python fetch_data.py`) |

Prompts (`code`, `prose`, `edit`) come from `../flash-next/`. `run.sh` and `ab.sh` start the engine; `codebench.py`
and `eval_chain.sh` need a server on `:6868` (`eval_chain.sh` starts its own stacks).

## Env and outputs

- `STRATA_DATA` and `STRATA_SRC` are required for `run.sh` and `ab.sh`; `codebench.py` runs the tests with
  `$STRATA_SRC/.venv/bin/python` (numpy) or `CODEBENCH_PYTHON`. `LCT_HOME` defaults to `<repo>/tmp`.
- `eval_chain.sh` drives the agent replay harness in `AB_EVAL` (default `$LCT_HOME/ab-eval`: `run_eval.py`, its
  tasks and a Pi config). It is not copied here: it holds a Pi auth file and a 4.5 GB workspace.
- Logs: `$LCT_HOME/experiments/dual-gpu/logs/`; codebench data, answers and results:
  `$LCT_HOME/experiments/dual-gpu/codebench/`; agent replay results: `$AB_EVAL/eval/results.jsonl`.
- Locks: the shared `$LCT_HOME/experiments/gpu.lock` (and `build.lock`, `STOP`), as in `../flash-next/README.md`.

Not copied: the one-off `ab_run_*.sh` / `batch_*.sh` / round scripts, their outputs, the hybrid GGUF builder and the
server A/B driver (`tmp/dual/serve/`).
