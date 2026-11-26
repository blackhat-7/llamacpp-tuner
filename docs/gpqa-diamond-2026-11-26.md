# GPQA Diamond: local Flash-Next vs local Inkling (2026-11-26)

One overnight run, 02:45-09:20. Question: how good are the two local models next to well-known ones?

## Why GPQA Diamond

- r/LocalLLaMA trusts no single public benchmark (old ones get trained on). GPQA Diamond is still the score most
  vendors publish, so it gives the most comparison points. It needs no judge model and no sandbox.
- It is near saturation: frontier models score ~96%. It ranks models, but cannot show the gap to the frontier well.
- Ruled out for one night: SWE-bench, Terminal-Bench (Docker, agent loops of 100K+ tokens a task), HLE (needs a
  judge), AIME/HMMT (saturated). LiveCodeBench v6 is the next candidate (needs a code sandbox).

## Setup

- Data: `fingertap/GPQA-Diamond` (198 questions, choices already shuffled). Prompt and `Answer: X` grading as in
  openai/simple-evals. Questions in one seeded order (`random.Random(0)`), so both models saw the same ones.
- Runner `experiments/benchmarks/gpqa.py`, scorer `experiments/benchmarks/score.py`; raw answers with full
  thinking in `$LCT_HOME/experiments/benchmarks/{flash,inkling}.jsonl`.
- Flash: stack `local-flash` (Swift = huihui abliterated v2 Qwen3.8-Flash-Next, mixB ~4-bit, Strata), default
  thinking, temperature 1.0, top-p 0.95, top-k 20. Inkling: stack `local-inkling` (Inkling-2-Mini, Loom
  `kv8-r3v`), default reasoning level `high`, temperature 1.0, top-p 1.0, top-k 0. Both `max_tokens` 32,000.
- Flash ran 165 questions, Inkling 153, in ~3 h 20 min each. Scores below use the 153 both answered.

## Results (153 shared questions)

| | right | score | 95% CI | median / mean tokens | s per question |
|---|---|---|---|---|---|
| local Flash-Next | 133 | 86.9% | 80.7-91.4 | 5,027 / 8,700 | 75 |
| local Inkling | 127 | 83.0% | 76.3-88.1 | 7,266 / 10,815 | 74 |

- Flash's own 165: 145 right (87.9%).
- The gap is not significant: Flash alone right on 11, Inkling alone on 5 (McNemar p = 0.21).
- **Inkling hit the 32K output cap on 15 questions** (10%), still reasoning, not looping; each counts as wrong.
  On the other 138 the two tie: Flash 126, Inkling 127. The cap is Loom's preset limit (`xtx-park`: 32K output).
  Flash never hit it. Flash got 7 of those 15 right.

## Next to published scores (vendor numbers unless marked; different harnesses, full 198 questions)

| model | GPQA Diamond |
|---|---|
| Gemini 4 Argon | 96.5 (OpenRouter listing) |
| GPT-6 Astra xhigh | 96.3 (Artificial Analysis) |
| Qwen3.8-Flash-Next, official BF16 | 91.7 |
| Claude Opus 4.6 Max | 91.3 |
| DeepSeek V4 Flash (0731) | 90.8 |
| Qwen3.8-27B | 89.2 |
| **local Flash-Next (this run)** | **86.9 (CI 80.7-91.4)** |
| Gemma 4 31B | 84.3 |
| **local Inkling (this run)** | **83.0 (CI 76.3-88.1)** |

- Thinking Machines publishes no GPQA score for Inkling-2-Mini; this run fills that gap.
- Local Flash is ~5 points under the official 91.7. Possible causes, not separated: the ~4-bit quant, the
  abliteration, the 32K budget, and sample noise (the CI reaches 91.4).
- Claude Opus 5.5 has no published GPQA score (Anthropic calls it saturated).

## Caveats

- One sample per question at temperature 1; a rerun can move each score by a few points.
- 153 of 198 questions. Inkling's score depends on the 32K output cap; a run with a larger cap would show its
  real ceiling.
- The CPU reached 94-96 °C under Strata (Loom ran at ~55 °C). From 05:27 the runner waited for the CPU to drop
  below 80 °C between questions (`--cool-below 80`).
