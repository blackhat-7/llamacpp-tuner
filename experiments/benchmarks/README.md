# Public benchmarks

Question: how do the local models score on public benchmarks next to published numbers?
Results: `docs/gpqa-diamond-2026-11-26.md`.

- `gpqa.py DATA.parquet OUT.jsonl --model ID [--sampling JSON] [--limit N] [--stop-at UNIX] [--cool-below °C]`:
  GPQA Diamond against an OpenAI-compatible server on :6868, one question at a time, seeded order, resumable.
  Data: `fingertap/GPQA-Diamond` (ungated mirror). Needs `uv run --no-project --with pandas --with pyarrow`.
- `score.py A.jsonl B.jsonl`: scores on the shared questions, 95% intervals, paired McNemar test.
- Outputs: `$LCT_HOME/experiments/benchmarks/`.
