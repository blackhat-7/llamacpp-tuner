"""Download codebench's test sets (EvalPlus HumanEval+ and MBPP+, pinned revisions) and write them as JSON lines to
$LCT_HOME/experiments/dual-gpu/codebench/data/, where codebench.py reads them.

usage: uv run --with pyarrow python fetch_data.py
"""
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OUT = Path(os.environ.get("LCT_HOME") or Path(__file__).resolve().parents[2] / "tmp") / "experiments/dual-gpu/codebench/data"
SETS = {
    "humanevalplus": ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
                      "data/test-00000-of-00001-5973903632b82d40.parquet"),
    "mbppplus": ("evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
                 "data/test-00000-of-00001-d5781c9c51e02795.parquet"),
}

OUT.mkdir(parents=True, exist_ok=True)
for name, (repo, revision, filename) in SETS.items():
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    rows = pq.read_table(path).to_pylist()
    with open(OUT / f"{name}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{name}: {len(rows)} tasks -> {OUT / name}.jsonl")
