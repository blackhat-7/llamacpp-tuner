"""A hybrid Strata pack: one pack's dense part with another quant's routed experts, of the same model.

Strata reads experts straight from their GGUF shards through native_experts.txt (absolute offsets, a shard name per
layer or per role, files beside --native). So the hybrid is: the dense pack's files, the expert pack's
native_experts.txt with every shard named, and a model folder that links both models' shards side by side.
The routers must be byte-identical in both packs (checked), or the two files are not one model.

usage: python3 -I make_hybrid.py DENSE_PACK EXPERT_PACK EXPERT_SHARD1 DENSE_SHARD1 OUT_PACK OUT_MODEL_DIR
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path


def routers(pack: Path) -> dict[str, str]:
    conv = json.loads((pack / "conversions.json").read_text())
    return {
        t["name"]: t["source_sha256"]
        for t in conv["tensors"]
        if "ffn_gate_inp" in t["name"]
    }


def main() -> None:
    dense_pack, expert_pack, expert_shard1, dense_shard1, out_pack, out_dir = map(
        Path, sys.argv[1:7]
    )
    rd, rx = routers(dense_pack), routers(expert_pack)
    if not rd or rd != rx:
        sys.exit(
            f"routers differ ({sum(rd.get(n) != rx.get(n) for n in rd | rx)} of {len(rd | rx)}): not one model"
        )

    out_pack.mkdir(parents=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("dense.bin", "index.txt", "conversions.json"):
        os.link(dense_pack / name, out_pack / name)
    shutil.copytree(dense_pack / "tokenizer", out_pack / "tokenizer")

    # an empty shard field means the --native shard, which in the hybrid is the dense model's: name them all
    lines = []
    for line in (expert_pack / "native_experts.txt").read_text().splitlines():
        f = line.split()
        if line.startswith("#") or not f:
            lines.append(line)
            continue
        if len(f) == 8:
            f.append(expert_shard1.name)
        else:
            f[8] = ",".join(s or expert_shard1.name for s in f[8].split(","))
        lines.append(" ".join(f))
    (out_pack / "native_experts.txt").write_text("\n".join(lines) + "\n")

    for shard1 in (dense_shard1, expert_shard1):
        stem = re.sub(r"-00001-of-(\d+)\.gguf$", "", shard1.name)
        for shard in sorted(shard1.parent.glob(stem + "-0000*-of-*.gguf")):
            link = out_dir / shard.name
            if not link.exists():
                link.symlink_to(shard.resolve())
    print(f"routers identical ({len(rd)}); pack {out_pack}; model folder {out_dir}")


if __name__ == "__main__":
    main()
