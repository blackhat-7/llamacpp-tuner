"""A per-layer, per-role mix of two expert sets of one model (Strata native packs).

DOWN_LAYERS take only the down projection from the upgrade pack (its type, offset and shard); their gate/up stay the
base pack's. FULL_LAYERS take the upgrade pack's whole line. Both packs quantize the same weights, and
native_experts.txt v4 names a shard per role, so one expert blob may span two files. The experts.bin offsets and
the header's total are recomputed. Example: mixD = the base GSQ-RCO pack with IQ4_NL downs in its 30 Q2_0 layers.

usage: python3 -I mix_down.py BASE_PACK UPGRADE_PACK OUT_PACK DOWN_LAYERS [FULL_LAYERS]   (layers: 1,3,5-9)
"""

import os
import re
import shutil
import sys
from pathlib import Path

# bytes per row: gate/up rows hold 2560 values (i-quants, 256 per block), down rows hold 640 (32 / 64 per block)
GU_BLOCK = {
    16: 66,
    17: 74,
    22: 82,
    18: 98,
    21: 110,
    23: 136,
}  # IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_XS
DOWN_ROW = {20: 18 * 20, 42: 18 * 10}  # IQ4_NL, Q2_0


def blob_bytes(gu_type: int, d_type: int) -> int:
    return 2 * 640 * 10 * GU_BLOCK[gu_type] + 2560 * DOWN_ROW[d_type]


def layer_set(spec: str) -> set[int]:
    out: set[int] = set()
    for part in filter(None, spec.split(",")):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def rows(pack: Path) -> tuple[str, dict[int, list[str]]]:
    head, table = "", {}
    for line in (pack / "native_experts.txt").read_text().splitlines():
        if line.startswith("#"):
            head = head or line
        elif line.strip():
            table[int(line.split()[0])] = line.split()
    return head, table


def shards(f: list[str]) -> list[str]:
    """The shard of gate, up and down ('' = the --native shard)."""
    if len(f) < 9:
        return ["", "", ""]
    return f[8].split(",") if "," in f[8] else [f[8]] * 3


def main() -> None:
    base, up, out = map(Path, sys.argv[1:4])
    down_layers = layer_set(sys.argv[4])
    full_layers = layer_set(sys.argv[5]) if len(sys.argv) > 5 else set()
    _, rb = rows(base)
    head, ru = rows(up)

    out.mkdir(parents=True)
    for name in ("dense.bin", "index.txt", "conversions.json"):
        os.link(base / name, out / name)
    shutil.copytree(base / "tokenizer", out / "tokenizer")

    offset, lines = 0, []
    for layer in sorted(rb):
        b, u = rb[layer], ru[layer]
        if layer in full_layers:
            f = list(u)
        elif layer in down_layers:
            gu_type, d_type = int(b[1]), int(u[2])
            if int(b[4]) != blob_bytes(gu_type, int(b[2])) or int(u[4]) != blob_bytes(
                int(u[1]), d_type
            ):
                sys.exit(f"layer {layer}: blob sizes do not match the known formats")
            sb, su = shards(b), shards(u)
            f = [
                str(layer),
                str(gu_type),
                str(d_type),
                "0",
                str(blob_bytes(gu_type, d_type)),
                b[5],
                b[6],
                u[7],
                ",".join([sb[0], sb[1], su[2]]),
            ]
        else:
            f = list(b)
        f[3] = str(offset)
        offset += int(f[4]) * 512
        lines.append(" ".join(f))
    head = re.sub(r"strata native experts v\d+", "strata native experts v4", head)
    head = re.sub(r"total \d+", f"total {offset}", head)
    (out / "native_experts.txt").write_text(head + "\n" + "\n".join(lines) + "\n")
    print(
        f"down from the upgrade: {len(down_layers)} layers; whole layer: {len(full_layers)}; "
        f"experts {offset / 2**30:.1f} GiB -> {out}"
    )


if __name__ == "__main__":
    main()
