"""Writes BASE.gguf with Inkling-2-Mini-Unbound v2's attention: the q, k, v, output and gate projections of all 24
layers (the 120 tensors v2 changes) are copied byte for byte from DONOR.gguf, a v2 file that keeps them at Q8_0
(every v2 size does). BASE (unsloth UD-Q4_K_XL) has them at Q8_0 too, so the output has BASE's exact layout and
the package's programs run it unchanged.

Both files must start from the same weights: every tensor v2 leaves alone and both files store in the same type
under the same name (the loop tensors' names changed in llama.cpp after unsloth's file)
(norms, and Q8_0 tensors when DONOR has them) must be byte-identical. Q8_0 needs no importance matrix, so this holds
across makers: unsloth's Q8_0 attn_rel equals v2 Q8_0's (checked 2026-11-22).

usage: unbound.py BASE.gguf DONOR.gguf OUT.gguf"""

import re
import shutil
import sys

import gguf

EDITED = re.compile(r"blk\.\d+\.attn_(q|k|v|output|gate)\.weight")


def main(base_path: str, donor_path: str, out: str) -> None:
    base = {t.name: t for t in gguf.GGUFReader(base_path).tensors}
    donor = {t.name: t for t in gguf.GGUFReader(donor_path).tensors}
    shared = [
        n
        for n in base
        if not EDITED.fullmatch(n)
        and n in donor
        and base[n].tensor_type == donor[n].tensor_type
    ]
    if not shared:
        raise SystemExit("unbound.py: no unedited tensor to compare")
    for name in shared:
        if bytes(base[name].data) != bytes(donor[name].data):
            raise SystemExit(
                f"unbound.py: {name} differs: the files do not share base weights"
            )
    edited = sorted(filter(EDITED.fullmatch, base))
    if len(edited) != 120:
        raise SystemExit(
            f"unbound.py: expected 120 edited tensors, found {len(edited)}"
        )
    for name in edited:
        b, d = base[name], donor[name]
        if (b.tensor_type, list(b.shape)) != (
            d.tensor_type,
            list(d.shape),
        ) or b.tensor_type != gguf.GGMLQuantizationType.Q8_0:
            raise SystemExit(
                f"unbound.py: {name}: {b.tensor_type.name} {list(b.shape)} vs {d.tensor_type.name} {list(d.shape)}"
            )
    shutil.copyfile(base_path, out)
    with open(out, "r+b") as f:
        for name in edited:
            f.seek(base[name].data_offset)
            f.write(bytes(donor[name].data))
    print(
        f"unbound.py: {len(shared)} unedited tensors equal; {len(edited)} attention tensors from {donor_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
