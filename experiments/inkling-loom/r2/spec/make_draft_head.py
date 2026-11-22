#!/usr/bin/env python3
"""Write the drafter GGUF plus a reduced proposal head (deterministic; CPU only).

The new file is the drafter checkpoint with three extra tensors:
  output.weight          the target's head, byte copy (Q8_0): the proposal-head root (head_weights.loom, which
                         converts it to the full Q4_0 copy for cohorts of 2+ spans) now reads this file, so every
                         proposal stage binds the same roots
  dsp_draft_head.weight  Q4_0 [2304, N]: rows `ids` of the target's output.weight (Q8_0), dequantized and
                         requantized with llama.cpp's quantize_row_q4_0_ref (what the load-time full Q4_0 copy does)
  dsp_draft_head.ids     N token ids as int32 bit patterns stored in an F32 tensor (loaded as plain bytes)

usage: make_draft_head.py TARGET.gguf DRAFTER.gguf IDS.json OUT.gguf
"""
import json
import sys

import gguf
import numpy as np

HIDDEN = 2304


def q8_0_rows(raw: np.ndarray) -> np.ndarray:
    blocks = raw.reshape(raw.shape[0], HIDDEN // 32, 34)
    d = blocks[:, :, :2].copy().view(np.float16).astype(np.float32)
    q = blocks[:, :, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(raw.shape[0], HIDDEN)


def q4_0(x: np.ndarray) -> np.ndarray:
    """llama.cpp quantize_row_q4_0_ref, row-major blocks of 32 -> 18 bytes."""
    b = x.reshape(-1, 32).astype(np.float32)
    idx = np.argmax(np.abs(b), axis=1)
    mx = b[np.arange(len(b)), idx]
    d = (mx / np.float32(-8)).astype(np.float32)
    inv = np.where(d != 0, np.float32(1) / np.where(d != 0, d, 1), 0).astype(np.float32)
    xi = np.minimum(15, (b * inv[:, None] + np.float32(8.5)).astype(np.int8)).astype(np.uint8)
    qs = xi[:, :16] | (xi[:, 16:] << 4)
    out = np.concatenate([d.astype(np.float16)[:, None].view(np.uint8), qs], axis=1)
    return out.reshape(x.shape[0], HIDDEN // 32 * 18)


def main() -> None:
    target, drafter, ids_path, out_path = sys.argv[1:5]
    ids = np.array(json.load(open(ids_path)), dtype=np.int32)
    assert len(ids) % 128 == 0 and np.all(np.diff(ids) > 0)
    head = next(t for t in gguf.GGUFReader(target).tensors if t.name == "output.weight")
    assert head.tensor_type == gguf.GGMLQuantizationType.Q8_0
    rows = q4_0(q8_0_rows(np.asarray(head.data[ids])))

    reader = gguf.GGUFReader(drafter)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(out_path, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), field.types[0], sub_type=sub)
    extra = [("output.weight", np.asarray(head.data), gguf.GGMLQuantizationType.Q8_0),
             ("dsp_draft_head.weight", rows, gguf.GGMLQuantizationType.Q4_0),
             ("dsp_draft_head.ids", ids.view(np.float32), gguf.GGMLQuantizationType.F32)]
    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for name, data, kind in extra:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    for _, data, _ in extra:
        writer.write_tensor_data(data)
    writer.close()


main()
