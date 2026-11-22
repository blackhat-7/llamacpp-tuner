"""Writes a Q8_0 copy of an mmproj GGUF: every 2-D F16/BF16 matrix becomes Q8_0; the patch embedding (4-D), the
F32 position table, norms, biases and embeddings keep their type. Deterministic (gguf-py's NumPy
Q8_0). The Q8_0 projector halves the encoder's VRAM (0.64 to 0.34 GB), which the 196,608 context needs.

usage: quantize_mmproj.py IN.gguf OUT.gguf"""

import sys

import gguf
import numpy as np


def main(src: str, dst: str) -> None:
    r = gguf.GGUFReader(src)
    w = gguf.GGUFWriter(
        dst, arch=bytes(r.fields["general.architecture"].parts[-1]).decode()
    )
    for name, field in r.fields.items():
        if name.startswith("GGUF.") or name in (
            "general.architecture",
            "general.file_type",
        ):
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            sub = field.types[-1]
            values = [
                bytes(field.parts[i]).decode()
                if sub == gguf.GGUFValueType.STRING
                else field.parts[i].tolist()[0]
                for i in field.data
            ]
            w.add_key_value(name, values, vtype, sub)
        elif vtype == gguf.GGUFValueType.STRING:
            w.add_key_value(name, bytes(field.parts[-1]).decode(), vtype)
        else:
            w.add_key_value(name, field.parts[-1].tolist()[0], vtype)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0)
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape)]  # NumPy order: rows first
        data = np.array(t.data)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            data = (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        if len(shape) == 2 and t.tensor_type in (
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        ):
            q = gguf.quants.quantize(
                data.astype(np.float32).reshape(shape), gguf.GGMLQuantizationType.Q8_0
            )
            w.add_tensor(t.name, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(t.name, data.reshape(shape), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
