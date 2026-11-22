#!/usr/bin/env python3
"""Microbenchmark of one prefill chunk of global attention: ts2 against split-K variants (+ combine).

usage: gen_skbench.py OUTDIR NAME=LLM_LOOM:KERNEL...
  e.g. sk=prefill-b/program/llm.loom:g_ink_attention_flash2_ts2_sk pf8=prefill-c/program/llm.loom:g_ink_attention_flash2_ts2_pf
Cases: 1024 tokens (64 uniform tiles) at positions 32768 and 131072 (the middle of 64K and 187K prompts), one
layer, kv8 arena, identity page map. Run: loom-benchmark OUTDIR/bench.loom --device=amdgpu --measure=dispatch_complete
"""
import os
import sys

out = sys.argv[1]
HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
os.makedirs(out, exist_ok=True)
srcs = {}


def src(path):
    srcs.setdefault(path, open(path).read())
    return srcs[path]


def func(s, name):
    i = s.index(f"func.def inline @{name}(")
    return s[i:s.index("\n}\n", i) + 3]


def kern(s, name, new):
    i = s.index(f'kernel.def export("{name}") @{name}(')
    k = s.index("\n}\n", s.index("launch(", i)) + 3
    return s[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")


variants = [a.split("=", 1) for a in sys.argv[2:]]
first = src(variants[0][1].split(":")[0])
parts = [HEAD] + [func(first, f) for f in ("g_kv8_scale", "g_kv8_offset", "g_kv8_load16", "g_tile_uniform", "g_tile_adjacent")]
parts += [kern(first, "g_ink_attention_flash2_ts2", "k_ts2"), kern(first, "g_ink_attn_combine_flash", "k_comb")]
for name, spec in variants:
    path, kernel = spec.split(":")
    parts.append(kern(src(path), kernel, f"k_{name}"))
T = 1024
for pos in (32768, 131072):
    pool = -(-(pos + T) // 4096) * 4096
    blocks = pool // 16
    A = f"tensor<{2 * pool * 1088}xi8>"
    ty = (f"index, index, i32, index, index, tensor<{T}x3072xf32>, tensor<{T}x384xf32>, tensor<16384xf32>, tensor<{T}x24xf32>, "
          f"tensor<{T}x3072xf32>, tensor<{T}x4xi32>, tensor<{T}xi32>, tensor<{10 + blocks}xi32>, {A}")
    args = "%t, %layer, %sliding, %context, %pool, %q, %r, %rb, %sinks, %gate, %info, %tiles, %orig, %arena"
    setup = f"""  %seed = check.literal value(7) : i64
  %t = check.literal value({T}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value(0) : i32
  %context = check.literal value({pool}) : index
  %pool = check.literal value({pool}) : index
  %info = check.file.read.npy path("info_{pos}.npy") : tensor<{T}x4xi32>
  %tiles = check.file.read.npy path("tiles.npy") : tensor<{T}xi32>
  %orig = check.file.read.npy path("orig_{pos}.npy") : tensor<{10 + blocks}xi32>
  %arena = check.generate.fill value(16) : {A}
  %q = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{T}x3072xf32>
  %r = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{T}x384xf32>
  %rb = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<16384xf32>
  %sinks = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{T}x24xf32>
  %gate = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{T}x3072xf32>
  %out = check.generate.fill value(0.0) : tensor<{T}x3072xf32>
"""
    cases = [("ts2", f"  kernel.launch @k_ts2[%t]({args}, %out) : [index]({ty}, tensor<{T}x3072xf32>)\n")]
    for name, _ in variants:
        cases.append((name, f"""  %acc = check.generate.fill value(0.0) : tensor<{T * 49152}xf32>
  %ml = check.generate.fill value(0.0) : tensor<{T * 768}xf32>
  kernel.launch @k_{name}[%t]({args}, %acc, %ml) : [index]({ty}, tensor<{T * 49152}xf32>, tensor<{T * 768}xf32>)
  kernel.launch @k_comb[%t](%t, %tiles, %acc, %ml, %sinks, %gate, %out) : [index](index, tensor<{T}xi32>, tensor<{T * 49152}xf32>, tensor<{T * 768}xf32>, tensor<{T}x24xf32>, tensor<{T}x3072xf32>, tensor<{T}x3072xf32>)
"""))
    for name, body in cases:
        parts.append(f"check.case public @{name}_{pos} {{\n{setup}{body}  check.return\n}}\ncheck.benchmark<@{name}_{pos}> @{name}_{pos}_bm\n")
    import numpy as np
    np.save(os.path.join(out, f"info_{pos}.npy"), np.array([[0, pos + i, 1, 0] for i in range(T)], np.int32))
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)
    np.save(os.path.join(out, f"orig_{pos}.npy"), orig)
import numpy as np
tiles = np.zeros(T, np.int32); tiles[0] = T // 16; tiles[1:1 + T // 16] = np.arange(0, T, 16)
np.save(os.path.join(out, "tiles.npy"), tiles)
open(os.path.join(out, "bench.loom"), "w").write("\n".join(parts))
print(f"{out}/bench.loom: ts2 + {[n for n, _ in variants]} at 32768 and 131072")
