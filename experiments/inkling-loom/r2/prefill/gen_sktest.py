#!/usr/bin/env python3
"""Kernel test of the split-K prefill flash attention against the original g_ink_attention_flash2_ts2.

usage: gen_sktest.py OUTDIR LLM_LOOM [KERNEL]   (LLM_LOOM: a program with prefill.py applied; KERNEL: the
       split kernel under test, default g_ink_attention_flash2_ts2_sk; g_ink_attention_flash2_ts2_pf for prefetch)
Writes OUTDIR/test.loom (+ .npy inputs). Each case runs ts2 -> out_b and ts2_sk + combine -> out on the same
inputs (one row of uniform 16-token tiles at position P, kv8 arena with random codes and power-of-two scales,
permuted page map, sinks, gate, relative bias) and expects them equal within 2e-4. Case `neg` feeds the split
path a different query and must fail.
"""
import os
import sys

import numpy as np

out, src_path = sys.argv[1], sys.argv[2]
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "g_ink_attention_flash2_ts2_sk"
SRC = open(src_path).read()
HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
rng = np.random.default_rng(11)
os.makedirs(out, exist_ok=True)


def func(name):
    i = SRC.index(f"func.def inline @{name}(")
    return SRC[i:SRC.index("\n}\n", i) + 3]


def kern(name, new):
    i = SRC.index(f'kernel.def export("{name}") @{name}(')
    k = SRC.index("\n}\n", SRC.index("launch(", i)) + 3
    return SRC[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")


def case(tag, pos, t, sliding=0, neg=False):
    pool = -(-(pos + t) // 4096) * 4096
    blocks = pool // 16
    orig = np.zeros(10 + blocks, np.int32)
    orig[8] = 40
    orig[10:] = rng.permutation(blocks)
    info = np.array([[0, pos + i, 1, 0] for i in range(t)], np.int32)
    tiles = np.zeros(t, np.int32)
    tiles[0] = t // 16
    tiles[1:1 + t // 16] = np.arange(0, t, 16)
    a8 = np.zeros((2, pool, 1088), np.uint8)
    a8[:, :, :1024] = rng.integers(-127, 128, (2, pool, 1024)).astype(np.int8).view(np.uint8)
    a8[:, :, 1024:] = (2.0 ** rng.integers(-9, -5, (2, pool, 32))).astype(np.float16).view(np.uint8).reshape(2, pool, 64)
    for n, a in [("orig", orig), ("info", info), ("tiles", tiles), ("a8", a8.view(np.int8).reshape(-1))]:
        np.save(os.path.join(out, f"{tag}_{n}.npy"), a)
    A = f"tensor<{2 * pool * 1088}xi8>"
    ty_in = (f"[index](index, index, i32, index, index, tensor<{t}x3072xf32>, tensor<{t}x384xf32>, tensor<16384xf32>, "
             f"tensor<{t}x24xf32>, tensor<{t}x3072xf32>, tensor<{t}x4xi32>, tensor<{t}xi32>, tensor<{10 + blocks}xi32>, {A}")
    args = "(%t, %layer, %sliding, %context, %pool, {q}, %r, %rb, %sinks, %gate, %info, %tiles, %orig, %arena"
    return f"""check.case public @{tag} {{
  %seed = check.literal value(7) : i64
  %seed2 = check.literal value(8) : i64
  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value({sliding}) : i32
  %context = check.literal value({pool}) : index
  %pool = check.literal value({pool}) : index
  %info = check.file.read.npy path("{tag}_info.npy") : tensor<{t}x4xi32>
  %tiles = check.file.read.npy path("{tag}_tiles.npy") : tensor<{t}xi32>
  %orig = check.file.read.npy path("{tag}_orig.npy") : tensor<{10 + blocks}xi32>
  %arena = check.file.read.npy path("{tag}_a8.npy") : {A}
  %q = check.generate.random.uniform seed(%seed) range(-4.0 to 4.0) : tensor<{t}x3072xf32>
  %q2 = check.generate.random.uniform seed(%seed2) range(-4.0 to 4.0) : tensor<{t}x3072xf32>
  %r = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{t}x384xf32>
  %rb = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<16384xf32>
  %sinks = check.generate.random.uniform seed(%seed) range(-1.0 to 3.0) : tensor<{t}x24xf32>
  %gate = check.generate.random.uniform seed(%seed2) range(-2.0 to 2.0) : tensor<{t}x3072xf32>
  %out_b = check.generate.fill value(0.0) : tensor<{t}x3072xf32>
  %out = check.generate.fill value(0.0) : tensor<{t}x3072xf32>
  %acc = check.generate.fill value(0.0) : tensor<{t * 49152}xf32>
  %ml = check.generate.fill value(0.0) : tensor<{t * 768}xf32>
  kernel.launch @kb[%t]{args.format(q="%q")}, %out_b) : {ty_in}, tensor<{t}x3072xf32>)
  kernel.launch @ks[%t]{args.format(q="%q2" if neg else "%q")}, %acc, %ml) : {ty_in}, tensor<{t * 49152}xf32>, tensor<{t * 768}xf32>)
  kernel.launch @kc[%t](%t, %tiles, %acc, %ml, %sinks, %gate, %out) : [index](index, tensor<{t}xi32>, tensor<{t * 49152}xf32>, tensor<{t * 768}xf32>, tensor<{t}x24xf32>, tensor<{t}x3072xf32>, tensor<{t}x3072xf32>)
  check.expect.approximate actual(%out) expected(%out_b) atol(2.0e-4) rtol(2.0e-4) nan(different) : tensor<{t}x3072xf32>, tensor<{t}x3072xf32>
  check.return
}}
"""


parts = [HEAD] + [func(f) for f in ("g_kv8_scale", "g_kv8_offset", "g_kv8_load16", "g_tile_uniform", "g_tile_adjacent")]
parts += [kern("g_ink_attention_flash2_ts2", "kb"), kern(SPLIT, "ks"),
          kern("g_ink_attn_combine_flash", "kc")]
cases = [("p0", 0, 64), ("p64", 64, 48), ("p1000", 992, 64), ("p4096", 4096, 64), ("p20k", 20000, 64),
         ("p70k", 70016, 32)]
parts += [case(*c) for c in cases] + [case("neg", 4096, 64, neg=True)]
open(os.path.join(out, "test.loom"), "w").write("\n".join(parts))
print(f"{out}/test.loom: {len(cases)} cases + neg")
