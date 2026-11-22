#!/usr/bin/env python3
"""Kernel benchmark and test of one global-attention decode step (split + combine) of Inkling-2-Mini.

usage: gen.py OUTDIR SPLIT_KERNEL SOURCE.loom [MORE.loom ...]
Writes OUTDIR/bench.loom: cases d64k / d195k (one decode token at depth 64000 / 195000, synthetic f16 KV of one
layer, identity page map; the original @g_ink_attn_combine follows the split kernel), and OUTDIR/test.loom:
SPLIT_KERNEL's partials (pm, pl, po) against @g_ink_attn_split's on the same inputs, with sharp scores, a
permuted page map and depths that cover partial pages, the relative-bias window and SWA.
"""
import os, sys
import numpy as np

HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
out, SPLIT = sys.argv[1], sys.argv[2]
SRC = "\n".join(open(p).read() for p in sys.argv[3:])


def grab(name, new):
    i = SRC.rindex(f'kernel.def export("{name}") @{name}(')  # last source wins
    j = SRC.index("launch(", i)
    k = SRC.index("\n}\n", j) + 3
    t = SRC[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")
    return t.replace("@g_kv_offset(", f"@{new}_kvo(")


def kvo(new, fn="g_kv_offset"):
    i = SRC.index(f"func.def inline @{fn}(")
    k = SRC.index("\n}\n", i) + 3
    return SRC[i:k].replace(f"@{fn}(", f"@{new}_kvo(")


KV8 = os.environ.get("FMT", "f16") == "kv8"
PB = 1088 if KV8 else 2048
AT = (lambda pool: f"tensor<{2 * pool * PB}xi8>") if KV8 else (lambda pool: f"tensor<{2 * pool * 1024}xf16>")


os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(5)


def inputs(tag, pos, pool, perm=False):
    blocks = pool // 16
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40
    orig[10:] = rng.permutation(blocks) if perm else np.arange(blocks)
    np.save(os.path.join(out, f"{tag}_orig.npy"), orig)
    np.save(os.path.join(out, f"{tag}_info.npy"), np.array([[0, pos, 1, 0]], np.int32))


def common(tag, pool, sliding=0, qr=1.0, arena_npy=False):
    blocks = pool // 16
    return f"""  %seed = check.literal value(7) : i64
  %t = check.literal value(1) : index
  %blocks = check.literal value({blocks}) : index
  %pool = check.literal value({pool}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value({sliding}) : i32
  %info = check.file.read.npy path("{tag}_info.npy") : tensor<1x4xi32>
  %orig = check.file.read.npy path("{tag}_orig.npy") : tensor<{10 + blocks}xi32>
  %q = check.generate.random.uniform seed(%seed) range(-{qr} to {qr}) : tensor<1x3072xf32>
  %r = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<1x384xf32>
  %rb = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<16384xf32>
""" + (f'  %arena = check.file.read.npy path("{tag}_a8.npy") : {AT(pool)}\n  %arena16 = check.file.read.npy path("{tag}_a16.npy") : tensor<{2 * pool * 1024}xf16>\n' if arena_npy
       else f"  %arena = check.generate.fill value(16) : {AT(pool)}\n" if KV8
       else f"  %arena = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : {AT(pool)}\n")


def split(name, tag, pool, sfx, arena="%arena", at=None):
    at = at or AT(pool)
    blocks = pool // 16
    return f"""  %pm{sfx} = check.generate.fill value(0.0) : tensor<1536xf32>
  %pl{sfx} = check.generate.fill value(0.0) : tensor<1536xf32>
  %po{sfx} = check.generate.fill value(0.0) : tensor<196608xf32>
  kernel.launch @{name}[%t](%t, %blocks, %pool, %layer, %sliding, %info, %orig, %q, %r, %rb, {arena}, %pm{sfx}, %pl{sfx}, %po{sfx}) : [index](index, index, index, index, i32, tensor<1x4xi32>, tensor<{10 + blocks}xi32>, tensor<1x3072xf32>, tensor<1x384xf32>, tensor<16384xf32>, {at}, tensor<1536xf32>, tensor<1536xf32>, tensor<196608xf32>)
"""


COMBINE = """  %sinks = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<1x24xf32>
  %gate = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<1x3072xf32>
  %o = check.generate.fill value(0) : tensor<3840xi8>
  kernel.launch @kc[%t](%t, %sliding, %info, %pm, %pl, %po, %sinks, %gate, %o) : [index](index, i32, tensor<1x4xi32>, tensor<1536xf32>, tensor<1536xf32>, tensor<196608xf32>, tensor<1x24xf32>, tensor<1x3072xf32>, tensor<3840xi8>)
"""

bench = [HEAD, (kvo("ks", "g_kv8_offset") if KV8 else kvo("ks")), grab(SPLIT, "ks").replace("@g_kv8_offset(", "@ks_kvo("), grab("g_ink_attn_combine", "kc")]
DEPTHS = [(f"d{p}", p, -(-(p + 1) // 4096) * 4096) for p in map(int, os.environ.get("DEPTHS", "64000,195000").split(","))]
for tag, pos, pool in DEPTHS:
    inputs(tag, pos, pool)
    bench.append(f"check.case public @{tag} {{\n" + common(tag, pool) + split("ks", tag, pool, "") + COMBINE
                 + f"  check.return\n}}\ncheck.benchmark<@{tag}> @{tag}_bm\n")
open(os.path.join(out, "bench.loom"), "w").write("\n".join(bench))

test = [HEAD, kvo("kb"), grab("g_ink_attn_split", "kb"), (kvo("ks", "g_kv8_offset") if KV8 else kvo("ks")), grab(SPLIT, "ks").replace("@g_kv8_offset(", "@ks_kvo(")]
T = [("t1", 0, 16, 0), ("t2", 37, 64, 0), ("t3", 700, 1024, 0), ("t4", 3000, 4096, 0), ("t5", 3000, 4096, 1),
     ("t6", 20011, 20480, 0)] + ([] if KV8 else [("t7", 64000, 65536, 0)])


def arenas(tag, pool):
    """KV8 arena with power-of-two scales (code x scale is exact in f16) and the same values as an f16 arena."""
    codes = rng.integers(-127, 128, (2, pool, 8, 128)).astype(np.int8)
    scales = (2.0 ** rng.integers(-9, -5, (2, pool, 8, 4))).astype(np.float16)
    a8 = np.zeros((2, pool, PB), np.uint8)
    a8[:, :, :1024] = codes.view(np.uint8).reshape(2, pool, 1024)
    a8[:, :, 1024:] = scales.view(np.uint8).reshape(2, pool, 64)
    vals = codes.astype(np.float32) * np.repeat(scales.astype(np.float32), 32, axis=3)
    np.save(os.path.join(out, f"{tag}_a8.npy"), a8.view(np.int8).reshape(-1))
    np.save(os.path.join(out, f"{tag}_a16.npy"), vals.astype(np.float16).reshape(-1))
for tag, pos, pool, sl in T:
    inputs(tag, pos, pool, perm=True)
    if KV8:
        arenas(tag, pool)
        body = (common(tag, pool, sl, qr=4.0, arena_npy=True) + split("kb", tag, pool, "_b", "%arena16", f"tensor<{2 * pool * 1024}xf16>")
                + split("ks", tag, pool, ""))
    else:
        body = common(tag, pool, sl, qr=4.0) + split("kb", tag, pool, "_b") + split("ks", tag, pool, "")
    for nm, n in [("pm", 1536), ("pl", 1536), ("po", 196608)]:
        body += f"  check.expect.approximate actual(%{nm}) expected(%{nm}_b) atol(1.0e-4) rtol(1.0e-4) nan(different) : tensor<{n}xf32>, tensor<{n}xf32>\n"
    test.append(f"check.case public @{tag} {{\n{body}  check.return\n}}\n")
open(os.path.join(out, "test.loom"), "w").write("\n".join(test))
