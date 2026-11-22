#!/usr/bin/env python3
"""Test and benchmark @g_ink_attn_split_span_v2 (verify spans) against @g_ink_attn_split_span.

usage: [FMT=f16|kv8] gen_spanbench.py OUTDIR KERNEL.loom SOURCE.loom [MORE.loom ...]  (last definition of a name wins)
OUTDIR/test.loom: span_v2's pm/pl/po against the original f16 span kernel (on the same values: kv8 data uses
power-of-two scales, so code x scale is exact in f16) for span lengths 1-8 at several depths, permuted pages.
OUTDIR/bench.loom: span_v2 + combine for spans of 1, 2 and 4 tokens at DEPTHS (default 64000,195000).
"""
import os, sys
import numpy as np

HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
out = sys.argv[1]
SRC = "\n".join(open(p).read() for p in sys.argv[3:]) + "\n" + open(sys.argv[2]).read()
KV8 = os.environ.get("FMT", "f16") == "kv8"
PB = 1088 if KV8 else 2048
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(11)


def grab(name, new, kvo):
    i = SRC.rindex(f'kernel.def export("{name}") @{name}(')
    k = SRC.index("\n}\n", SRC.index("launch(", i)) + 3
    t = SRC[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")
    return t.replace("@g_kv_offset(", f"@{kvo}(").replace("@g_kv8_offset(", f"@{kvo}(").replace("@g_gdn_span(", "@gspan(")


def fn(name, new):
    i = SRC.index(f"func.def inline @{name}(")
    k = SRC.index("\n}\n", i) + 3
    return SRC[i:k].replace(f"@{name}(", f"@{new}(")


def save(name, a):
    np.save(os.path.join(out, name + ".npy"), a)


def at(pool, f16=False):
    return f"tensor<{2 * pool * 1024}xf16>" if f16 or not KV8 else f"tensor<{2 * pool * PB}xi8>"


def launch(k, n, pool, blocks, arena, atype, sfx):
    OT = f"tensor<{10 + blocks}xi32>"
    return f"""  %pm{sfx} = check.generate.fill value(0.0) : tensor<{n * 1536}xf32>
  %pl{sfx} = check.generate.fill value(0.0) : tensor<{n * 1536}xf32>
  %po{sfx} = check.generate.fill value(0.0) : tensor<{n * 196608}xf32>
  kernel.launch @{k}[%one](%t, %one, %blocks, %pool, %layer, %meta, %orig, %q, %r, %rb, {arena}, %pm{sfx}, %pl{sfx}, %po{sfx}) : [index](index, index, index, index, index, tensor<13xi32>, {OT}, tensor<{n}x3072xf32>, tensor<{n}x384xf32>, tensor<16384xf32>, {atype}, tensor<{n * 1536}xf32>, tensor<{n * 1536}xf32>, tensor<{n * 196608}xf32>)
"""


def head(tag, n, pos0, pool, qr):
    blocks = pool // 16
    meta = np.zeros(13, np.int32); meta[0] = 1; meta[4:9] = [n, pos0, 0, 0, 0]; save(f"m_{tag}", meta)
    info = np.zeros((n, 4), np.int32); info[:, 1] = pos0 + np.arange(n); info[:, 2] = 1; save(f"i_{tag}", info)
    return blocks, f"""  %seed = check.literal value(5) : i64
  %t = check.literal value({n}) : index
  %one = check.literal value(1) : index
  %pool = check.literal value({pool}) : index
  %blocks = check.literal value({blocks}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value(0) : i32
  %meta = check.file.read.npy path("m_{tag}.npy") : tensor<13xi32>
  %info = check.file.read.npy path("i_{tag}.npy") : tensor<{n}x4xi32>
  %orig = check.file.read.npy path("o_{tag}.npy") : tensor<{10 + blocks}xi32>
  %q = check.generate.random.uniform seed(%seed) range(-{qr} to {qr}) : tensor<{n}x3072xf32>
  %r = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x384xf32>
  %rb = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<16384xf32>
"""


def orig(tag, pool, perm):
    blocks = pool // 16
    o = np.zeros(10 + blocks, np.int32); o[8] = 40; o[10:] = rng.permutation(blocks) if perm else np.arange(blocks)
    save(f"o_{tag}", o)


common = [HEAD, fn("g_gdn_span", "gspan")]
i = SRC.index("func.def inline @g_kv_offset("); k = SRC.index("\n}\n", i) + 3
common.append(SRC[i:k].replace("@g_kv_offset(", "@kvo16("))
if KV8:
    common.append(fn("g_kv8_offset", "kvo8"))
NEWKVO = "kvo8" if KV8 else "kvo16"

# tests
test = common + [grab("g_ink_attn_split_span", "kold", "kvo16"), grab("g_ink_attn_split_span_v2", "knew", NEWKVO)]
for n, last, pool in [(1, 37, 64), (2, 37, 64), (3, 700, 1024), (2, 3000, 4096), (5, 3000, 4096), (8, 3001, 4096), (2, 20011, 20480), (4, 20011, 20480)]:
    tag = f"t{n}_{last}"
    orig(tag, pool, True)
    blocks, body = head(tag, n, last - n + 1, pool, 4.0)
    if KV8:
        codes = rng.integers(-127, 128, (2, pool, 8, 128)).astype(np.int8)
        scales = (2.0 ** rng.integers(-9, -5, (2, pool, 8, 4))).astype(np.float16)
        a8 = np.zeros((2, pool, PB), np.uint8)
        a8[:, :, :1024] = codes.view(np.uint8).reshape(2, pool, 1024)
        a8[:, :, 1024:] = scales.view(np.uint8).reshape(2, pool, 64)
        vals = codes.astype(np.float32) * np.repeat(scales.astype(np.float32), 32, axis=3)
        save(f"a8_{tag}", a8.view(np.int8).reshape(-1)); save(f"a16_{tag}", vals.astype(np.float16).reshape(-1))
        body += f'  %arena = check.file.read.npy path("a8_{tag}.npy") : {at(pool)}\n'
        body += f'  %arena16 = check.file.read.npy path("a16_{tag}.npy") : {at(pool, True)}\n'
        a16 = "%arena16"
    else:
        body += f"  %arena = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : {at(pool)}\n"
        a16 = "%arena"
    body += launch("kold", n, pool, blocks, a16, at(pool, True), "_b") + launch("knew", n, pool, blocks, "%arena", at(pool), "")
    for nm, sz in [("pm", n * 1536), ("pl", n * 1536), ("po", n * 196608)]:
        body += f"  check.expect.approximate actual(%{nm}) expected(%{nm}_b) atol(1.0e-4) rtol(1.0e-4) nan(different) : tensor<{sz}xf32>, tensor<{sz}xf32>\n"
    test.append(f"check.case public @{tag} {{\n{body}  check.return\n}}\n")
open(os.path.join(out, "test.loom"), "w").write("\n".join(test))

# benchmarks
bench = common + [grab("g_ink_attn_split_span_v2", "knew", NEWKVO), grab("g_ink_attn_combine", "kc", "kvo16")]
for depth in map(int, os.environ.get("DEPTHS", "64000,195000").split(",")):
    pool = -(-(depth + 16) // 4096) * 4096
    for n in (1, 2, 4):
        tag = f"b{n}_{depth // 1000}k"
        orig(tag, pool, False)
        blocks, body = head(tag, n, depth - n + 1, pool, 1.0)
        fill = "check.generate.fill value(16)" if KV8 else "check.generate.random.uniform seed(%seed) range(-1.0 to 1.0)"
        body += f"  %arena = {fill} : {at(pool)}\n" + launch("knew", n, pool, blocks, "%arena", at(pool), "")
        body += f"""  %sinks = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x24xf32>
  %gate = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x3072xf32>
  %o = check.generate.fill value(0) : tensor<{n * 3840}xi8>
  kernel.launch @kc[%t](%t, %sliding, %info, %pm, %pl, %po, %sinks, %gate, %o) : [index](index, i32, tensor<{n}x4xi32>, tensor<{n * 1536}xf32>, tensor<{n * 1536}xf32>, tensor<{n * 196608}xf32>, tensor<{n}x24xf32>, tensor<{n}x3072xf32>, tensor<{n * 3840}xi8>)
"""
        bench.append(f"check.case public @{tag} {{\n{body}  check.return\n}}\ncheck.benchmark<@{tag}> @{tag}_bm\n")
open(os.path.join(out, "bench.loom"), "w").write("\n".join(bench))
