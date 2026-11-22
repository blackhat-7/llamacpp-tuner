#!/usr/bin/env python3
"""Kernel benchmarks of the drafter-on attention costs of Inkling-2-Mini, one span, f16 KV.

usage: gen_dsp.py OUTDIR SOURCE.loom [MORE.loom ...]   (env DEPTHS=64000,195000; last definition of a name wins)
Cases per depth D:
  dsp_D    @g_dsp_attn_split_full + @g_dsp_attn_combine: the drafter's 8-token block over its D-position context
           (one layer, 2 planes x 2048 bytes per position).
  ver1_D   @g_ink_attn_split_span + @g_ink_attn_combine, a 1-token span at D (decode through the span path).
  ver2_D   the same with a 2-token verify span (anchor + 1 draft, draft_depth 1).
  dec_D    SPLIT (env, default g_ink_attn_split_v2) + combine, plain decode at D, for comparison.
"""
import os, sys
import numpy as np

HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
out = sys.argv[1]
SRC = "\n".join(open(p).read() for p in sys.argv[2:])
SPLIT = os.environ.get("SPLIT", "g_ink_attn_split_v2")
DSP = os.environ.get("DSP", "g_dsp_attn_split_full")
os.makedirs(out, exist_ok=True)


def grab(name, new):
    i = SRC.rindex(f'kernel.def export("{name}") @{name}(')
    j = SRC.index("launch(", i)
    k = SRC.index("\n}\n", j) + 3
    t = SRC[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")
    return t.replace("@g_kv_offset(", "@kvo(").replace("@g_gdn_span(", "@gspan(")


def fn(name, new):
    i = SRC.index(f"func.def inline @{name}(")
    k = SRC.index("\n}\n", i) + 3
    return SRC[i:k].replace(f"@{name}(", f"@{new}(")


def save(name, a):
    np.save(os.path.join(out, name + ".npy"), a)


parts = [HEAD, fn("g_kv_offset", "kvo"), fn("g_gdn_span", "gspan"), grab(DSP, "kdsp"), grab("g_dsp_attn_combine", "kdc"),
         grab("g_ink_attn_split_span", "kspan"), grab(SPLIT, "ks"), grab("g_ink_attn_combine", "kc")]
for depth in map(int, os.environ.get("DEPTHS", "64000,195000").split(",")):
    pool = -(-(depth + 16) // 4096) * 4096
    blocks = pool // 16
    tag = f"{depth // 1000}k"
    orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks); save(f"o{tag}", orig)
    AT, OT = f"tensor<{2 * pool * 1024}xf16>", f"tensor<{10 + blocks}xi32>"
    com = f"""  %seed = check.literal value(7) : i64
  %pool = check.literal value({pool}) : index
  %blocks = check.literal value({blocks}) : index
  %ctx = check.literal value({pool}) : index
  %one = check.literal value(1) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value(0) : i32
  %orig = check.file.read.npy path("o{tag}.npy") : {OT}
  %arena = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : {AT}
  %rb = check.generate.random.uniform seed(%seed) range(-0.1 to 0.1) : tensor<16384xf32>
"""
    # drafter: 8 block tokens of span 0 anchored at `depth`
    meta = np.zeros(4 + 9, np.int32); meta[0] = 1; meta[4:9] = [8, depth, 0, 0, 0]; save(f"md{tag}", meta)
    info = np.zeros((8, 4), np.int32); info[:, 1] = depth + np.arange(8); info[:, 2] = 1; save(f"id{tag}", info)
    parts.append(f"""check.case public @dsp_{tag} {{
{com}  %t = check.literal value(8) : index
  %meta = check.file.read.npy path("md{tag}.npy") : tensor<13xi32>
  %info = check.file.read.npy path("id{tag}.npy") : tensor<8x4xi32>
  %q = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<8x3072xf32>
  %k = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<8x1024xf32>
  %v = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<8x1024xf32>
  %pm = check.generate.fill value(0.0) : tensor<3072xf32>
  %pl = check.generate.fill value(0.0) : tensor<3072xf32>
  %po = check.generate.fill value(0.0) : tensor<393216xf32>
  %o = check.generate.fill value(0.0) : tensor<8x3072xf32>
  kernel.launch @kdsp[%t](%t, %one, %ctx, %pool, %meta, %info, %q, %k, %v, %orig, %arena, %pm, %pl, %po) : [index](index, index, index, index, tensor<13xi32>, tensor<8x4xi32>, tensor<8x3072xf32>, tensor<8x1024xf32>, tensor<8x1024xf32>, {OT}, {AT}, tensor<3072xf32>, tensor<3072xf32>, tensor<393216xf32>)
  kernel.launch @kdc[%t](%t, %pm, %pl, %po, %o) : [index](index, tensor<3072xf32>, tensor<3072xf32>, tensor<393216xf32>, tensor<8x3072xf32>)
  check.return
}}
check.benchmark<@dsp_{tag}> @dsp_{tag}_bm
""")
    for n in (1, 2):
        meta = np.zeros(4 + 9, np.int32); meta[0] = 1; meta[4:9] = [n, depth - n + 1, 0, 0, 0]; save(f"mv{n}{tag}", meta)
        info = np.zeros((n, 4), np.int32); info[:, 1] = depth - n + 1 + np.arange(n); info[:, 2] = 1; save(f"iv{n}{tag}", info)
        parts.append(f"""check.case public @ver{n}_{tag} {{
{com}  %t = check.literal value({n}) : index
  %meta = check.file.read.npy path("mv{n}{tag}.npy") : tensor<13xi32>
  %info = check.file.read.npy path("iv{n}{tag}.npy") : tensor<{n}x4xi32>
  %q = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x3072xf32>
  %r = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x384xf32>
  %pm = check.generate.fill value(0.0) : tensor<{n * 1536}xf32>
  %pl = check.generate.fill value(0.0) : tensor<{n * 1536}xf32>
  %po = check.generate.fill value(0.0) : tensor<{n * 196608}xf32>
  %sinks = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x24xf32>
  %gate = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x3072xf32>
  %o = check.generate.fill value(0) : tensor<{n * 3840}xi8>
  kernel.launch @kspan[%one](%t, %one, %blocks, %pool, %layer, %meta, %orig, %q, %r, %rb, %arena, %pm, %pl, %po) : [index](index, index, index, index, index, tensor<13xi32>, {OT}, tensor<{n}x3072xf32>, tensor<{n}x384xf32>, tensor<16384xf32>, {AT}, tensor<{n * 1536}xf32>, tensor<{n * 1536}xf32>, tensor<{n * 196608}xf32>)
  kernel.launch @kc[%t](%t, %sliding, %info, %pm, %pl, %po, %sinks, %gate, %o) : [index](index, i32, tensor<{n}x4xi32>, tensor<{n * 1536}xf32>, tensor<{n * 1536}xf32>, tensor<{n * 196608}xf32>, tensor<{n}x24xf32>, tensor<{n}x3072xf32>, tensor<{n * 3840}xi8>)
  check.return
}}
check.benchmark<@ver{n}_{tag}> @ver{n}_{tag}_bm
""")
    info = np.array([[0, depth, 1, 0]], np.int32); save(f"i1{tag}", info)
    parts.append(f"""check.case public @dec_{tag} {{
{com}  %t = check.literal value(1) : index
  %info = check.file.read.npy path("i1{tag}.npy") : tensor<1x4xi32>
  %q = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<1x3072xf32>
  %r = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<1x384xf32>
  %pm = check.generate.fill value(0.0) : tensor<1536xf32>
  %pl = check.generate.fill value(0.0) : tensor<1536xf32>
  %po = check.generate.fill value(0.0) : tensor<196608xf32>
  %sinks = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<1x24xf32>
  %gate = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<1x3072xf32>
  %o = check.generate.fill value(0) : tensor<3840xi8>
  kernel.launch @ks[%t](%t, %blocks, %pool, %layer, %sliding, %info, %orig, %q, %r, %rb, %arena, %pm, %pl, %po) : [index](index, index, index, index, i32, tensor<1x4xi32>, {OT}, tensor<1x3072xf32>, tensor<1x384xf32>, tensor<16384xf32>, {AT}, tensor<1536xf32>, tensor<1536xf32>, tensor<196608xf32>)
  kernel.launch @kc[%t](%t, %sliding, %info, %pm, %pl, %po, %sinks, %gate, %o) : [index](index, i32, tensor<1x4xi32>, tensor<1536xf32>, tensor<1536xf32>, tensor<196608xf32>, tensor<1x24xf32>, tensor<1x3072xf32>, tensor<3840xi8>)
  check.return
}}
check.benchmark<@dec_{tag}> @dec_{tag}_bm
""")
open(os.path.join(out, "bench.loom"), "w").write("\n".join(parts))
