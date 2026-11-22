#!/usr/bin/env python3
"""Global-layer attention (@g_ink_attn_split_v2 on the kv8 cache) with GG keys per split + the new combine,
against the shipped v2 (8 keys per split) + the shipped combine.

usage: gen_v2bench.py OUTDIR LLM.loom COMB.loom GG
OUTDIR/test_e.loom: both pipelines write their q8 rows to npy (check_e.py compares).
OUTDIR/bench.loom: spans 1, 2, 4 at 2.5K, 64K, 195K for both.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/illusion/Documents/projects/llamacpp-tuner/tmp/loom/attn/deliver")
import attn  # noqa: E402

out, llm, comb, GG = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
SRC = open(llm).read()
os.makedirs(out, exist_ok=True)
HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
rng = np.random.default_rng(3)


def grab(text, name, new):
    i = text.rindex(f'kernel.def export("{name}") @{name}(')
    k = text.index("\n}\n", text.index("launch(", i)) + 3
    return text[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")


def fn(name):
    i = SRC.index(f"func.def inline @{name}(")
    return SRC[i : SRC.index("\n}\n", i) + 3]


a, b = attn.kernel_span(SRC, "g_ink_attn_split_v2")
attn.GG = GG
v2new = attn.transform_v2(SRC[a:b])
common = [HEAD, fn("g_kv8_offset"), grab(SRC, "g_ink_attn_split_v2", "kold"), grab(SRC, "g_ink_attn_combine", "kcold"),
          grab(v2new, "g_ink_attn_split_v2", "knew"), grab(open(comb).read(), "g_ink_attn_combine", "kcnew")]


def case(tag, n, last, which, write, shape=0):
    """n live tokens ending at position last, in a launch of `shape` token rows (padding rows not live)."""
    pool = -(-(last + 16) // 4096) * 4096
    blocks = pool // 16
    n, live = max(n, shape), n
    info = np.zeros((n, 4), np.int32); info[:live, 1] = last - live + 1 + np.arange(live); info[:live, 2] = 1
    np.save(os.path.join(out, f"i_{tag}.npy"), info)
    o = np.zeros(10 + blocks, np.int32); o[8] = 40; o[10:] = rng.permutation(blocks)
    np.save(os.path.join(out, f"o_{tag}.npy"), o)
    if write:
        codes = rng.integers(-127, 128, (2, pool, 1024)).astype(np.int8).view(np.uint8)
        scales = (2.0 ** rng.integers(-9, -5, (2, pool, 32))).astype(np.float16).view(np.uint8).reshape(2, pool, 64)
        np.save(os.path.join(out, f"a_{tag}.npy"), np.concatenate([codes, scales], axis=2).view(np.int8).reshape(-1))
        arena = f'  %arena = check.file.read.npy path("a_{tag}.npy") : tensor<{2 * pool * 1088}xi8>\n'
    else:
        arena = f"  %arena = check.generate.fill value(16) : tensor<{2 * pool * 1088}xi8>\n"
    P, O, Q = n * 1536, n * 196608, n * 3840
    ty = (f"index, index, index, index, i32, tensor<{n}x4xi32>, tensor<{10 + blocks}xi32>, tensor<{n}x3072xf32>, "
          f"tensor<{n}x384xf32>, tensor<16384xf32>, tensor<{2 * pool * 1088}xi8>, tensor<{P}xf32>, tensor<{P}xf32>, tensor<{O}xf32>")
    tc = f"index, i32, tensor<{n}x4xi32>, tensor<{P}xf32>, tensor<{P}xf32>, tensor<{O}xf32>, tensor<{n}x24xf32>, tensor<{n}x3072xf32>, tensor<{Q}xi8>"
    body = f"""  %seed = check.literal value(5) : i64
  %t = check.literal value({n}) : index
  %pool = check.literal value({pool}) : index
  %blocks = check.literal value({blocks}) : index
  %layer = check.literal value(0) : index
  %sliding = check.literal value(0) : i32
  %info = check.file.read.npy path("i_{tag}.npy") : tensor<{n}x4xi32>
  %orig = check.file.read.npy path("o_{tag}.npy") : tensor<{10 + blocks}xi32>
  %q = check.generate.random.uniform seed(%seed) range(-4.0 to 4.0) : tensor<{n}x3072xf32>
  %r = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x384xf32>
  %rb = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<16384xf32>
{arena}  %sinks = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x24xf32>
  %gate = check.generate.random.uniform seed(%seed) range(-3.0 to 3.0) : tensor<{n}x3072xf32>
"""
    for w in which:
        k, kc = ("kold", "kcold") if w == "old" else ("knew", "kcnew")
        body += f"""  %pm{w} = check.generate.fill value(-3.40282347e+38) : tensor<{P}xf32>
  %pl{w} = check.generate.fill value(0.0) : tensor<{P}xf32>
  %po{w} = check.generate.fill value(0.0) : tensor<{O}xf32>
  %o{w} = check.generate.fill value(0) : tensor<{Q}xi8>
  kernel.launch @{k}[%t](%t, %blocks, %pool, %layer, %sliding, %info, %orig, %q, %r, %rb, %arena, %pm{w}, %pl{w}, %po{w}) : [index]({ty})
  kernel.launch @{kc}[%t](%t, %sliding, %info, %pm{w}, %pl{w}, %po{w}, %sinks, %gate, %o{w}) : [index]({tc})
"""
        if write:
            body += f'  check.file.write.npy value(%o{w}) path("{tag}_{w}.npy") mode(always) : tensor<{Q}xi8>\n'
    return f"check.case public @{tag} {{\n{body}  check.return\n}}\n"


open(os.path.join(out, "test_e.loom"), "w").write("\n".join(
    common + [case(f"e{n}_{last}_{sh}", n, last, ("old", "new"), True, sh)
    for n, last, sh in ((1, 5, 1), (2, 300, 2), (4, 2600, 4), (2, 20011, 2), (3, 9000, 4), (5, 3000, 8), (7, 9000, 8))]))
open(os.path.join(out, "bench.loom"), "w").write("\n".join(common + [
    case(f"{w}{n}of{sh}_{last // 1000}k", n, last, (w,), False, sh) + f"check.benchmark<@{w}{n}of{sh}_{last // 1000}k> @{w}{n}of{sh}_{last // 1000}k_bm\n"
    for last in (2500, 64000, 195000) for n, sh in ((1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8)) for w in ("old", "new")]))
