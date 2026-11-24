#!/usr/bin/env python3
"""GPU check of g_head_select's T > 0 path after engines/loom/tools/spec/gumbel.py: the draw's token frequencies
against softmax over the kept candidates, the fast top-k against the old one draw for draw, negative controls.

usage: gen_select_test.py OUTDIR NEW_LLM.loom OLD_LLM.loom   then: cd OUTDIR && loom-test test.loom --device=amdgpu
       --file-output-root=. ; check_select.py OUTDIR
Rows (one fixed logits row per case, tiled over N spans; partials = per-chunk max and first argmax, as the heads):
  a  120 live ids scattered over the vocabulary, two tied at the top
  b  ties: 430 ids in 4 chunks (one is the last, partial chunk) and 30 scattered, values on a 0.5 grid
  c  dense: every id normal(0, 2), a few peaks
  d  1,000 consecutive ids at exactly the maximum (the top-64 is a pure id-order tie)
Configs: c1 T 1 top_p 1 top_k 0 (64) | c2 T 0.7 top_p 0.9 top_k 40 | c3 T 1.5 top_p 1 top_k 10.
Kernels: new (the package's), slow (old top-k + the new draw: must match new draw for draw), neg (new with the
noise -log(u), an exponential: must fail the chi-square), negtk (new scanning half the chosen chunks: must
disagree with slow on row c).
bench.loom: g_head_select alone, old and new, T 0 and T 1, 1 and 4 spans (loom-benchmark).
"""
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [here, os.path.join(here, "../../../engines/loom/tools/spec")]
import gumbel  # noqa: E402

HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
N, V, VP, CH = 2048, 200058, 200064, 1563
FAST_LAUNCHES, SLOW_LAUNCHES = 12, 2
out, new_path, old_path = sys.argv[1:4]
os.makedirs(out, exist_ok=True)


def grab(src, name, new):
    i = src.index(f'kernel.def export("{name}") @{name}(')
    k = src.index("\n}\n", src.index("launch(", i)) + 3
    return src[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")


old_src = open(old_path).read()
new_k = grab(open(new_path).read(), "g_head_select", "sel_new")
old_k = grab(old_src, "g_head_select", "sel_old")
slow_k = gumbel.edit_kernel(old_src, "g_head_select", gumbel.DRAW_OLD, gumbel.DRAW_NEW)
slow_k = grab(slow_k, "g_head_select", "sel_slow")
broken = "%score = scalar.subf %ds, %log_e : f32"
assert new_k.count(broken) == 1
neg_k = new_k.replace("@sel_new(", "@sel_neg(").replace(broken, "%score = scalar.subf %ds, %log_u : f32")
half = "%quads = index.mul %k, %c32_index : index"
assert new_k.count(half) == 1
negtk_k = new_k.replace("@sel_new(", "@sel_negtk(").replace(
    half, "%c16_index = index.constant 16 : index\n        %quads = index.mul %k, %c16_index : index")

TILE = """kernel.def target(@w32) @tile_rows(%rows: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%rows_arg: index, %src: buffer, %dst: buffer, %psrc: buffer, %pdst: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 8192)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c256 = index.constant 256 : index
  %n = index.constant 200064 : index
  %np = index.constant 3126 : index
  %sv = buffer.view %src[%zero] : buffer -> view<200064xf32>
  %dv = buffer.view %dst[%zero] : buffer -> view<[%rows]x200064xf32>
  scf.for %i = [%lane to %n step %c256] {
    %bi = index.assume %i [range(%i, 0, 200063)] : index
    %v = view.load %sv[%bi] : view<200064xf32> -> f32
    view.store %v, %dv[%row, %bi] : f32, view<[%rows]x200064xf32>
  }
  %psv = buffer.view %psrc[%zero] : buffer -> view<3126xi32>
  %pdv = buffer.view %pdst[%zero] : buffer -> view<[%rows]x3126xi32>
  scf.for %i = [%lane to %np step %c256] {
    %bi = index.assume %i [range(%i, 0, 3125)] : index
    %v = view.load %psv[%bi] : view<3126xi32> -> i32
    view.store %v, %pdv[%row, %bi] : i32, view<[%rows]x3126xi32>
  }
  kernel.return
}
"""

rng = np.random.default_rng(11)


def make_row(kind):
    row = np.full(VP, -60.0, np.float32)
    if kind == "a":
        ids = rng.choice(V, 120, replace=False)
        row[ids] = np.sort(rng.uniform(-6.0, 3.0, 120))[::-1].astype(np.float32) + rng.normal(0, 0.02, 120).astype(np.float32)
        row[ids[:2]] = [4.0, 4.0]
    elif kind == "b":
        for chunk, count in ((10, 128), (11, 100), (500, 80), (CH - 1, 122)):
            ids = chunk * 128 + rng.choice(min(128, V - chunk * 128), count, replace=False)
            row[ids] = np.round(rng.uniform(-4.0, 3.0, count) * 2) / 2
        ids = rng.choice(V, 30, replace=False)
        row[ids] = np.round(rng.uniform(-2.0, 3.0, 30) * 2) / 2
    elif kind == "c":
        row[:V] = rng.normal(0, 2, V).astype(np.float32)
        row[rng.choice(V, 5, replace=False)] = [12.0, 11.5, 11.0, 10.0, 9.0]
    else:
        row[50000:51000] = 5.0
        row[rng.choice(V, 200, replace=False)] = rng.uniform(-3.0, 4.9, 200).astype(np.float32)
    row[V:] = -3.0e38
    parts = np.zeros((CH, 2), np.int32)
    for c in range(CH):
        seg = row[c * 128:min((c + 1) * 128, V)]
        j = int(np.argmax(seg))  # first occurrence of the maximum, as the heads' strict greater-than
        parts[c] = [np.float32(seg[j]).view(np.int32), c * 128 + j]
    return row, parts


CONFIGS = {"c1": (1.0, 1.0, 0), "c2": (0.7, 0.9, 40), "c3": (1.5, 1.0, 10)}


def meta(tag, launch, temperature, top_p, top_k, salt):
    m = np.zeros(4 + N * 9, np.int32)
    m[0], m[1] = N, 200001
    p = m[4:].reshape(N, 9)
    span = np.arange(N)
    p[:, 0] = 1 + span % 4                       # length (verify rows have 1..4)
    p[:, 1] = 1000 + 37 * span + launch          # start; position = start + length
    p[:, 2] = np.float32(temperature).view(np.int32)
    p[:, 5] = np.float32(top_p).view(np.int32)
    p[:, 6] = 1                                  # selected
    p[:, 7] = top_k
    p[:, 8] = 7919 * launch + 13 * span + salt   # seed
    np.save(f"{out}/meta_{tag}_{launch}.npy", m)


def case(row, config, kernels):
    """kernels: [(kernel name, file tag, launches)]; all share the metas of (row, config)."""
    t, tp, tk = CONFIGS[config]
    tag = f"{row}{config}"
    lines = [f"""  %n = check.literal value({N}) : index
  %row = check.file.read.npy path("row_{row}.npy") : tensor<{VP}xf32>
  %prow = check.file.read.npy path("part_{row}.npy") : tensor<{CH * 2}xi32>
  %logits = check.generate.fill value(0.0) : tensor<{N}x{VP}xf32>
  %parts = check.generate.fill value(0) : tensor<{N * CH * 2}xi32>
  kernel.launch @tile_rows[%n](%n, %row, %logits, %prow, %parts) : [index](index, tensor<{VP}xf32>, tensor<{N}x{VP}xf32>, tensor<{CH * 2}xi32>, tensor<{N * CH * 2}xi32>)
"""]
    for i in range(max(launches for _, _, launches in kernels)):
        meta(tag, i, t, tp, tk, "abcd".index(row) * 10 + int(config[1]))
        lines.append(f'  %meta{i} = check.file.read.npy path("meta_{tag}_{i}.npy") : tensor<{4 + N * 9}xi32>\n')
    for kernel, ktag, launches in kernels:
        for i in range(launches):
            lines.append(f"""  %tok_{ktag}{i} = check.generate.fill value(-1) : tensor<{N}xi32>
  kernel.launch @{kernel}[%n](%n, %meta{i}, %parts, %logits, %tok_{ktag}{i}) : [index](index, tensor<{4 + N * 9}xi32>, tensor<{N * CH * 2}xi32>, tensor<{N}x{VP}xf32>, tensor<{N}xi32>)
  check.file.write.npy value(%tok_{ktag}{i}) path("tok_{ktag}_{tag}_{i}.npy") mode(always) : tensor<{N}xi32>
""")
    return f"check.case public @{tag} {{\n{''.join(lines)}  check.return\n}}\n"


cases = []
for row in "abcd":
    r, p = make_row(row)
    np.save(f"{out}/row_{row}.npy", r)
    np.save(f"{out}/part_{row}.npy", p.reshape(-1))
    for config in CONFIGS:
        kernels = [("sel_new", "new", FAST_LAUNCHES), ("sel_slow", "slow", SLOW_LAUNCHES)]
        if (row, config) == ("a", "c1"):
            kernels.append(("sel_neg", "neg", FAST_LAUNCHES))
        if (row, config) == ("c", "c1"):
            kernels.append(("sel_negtk", "negtk", SLOW_LAUNCHES))
        cases.append(case(row, config, kernels))
open(f"{out}/test.loom", "w").write("\n".join([HEAD, TILE, new_k, slow_k, neg_k, negtk_k] + cases))


def bench(name, kernel, spans, temperature):
    m = np.zeros(4 + spans * 9, np.int32)
    m[0], m[1] = spans, 200001
    p = m[4:].reshape(spans, 9)
    p[:, 0], p[:, 1], p[:, 6], p[:, 8] = 1, 5000, 1, 3
    p[:, 2] = np.float32(temperature).view(np.int32)
    p[:, 5] = np.float32(1.0).view(np.int32)
    np.save(f"{out}/bmeta_{name}.npy", m)
    return f"""check.case public @{name} {{
  %n = check.literal value({spans}) : index
  %meta = check.file.read.npy path("bmeta_{name}.npy") : tensor<{4 + spans * 9}xi32>
  %row = check.file.read.npy path("row_c.npy") : tensor<{VP}xf32>
  %prow = check.file.read.npy path("part_c.npy") : tensor<{CH * 2}xi32>
  %logits = check.generate.fill value(0.0) : tensor<{spans}x{VP}xf32>
  %parts = check.generate.fill value(0) : tensor<{spans * CH * 2}xi32>
  kernel.launch @tile_rows[%n](%n, %row, %logits, %prow, %parts) : [index](index, tensor<{VP}xf32>, tensor<{spans}x{VP}xf32>, tensor<{CH * 2}xi32>, tensor<{spans * CH * 2}xi32>)
  %tok = check.generate.fill value(-1) : tensor<{spans}xi32>
  kernel.launch @{kernel}[%n](%n, %meta, %parts, %logits, %tok) : [index](index, tensor<{4 + spans * 9}xi32>, tensor<{spans * CH * 2}xi32>, tensor<{spans}x{VP}xf32>, tensor<{spans}xi32>)
  check.return
}}
check.benchmark<@{name}> @{name}_bm
"""


benches = [bench(f"{k}_t{int(t)}_s{s}", f"sel_{k}", s, t) for k in ("old", "new") for t in (0.0, 1.0) for s in (1, 4)]
open(f"{out}/bench.loom", "w").write("\n".join([HEAD, TILE, new_k, old_k] + benches))
print(f"{out}/test.loom: {len(cases)} cases; {out}/bench.loom: {len(benches)} cases")
