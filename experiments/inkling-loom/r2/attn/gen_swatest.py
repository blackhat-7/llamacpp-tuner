#!/usr/bin/env python3
"""Test and benchmark the SWA attention v2 kernel and the new combine against the shipped kernels.

usage: gen_swatest.py OUTDIR LLM.loom SWA8.loom SWA.loom COMB.loom
  LLM.loom   shipped program (old kernels: g_ink_attn_split_swa, g_ink_attn_split_swa_span, g_ink_attn_combine)
  SWA8.loom  new SWA kernel generated with G=8 SMAX=64 (the old split ranges): partials must match the old single kernel
  SWA.loom   new SWA kernel at the chosen G/SMAX; COMB.loom the matching combine
OUTDIR/test.loom:
  p_* : new(G=8) pm/pl/po == old single-token kernel (atol/rtol 1e-4)
  e_* : old span kernel + old combine and new SWA + new combine both write their q8 rows to npy (check_e.py compares)
OUTDIR/bench.loom: old span+combine vs new swa+combine, spans 1, 2, 4 at a full window.
"""
import os
import sys

import numpy as np

out, llm, swa8, swa, comb = sys.argv[1:6]
SRC = open(llm).read()
os.makedirs(out, exist_ok=True)
HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
RING = 2 * 1040 * 1024  # f16 values in the K and V rings
rng = np.random.default_rng(7)


def grab(text, name, new):
    i = text.rindex(f'kernel.def export("{name}") @{name}(')
    k = text.index("\n}\n", text.index("launch(", i)) + 3
    return text[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")


def fn(name):
    i = SRC.index(f"func.def inline @{name}(")
    return SRC[i : SRC.index("\n}\n", i) + 3]


def save(name, a):
    np.save(os.path.join(out, name + ".npy"), a)


common = [HEAD, fn("g_gdn_span"), fn("g_gdn_slots"),
          grab(SRC, "g_ink_attn_split_swa", "kold1"), grab(SRC, "g_ink_attn_split_swa_span", "kolds"),
          grab(SRC, "g_ink_attn_combine", "kcold"), grab(open(swa).read(), "g_ink_attn_split_swa_v2", "knew"),
          grab(open(comb).read(), "g_ink_attn_combine", "kcnew")]
T = "index, index, index, tensor<{M}xi32>, tensor<{n}x4xi32>, tensor<5xi64>, tensor<{n}x3072xf32>, tensor<{n}x384xf32>, tensor<16384xf32>, tensor<{R}xf16>, tensor<{P}xf32>, tensor<{P}xf32>, tensor<{O}xf32>"
TS = "index, index, index, tensor<{M}xi32>, tensor<5xi64>, tensor<{n}x3072xf32>, tensor<{n}x384xf32>, tensor<16384xf32>, tensor<{R}xf16>, tensor<{P}xf32>, tensor<{P}xf32>, tensor<{O}xf32>"
TC = "index, i32, tensor<{n}x4xi32>, tensor<{P}xf32>, tensor<{P}xf32>, tensor<{O}xf32>, tensor<{n}x24xf32>, tensor<{n}x3072xf32>, tensor<{Q}xi8>"


def case(tag, n, last, mode, shape=0):
    """n live tokens of a span ending at position last, in a launch of `shape` token rows (padding rows not live)."""
    start = last - n + 1
    meta = np.zeros(4 + 9, np.int32); meta[0] = 1; meta[4:9] = [n, start, 0, 0, 0]; save(f"m_{tag}", meta)
    n, live = max(n, shape), n
    info = np.zeros((n, 4), np.int32); info[:live, 1] = start + np.arange(live); info[:live, 2] = 1; save(f"i_{tag}", info)
    d = dict(n=n, M=13, R=RING, P=n * 1536, O=n * 196608, Q=n * 3840)
    b = f"""  %seed = check.literal value(5) : i64
  %t = check.literal value({n}) : index
  %one = check.literal value(1) : index
  %off = check.literal value(0) : index
  %sliding = check.literal value(1) : i32
  %meta = check.file.read.npy path("m_{tag}.npy") : tensor<13xi32>
  %info = check.file.read.npy path("i_{tag}.npy") : tensor<{n}x4xi32>
  %orig = check.generate.fill value(0) : tensor<5xi64>
  %q = check.generate.random.uniform seed(%seed) range(-4.0 to 4.0) : tensor<{n}x3072xf32>
  %r = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x384xf32>
  %rb = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<16384xf32>
  %arena = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{RING}xf16>
"""
    def parts(s):
        return f"""  %pm{s} = check.generate.fill value(-3.40282347e+38) : tensor<{d['P']}xf32>
  %pl{s} = check.generate.fill value(0.0) : tensor<{d['P']}xf32>
  %po{s} = check.generate.fill value(0.0) : tensor<{d['O']}xf32>
"""
    args1 = "(%t, %one, %off, %meta, %info, %orig, %q, %r, %rb, %arena, %pm{s}, %pl{s}, %po{s})"
    if mode == "p":
        b += parts("a") + parts("b")
        b += f"  kernel.launch @kold1[%t]{args1.format(s='a')} : [index]({T.format(**d)})\n"
        b += f"  kernel.launch @knew[%t]{args1.format(s='b')} : [index]({T.format(**d)})\n"
        for nm, sz in (("pm", d["P"]), ("pl", d["P"]), ("po", d["O"])):
            b += f"  check.expect.approximate actual(%{nm}b) expected(%{nm}a) atol(1.0e-4) rtol(1.0e-4) nan(different) : tensor<{sz}xf32>, tensor<{sz}xf32>\n"
    else:
        b += parts("a") + parts("b") + f"""  %sinks = check.generate.random.uniform seed(%seed) range(-1.0 to 1.0) : tensor<{n}x24xf32>
  %gate = check.generate.random.uniform seed(%seed) range(-3.0 to 3.0) : tensor<{n}x3072xf32>
  %oa = check.generate.fill value(0) : tensor<{d['Q']}xi8>
  %ob = check.generate.fill value(0) : tensor<{d['Q']}xi8>
  kernel.launch @kolds[%one](%t, %one, %off, %meta, %orig, %q, %r, %rb, %arena, %pma, %pla, %poa) : [index]({TS.format(**d)})
  kernel.launch @kcold[%t](%t, %sliding, %info, %pma, %pla, %poa, %sinks, %gate, %oa) : [index]({TC.format(**d)})
  kernel.launch @knew[%t]{args1.format(s='b')} : [index]({T.format(**d)})
  kernel.launch @kcnew[%t](%t, %sliding, %info, %pmb, %plb, %pob, %sinks, %gate, %ob) : [index]({TC.format(**d)})
  check.file.write.npy value(%oa) path("{tag}_old.npy") mode(always) : tensor<{d['Q']}xi8>
  check.file.write.npy value(%ob) path("{tag}_new.npy") mode(always) : tensor<{d['Q']}xi8>
"""
    return f"check.case public @{tag} {{\n{b}  check.return\n}}\n"


SPOTS = [(1, 0), (1, 37), (2, 37), (3, 700), (4, 1022), (2, 1024), (4, 1030), (1, 1500), (4, 5000), (2, 195000), (3, 64001),
         (3, 2000, 4), (5, 3000, 8), (6, 1030, 8), (7, 195000, 8), (8, 5000, 8)]  # (live, last position[, shape])
test = common.copy()
test[6] = grab(open(swa8).read(), "g_ink_attn_split_swa_v2", "knew")
test.append("".join(case(f"p{n}_{last}_{sh[0] if sh else n}", n, last, "p", *sh) for n, last, *sh in SPOTS))
open(os.path.join(out, "test_p.loom"), "w").write("\n".join(test))
open(os.path.join(out, "test_e.loom"), "w").write("\n".join(common + [case(f"e{n}_{last}_{sh[0] if sh else n}", n, last, "e", *sh) for n, last, *sh in SPOTS]))


def bench(tag, n, last, which, shape=0):
    c = case(tag, n, last, "e", shape).split("\n")
    keep = [l for l in c if "check.file.write" not in l and ("@kold" not in l and "@kcold" not in l if which == "new" else "@knew" not in l and "@kcnew" not in l)]
    return "\n".join(keep) + f"check.benchmark<@{tag}> @{tag}_bm\n"


open(os.path.join(out, "bench.loom"), "w").write("\n".join(
    common + [bench(f"{w}{n}of{sh}_{last // 1000}k", n, last, w, sh) for last in (3000, 195000) for n, sh in ((1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8)) for w in ("old", "new")]))
