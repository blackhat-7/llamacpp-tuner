#!/usr/bin/env python3
"""Kernel test for the reduced draft head (g_head_mv_q8_p40_sub + g_head_select_sub). Writes OUTDIR/test.loom + npy.

usage: gen_dh_test.py OUTDIR PATCHED_LLM.loom DRAFTER_WITH_HEAD.gguf
Cases (loom-benchmark OUTDIR/test.loom --device=amdgpu):
  match   sub path token == original g_head_mv_q8_p40 + g_head_select token over a full-vocabulary head whose
          rows outside the draft vocabulary are zero (logit 0; the subset maximum is > 0 by construction)
  numpy   sub path token == numpy argmax over the subset rows (inputs chosen with a clear top-1 margin)
  neg     negative control: the id map shifted by one row; expected to FAIL
"""
import os
import sys

import gguf
import numpy as np

HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
out, llm_path, gguf_path = sys.argv[1:4]
SRC = open(llm_path).read()
os.makedirs(out, exist_ok=True)


def grab(name):
    i = SRC.index(f'kernel.def export("{name}") @{name}(')
    k = SRC.index("\n}\n", SRC.index("launch(", i)) + 3
    return SRC[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{name}(")


def func(name):
    i = SRC.index(f"func.def inline @{name}(")
    return SRC[i:SRC.index("\n}\n", i) + 3]


t = {x.name: x for x in gguf.GGUFReader(gguf_path).tensors}
sub = np.asarray(t["dsp_draft_head.weight"].data).reshape(-1, 1296)
ids = np.asarray(t["dsp_draft_head.ids"].data).view(np.int32)
n = len(ids)
full = np.zeros((200058, 1296), np.uint8)
full[ids] = sub
np.save(f"{out}/full.npy", full.view(np.int8).reshape(-1))
np.save(f"{out}/sub.npy", sub.view(np.int8).reshape(-1))
np.save(f"{out}/ids.npy", ids.view(np.float32))
np.save(f"{out}/ids_neg.npy", np.roll(ids, 1).view(np.float32))

# subset logits in float (Q4_0 rows dequantized) for a hidden with a clear winner
d = sub[:, 0::18].copy()
blocks = sub.reshape(n, 72, 18)
scale = blocks[:, :, :2].copy().view(np.float16).astype(np.float32)[:, :, 0]
qs = blocks[:, :, 2:]
codes = np.concatenate([qs & 15, qs >> 4], axis=2).astype(np.float32) - 8
w = (codes * scale[:, :, None]).reshape(n, 2304)
rng = np.random.default_rng(3)
for _ in range(100):
    h = rng.standard_normal(2304).astype(np.float32)
    logits = w @ h
    top = np.sort(logits)[-2:]
    if top[1] > 0 and top[1] - top[0] > 0.05 * abs(top[1]):
        break
np.save(f"{out}/h.npy", h.reshape(1, 2304))
np.save(f"{out}/expected.npy", np.array([ids[int(np.argmax(logits))]], np.int32))
meta = np.zeros(13, np.int32)
meta[0], meta[1] = 1, 200001   # one span; an eos id
meta[4 + 0], meta[4 + 6] = 1, 1  # length 1, selected; temperature 0 (greedy), input index 0
np.save(f"{out}/meta.npy", meta)

HB = n * 1296
TYPES = "tensor<13xi32>, tensor<4608xi8>"


def path(kind, idfile):
    if kind == "full":
        return f"""  %w = check.file.read.npy path("full.npy") : tensor<259275168xi8>
  kernel.launch @g_head_mv_q8_p40[%one](%one, %one, %meta, %hq, %w, %parts, %logits) : [index](index, index, {TYPES}, tensor<259275168xi8>, tensor<3126xi32>, tensor<200064xf32>)
  kernel.launch @g_head_select[%one](%one, %meta, %parts, %logits, %tok) : [index](index, tensor<13xi32>, tensor<3126xi32>, tensor<200064xf32>, tensor<1xi32>)
"""
    return f"""  %w = check.file.read.npy path("sub.npy") : tensor<{HB}xi8>
  %ids = check.file.read.npy path("{idfile}") : tensor<{n}xf32>
  kernel.launch @g_head_mv_q8_p40_sub[%one](%one, %one, %meta, %hq, %w, %parts, %logits, %ids) : [index](index, index, {TYPES}, tensor<{HB}xi8>, tensor<3126xi32>, tensor<200064xf32>, tensor<{n}xf32>)
  kernel.launch @g_head_select_sub[%one](%one, %meta, %parts, %logits, %tok) : [index](index, tensor<13xi32>, tensor<3126xi32>, tensor<200064xf32>, tensor<1xi32>)
"""


def run(kind, idfile, sfx):
    return (f"""  %parts = check.generate.fill value(0) : tensor<3126xi32>
  %logits = check.generate.fill value(0.0) : tensor<200064xf32>
  %tok = check.generate.fill value(-1) : tensor<1xi32>
""" + path(kind, idfile)).replace("%parts", f"%parts{sfx}").replace("%logits", f"%logits{sfx}").replace(
        "%tok", f"%tok{sfx}").replace("%w ", f"%w{sfx} ").replace("%w,", f"%w{sfx},").replace("%ids", f"%ids{sfx}")


COMMON = """  %one = check.literal value(1) : index
  %width = check.literal value(2304) : index
  %meta = check.file.read.npy path("meta.npy") : tensor<13xi32>
  %h = check.file.read.npy path("h.npy") : tensor<1x2304xf32>
  %hq = check.generate.fill value(0) : tensor<4608xi8>
  kernel.launch @g_quant_q8[%one, %width](%one, %width, %h, %hq) : [index, index](index, index, tensor<1x2304xf32>, tensor<4608xi8>)
"""
cases = {
    "match": COMMON + run("full", "", "_a") + run("sub", "ids.npy", "_b")
    + "  check.expect.equal actual(%tok_b) expected(%tok_a) : tensor<1xi32>\n",
    "numpy": COMMON + run("sub", "ids.npy", "_b")
    + '  %expected = check.file.read.npy path("expected.npy") : tensor<1xi32>\n'
    + "  check.expect.equal actual(%tok_b) expected(%expected) : tensor<1xi32>\n",
    "neg": COMMON + run("sub", "ids_neg.npy", "_b")
    + '  %expected = check.file.read.npy path("expected.npy") : tensor<1xi32>\n'
    + "  check.expect.equal actual(%tok_b) expected(%expected) : tensor<1xi32>\n",
}
body = [HEAD, func("g_q4_0_icodes")] + [grab(k) for k in ("g_quant_q8", "g_head_mv_q8_p40", "g_head_select",
                                                          "g_head_mv_q8_p40_sub", "g_head_select_sub")]
body += [f"check.case public @{name} {{\n{text}  check.return\n}}\n" for name, text in cases.items()]
open(f"{out}/test.loom", "w").write("\n".join(body))
print(f"{out}/test.loom: cases match, numpy, neg (neg must fail); expected token {int(np.load(f'{out}/expected.npy')[0])}")
