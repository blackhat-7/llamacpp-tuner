#!/usr/bin/env python3
"""Test of the padding skip (moe_opt.py step 3): new router and expert kernels vs the originals.

usage: gen_pad_test.py OUTDIR ORIG_LLM.loom NEW_LLM.loom   then run OUTDIR/test.loom with loom-test
       (--file-output-root=OUTDIR) and check with check_pad.py OUTDIR.
Router: all tokens live -> bitwise equal to g_ink_moe_topk (checked in loom-test); some tokens not live -> live rows
equal, others ids -1 / weights 0 (check_pad.py). Experts (NVFP4 gate/up r2 + down, Q8_0 gate/up + down, 4 experts):
ids with -1 slots; live slots must equal the original kernel bit for bit, -1 slots must be 0 (check_pad.py).
"""
import os, re, sys
import numpy as np

out, ORIG, NEW = sys.argv[1], open(sys.argv[2]).read(), open(sys.argv[3]).read()
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(21)


def kernel(src, name, new):
    i = src.index(f'kernel.def export("{name}") @{name}(')
    k = src.index("\n}\n", src.index("launch(", i)) + 3
    return src[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")


def funcs(src, text, seen):
    res = ""
    for f in sorted(set(re.findall(r"func\.call @(\w+)\(", text)) - seen):
        seen.add(f)
        i = src.index(f"func.def inline @{f}(")
        body = src[i:src.index("\n}\n", i) + 3]
        res += funcs(src, body, seen) + body + "\n"
    return res


KS = [("g_ink_moe_topk", "g_ink_moe_topk_w32l", "tk"), ("g_ink_moe_gu_nvfp4_w1_k2304_r4", "g_ink_moe_gu_nvfp4_w1_k2304_r2", "gn"),
      ("g_ink_moe_dn_nvfp4_w1_k1152_r4", "g_ink_moe_dn_nvfp4_w1_k1152_r4", "dn"), ("g_moe_gu_q8_0_fused_q8_k2304_r4", "g_moe_gu_q8_0_fused_q8_k2304_r4", "gq"),
      ("g_moe_dn_q8_0_q8_k1152_r4", "g_moe_dn_q8_0_q8_k1152_r4", "dq")]
lib = "".join(kernel(ORIG, o, "o" + t) + kernel(NEW, n, "n" + t) for o, n, t in KS)
lib = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n" + funcs(ORIG, lib, set()) + lib
NE = 4
SZ = {"gn": NE * 2985984, "dn": NE * 1492992, "gq": NE * 5640192, "dq": NE * 2820096}


def nvfp4(n):
    w = rng.integers(-128, 128, n, dtype=np.int16).astype(np.int8)
    w.reshape(-1, 36)[:, :4] = rng.integers(40, 72, (n // 36, 4))
    return w


def q8(n):
    w = rng.integers(-127, 128, n, dtype=np.int16).astype(np.int8).reshape(-1, 34)
    w[:, :2] = np.float16(rng.uniform(1e-3, 1e-2, len(w))).reshape(-1, 1).view(np.int8)
    return w.reshape(-1)


def xq(rows, k):
    codes = rng.integers(-127, 128, (rows, k), dtype=np.int16).astype(np.int8)
    meta = np.zeros((rows, k // 32, 2), np.float32)
    meta[:, :, 0] = rng.uniform(0.001, 0.02, (rows, k // 32)); meta[:, :, 1] = codes.reshape(rows, -1, 32).sum(-1)
    return np.concatenate([codes.view(np.uint8).reshape(-1), meta.view(np.uint8).reshape(-1)]).view(np.int8)


for t, f in (("gn", nvfp4), ("dn", nvfp4), ("gq", q8), ("dq", q8)):
    np.save(f"{out}/w{t}.npy", f(SZ[t]))
np.save(f"{out}/gus.npy", rng.uniform(0.5, 2.0, 128).astype(np.float32))
np.save(f"{out}/bias.npy", rng.uniform(-0.05, 0.05, 128).astype(np.float32))
np.save(f"{out}/ds.npy", rng.uniform(0.5, 2, 128).astype(np.float32))
cases = []
for T, nlive in ((4, 3), (8, 5), (8, 7), (2, 2)):
    A = 8 * T
    live = (np.arange(T) < nlive).astype(np.int32)
    info = np.zeros((T, 4), np.int32); info[:, 2] = live
    np.save(f"{out}/info{T}_{nlive}.npy", info); np.save(f"{out}/infoall{T}.npy", np.stack([np.zeros(T), np.zeros(T), np.ones(T), np.zeros(T)], 1).astype(np.int32))
    np.save(f"{out}/lg{T}_{nlive}.npy", rng.normal(0, 2, (T, 128)).astype(np.float32))
    ids = np.stack([rng.permutation(np.arange(8) % NE) for _ in range(T)]).astype(np.int32)
    ids[nlive:] = -1
    ids[0, 3] = -1  # a -1 slot inside a live token too: only the slot is skipped
    np.save(f"{out}/ids{T}_{nlive}.npy", ids.reshape(-1))
    np.save(f"{out}/xg{T}_{nlive}.npy", xq(T, 2304)); np.save(f"{out}/xd{T}_{nlive}.npy", xq(A, 1152))
    g = f"{T}_{nlive}"
    tg = f"tensor<{A}xi32>"
    tk_t = f"(index, i32, tensor<{T}x128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<{T}x8xi32>, tensor<{T}x8xf32>"
    body = f"""  %t = check.literal value({T}) : index
  %ta = check.literal value({A}) : index
  %nF = check.literal value(1152) : index
  %nH = check.literal value(2304) : index
  %use1 = check.literal value(1) : i32
  %lg = check.file.read.npy path("lg{g}.npy") : tensor<{T}x128xf32>
  %bias = check.file.read.npy path("bias.npy") : tensor<128xf32>
  %ds = check.file.read.npy path("ds.npy") : tensor<128xf32>
  %info = check.file.read.npy path("info{g}.npy") : tensor<{T}x4xi32>
  %infoall = check.file.read.npy path("infoall{T}.npy") : tensor<{T}x4xi32>
  %io = check.generate.fill value(0) : tensor<{T}x8xi32>
  %wo = check.generate.fill value(0.0) : tensor<{T}x8xf32>
  %ia = check.generate.fill value(0) : tensor<{T}x8xi32>
  %wa = check.generate.fill value(0.0) : tensor<{T}x8xf32>
  %il = check.generate.fill value(0) : tensor<{T}x8xi32>
  %wl = check.generate.fill value(0.0) : tensor<{T}x8xf32>
  kernel.launch @otk[%t](%t, %use1, %lg, %bias, %ds, %io, %wo) : [index]{tk_t})
  kernel.launch @ntk[%t](%t, %use1, %lg, %bias, %ds, %ia, %wa, %infoall) : [index]{tk_t}, tensor<{T}x4xi32>)
  kernel.launch @ntk[%t](%t, %use1, %lg, %bias, %ds, %il, %wl, %info) : [index]{tk_t}, tensor<{T}x4xi32>)
  check.expect.bitwise actual(%ia) expected(%io) : tensor<{T}x8xi32>
  check.expect.bitwise actual(%wa) expected(%wo) : tensor<{T}x8xf32>
  check.file.write.npy value(%io) path("{g}_tk_ids_o.npy") mode(always) : tensor<{T}x8xi32>
  check.file.write.npy value(%wo) path("{g}_tk_w_o.npy") mode(always) : tensor<{T}x8xf32>
  check.file.write.npy value(%il) path("{g}_tk_ids_n.npy") mode(always) : tensor<{T}x8xi32>
  check.file.write.npy value(%wl) path("{g}_tk_w_n.npy") mode(always) : tensor<{T}x8xf32>
  %ids = check.file.read.npy path("ids{g}.npy") : {tg}
  %gus = check.file.read.npy path("gus.npy") : tensor<128xf32>
  %xg = check.file.read.npy path("xg{g}.npy") : tensor<{T * 2880}xi8>
  %xd = check.file.read.npy path("xd{g}.npy") : tensor<{A * 1440}xi8>
"""
    for t, n, x, xt, extra in (("gn", "%nF", "%xg", f"tensor<{T * 2880}xi8>", ["%gus"]), ("dn", "%nH", "%xd", f"tensor<{A * 1440}xi8>", []),
                               ("gq", "%nF", "%xg", f"tensor<{T * 2880}xi8>", ["%w2"]), ("dq", "%nH", "%xd", f"tensor<{A * 1440}xi8>", [])):
        N = 1152 if n == "%nF" else 2304
        wt = f"tensor<{SZ[t]}xi8>"
        ex_v = "".join(", " + (f"%w{t}" if e == "%w2" else e) for e in extra)
        ex_t = "".join(", " + (wt if e == "%w2" else "tensor<128xf32>") for e in extra)
        body += f"  %w{t} = check.file.read.npy path(\"w{t}.npy\") : {wt}\n"
        for k in "on":
            body += f"""  %y{t}{k} = check.generate.fill value(7.0) : tensor<{A * N}xf32>
  kernel.launch @{k}{t}[{n}, %ta]({n}, %ta, %ids, {x}, %w{t}{ex_v}, %y{t}{k}) : [index, index](index, index, {tg}, {xt}, {wt}{ex_t}, tensor<{A * N}xf32>)
  check.file.write.npy value(%y{t}{k}) path("{g}_{t}_{k}.npy") mode(always) : tensor<{A * N}xf32>
"""
    cases.append(f"check.case public @c{g} {{\n{body}  check.return\n}}\n")
open(f"{out}/test.loom", "w").write(lib + "\n".join(cases))
