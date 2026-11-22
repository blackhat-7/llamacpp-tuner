#!/usr/bin/env python3
"""Kernel test: new NVFP4 MoE kernels vs the originals on identical random inputs (4 experts, spans 1/2/4).

usage: gen_test.py OUTDIR LLM.loom NEW.loom...  (env GU=name DN=name of the new kernels; NEG=1 adds a negative
control: the new kernel against the original run with other expert ids, which must fail)
"""
import os, sys
import numpy as np

out = sys.argv[1]
SRC = "\n".join(open(p).read() for p in sys.argv[2:])
GU, DN = os.environ.get("GU", "g_moe_gu_nvb"), os.environ.get("DN", "g_moe_dn_nvb")


def grab(name, new):
    i = SRC.rindex(f'kernel.def export("{name}") @{name}(')
    j = SRC.index("launch(", i)
    k = SRC.index("\n}\n", j) + 3
    return SRC[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")


def func(name):
    i = SRC.index(f"func.def inline @{name}(")
    return SRC[i:SRC.index("\n}\n", i) + 3]


os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(3)
NE = 4
GU_W, DN_W = NE * 2985984, NE * 1492992


def weights(nbytes, row_bytes):
    w = rng.integers(-128, 128, nbytes, dtype=np.int16).astype(np.int8)
    blocks = w.reshape(-1, 36)
    blocks[:, :4] = rng.integers(40, 72, (blocks.shape[0], 4)).astype(np.int8)  # sane e4m3 scales
    return w


def xq(rows, k):
    codes = rng.integers(-127, 128, (rows, k), dtype=np.int16).astype(np.int8)
    meta = np.zeros((rows, k // 32, 2), np.float32)
    meta[:, :, 0] = rng.uniform(0.001, 0.02, (rows, k // 32))
    return np.concatenate([codes.view(np.uint8).reshape(-1), meta.view(np.uint8).reshape(-1)]).view(np.int8)


np.save(f"{out}/wgu.npy", weights(GU_W, 1296))
np.save(f"{out}/wdn.npy", weights(DN_W, 648))
np.save(f"{out}/gus.npy", rng.uniform(0.5, 2.0, 128).astype(np.float32))
HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
lib = [HEAD, func("g_ue4m3_to_f32"), grab("g_ink_moe_gu_nvfp4_w1_k2304_r4", "ogu"), grab("g_ink_moe_dn_nvfp4_w1_k1152_r4", "odn"),
       grab(GU, "ngu"), grab(DN, "ndn")]
cases = []
for T in (1, 2, 4):
    A = 8 * T
    ids = np.stack([rng.permutation(NE)[:NE].repeat(2)[:8] % NE for _ in range(T)]).astype(np.int32).reshape(-1)
    ids2 = (ids + 1) % NE
    np.save(f"{out}/ids{T}.npy", ids); np.save(f"{out}/idsb{T}.npy", ids2)
    np.save(f"{out}/xg{T}.npy", xq(T, 2304)); np.save(f"{out}/xd{T}.npy", xq(A, 1152))
    neg = os.environ.get("NEG") == "1"
    o_ids = "%idsb" if neg else "%ids"
    tg, td = f"tensor<{A}xi32>", f"tensor<{T * 2880}xi8>"
    cases.append(f"""check.case public @t{T} {{
  %nF = check.literal value(1152) : index
  %nH = check.literal value(2304) : index
  %ta = check.literal value({A}) : index
  %ids = check.file.read.npy path("ids{T}.npy") : {tg}
  %idsb = check.file.read.npy path("idsb{T}.npy") : {tg}
  %wgu = check.file.read.npy path("wgu.npy") : tensor<{GU_W}xi8>
  %wdn = check.file.read.npy path("wdn.npy") : tensor<{DN_W}xi8>
  %gus = check.file.read.npy path("gus.npy") : tensor<128xf32>
  %xg = check.file.read.npy path("xg{T}.npy") : {td}
  %xd = check.file.read.npy path("xd{T}.npy") : tensor<{A * 1440}xi8>
  %act_o = check.generate.fill value(0.0) : tensor<{A * 1152}xf32>
  %act_n = check.generate.fill value(0.0) : tensor<{A * 1152}xf32>
  %ys_o = check.generate.fill value(0.0) : tensor<{A * 2304}xf32>
  %ys_n = check.generate.fill value(0.0) : tensor<{A * 2304}xf32>
  kernel.launch @ogu[%nF, %ta](%nF, %ta, {o_ids}, %xg, %wgu, %gus, %act_o) : [index, index](index, index, {tg}, {td}, tensor<{GU_W}xi8>, tensor<128xf32>, tensor<{A * 1152}xf32>)
  kernel.launch @ngu[%nF, %ta](%nF, %ta, %ids, %xg, %wgu, %gus, %act_n) : [index, index](index, index, {tg}, {td}, tensor<{GU_W}xi8>, tensor<128xf32>, tensor<{A * 1152}xf32>)
  kernel.launch @odn[%nH, %ta](%nH, %ta, {o_ids}, %xd, %wdn, %ys_o) : [index, index](index, index, {tg}, tensor<{A * 1440}xi8>, tensor<{DN_W}xi8>, tensor<{A * 2304}xf32>)
  kernel.launch @ndn[%nH, %ta](%nH, %ta, %ids, %xd, %wdn, %ys_n) : [index, index](index, index, {tg}, tensor<{A * 1440}xi8>, tensor<{DN_W}xi8>, tensor<{A * 2304}xf32>)
  check.file.write.npy value(%act_o) path("t{T}_act_o.npy") mode(always) : tensor<{A * 1152}xf32>
  check.file.write.npy value(%act_n) path("t{T}_act_n.npy") mode(always) : tensor<{A * 1152}xf32>
  check.file.write.npy value(%ys_o) path("t{T}_ys_o.npy") mode(always) : tensor<{A * 2304}xf32>
  check.file.write.npy value(%ys_n) path("t{T}_ys_n.npy") mode(always) : tensor<{A * 2304}xf32>
  check.expect.approximate actual(%act_n) expected(%act_o) atol(3.0e-3) rtol(1.0e-3) nan(different) : tensor<{A * 1152}xf32>, tensor<{A * 1152}xf32>
  check.expect.approximate actual(%ys_n) expected(%ys_o) atol(3.0e-4) rtol(1.0e-5) nan(different) : tensor<{A * 2304}xf32>, tensor<{A * 2304}xf32>
  check.return
}}
""")
open(f"{out}/test.loom", "w").write("\n".join(lib + cases))
