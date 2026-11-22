#!/usr/bin/env python3
"""MoE decode/verify microbenchmark for Inkling-2-Mini's NVFP4 expert kernels.

usage: gen_moe.py OUTDIR LLM.loom [EXTRA.loom ...]  (env KERNELS="gu=NAME,dn=NAME" picks the kernels, default the
originals; CASES="T:s,T:s" span and shared experts between consecutive tokens)
Each case runs L launches (one per fake layer) of gate/up then down, each launch with its own expert ids, so the
expert sets sweep all 128 experts and the 96 MB infinity cache does not keep weights between launches.
"""
import os, sys
import numpy as np

HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
out = sys.argv[1]
SRC = "\n".join(open(p).read() for p in sys.argv[2:])
# KERNELS="gu=A,dn=B" or several variants "v1:gu=A,dn=B;v2:gu=C,dn=D" (cases interleaved per variant)
_spec = os.environ.get("KERNELS", "gu=g_ink_moe_gu_nvfp4_w1_k2304_r4,dn=g_ink_moe_dn_nvfp4_w1_k1152_r4")
VARS = [(v.split(":")[0] if ":" in v else "", dict(kv.split("=") for kv in v.split(":")[-1].split(","))) for v in _spec.split(";")]
CASES = [tuple(map(int, c.split(":"))) for c in os.environ.get("CASES", "1:0,2:0,2:3,2:8,4:0,4:3,4:8").split(",")]
L = int(os.environ.get("L", "16"))


def grab(name, new):
    i = SRC.rindex(f'kernel.def export("{name}") @{name}(')  # last source wins
    j = SRC.index("launch(", i)
    k = SRC.index("\n}\n", j) + 3
    return SRC[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")


def func(name):
    i = SRC.index(f"func.def inline @{name}(")
    return SRC[i:SRC.index("\n}\n", i) + 3]


os.makedirs(out, exist_ok=True)


def routing(T, s, base):
    """ids [T*8]: token j keeps the first s experts of token j-1 and takes 8-s new ones; returns ids, next base."""
    ids, prev = [], None
    for j in range(T):
        cur = (prev[:s] if prev else []) + [(base + i) % 128 for i in range(8 - (len(prev[:s]) if prev else 0))]
        base += 8 - (len(prev[:s]) if prev else 0)
        ids += cur
        prev = cur
    return np.array(ids, np.int32), base


GU_W, DN_W = 382205952, 191102976
bench = [HEAD, func("g_ue4m3_to_f32")]
for vn, K in VARS:
    bench += [grab(K["gu"], f"kgu{vn}"), grab(K["dn"], f"kdn{vn}")]
for (T, s), (vn, K) in [(c, v) for c in CASES for v in VARS]:
    A = 8 * T
    tag = f"t{T}s{s}{vn}"
    body = f"""  %seed = check.literal value(7) : i64
  %nF = check.literal value(1152) : index
  %nH = check.literal value(2304) : index
  %ta = check.literal value({A}) : index
  %wgu = check.generate.fill value(17) : tensor<{GU_W}xi8>
  %wdn = check.generate.fill value(17) : tensor<{DN_W}xi8>
  %gus = check.generate.fill value(1.0) : tensor<128xf32>
  %xg = check.generate.fill value(3) : tensor<{T * 2880}xi8>
  %xd = check.generate.fill value(3) : tensor<{A * 1440}xi8>
  %act = check.generate.fill value(0.0) : tensor<{A * 1152}xf32>
  %ys = check.generate.fill value(0.0) : tensor<{A * 2304}xf32>
"""
    base = 0
    for l in range(L):
        ids, base = routing(T, s, base)
        np.save(os.path.join(out, f"{tag}_ids{l}.npy"), ids)
        body += f"""  %ids{l} = check.file.read.npy path("{tag}_ids{l}.npy") : tensor<{A}xi32>
  kernel.launch @kgu{vn}[%nF, %ta](%nF, %ta, %ids{l}, %xg, %wgu, %gus, %act) : [index, index](index, index, tensor<{A}xi32>, tensor<{T * 2880}xi8>, tensor<{GU_W}xi8>, tensor<128xf32>, tensor<{A * 1152}xf32>)
  kernel.launch @kdn{vn}[%nH, %ta](%nH, %ta, %ids{l}, %xd, %wdn, %ys) : [index, index](index, index, tensor<{A}xi32>, tensor<{A * 1440}xi8>, tensor<{DN_W}xi8>, tensor<{A * 2304}xf32>)
"""
    bench.append(f"check.case public @{tag} {{\n{body}  check.return\n}}\ncheck.benchmark<@{tag}> @{tag}_bm\n")
open(os.path.join(out, "bench.loom"), "w").write("\n".join(bench))
