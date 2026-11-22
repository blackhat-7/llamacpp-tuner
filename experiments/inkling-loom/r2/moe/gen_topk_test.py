#!/usr/bin/env python3
"""test.loom: new topk vs g_ink_moe_topk, exact ids and weights (ties included); bench.loom: both at 1/2/4 tokens.
usage: gen_topk_test.py OUTDIR LLM.loom NEW.loom  (env NEW=kernel name; NEG=1: compare against other logits)"""
import os, sys
import numpy as np
out = sys.argv[1]; SRC = "\n".join(open(p).read() for p in sys.argv[2:]); NEW = os.environ.get("NEW", "g_ink_moe_topk_w32")
def grab(name, new):
    i = SRC.rindex(f'kernel.def export("{name}") @{name}(')
    k = SRC.index("\n}\n", SRC.index("launch(", i)) + 3
    return SRC[i:k].replace(f'kernel.def export("{name}") @{name}(', f"kernel.def target(@w32) @{new}(")
os.makedirs(out, exist_ok=True); rng = np.random.default_rng(11)
HEAD = "amdgpu.target<gfx11-generic> @w32 {subgroup_size = 32}\n\n"
lib = [HEAD, grab("g_ink_moe_topk", "ko"), grab(NEW, "kn")]
np.save(f"{out}/bias.npy", rng.uniform(-0.05, 0.05, 128).astype(np.float32))
np.save(f"{out}/bias0.npy", np.zeros(128, np.float32))
np.save(f"{out}/ds.npy", rng.uniform(0.5, 2, 128).astype(np.float32))
cases, bench = [], []
def launch(k, T, lg, bias, use, ids, wts):
    return (f"  kernel.launch @{k}[%t](%t, %use{use}, %{lg}, %{bias}, %ds, %{ids}, %{wts}) : [index](index, i32, tensor<{T}x128xf32>, "
            f"tensor<128xf32>, tensor<128xf32>, tensor<{T}x8xi32>, tensor<{T}x8xf32>)\n")
def common(T, tag):
    return f"""  %t = check.literal value({T}) : index
  %use0 = check.literal value(0) : i32
  %use1 = check.literal value(1) : i32
  %lg = check.file.read.npy path("{tag}_lg.npy") : tensor<{T}x128xf32>
  %lg2 = check.file.read.npy path("{tag}_lg2.npy") : tensor<{T}x128xf32>
  %bias = check.file.read.npy path("bias.npy") : tensor<128xf32>
  %bias0 = check.file.read.npy path("bias0.npy") : tensor<128xf32>
  %ds = check.file.read.npy path("ds.npy") : tensor<128xf32>
  %io = check.generate.fill value(0) : tensor<{T}x8xi32>
  %in = check.generate.fill value(0) : tensor<{T}x8xi32>
  %wo = check.generate.fill value(0.0) : tensor<{T}x8xf32>
  %wn = check.generate.fill value(0.0) : tensor<{T}x8xf32>
"""
for T, kind in [(1, "rand"), (2, "rand"), (4, "ties"), (8, "ties"), (3, "allsame")]:
    tag = f"t{T}{kind}"
    if kind == "rand":
        lg = rng.normal(0, 2, (T, 128))
    elif kind == "ties":
        lg = rng.integers(-3, 4, (T, 128)) * 0.5
    else:
        lg = np.zeros((T, 128))
    np.save(f"{out}/{tag}_lg.npy", lg.astype(np.float32)); np.save(f"{out}/{tag}_lg2.npy", rng.normal(0, 2, (T, 128)).astype(np.float32))
    for use, bias in [(0, "bias"), (1, "bias"), (1, "bias0")]:
        olg = "lg2" if os.environ.get("NEG") else "lg"
        body = common(T, tag) + launch("ko", T, olg, bias, use, "io", "wo") + launch("kn", T, "lg", bias, use, "in", "wn")
        body += f"""  check.expect.bitwise actual(%in) expected(%io) : tensor<{T}x8xi32>
  check.expect.bitwise actual(%wn) expected(%wo) : tensor<{T}x8xf32>
"""
        cases.append(f"check.case public @{tag}_u{use}{bias} {{\n{body}  check.return\n}}\n")
for T in (1, 2, 4):
    tag = f"t{T}b"
    np.save(f"{out}/{tag}_lg.npy", rng.normal(0, 2, (T, 128)).astype(np.float32)); np.save(f"{out}/{tag}_lg2.npy", rng.normal(0, 2, (T, 128)).astype(np.float32))
    for k in ("ko", "kn"):
        body = common(T, tag) + "".join(launch(k, T, "lg", "bias", 1, "io", "wo") for _ in range(16))
        bench.append(f"check.case public @{tag}{k} {{\n{body}  check.return\n}}\ncheck.benchmark<@{tag}{k}> @{tag}{k}_bm\n")
open(f"{out}/test.loom", "w").write("\n".join(lib + cases))
open(f"{out}/bench.loom", "w").write("\n".join(lib + bench))
