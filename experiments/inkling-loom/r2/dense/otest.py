"""otest.py OUTDIR: O projection (Q8_0, K=3072, n=2304): the branch-free R=2 kernel vs g_mv_q8_0_k3072_t4_q8_r1,
bitwise (t = 2, 3, 4) in test.loom; bench.loom times one visit per layer copy (NL copies)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kx
import mvgen

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
S = kx.src()
rng = np.random.default_rng(5)
ref = kx.grab(S, "g_mv_q8_0_k3072_t4_q8_r1", "oref")
mine = mvgen.body(3072, "ob", 2) + mvgen.kernel(3072, "onew", "ob", [0], 2).replace('kernel.def export("onew") @onew(', "kernel.def target(@w32) @onew(")
lib = [kx.HEAD, "".join(kx.funcs_used(S, ref + mine).values()), ref, mine]
WB = 2304 * 3264


def npy(name, a):
    np.save(os.path.join(out, name), a)
    return name


def launches(t, w, y, sfx=""):
    return (f"  kernel.launch @oref[%n, %t](%n, %t, %acc, %x, {w}, %e{sfx}) : [index, index](index, index, i32, tensor<{t * 3840}xi8>, tensor<{WB}xi8>, tensor<{t}x2304xf32>)\n",
            f"  kernel.launch @onew[%nw, %t](%nw, %t, %acc, %n, %x, {w}, {y}) : [index, index](index, index, i32, index, tensor<{t * 3840}xi8>, tensor<{WB}xi8>, tensor<{t}x2304xf32>)\n")


test = list(lib)
for t in (2, 3, 4):
    blk = np.zeros((2304, 96, 34), np.uint8)
    blk[:, :, :2] = rng.uniform(0.002, 0.02, (2304, 96)).astype(np.float16).view(np.uint8).reshape(2304, 96, 2)
    blk[:, :, 2:] = rng.integers(-127, 128, (2304, 96, 32)).astype(np.int8).view(np.uint8)
    codes = rng.integers(-127, 128, (t, 3072)).astype(np.int8)
    meta = np.stack([rng.uniform(0.001, 0.05, (t, 96)).astype(np.float16).astype(np.float32), codes.reshape(t, 96, 32).sum(-1).astype(np.float32)], -1)
    x = np.concatenate([codes.view(np.uint8).reshape(-1), meta.astype(np.float32).view(np.uint8).reshape(-1)]).view(np.int8)
    s = (f"  %t = check.literal value({t}) : index\n  %n = check.literal value(2304) : index\n  %nw = check.literal value(1152) : index\n"
         f"  %acc = check.literal value(0) : i32\n  %x = check.file.read.npy path(\"{npy(f't{t}x.npy', x)}\") : tensor<{t * 3840}xi8>\n"
         f"  %w = check.file.read.npy path(\"{npy(f't{t}w.npy', blk.view(np.int8).reshape(-1))}\") : tensor<{WB}xi8>\n"
         f"  %e = check.generate.fill value(0.0) : tensor<{t}x2304xf32>\n  %y = check.generate.fill value(0.0) : tensor<{t}x2304xf32>\n")
    a, b = launches(t, "%w", "%y")
    s += a + b + f"  check.expect.bitwise actual(%y) expected(%e) : tensor<{t}x2304xf32>\n"
    test.append(f"check.case public @t{t} {{\n{s}  check.return\n}}\n")
open(os.path.join(out, "test.loom"), "w").write("\n".join(test))

NL = 24
bench = list(lib)
for t in (2, 4):
    s = (f"  %t = check.literal value({t}) : index\n  %n = check.literal value(2304) : index\n  %nw = check.literal value(1152) : index\n"
         f"  %acc = check.literal value(0) : i32\n  %x = check.generate.fill value(1) : tensor<{t * 3840}xi8>\n"
         f"  %e = check.generate.fill value(0.0) : tensor<{t}x2304xf32>\n  %y = check.generate.fill value(0.0) : tensor<{t}x2304xf32>\n")
    s += "".join(f"  %w{L} = check.generate.fill value(1) : tensor<{WB}xi8>\n" for L in range(NL))
    base = "".join(launches(t, f"%w{L}", "%y")[0] for L in range(NL))
    new = "".join(launches(t, f"%w{L}", "%y")[1] for L in range(NL))
    bench.append(f"check.case public @base_t{t} {{\n{s}{base}  check.return\n}}\ncheck.benchmark<@base_t{t}> @base_t{t}_bm\n")
    bench.append(f"check.case public @new_t{t} {{\n{s}{new}  check.return\n}}\ncheck.benchmark<@new_t{t}> @new_t{t}_bm\n")
open(os.path.join(out, "bench.loom"), "w").write("\n".join(bench))
