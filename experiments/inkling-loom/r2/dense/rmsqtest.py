"""rmsqtest.py OUTDIR: test.loom (rmsq1/rmsq3 vs g_ink_rms_norm_plane + g_quant_q8, bitwise, t = 1..4) and
bench.loom (3 norms of a looped visit: base 6 launches vs one rmsq3; one norm: base 2 launches vs rmsq1)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kx
import rmsqgen

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
S = kx.src()
rng = np.random.default_rng(3)
nrm = kx.grab(S, "g_ink_rms_norm_plane", "nrm").replace("[lt(%idx_arg, %pl)]", "[range(%idx_arg, 0, 3), lt(%idx_arg, %pl)]")
qnt = kx.grab(S, "g_quant_q8", "qnt")
mine = rmsqgen.module("rq").replace('kernel.def export("rq1") @rq1(', "kernel.def target(@w32) @rq1(").replace(
    'kernel.def export("rq3") @rq3(', "kernel.def target(@w32) @rq3(")
lib = [kx.HEAD, nrm, qnt, mine]
NEG = os.environ.get("NEG")


def npy(name, a):
    np.save(os.path.join(out, name), a)
    return name


test = list(lib)
for t in (1, 2, 3, 4):
    s = f"  %t = check.literal value({t}) : index\n  %nH = check.literal value(2304) : index\n  %c3 = check.literal value(3) : index\n"
    for p in range(3):
        x = rng.normal(0, 1 + p, (3, t, 2304)).astype(np.float32)
        if p == 1:
            x[:, :, 64:96] = 0.0  # an all-zero group
        s += f'  %x{p} = check.file.read.npy path("{npy(f"t{t}x{p}.npy", x)}") : tensor<3x{t}x2304xf32>\n'
        w = rng.uniform(0.5, 1.5, 2304).astype(np.float32)
        s += f'  %w{p} = check.file.read.npy path("{npy(f"t{t}w{p}.npy", w)}") : tensor<2304xf32>\n'
        if NEG and p == 2:
            w = w.copy(); w[100] *= 1.0001
        s += f'  %wf{p} = check.file.read.npy path("{npy(f"t{t}wf{p}.npy", w)}") : tensor<2304xf32>\n'
        s += f"  %i{p} = check.literal value({p}) : index\n"
        for nm in ("h", "e"):
            s += f"  %{nm}{p} = check.generate.fill value(0.0) : tensor<{t}x2304xf32>\n  %{nm}q{p} = check.generate.fill value(0) : tensor<{t * 2880}xi8>\n"
        s += (f"  kernel.launch @nrm[%t](%t, %c3, %i{p}, %x{p}, %w{p}, %e{p}) : [index](index, index, index, tensor<3x{t}x2304xf32>, tensor<2304xf32>, tensor<{t}x2304xf32>)\n"
              f"  kernel.launch @qnt[%t, %nH](%t, %nH, %e{p}, %eq{p}) : [index, index](index, index, tensor<{t}x2304xf32>, tensor<{t * 2880}xi8>)\n")
    a3 = ", ".join(f"%c3, %i{p}, %x{p}, %wf{p}, %h{p}, %hq{p}" for p in range(3))
    ty = ", ".join(f"index, index, tensor<3x{t}x2304xf32>, tensor<2304xf32>, tensor<{t}x2304xf32>, tensor<{t * 2880}xi8>" for p in range(3))
    s += f"  kernel.launch @rq3[%t](%t, {a3}) : [index](index, {ty})\n"
    for p in range(3):
        s += f"  check.expect.bitwise actual(%h{p}) expected(%e{p}) : tensor<{t}x2304xf32>\n  check.expect.bitwise actual(%hq{p}) expected(%eq{p}) : tensor<{t * 2880}xi8>\n"
    # rq1 on plane 1 into fresh buffers
    s += f"  %g1 = check.generate.fill value(0.0) : tensor<{t}x2304xf32>\n  %gq1 = check.generate.fill value(0) : tensor<{t * 2880}xi8>\n"
    s += f"  kernel.launch @rq1[%t](%t, %c3, %i1, %x1, %wf1, %g1, %gq1) : [index](index, index, index, tensor<3x{t}x2304xf32>, tensor<2304xf32>, tensor<{t}x2304xf32>, tensor<{t * 2880}xi8>)\n"
    s += f"  check.expect.bitwise actual(%g1) expected(%e1) : tensor<{t}x2304xf32>\n  check.expect.bitwise actual(%gq1) expected(%eq1) : tensor<{t * 2880}xi8>\n"
    test.append(f"check.case public @t{t} {{\n{s}  check.return\n}}\n")
open(os.path.join(out, "test.loom"), "w").write("\n".join(test))

bench = list(lib)
for t in (2, 4):
    s = f"  %t = check.literal value({t}) : index\n  %nH = check.literal value(2304) : index\n  %c3 = check.literal value(3) : index\n"
    for p in range(3):
        s += (f"  %x{p} = check.generate.fill value(0.5) : tensor<3x{t}x2304xf32>\n  %w{p} = check.generate.fill value(1.0) : tensor<2304xf32>\n"
              f"  %i{p} = check.literal value({p}) : index\n  %h{p} = check.generate.fill value(0.0) : tensor<{t}x2304xf32>\n"
              f"  %hq{p} = check.generate.fill value(0) : tensor<{t * 2880}xi8>\n")
    b = "".join(f"  kernel.launch @nrm[%t](%t, %c3, %i{p}, %x{p}, %w{p}, %h{p}) : [index](index, index, index, tensor<3x{t}x2304xf32>, tensor<2304xf32>, tensor<{t}x2304xf32>)\n"
                f"  kernel.launch @qnt[%t, %nH](%t, %nH, %h{p}, %hq{p}) : [index, index](index, index, tensor<{t}x2304xf32>, tensor<{t * 2880}xi8>)\n" for p in range(3))
    a3 = ", ".join(f"%c3, %i{p}, %x{p}, %w{p}, %h{p}, %hq{p}" for p in range(3))
    ty = ", ".join(f"index, index, tensor<3x{t}x2304xf32>, tensor<2304xf32>, tensor<{t}x2304xf32>, tensor<{t * 2880}xi8>" for p in range(3))
    f3 = f"  kernel.launch @rq3[%t](%t, {a3}) : [index](index, {ty})\n"
    f1 = "".join(f"  kernel.launch @rq1[%t](%t, %c3, %i{p}, %x{p}, %w{p}, %h{p}, %hq{p}) : [index](index, index, index, tensor<3x{t}x2304xf32>, tensor<2304xf32>, tensor<{t}x2304xf32>, tensor<{t * 2880}xi8>)\n" for p in range(3))
    for nm, body in (("base", b), ("rq3", f3), ("rq1x3", f1)):
        bench.append(f"check.case public @{nm}_t{t} {{\n{s}{body * 4}  check.return\n}}\ncheck.benchmark<@{nm}_t{t}> @{nm}_t{t}_bm\n")
open(os.path.join(out, "bench.loom"), "w").write("\n".join(bench))
