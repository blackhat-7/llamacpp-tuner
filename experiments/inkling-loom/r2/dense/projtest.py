"""projtest.py OUTDIR: test.loom (fused projection kernels vs g_mv_q8_0_k2304_t4_q8_r1 + g_ink_bf16_proj, bitwise,
t = 2, 3, 4, one dead token) and bench.loom (one visit, weights rotated over NL layer copies; base vs fused)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kx
import projgen

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
S = kx.src()
rng = np.random.default_rng(11)
NEG = os.environ.get("NEG")  # negative control: perturb one fused weight byte

mv = kx.grab(S, "g_mv_q8_0_k2304_t4_q8_r1", "mv4")
bf = kx.grab(S, "g_ink_bf16_proj", "bfp")
fused = projgen.module(("p6", "p8")).replace('kernel.def export("p6") @p6(', "kernel.def target(@w32) @p6(").replace(
    'kernel.def export("p8") @p8(', "kernel.def target(@w32) @p8(")
lib = [kx.HEAD, "".join(kx.funcs_used(S, mv + bf + fused).values()), mv, bf, fused]
Q8 = projgen.Q8
BFS = projgen.BF[8]


def npy(name, a):
    np.save(os.path.join(out, name), a)
    return name


def q8w(rows):
    blk = np.zeros((rows, 72, 34), np.uint8)
    sc = rng.uniform(0.002, 0.02, (rows, 72)).astype(np.float16)
    blk[:, :, :2] = sc.view(np.uint8).reshape(rows, 72, 2)
    blk[:, :, 2:] = rng.integers(-127, 128, (rows, 72, 32)).astype(np.int8).view(np.uint8)
    return blk.view(np.int8).reshape(-1)


def xq8(t):
    codes = rng.integers(-127, 128, (t, 2304)).astype(np.int8)
    meta = np.stack([rng.uniform(0.001, 0.05, (t, 72)).astype(np.float16).astype(np.float32),
                     codes.reshape(t, 72, 32).sum(-1).astype(np.float32)], -1)
    return np.concatenate([codes.view(np.uint8).reshape(-1), meta.astype(np.float32).view(np.uint8).reshape(-1)]).view(np.int8)


def bfw(rows):
    f = rng.uniform(-0.1, 0.1, (rows, 2304)).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16).view(np.int8).reshape(-1)


def tens(t, tag):
    s = ""
    for i in range(3):
        s += f'  %x{i} = check.file.read.npy path("{npy(f"{tag}_x{i}.npy", xq8(t))}") : tensor<{t * 2880}xi8>\n'
        s += f'  %h{i} = check.file.read.npy path("{npy(f"{tag}_h{i}.npy", rng.uniform(-2, 2, (t, 2304)).astype(np.float32))}") : tensor<{t}x2304xf32>\n'
    info = np.zeros((t, 4), np.int32)
    info[:, 2] = 1
    info[t - 1, 2] = 0 if t == 3 else 1
    s += f'  %info = check.file.read.npy path("{npy(f"{tag}_info.npy", info)}") : tensor<{t}x4xi32>\n'
    for n, rows, _ in Q8:
        w = q8w(rows)
        s += f'  %w_{n} = check.file.read.npy path("{npy(f"{tag}_w{n}.npy", w)}") : tensor<{rows * 2448}xi8>\n'
        if NEG and n == "v":
            w = w.copy()
            w[rows * 2448 // 2 + 7] ^= 1
        s += f'  %wf_{n} = check.file.read.npy path("{npy(f"{tag}_wf{n}.npy", w)}") : tensor<{rows * 2448}xi8>\n'
    for n, rows, _ in BFS:
        s += f'  %w_{n} = check.file.read.npy path("{npy(f"{tag}_w{n}.npy", bfw(rows))}") : tensor<{rows * 4608}xi8>\n'
    for n, rows, _ in Q8 + BFS:
        s += f"  %y_{n} = check.generate.fill value(0.0) : tensor<{t}x{rows}xf32>\n"
        s += f"  %e_{n} = check.generate.fill value(0.0) : tensor<{t}x{rows}xf32>\n"
    s += f"  %t = check.literal value({t}) : index\n  %acc = check.literal value(0) : i32\n"
    for n, rows, _ in Q8 + BFS:
        s += f"  %n_{n} = check.literal value({rows}) : index\n"
    return s


def base(t, wpre="%w_", ypre="%e_", sfx=""):
    s = ""
    for n, rows, x in Q8:
        s += (f"  kernel.launch @mv4[%n_{n}, %t](%n_{n}, %t, %acc, %x{x}, {wpre}{n}{sfx}, {ypre}{n}) : [index, index](index, index, i32, "
              f"tensor<{t * 2880}xi8>, tensor<{rows * 2448}xi8>, tensor<{t}x{rows}xf32>)\n")
    return s


def base_bf(t, segs, sfx="", ypre="%e_"):
    return "".join(f"  kernel.launch @bfp[%t, %n_{n}](%t, %n_{n}, %info, %h{x}, %w_{n}{sfx}, {ypre}{n}) : [index, index](index, index, "
                   f"tensor<{t}x4xi32>, tensor<{t}x2304xf32>, tensor<{rows * 4608}xi8>, tensor<{t}x{rows}xf32>)\n" for n, rows, x in segs)


def fused_launch(t, nseg, wq="%wf_", sfx=""):
    bfs = projgen.BF[nseg]
    segs = [(n, rows, "bf") for n, rows, _ in bfs] + [(n, rows, "q8") for n, rows, _ in Q8]
    nw = projgen.workgroups(nseg)
    ws = ", ".join((f"%w_{n}{sfx}" if k == "bf" else f"{wq}{n}{sfx}") for n, _, k in segs)
    ys = ", ".join(f"%y_{n}" for n, *_ in segs)
    wt = ", ".join(f"tensor<{rows * (4608 if k == 'bf' else 2448)}xi8>" for _, rows, k in segs)
    yt = ", ".join(f"tensor<{t}x{rows}xf32>" for _, rows, k in segs)
    xt = f"tensor<{t * 2880}xi8>, " * 3 + f"tensor<{t}x2304xf32>, " * 3 + f"tensor<{t}x4xi32>"
    return (f"  %nw{nseg}{sfx} = check.literal value({nw}) : index\n"
            f"  kernel.launch @p{nseg}[%nw{nseg}{sfx}, %t](%nw{nseg}{sfx}, %t, %x0, %x1, %x2, %h0, %h1, %h2, %info, {ws}, {ys}) : "
            f"[index, index](index, index, {xt}, {wt}, {yt})\n")


test = list(lib)
for t in (2, 3, 4):
    for nseg in (6, 8):
        tag = f"t{t}s{nseg}"
        body = tens(t, tag) + base(t) + base_bf(t, projgen.BF[nseg]) + fused_launch(t, nseg)
        for n, rows, _ in Q8 + projgen.BF[nseg]:
            body += f"  check.expect.bitwise actual(%y_{n}) expected(%e_{n}) : tensor<{t}x{rows}xf32>\n"
        test.append(f"check.case public @{tag} {{\n{body}  check.return\n}}\n")
open(os.path.join(out, "test.loom"), "w").write("\n".join(test))

NL = int(os.environ.get("NL", "8"))
bench = list(lib)
for t in (2, 4):
    s = f"  %t = check.literal value({t}) : index\n  %acc = check.literal value(0) : i32\n"
    s += "".join(f"  %x{i} = check.generate.fill value(1) : tensor<{t * 2880}xi8>\n  %h{i} = check.generate.fill value(0.5) : tensor<{t}x2304xf32>\n" for i in range(3))
    s += f"  %info = check.generate.fill value(1) : tensor<{t}x4xi32>\n"
    for n, rows, _ in Q8 + BFS:
        s += f"  %n_{n} = check.literal value({rows}) : index\n  %y_{n} = check.generate.fill value(0.0) : tensor<{t}x{rows}xf32>\n"
        for L in range(NL):
            s += f"  %w_{n}{L} = check.generate.fill value(1) : tensor<{rows * (2448 if n in ("q", "g", "k", "v", "r") else 4608)}xi8>\n"
    for nseg in (6, 8):
        b = "".join(base(t, "%w_", "%y_", L) + base_bf(t, projgen.BF[nseg], L, "%y_") for L in range(NL))
        f = "".join(fused_launch(t, nseg, "%w_", L) for L in range(NL))
        bench.append(f"check.case public @base{nseg}_t{t} {{\n{s}{b}  check.return\n}}\ncheck.benchmark<@base{nseg}_t{t}> @base{nseg}_t{t}_bm\n")
        bench.append(f"check.case public @fused{nseg}_t{t} {{\n{s}{f}  check.return\n}}\ncheck.benchmark<@fused{nseg}_t{t}> @fused{nseg}_t{t}_bm\n")
open(os.path.join(out, "bench.loom"), "w").write("\n".join(bench))
