"""mixtest.py OUTDIR: test.loom: g_ink_mixN_nqP vs g_ink_mixN + g_ink_rmsqP (N = 2..5, P = q_cnt = 1, 3) and
g_ink_mix2x2 vs two g_ink_mix2, bitwise, t = 1..4; bench.loom: base vs fused at t = 2, 4. NEG=1 perturbs a weight."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kx
import mixgen
import rmsqgen

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
S = kx.src("/home/illusion/Documents/projects/llamacpp-tuner/tmp/loom/int-r2/program/llm.loom")
rng = np.random.default_rng(21)
NEG = os.environ.get("NEG")
orig = [kx.grab(S, f"g_ink_mix{n}", f"o_ink_mix{n}") for n in (2, 3, 4, 5)]
mine = rmsqgen.module("rq") + mixgen.module(S, rmsqgen.body("rq_body"))
for n in ["rq1", "rq3", "g_ink_mix2x2"] + [f"g_ink_mix{n}_nq{p}" for n in (2, 3, 4, 5) for p in (1, 3)]:
    mine = mine.replace(f'kernel.def export("{n}") @{n}(', f"kernel.def target(@w32) @{n}(")
lib = [kx.HEAD, "".join(kx.funcs_used(S, "".join(orig) + mine).values()), *orig, mine]
MIX_T = "index, index, index, index, index, {B}, {C}, {Q}, {O}"


def npy(name, a):
    np.save(os.path.join(out, name), a)
    return name


def tens(tag, t, P, rand=True):
    def F(name, shape, sd):
        n = "x".join(map(str, shape))
        if rand:
            return f'  %{name} = check.file.read.npy path("{npy(f"{tag}_{name}.npy", rng.normal(0, sd, shape).astype(np.float32))}") : tensor<{n}xf32>\n'
        return f"  %{name} = check.generate.fill value(0.5) : tensor<{n}xf32>\n"
    s = f"  %t = check.literal value({t}) : index\n  %P = check.literal value({P}) : index\n  %c0 = check.literal value(0) : index\n"
    s += "".join(f"  %i{p} = check.literal value({p}) : index\n" for p in range(3))
    s += F("blk", (8, t, 2304), 1.0) + F("cur", (t, 2304), 1.0) + F("qry", (P, 2304), 0.05) + F("qry2", (P, 2304), 0.05)
    for p in range(P):
        s += F(f"w{p}", (2304,), 1.0)
        for nm in ("h", "e"):
            s += f"  %{nm}{p} = check.generate.fill value(0.0) : tensor<{t}x2304xf32>\n  %{nm}q{p} = check.generate.fill value(0) : tensor<{t * 2880}xi8>\n"
    for nm in ("mo", "me", "mo2", "me2"):
        s += f"  %{nm} = check.generate.fill value(0.0) : tensor<{P}x{t}x2304xf32>\n"
    return s


def types(t, P, q="qry"):
    return MIX_T.format(B=f"tensor<8x{t}x2304xf32>", C=f"tensor<{t}x2304xf32>", Q=f"tensor<{P}x2304xf32>", O=f"tensor<{P}x{t}x2304xf32>")


NQ_T = "tensor<2304xf32>, tensor<{t}x2304xf32>, tensor<{q}xi8>"


def base(n, t, P, mo="%me", e="%e"):
    s = (f"  kernel.launch @o_ink_mix{n}[%t, %P](%t, %P, %c0, %c0, %P, %blk, %cur, %qry, {mo}) : [index, index]({types(t, P)})\n")
    rt = f"index, index, tensor<{P}x{t}x2304xf32>, tensor<2304xf32>, tensor<{t}x2304xf32>, tensor<{t * 2880}xi8>"
    args = ", ".join(f"%P, %i{p}, {mo}, %w{p}, {e}{p}, {e}q{p}" for p in range(P))
    return s + f"  kernel.launch @rq{P}[%t](%t, {args}) : [index](index, {', '.join([rt] * P)})\n"


def fused(n, t, P, mo="%mo", h="%h"):
    w = ", ".join(f"%w{p}, {h}{p}, {h}q{p}" for p in range(P))
    wt = ", ".join(f"tensor<2304xf32>, tensor<{t}x2304xf32>, tensor<{t * 2880}xi8>" for _ in range(P))
    return f"  kernel.launch @g_ink_mix{n}_nq{P}[%t, %P](%t, %P, %c0, %c0, %P, %blk, %cur, %qry, {mo}, {w}) : [index, index]({types(t, P)}, {wt})\n"


test = list(lib)
for t in (1, 2, 3, 4):
    for P in (1, 3):
        for n in (2, 3, 4, 5):
            tag = f"m{n}p{P}t{t}"
            s = tens(tag, t, P) + base(n, t, P)
            if NEG and n == 3:
                s += "  %w0 = check.generate.fill value(1.0) : tensor<2304xf32>\n".replace("%w0", "%wneg")
                s += fused(n, t, P).replace("%w0,", "%wneg,")
            else:
                s += fused(n, t, P)
            s += f"  check.expect.bitwise actual(%mo) expected(%me) : tensor<{P}x{t}x2304xf32>\n"
            for p in range(P):
                s += f"  check.expect.bitwise actual(%h{p}) expected(%e{p}) : tensor<{t}x2304xf32>\n  check.expect.bitwise actual(%hq{p}) expected(%eq{p}) : tensor<{t * 2880}xi8>\n"
            if n == 2:  # mix2 pair
                s += f"  kernel.launch @o_ink_mix2[%t, %P](%t, %P, %c0, %c0, %P, %blk, %cur, %qry2, %me2) : [index, index]({types(t, P)})\n"
                s += (f"  kernel.launch @g_ink_mix2x2[%t, %P](%t, %P, %c0, %c0, %P, %blk, %cur, %qry, %mo, %P, %c0, %c0, %P, %blk, %cur, %qry2, %mo2) : "
                      f"[index, index](index, {types(t, P)[7:]}, {types(t, P)[7:]})\n")
                s += f"  check.expect.bitwise actual(%mo2) expected(%me2) : tensor<{P}x{t}x2304xf32>\n"
            test.append(f"check.case public @{tag} {{\n{s}  check.return\n}}\n")
open(os.path.join(out, "test.loom"), "w").write("\n".join(test))

bench = list(lib)
for t in (2, 4):
    for n, P in ((2, 1), (3, 1), (4, 1), (2, 3)):
        s = tens("b", t, P, rand=False)
        bench.append(f"check.case public @base_m{n}p{P}_t{t} {{\n{s}{base(n, t, P) * 4}  check.return\n}}\ncheck.benchmark<@base_m{n}p{P}_t{t}> @base_m{n}p{P}_t{t}_bm\n")
        bench.append(f"check.case public @fused_m{n}p{P}_t{t} {{\n{s}{fused(n, t, P) * 4}  check.return\n}}\ncheck.benchmark<@fused_m{n}p{P}_t{t}> @fused_m{n}p{P}_t{t}_bm\n")
open(os.path.join(out, "bench.loom"), "w").write("\n".join(bench))
