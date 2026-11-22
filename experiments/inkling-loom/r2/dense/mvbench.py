"""mvbench.py OUTDIR [EXTRA.loom...]: bench of one layer visit's Q/G/K/V/R projections (t = 2, 4), weights rotated
over NL layer copies so they stream from DRAM as in the model. Cases: base_tT (5 launches of the original kernel)
and any VARIANTS listed in env VARIANTS (python file defining variant(t, L) -> launch text, kernels -> text)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import kx

out = sys.argv[1]
extra = sys.argv[2:]
S = kx.src()
NL = int(os.environ.get("NL", "8"))
ROWS = [("q", 3072), ("g", 3072), ("k", 1024), ("v", 1024), ("r", 384)]
WB = {n: rows * 2448 for n, rows in ROWS}
body = [kx.HEAD]
k = kx.grab(S, "g_mv_q8_0_k2304_t4_q8_r1", "mv4")
body.append("".join(kx.funcs_used(S, k).values()))
body.append(k)
VAR = {}
if os.environ.get("VARIANTS"):
    exec(open(os.environ["VARIANTS"]).read(), VAR)
    body.append(VAR["kernels"])


def tensors(t):
    s = "  %seed = check.literal value(7) : i64\n"
    s += f"  %t = check.literal value({t}) : index\n  %acc = check.literal value(0) : i32\n"
    s += f"  %x = check.generate.fill value(1) : tensor<{t * 2880}xi8>\n"
    for n, rows in ROWS:
        s += f"  %n{n} = check.literal value({rows}) : index\n"
        s += f"  %y{n} = check.generate.fill value(0.0) : tensor<{t * rows}xf32>\n"
        for L in range(NL):
            s += f"  %w{n}{L} = check.generate.fill value(1) : tensor<{WB[n]}xi8>\n"
    return s


for t in (2, 4):
    s = tensors(t)
    for L in range(NL):
        for n, rows in ROWS:
            s += (f"  kernel.launch @mv4[%n{n}, %t](%n{n}, %t, %acc, %x, %w{n}{L}, %y{n}) : [index, index](index, index, i32, "
                  f"tensor<{t * 2880}xi8>, tensor<{WB[n]}xi8>, tensor<{t * rows}xf32>)\n")
    body.append(f"check.case public @base_t{t} {{\n{s}  check.return\n}}\ncheck.benchmark<@base_t{t}> @base_t{t}_bm\n")
    for vname, fn in VAR.get("variants", {}).items():
        s = tensors(t) + "".join(fn(t, L) for L in range(NL))
        body.append(f"check.case public @{vname}_t{t} {{\n{s}  check.return\n}}\ncheck.benchmark<@{vname}_t{t}> @{vname}_t{t}_bm\n")
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "bench.loom"), "w").write("\n".join(body))
