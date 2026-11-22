"""ctrltest.py OUTDIR: test.loom (concat_rms3 + mvf k4608 x3 + ctrl_apply3 vs 3 x original chain; mvf k4608/k2304
single vs g_mv_f32_kK_t4_r1; bitwise, t = 2..4 and 7) and bench.loom (base vs fused, weights rotated over NL copies)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctrlgen
import kx

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
S = kx.src()
rng = np.random.default_rng(9)
NEG = os.environ.get("NEG")
orig = [kx.grab(S, n, n.replace("g_", "o_")).replace("[lt(%c_idx_arg, %cp)]", "[range(%c_idx_arg, 0, 3), lt(%c_idx_arg, %cp)]").replace(
    "[lt(%m_idx_arg, %mp)]", "[range(%m_idx_arg, 0, 3), lt(%m_idx_arg, %mp)]") for n in ("g_ink_concat_rms", "g_ink_ctrl_apply", "g_mv_f32_k4608_t4_r1", "g_mv_f32_k2304_t4_r1")]
mine = ctrlgen.module(S)
for n in ("g_ink_concat_rms3", "g_ink_ctrl_apply3", "g_mvf_k4608_t4", "g_mvf_k2304_t4", "g_mvf_k4608_t4x3"):
    mine = mine.replace(f'kernel.def export("{n}") @{n}(', f"kernel.def target(@w32) @{n}(")
lib = [kx.HEAD, "".join(kx.funcs_used(S, "".join(orig) + mine).values()), *orig, mine]
CH = [("a", 0, "msg_attn", 0), ("k", 0, "msg_kv", 0), ("v", 0, "msg_kv", 1)]  # chain, cur idx, msg buffer, msg idx


def npy(name, a):
    np.save(os.path.join(out, name), a)
    return name


def tens(t, tag, rand=True):
    def T(name, shape, gen):
        n = "x".join(map(str, shape))
        if rand:
            return f'  %{name} = check.file.read.npy path("{npy(f"{tag}_{name}.npy", gen().astype(np.float32))}") : tensor<{n}xf32>\n'
        return f"  %{name} = check.generate.fill value(0.25) : tensor<{n}xf32>\n"
    s = f"  %t = check.literal value({t}) : index\n  %c0 = check.literal value(0) : index\n  %c1 = check.literal value(1) : index\n"
    s += "  %c2 = check.literal value(2) : index\n  %nR = check.literal value(64) : index\n  %nE = check.literal value(128) : index\n"
    s += "  %acc = check.literal value(0) : i32\n  %nw3 = check.literal value(192) : index\n  %nwr = check.literal value(128) : index\n"
    s += T("mix_out", (1, t, 2304), lambda: rng.normal(0, 1, (1, t, 2304)))
    s += T("msg_attn", (2, t, 2304), lambda: rng.normal(0, 2, (2, t, 2304)))
    s += T("msg_kv", (2, t, 2304), lambda: rng.normal(0, 0.5, (2, t, 2304)))
    s += T("h", (t, 2304), lambda: rng.normal(0, 1, (t, 2304)))
    s += T("wrt", (128, 2304), lambda: rng.normal(0, 0.05, (128, 2304)))
    for c, *_ in CH:
        s += T(f"dn_{c}", (64, 4608), lambda: rng.normal(0, 0.05, (64, 4608)))
        s += T(f"gn_{c}", (2304, 64), lambda: rng.normal(0, 0.2, (2304, 64)))
        s += T(f"fg_{c}", (2304, 64), lambda: rng.normal(0, 0.2, (2304, 64)))
        for nm, shape in ((f"cat_{c}", (t, 4608)), (f"z_{c}", (t, 64)), (f"o_{c}", (t, 2304))):
            n = "x".join(map(str, shape))
            s += f"  %{nm} = check.generate.fill value(0.0) : tensor<{n}xf32>\n  %e{nm} = check.generate.fill value(0.0) : tensor<{n}xf32>\n"
    s += f"  %rl = check.generate.fill value(0.0) : tensor<{t}x128xf32>\n  %erl = check.generate.fill value(0.0) : tensor<{t}x128xf32>\n"
    return s


def base_chain(t, c, ci, msg, mi, pre="%e"):
    cat, z, o = f"{pre}cat_{c}", f"{pre}z_{c}", f"{pre}o_{c}"
    return (f"  kernel.launch @o_ink_concat_rms[%t](%t, %c1, %c{ci}, %c2, %c{mi}, %mix_out, %{msg}, {cat}) : [index](index, index, index, index, index, "
            f"tensor<1x{t}x2304xf32>, tensor<2x{t}x2304xf32>, tensor<{t}x4608xf32>)\n"
            f"  kernel.launch @o_mv_f32_k4608_t4_r1[%nR, %t](%nR, %t, %acc, {cat}, %dn_{c}, {z}) : [index, index](index, index, i32, tensor<{t}x4608xf32>, tensor<64x4608xf32>, tensor<{t}x64xf32>)\n"
            f"  kernel.launch @o_ink_ctrl_apply[%t](%t, %c1, %c{ci}, %c2, %c{mi}, %mix_out, %{msg}, {z}, %gn_{c}, %fg_{c}, {o}) : [index](index, index, index, index, index, "
            f"tensor<1x{t}x2304xf32>, tensor<2x{t}x2304xf32>, tensor<{t}x64xf32>, tensor<2304x64xf32>, tensor<2304x64xf32>, tensor<{t}x2304xf32>)\n")


def fused(t, pre="%"):
    ca = ", ".join(f"%c1, %c{ci}, %c2, %c{mi}, %mix_out, %{msg}, {pre}cat_{c}" for c, ci, msg, mi in CH)
    cat_t = ", ".join(f"index, index, index, index, tensor<1x{t}x2304xf32>, tensor<2x{t}x2304xf32>, tensor<{t}x4608xf32>" for _ in CH)
    mv = (f"  kernel.launch @g_mvf_k4608_t4x3[%nw3, %t](%nw3, %t, %acc, %nR, %nR, %nR, " + ", ".join(f"{pre}cat_{c}" for c, *_ in CH) + ", "
          + ", ".join(f"%dn_{c}" for c, *_ in CH) + ", " + ", ".join(f"{pre}z_{c}" for c, *_ in CH)
          + f") : [index, index](index, index, i32, index, index, index, " + ", ".join([f"tensor<{t}x4608xf32>"] * 3 + ["tensor<64x4608xf32>"] * 3 + [f"tensor<{t}x64xf32>"] * 3) + ")\n")
    ap = ", ".join(f"%c1, %c{ci}, %c2, %c{mi}, %mix_out, %{msg}, {pre}z_{c}, %gn_{c}, %fg_{c}, {pre}o_{c}" for c, ci, msg, mi in CH)
    ap_t = ", ".join(f"index, index, index, index, tensor<1x{t}x2304xf32>, tensor<2x{t}x2304xf32>, tensor<{t}x64xf32>, tensor<2304x64xf32>, tensor<2304x64xf32>, tensor<{t}x2304xf32>" for _ in CH)
    return (f"  kernel.launch @g_ink_concat_rms3[%t](%t, {ca}) : [index](index, {cat_t})\n" + mv
            + f"  kernel.launch @g_ink_ctrl_apply3[%t](%t, {ap}) : [index](index, {ap_t})\n")


def router(t, new, out):
    if new:
        return (f"  kernel.launch @g_mvf_k2304_t4[%nwr, %t](%nwr, %t, %acc, %nE, %h, %wrt, {out}) : "
                f"[index, index](index, index, i32, index, tensor<{t}x2304xf32>, tensor<128x2304xf32>, tensor<{t}x128xf32>)\n")
    return (f"  kernel.launch @o_mv_f32_k2304_t4_r1[%nE, %t](%nE, %t, %acc, %h, %wrt, {out}) : "
            f"[index, index](index, index, i32, tensor<{t}x2304xf32>, tensor<128x2304xf32>, tensor<{t}x128xf32>)\n")


test = list(lib)
for t in (2, 3, 4, 7):
    tag = f"t{t}"
    s = tens(t, tag) + "".join(base_chain(t, *c) for c in CH) + router(t, False, "%erl")
    if NEG:
        s += "  %gn_neg = check.generate.fill value(0.01) : tensor<2304x64xf32>\n"
    s += ((fused(t).replace("%gn_k,", "%gn_neg,") if NEG else fused(t)) if t <= 4 else "") + router(t, True, "%rl")
    for c, *_ in (CH if t <= 4 else []):
        for nm, shape in ((f"cat_{c}", (t, 4608)), (f"z_{c}", (t, 64)), (f"o_{c}", (t, 2304))):
            n = "x".join(map(str, shape))
            s += f"  check.expect.bitwise actual(%{nm}) expected(%e{nm}) : tensor<{n}xf32>\n"
    s += f"  check.expect.bitwise actual(%rl) expected(%erl) : tensor<{t}x128xf32>\n"
    test.append(f"check.case public @{tag} {{\n{s}  check.return\n}}\n")
open(os.path.join(out, "test.loom"), "w").write("\n".join(test))

bench = list(lib)
for t in (2, 4):
    s = tens(t, "b", rand=False)
    b = "".join(base_chain(t, *c) for c in CH)
    bench.append(f"check.case public @base_t{t} {{\n{s}{b * 4}  check.return\n}}\ncheck.benchmark<@base_t{t}> @base_t{t}_bm\n")
    bench.append(f"check.case public @fused_t{t} {{\n{s}{fused(t) * 4}  check.return\n}}\ncheck.benchmark<@fused_t{t}> @fused_t{t}_bm\n")
    bench.append(f"check.case public @rbase_t{t} {{\n{s}{router(t, False, '%rl') * 4}  check.return\n}}\ncheck.benchmark<@rbase_t{t}> @rbase_t{t}_bm\n")
    bench.append(f"check.case public @rnew_t{t} {{\n{s}{router(t, True, '%rl') * 4}  check.return\n}}\ncheck.benchmark<@rnew_t{t}> @rnew_t{t}_bm\n")
open(os.path.join(out, "bench.loom"), "w").write("\n".join(bench))
