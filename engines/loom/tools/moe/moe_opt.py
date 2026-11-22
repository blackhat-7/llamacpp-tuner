#!/usr/bin/env python3
"""Apply the moe workstream's changes to a package's program/llm.loom in place.

usage: moe_opt.py PACKAGE_DIR [--no-gu-r2] [--no-skip-pad]

1. Router top-8: add g_ink_moe_topk_w32 (one wave, subgroup reductions; same ABI, bit-exact) from
   g_ink_moe_topk_w32.loom next to this file, and point every `kernel.launch @g_ink_moe_topk[` at it.
2. NVFP4 gate/up: g_ink_moe_gu_nvfp4_w1_k2304_r4 handles 2 rows per workgroup instead of 4 (renamed _r2; twice
   the workgroups; each row's math unchanged, so bit-exact). Skipped with --no-gu-r2.
3. Padding tokens (verify spans run in power-of-two shapes: span 3 as 4 tokens, 5-7 as 8) stop reading experts:
   the router (g_ink_moe_topk_w32l, launched with %info) gives tokens that g_token_info marks not live ids -1 and
   weights 0; the four expert kernels (NVFP4 and Q8_0 gate/up and down) skip the weight reads of a -1 slot and
   write 0. Live tokens are unchanged bit for bit. Skipped with --no-skip-pad.
Idempotent: refuses to run twice. Writes through a fresh file (never through a hard link).
"""

import os
import sys
from pathlib import Path

pkg = Path(sys.argv[1])
gu_r2 = "--no-gu-r2" not in sys.argv[2:]
skip_pad = "--no-skip-pad" not in sys.argv[2:]
llm = pkg / "program" / "llm.loom"
src = llm.read_text()
if "g_ink_moe_topk_w32" in src:
    sys.exit("moe_opt: already applied")

here = Path(__file__).resolve().parent
topk_name = "g_ink_moe_topk_w32l" if skip_pad else "g_ink_moe_topk_w32"
topk = (here / f"{topk_name}.loom").read_text()
n = src.count("kernel.launch @g_ink_moe_topk[")
assert n > 0
if skip_pad:
    import re

    pat = r"kernel\.launch @g_ink_moe_topk\[%t\]\(%t, ([^()]*), %ids, %wts\) : \[index\]\(index, i32, buffer, view<128xf32>, view<128xf32>, buffer, buffer\)"
    src, k = re.subn(
        pat,
        r"kernel.launch @g_ink_moe_topk_w32l[%t](%t, \1, %ids, %wts, %info) : [index](index, i32, buffer, view<128xf32>, view<128xf32>, buffer, buffer, buffer)",
        src,
    )
    assert k == n, (k, n)
else:
    src = src.replace(
        "kernel.launch @g_ink_moe_topk[", "kernel.launch @g_ink_moe_topk_w32["
    )
i = src.index('kernel.def export("g_ink_moe_topk") @g_ink_moe_topk(')
src = src[:i] + topk.rstrip("\n") + "\n\n\n" + src[i:]

if gu_r2:
    old, new = "g_ink_moe_gu_nvfp4_w1_k2304_r4", "g_ink_moe_gu_nvfp4_w1_k2304_r2"
    i = src.index(f'kernel.def export("{old}") @{old}(')
    j = src.index("\n}\n", src.index("launch(", i)) + 3
    k = src[i:j].replace(old, new)
    for a, b in [
        (
            "%rows_per = index.constant 4 : index",
            "%rows_per = index.constant 2 : index",
        ),
        ("%cR = index.constant 4 : index", "%cR = index.constant 2 : index"),
        (
            "[range(%n_arg, 4, 1152), multiple_of(%n_arg, 4)]",
            "[range(%n_arg, 2, 1152), multiple_of(%n_arg, 2)]",
        ),
    ]:
        assert k.count(a) == 1, a
        k = k.replace(a, b)
    src = src[:i] + k + src[j:]
    assert src.count(f"@{old}[") > 0
    src = src.replace(f"kernel.launch @{old}[", f"kernel.launch @{new}[")
    assert old not in src


def kernel_span(src, name):
    i = src.index(f'kernel.def export("{name}") @{name}(')
    return i, src.index("\n}\n", src.index("launch(", i)) + 3


EID = "  %eid = view.load %ids_view[%a] : view<[%tokens]xi32> -> i32\n"
LIVE = EID + "  %e_live = scalar.cmpi sge, %eid, %c0_i32 : i32\n"


def skip_nvfp4(src, name, n_rows):
    """Wrap the row loop in `scf.if %e_live`; a padding slot stores 0 to its rows instead."""
    i, j = kernel_span(src, name)
    k = src[i:j]
    loop = "  scf.for %r = [%c0 to %cR step %c1] {\n"
    end = "\n  }\n  kernel.return\n}\n"
    assert k.count(EID) == 1 and k.count(loop) == 1 and k.endswith(end)
    body = k[k.index(loop) : -len("  kernel.return\n}\n")]
    body = "\n".join("  " + line if line else line for line in body.split("\n"))
    zero = f"""  }} else {{
    scf.for %zr = [%c0 to %cR step %c1] {{
      %zrow_v = index.add %row_base, %zr : index
      %zrow = index.assume %zrow_v [range(%zrow_v, 0, {n_rows - 1})] : index
      scf.if %is_l0 {{
        view.store %zero_f, %yv[%a, %zrow] : f32, view<[%tokens]x{n_rows}xf32>
      }}
    }}
  }}
"""
    k = (
        k[: k.index(loop)]
        + "  scf.if %e_live {\n"
        + body.rstrip(" ")
        + zero
        + "  kernel.return\n}\n"
    )
    return src[:i] + k.replace(EID, LIVE) + src[j:]


def skip_q8(src, name):
    """Skip the weight loop of a padding slot (its LDS partials are then garbage) and store 0 for it."""
    i, j = kernel_span(src, name)
    k = src[i:j]
    loop = "  scf.for %r = [%c0 to %rows_per step %c1] {\n"
    bar = "\n  }\n  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    store = "func.call @g_store_row(%y, %tokens, %n, %a, %out_row, %value, %c0_i32)"
    assert (
        k.count(EID) == 1
        and k.count(loop) == 1
        and k.count(bar) == 1
        and k.count(store) == 1
    )
    a, b = k.index(loop), k.index(bar)
    body = "\n".join("  " + line if line else line for line in k[a : b + 5].split("\n"))
    k = k[:a] + "  scf.if %e_live {\n" + body.rstrip(" ") + "  }\n" + k[b + 5 :]
    k = k.replace(
        store,
        "%value_live = scf.select %e_live, %value, %zero_f : f32\n    "
        + store.replace("%value,", "%value_live,"),
    )
    return src[:i] + k.replace(EID, LIVE) + src[j:]


if skip_pad:
    gu = "g_ink_moe_gu_nvfp4_w1_k2304_r2" if gu_r2 else "g_ink_moe_gu_nvfp4_w1_k2304_r4"
    src = skip_nvfp4(src, gu, 1152)
    src = skip_nvfp4(src, "g_ink_moe_dn_nvfp4_w1_k1152_r4", 2304)
    src = skip_q8(src, "g_moe_gu_q8_0_fused_q8_k2304_r4")
    src = skip_q8(src, "g_moe_dn_q8_0_q8_k1152_r4")

tmp = llm.with_suffix(".loom.moe_tmp")
tmp.write_text(src)
os.replace(tmp, llm)  # new inode: a hard-linked original stays untouched
print(f"moe_opt: topk launches {n}, gu r2 {gu_r2}, skip padding {skip_pad}")
