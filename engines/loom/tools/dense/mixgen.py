"""Mix + RMSNorm + planar Q8 in one launch, bit for bit: g_ink_mixN (q_cnt planes) followed by g_ink_rmsq1/3 on
its output. Each workgroup computes one (token, plane) of the mix exactly as before and writes it to global memory
and to workgroup memory; after a workgroup barrier it runs g_ink_rmsq's body on that row, reading workgroup memory
(the same f32 values the separate norm would load). Also g_ink_mix2x2: two g_ink_mix2 launches in one (grid z)."""

import re

from ctrlgen import kernel_func, multi


def lds_norm_body(rmsq_src, fn):
    """g_ink_rmsq's body reading its row from workgroup memory %lds (576 x vec4) instead of x[pi, tok]."""
    b = rmsq_src.replace(
        "(%tokens: index, %pl: index, %pi: index, %x: buffer,",
        "(%tokens: index, %lds: buffer,",
        1,
    )
    b = b.replace(
        "  %xv = buffer.view %x[%zero] : buffer -> view<[%pl]x[%tokens]x576x4xf32>\n",
        "  %xv = buffer.view %lds[%zero] : buffer -> view<576x4xf32>\n",
        1,
    )
    b = b.replace(
        "vector.load %xv[%pi, %tok, %ch, %c0] : view<[%pl]x[%tokens]x576x4xf32>",
        "vector.load %xv[%ch, %c0] : view<576x4xf32>",
    )
    assert b.count("vector.load %xv[%ch, %c0]") == 2 and "%pi" not in b
    return re.sub(r"func\.def inline @\w+\(", f"func.def inline @{fn}(", b, count=1)


def module(src, rmsq_src):
    out = [lds_norm_body(rmsq_src, "g_ink_mix_nq_norm")]
    for n in (2, 3, 4, 5):
        body, ps = kernel_func(src, f"g_ink_mix{n}", f"g_ink_mix{n}_nq_mix")
        body = body.replace("%out: buffer) {", "%out: buffer, %lds: buffer) {", 1)
        body = body.replace(
            "  %zero = index.constant 0 : offset\n",
            "  %zero = index.constant 0 : offset\n  %lds_v = buffer.view %lds[%zero] : buffer -> view<576x4xf32>\n",
            1,
        )
        body, k = re.subn(
            r"( *)vector\.store (%\w+), %ov\[%qi, %tok, (%i\d), %c0\] : ([^\n]*)\n",
            r"\g<0>\1vector.store \2, %lds_v[\3, %c0] : vector<4xf32>, view<576x4xf32>\n",
            body,
        )
        assert k == 3, k
        out.append(body)
        names = [p.split(":")[0].strip() for p in ps] + ["%lds"]
        types = ", ".join([p.split(":")[1].strip() for p in ps] + ["buffer"])
        for nnorm in (1, 3):
            norm = ", ".join(
                f"%w{p}: buffer, %h{p}: buffer, %hq{p}: buffer" for p in range(nnorm)
            )
            calls = [
                f"func.call @g_ink_mix_nq_norm(%tokens, %lds, %w{p}, %h{p}, %hq{p}) : (index, buffer, buffer, buffer, buffer)\n"
                for p in range(nnorm)
            ]
            chain = calls[0]
            if nnorm == 3:
                chain = (
                    "%is0 = index.cmp eq, %qi, %pi0 : index\nscf.if %is0 {\n"
                    + calls[0]
                    + "} else {\n"
                    "%is1 = index.cmp eq, %qi, %pi1 : index\nscf.if %is1 {\n"
                    + calls[1]
                    + "} else {\n"
                    + calls[2]
                    + "}\n}\n"
                )
            out.append(
                f"""kernel.def export("g_ink_mix{n}_nq{nnorm}") @g_ink_mix{n}_nq{nnorm}(%tokens: index, %q_cnt: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %q_cnt, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch({", ".join(ps)}, {norm}) {{
  %lds_bytes = index.constant 9216 : offset
  %lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer
  func.call @g_ink_mix{n}_nq_mix({", ".join(names)}) : ({types})
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %qi = kernel.workgroup.id<y> : index
"""
                + "".join(
                    f"  %pi{p} = index.constant {p} : index\n" for p in range(nnorm - 1)
                )
                + chain
                + "  kernel.return\n}\n"
            )
    body, ps = kernel_func(src, "g_ink_mix2", "g_ink_mix2x2_body")
    out += [
        body,
        multi(
            "g_ink_mix2x2",
            "g_ink_mix2x2_body",
            ps,
            2,
            "z",
            "%c256 = index.constant 256 : index\n  kernel.launch.config workgroups(%tokens, %q_cnt, %cP) workgroup_size(%c256, %c1, %c1) : index",
            "%tokens: index, %q_cnt: index",
        ),
    ]
    return "\n".join(out)
