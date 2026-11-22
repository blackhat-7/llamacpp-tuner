"""Loop-controller kernels of a looped Inkling visit, three chains (attn, k, v) per launch, bit for bit:
- g_ink_concat_rms3 / g_ink_ctrl_apply3: the original kernel bodies as inline functions, grid y (z) picks the chain.
- mv f32 (K = 4608 controller down, K = 2304 router): mvfgen bodies, every load issued up front."""

import mvfgen
import mvgen


def kernel_func(src, name, fn):
    """The launch body of kernel `name` as `func.def inline @fn(<launch params>)`; returns (text, params)."""
    i = src.index(f'kernel.def export("{name}") @{name}(')
    j = src.index("} launch(", i)
    k = src.index(") {\n", j)
    params = src[j + len("} launch(") : k]
    end = src.index("\n}\n", k) + 3
    body = src[k + 4 : end].replace("kernel.return", "func.return")
    body = body.replace(
        "[lt(%c_idx_arg, %cp)]", "[range(%c_idx_arg, 0, 3), lt(%c_idx_arg, %cp)]"
    ).replace("[lt(%m_idx_arg, %mp)]", "[range(%m_idx_arg, 0, 3), lt(%m_idx_arg, %mp)]")
    body = body.replace(
        "%chan_off = index.cast %chan_off_arg : index to offset",
        "%chan_off_nn = index.assume %chan_off_arg [range(%chan_off_arg, 0, 1048575)] : index\n"
        "  %chan_off = index.cast %chan_off_nn : index to offset",
    ).replace("index.div %chan_off_arg, %c12", "index.div %chan_off_nn, %c12")
    return f"func.def inline @{fn}({params}) {{\n{body}", [
        p.strip() for p in params.split(",")
    ]


def multi(name, fn, params, nchain, dim, grid, cfg="%tokens: index"):
    """Kernel `name` taking params[0] (%tokens_arg) once and the other params nchain times (suffix _p), calling fn for
    the chain workgroup.id<dim> selects. grid: the launch.config workgroups text with %cP for the chain count."""
    first, rest = params[0], params[1:]
    names = [p.split(":")[0].strip() for p in rest]
    args = ", ".join(
        f"{n}_{p}: {t.split(':')[1].strip()}"
        for p in range(nchain)
        for n, t in zip(names, rest, strict=True)
    )
    tok = first.split(":")[0].strip()
    lines = [
        f"""kernel.def export("{name}") @{name}({cfg}) {{
  %c1 = index.constant 1 : index
  %cP = index.constant {nchain} : index
  {grid}
}} launch({first}, {args}) {{
  %py = kernel.workgroup.id<{dim}> : index"""
    ]
    types = ", ".join(["index"] + [p.split(":")[1].strip() for p in rest])

    def call(p):
        return f"func.call @{fn}({tok}, {', '.join(f'{n}_{p}' for n in names)}) : ({types})\n"

    def chain(p):
        if p == nchain - 1:
            return call(p)
        return f"%isp{p} = index.cmp eq, %py, %cp{p} : index\nscf.if %isp{p} {{\n{call(p)}}} else {{\n{chain(p + 1)}}}\n"

    lines += [f"  %cp{p} = index.constant {p} : index" for p in range(nchain - 1)]
    lines.append(chain(0) + "kernel.return\n}\n")
    return "\n".join(lines)


def module(src):
    out = []
    t, ps = kernel_func(src, "g_ink_concat_rms", "g_ink_concat_rms3_body")
    out += [
        t,
        multi(
            "g_ink_concat_rms3",
            "g_ink_concat_rms3_body",
            ps,
            3,
            "y",
            "%c256 = index.constant 256 : index\n  kernel.launch.config workgroups(%tokens, %cP, %c1) workgroup_size(%c256, %c1, %c1) : index",
        ),
    ]
    t, ps = kernel_func(src, "g_ink_ctrl_apply", "g_ink_ctrl_apply3_body")
    out += [
        t,
        multi(
            "g_ink_ctrl_apply3",
            "g_ink_ctrl_apply3_body",
            ps,
            3,
            "z",
            "%c64 = index.constant 64 : index\n  %cb = index.constant 36 : index\n  kernel.launch.config workgroups(%tokens, %cb, %cP) workgroup_size(%c64, %c1, %c1) : index",
        ),
    ]
    for k in (4608, 2304):
        w = mvfgen.waves(k)
        out.append(mvfgen.body(k, f"g_mvf_k{k}_body"))
        out.append(
            mvgen.kernel(
                k, f"g_mvf_k{k}_t4", f"g_mvf_k{k}_body", [0], 1, 32 * w, 16 * w
            )
        )
    out.append(
        mvgen.kernel(
            4608,
            "g_mvf_k4608_t4x3",
            "g_mvf_k4608_body",
            [0, 1, 2],
            1,
            32 * mvfgen.waves(4608),
            16 * mvfgen.waves(4608),
        )
    )
    return "\n".join(out)
