#!/usr/bin/env python3
"""dense_edit.py PACKAGE_DIR: speed up the dense part of Inkling-2-Mini's decode/verify steps on Loom by editing
PACKAGE_DIR/program/llm.loom in place (run it after engines/loom/build.sh's steps). Deterministic; refuses a file it
already edited; replaces the file instead of writing through a hard link.

Every new kernel keeps the original kernels' arithmetic, so results are bit-identical; the edits only regroup work
and issue loads earlier. In @llm_verify and @llm_advance:
- proj (2..4 tokens): per layer visit, one launch (g_ink_proj6_t4 / g_ink_proj8_t4) does the q, gate, k, v, rel
  Q8_0 projections and the bf16 attn_sink_proj (global visits: also attn_k/v_conv_proj): 6 (8) launches -> 1.
- norm (1..63 tokens): g_ink_rms_norm_plane + g_quant_q8 -> g_ink_rmsq1; three adjacent norms (looped and global
  visits) -> one g_ink_rmsq3.
- ctrl (2..4 tokens): a looped visit's three attention-side loop controllers (attn, k, v) run stage by stage:
  g_ink_concat_rms3, g_mvf_k4608_t4x3, g_ink_ctrl_apply3: 9 launches -> 3 (new scratch %cat_b/c, %z_b/c).
- mvf (2..63 tokens): g_mv_f32_k4608_t4_r1 (controller down) and g_mv_f32_k2304_t4_r1 (MoE router logits) ->
  g_mvf_k4608_t4 / g_mvf_k2304_t4, the same math with all loads issued before the first wait.
- mix (@llm_verify): g_ink_mixN + the norm right after it (rmsq1/rmsq3 on its output) -> g_ink_mixN_nqP on the
  t < 64 path (the row passes through workgroup memory); two adjacent same-shape g_ink_mix2 -> g_ink_mix2x2.
- sconv (only with --sconv, @llm_verify, 2..63 tokens): K/V short conv stat x2 -> g_ink_sconv_stat2_c1024; SWA ring
  store + K/V conv commits -> g_ink_kvring_commit2 (5 launches -> 2); post-attn/post-FFN conv stat + residual add
  -> g_ink_sconv_stat_add_c2304 (3 -> 2)."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrlgen  # noqa: E402
import mixgen  # noqa: E402
import projgen  # noqa: E402
import rmsqgen  # noqa: E402
import sconvgen  # noqa: E402

MARK = "// dense_edit applied"
COMMANDS = ("@llm_verify()", "@llm_advance()")

PRO = """  %c4_fz = index.constant 4 : index
  %le4_fz = index.cmp ule, %t, %c4_fz : index
  %true_fz = scalar.constant true : i1
  %multi_fz = scalar.xori %single, %true_fz : i1
  %fz = scalar.andi %le4_fz, %multi_fz : i1
  %nfz = scalar.xori %fz, %true_fz : i1
  %nw_ctrl3 = index.constant 192 : index
"""
SCRATCH = """  %cat_b = buffer.alloca<global> align(256) %cat_size : buffer
  %cat_c = buffer.alloca<global> align(256) %cat_size : buffer
  %z_b = buffer.alloca<global> align(256) %z_size : buffer
  %z_c = buffer.alloca<global> align(256) %z_size : buffer
"""
PARAM = re.compile(r"  (%\w+) = command\.parameter [^\n]*\n")


def gate(m, line, cond="%nfz"):
    ind = m[1]
    return f"{ind}scf.if {cond} {{\n{ind}  {line}\n{ind}}}\n"


def hoist(chunk, names, before):
    """Move the top-level command.parameter lines defining `names` that sit after offset `before` up to it."""
    late = [p for p in PARAM.finditer(chunk) if p[1] in names and p.start() >= before]
    for p in sorted(late, key=lambda p: p.start(), reverse=True):
        chunk = chunk[: p.start()] + chunk[p.end() :]
    return chunk[:before] + "".join(p[0] for p in late) + chunk[before:]


# ---- proj -------------------------------------------------------------------------------------------------------
T4 = re.compile(
    r"( *)kernel\.launch @g_mv_q8_0_k2304_t4_q8_r1\[%(?:nQ|nKV|nR), %t\]\(%\w+, %t, %zero_i32, (%\w+), (%\w+), (%\w+)\) : "
    r"\[index, index\]\(index, index, i32, buffer, (view<\d+xi8>), buffer\)\n"
)
BF = re.compile(
    r"( *)(kernel\.launch @g_ink_bf16_proj\[%t, %(?:nSINK|nDYN)\]\(%t, %\w+, %info, (%\w+), (%\w+), (%\w+)\) : [^\n]*)\n"
)


def edit_proj(chunk, step):
    t4 = {m[4]: m for m in T4.finditer(chunk)}
    assert sorted(t4) == sorted(["%q", "%gate", "%k_raw", "%v_raw", "%r_buf"]), list(t4)
    bfs = list(BF.finditer(chunk))
    assert [m[5] for m in bfs] in (["%sink_buf"], ["%sink_buf", "%dyn_k", "%dyn_v"]), [
        m[5] for m in bfs
    ]
    nseg = 6 if len(bfs) == 1 else 8
    q, g, k, v, r = (t4[o] for o in ["%q", "%gate", "%k_raw", "%v_raw", "%r_buf"])
    assert q[2] == g[2] == r[2]
    hs = [m[3] for m in bfs] + [bfs[0][3]] * (3 - len(bfs))
    ws = [m[4] for m in bfs] + [m[3] for m in (q, g, k, v, r)]
    ys = [m[5] for m in bfs] + [m[4] for m in (q, g, k, v, r)]
    wtypes = {m[4]: re.search(r"(view<\d+xi8>), buffer\)$", m[2])[1] for m in bfs}
    wtypes.update({m[3]: m[5] for m in (q, g, k, v, r)})
    name = f"g_ink_proj{nseg}_t4"
    types = ", ".join(
        ["index", "index"]
        + ["buffer"] * 7
        + [wtypes[w] for w in ws]
        + ["buffer"] * nseg
    )
    nw = f"%nw_proj{step}"
    launch = (
        f"{q[1]}{nw} = index.constant {projgen.workgroups(nseg)} : index\n"
        f"{q[1]}kernel.launch @{name}[{nw}, %t]({nw}, %t, {q[2]}, {k[2]}, {v[2]}, {', '.join(hs)}, %info, "
        f"{', '.join(ws)}, {', '.join(ys)}) : [index, index]({types})\n"
    )
    edits = [(q.start(), q.end(), launch)] + [
        (m.start(), m.end(), "") for m in (g, k, v, r)
    ]
    edits += [(m.start(), m.end(), gate(m, m[2])) for m in bfs]
    for s, e, rep in sorted(edits, reverse=True):
        chunk = chunk[:s] + rep + chunk[e:]
    wq_def = re.search(rf"  {re.escape(q[3])} = command\.parameter [^\n]*\n", chunk)
    return hoist(chunk, set(ws), wq_def.end())


# ---- norm -------------------------------------------------------------------------------------------------------
NQ = re.compile(
    r"    kernel\.launch @g_ink_rms_norm_plane\[%t\]\(%t, (%\w+), (%\w+), (%\w+), (%\w+), (%\w+)\) : "
    r"\[index\]\(index, index, index, buffer, view<2304xf32>, buffer\)\n"
    r"    kernel\.launch @g_quant_q8\[%t, %nH\]\(%t, %nH, \5, (%\w+)\) : \[index, index\]\(index, index, buffer, buffer\)\n"
)
BETWEEN = re.compile(
    r"  \}\n  scf\.if %gemm \{\n    kernel\.launch @g_ink_rms_norm_plane_f16\[[^\n]*\n  \} else \{\n"
)
NQ_T = "index, index, buffer, view<2304xf32>, buffer, buffer"


def edit_norms(chunk):
    ms = list(NQ.finditer(chunk))
    edits, i = [], 0
    while i < len(ms):
        grp = ms[i : i + 3]
        if len(grp) == 3 and all(
            BETWEEN.fullmatch(chunk, a.end(), b.start())
            for a, b in zip(grp, grp[1:], strict=False)
        ):
            args = ", ".join(", ".join(m.groups()) for m in grp)
            edits += [(m.start(), m.end(), "") for m in grp[:2]]
            edits.append(
                (
                    grp[2].start(),
                    grp[2].end(),
                    f"    kernel.launch @g_ink_rmsq3[%t](%t, {args}) : [index](index, {', '.join([NQ_T] * 3)})\n",
                )
            )
            i += 3
        else:
            m = ms[i]
            edits.append(
                (
                    m.start(),
                    m.end(),
                    f"    kernel.launch @g_ink_rmsq1[%t](%t, {', '.join(m.groups())}) : [index](index, {NQ_T})\n",
                )
            )
            i += 1
    for s, e, rep in sorted(edits, reverse=True):
        chunk = chunk[:s] + rep + chunk[e:]
    return chunk


# ---- ctrl -------------------------------------------------------------------------------------------------------
CAT = re.compile(
    r"( *)(kernel\.launch @g_ink_concat_rms\[%t\]\(%t, (%\w+), (%\w+), (%\w+), (%\w+), (%\w+), (%\w+), %cat\) : [^\n]*)\n"
)
MV4 = re.compile(
    r"( *)(kernel\.launch @g_mv_f32_k4608_t4_r1\[%nRANK, %t\]\(%nRANK, %t, %zero_i32, %cat, (%\w+), %z_buf\) : [^\n]*)\n"
)
APP = re.compile(
    r"( *)(kernel\.launch @g_ink_ctrl_apply\[%t\]\(%t, (%\w+), (%\w+), (%\w+), (%\w+), (%\w+), (%\w+), %z_buf, (%\w+), (%\w+), (%\w+)\) : [^\n]*)\n"
)
CAT_T = "index, index, index, index, buffer, buffer, buffer"
APP_T = "index, index, index, index, buffer, buffer, buffer, view<589824xi8>, view<589824xi8>, buffer"


def edit_ctrl(chunk):
    cats, mvs, apps = (
        list(CAT.finditer(chunk)),
        list(MV4.finditer(chunk)),
        list(APP.finditer(chunk)),
    )
    if len(cats) < 4 or any(
        m[1] != "  " for m in cats
    ):  # not a looped visit, or not top level (@llm_advance)
        return chunk
    assert len(cats) == len(mvs) == len(apps) == 4, (len(cats), len(mvs), len(apps))
    assert [m[11] for m in apps[:3]] == ["%ctrl_out0", "%ctrl_out1", "%ctrl_out2"]
    cats, mvs, apps = cats[:3], mvs[:3], apps[:3]
    assert cats[0].start() < mvs[0].start() < apps[0].start() < cats[1].start()
    cbuf, zbuf = ["%cat", "%cat_b", "%cat_c"], ["%z_buf", "%z_b", "%z_c"]
    ca = ", ".join(
        f"{c[3]}, {c[4]}, {c[5]}, {c[6]}, {c[7]}, {c[8]}, {cb}"
        for c, cb in zip(cats, cbuf, strict=True)
    )
    mv = f"%nw_ctrl3, %t, %zero_i32, %nRANK, %nRANK, %nRANK, {', '.join(cbuf)}, {', '.join(m[3] for m in mvs)}, {', '.join(zbuf)}"
    ap = ", ".join(
        f"{a[3]}, {a[4]}, {a[5]}, {a[6]}, {a[7]}, {a[8]}, {zb}, {a[9]}, {a[10]}, {a[11]}"
        for a, zb in zip(apps, zbuf, strict=True)
    )
    fused = (
        "  scf.if %fz {\n"
        f"    kernel.launch @g_ink_concat_rms3[%t](%t, {ca}) : [index](index, {', '.join([CAT_T] * 3)})\n"
        f"    kernel.launch @g_mvf_k4608_t4x3[%nw_ctrl3, %t]({mv}) : [index, index](index, index, i32, index, index, index, "
        f"buffer, buffer, buffer, view<1179648xi8>, view<1179648xi8>, view<1179648xi8>, buffer, buffer, buffer)\n"
        f"    kernel.launch @g_ink_ctrl_apply3[%t](%t, {ap}) : [index](index, {', '.join([APP_T] * 3)})\n"
        "  }\n"
    )
    edits = [(m.start(), m.end(), gate(m, m[2])) for m in cats + mvs + apps]
    for s, e, rep in sorted(edits, reverse=True):
        chunk = chunk[:s] + rep + chunk[e:]
    at = chunk.rindex("  scf.if %nfz {\n", 0, CAT.search(chunk).start())
    names = {m[3] for m in mvs} | {a[9] for a in apps} | {a[10] for a in apps}
    chunk = hoist(chunk, names, at)
    at = chunk.rindex("  scf.if %nfz {\n", 0, CAT.search(chunk).start())
    return chunk[:at] + fused + chunk[at:]


# ---- sconv (optional, --sconv) ------------------------------------------------------------------------------------
KV5 = re.compile(
    r"( *)kernel\.launch @g_ink_sconv_stat_c1024\[%t\]\(%t, %spans, (%\w+), %off_k, %metadata, %info, %origins, %arena, %k_raw, (%\w+), %k_conv\) : [^\n]*\n"
    r" *kernel\.launch @g_ink_sconv_commit_cap_c1024\[%spans\]\(%t, %spans, \2, %off_k, %metadata, %origins, %arena, %k_raw, %sc_cap\) : [^\n]*\n"
    r" *kernel\.launch @g_ink_sconv_stat_c1024\[%t\]\(%t, %spans, \2, %off_v, %metadata, %info, %origins, %arena, %v_raw, (%\w+), %v_conv\) : [^\n]*\n"
    r" *kernel\.launch @g_ink_sconv_commit_cap_c1024\[%spans\]\(%t, %spans, \2, %off_v, %metadata, %origins, %arena, %v_raw, %sc_cap\) : [^\n]*\n"
    r" *kernel\.launch @g_ink_kv_prep_store_swa_ring\[%t\]\(%t, %spans, (%\w+), %metadata, %info, %origins, %k_conv, %v_conv, %arena\) : [^\n]*\n"
)
SA = re.compile(
    r"( *)kernel\.launch @g_ink_sconv_stat_c2304\[%t\]\(%t, %spans, (%\w+), (%\w+), %metadata, %info, %origins, %arena, (%\w+), (%\w+), %y2\) : "
    r"\[index\]\(index, index, index, index, buffer, buffer, buffer, buffer, buffer, (view<\d+xi8>), buffer\)\n"
    r"( *kernel\.launch @g_ink_sconv_commit_cap_c2304\[%spans\][^\n]*\n)"
    r" *kernel\.launch @g_ink_add\[%t\]\(%t, (%\w+), %y2, (%\w+)\) : [^\n]*\n"
)
ST_T = "index, index, index, buffer, buffer, buffer, buffer, buffer, view<16384xi8>, buffer"


def edit_sconv(chunk):
    def kv(m):
        i, slot, kc, vc, swa = m.groups()
        return (
            f"{i}kernel.launch @g_ink_sconv_stat2_c1024[%t](%t, %spans, {slot}, %off_k, %metadata, %info, %origins, %arena, %k_raw, {kc}, %k_conv, "
            f"%spans, {slot}, %off_v, %metadata, %info, %origins, %arena, %v_raw, {vc}, %v_conv) : [index](index, {ST_T}, {ST_T})\n"
            f"{i}kernel.launch @g_ink_kvring_commit2[%t, %spans](%t, %spans, {swa}, %metadata, %info, %origins, %k_conv, %v_conv, %arena, "
            f"{slot}, %off_k, %off_v, %k_raw, %v_raw, %sc_cap) : [index, index](index, index, index, buffer, buffer, buffer, buffer, buffer, "
            f"buffer, index, index, index, buffer, buffer, buffer)\n"
        )

    def sa(m):
        i, slot, off, cur, w, wt, commit, a, out = m.groups()
        return (
            f"{i}kernel.launch @g_ink_sconv_stat_add_c2304[%t](%t, %spans, {slot}, {off}, %metadata, %info, %origins, %arena, {cur}, {w}, {a}, {out}) : "
            f"[index](index, index, index, index, buffer, buffer, buffer, buffer, buffer, {wt}, buffer, buffer)\n{commit}"
        )

    return SA.sub(sa, KV5.sub(kv, chunk))


# ---- mix (@llm_verify, any t) ---------------------------------------------------------------------------------------
MIX = re.compile(
    r"  kernel\.launch @g_ink_mix([2-5])\[%t, (%\w+)\]\((%t, \2, %\w+, %\w+, %\w+, %\w+, %\w+, %\w+, (%\w+))\) : "
    r"\[index, index\]\(([^)]*)\)\n"
)
NEXT = re.compile(r"kernel\.launch @(\w+)\[")
RQ = re.compile(r"    kernel\.launch @g_ink_rmsq([13])\[%t\]\(%t, ([^)]*)\) : [^\n]*\n")
PAIR = re.compile(
    r"  kernel\.launch @g_ink_mix2\[%t, (%\w+)\]\(%t, ([^)]*)\) : \[index, index\]\(index, ([^)]*)\)\n"
    r"  kernel\.launch @g_ink_mix2\[%t, \1\]\(%t, ([^)]*)\) : \[index, index\]\(index, \3\)\n"
)


def edit_mix(chunk):
    """mixN then its norm (only f16 prefill norms and parameters between) -> g_ink_mixN_nqP on the t < 64 path;
    two adjacent same-shape g_ink_mix2 -> g_ink_mix2x2."""

    def pair(m):
        a, b = m[2].split(", "), m[4].split(", ")
        if a[:6] != b[:6]:
            return m[0]
        return f"  kernel.launch @g_ink_mix2x2[%t, {m[1]}](%t, {m[2]}, {m[4]}) : [index, index](index, {m[3]}, {m[3]})\n"

    chunk = PAIR.sub(pair, chunk)
    edits = []
    for m in MIX.finditer(chunk):
        n, q, args, out, types = m.groups()
        pos = m.end()
        while (k := NEXT.search(chunk, pos)) and k[1] == "g_ink_rms_norm_plane_f16":
            pos = k.end()
        r = RQ.search(chunk, m.end())
        if (
            not k
            or k[1] not in ("g_ink_rmsq1", "g_ink_rmsq3")
            or not r
            or r.start() > k.start()
        ):
            continue
        nnorm = int(r[1])
        a = [x.strip() for x in r[2].split(",")]
        planes = [a[i : i + 6] for i in range(0, len(a), 6)]
        if q != {1: "%c1", 3: "%c3"}[nnorm] or any(
            pl != [q, f"%c{i}", out] for i, pl in enumerate(p[:3] for p in planes)
        ):
            continue
        norm = ", ".join(", ".join(p[3:]) for p in planes)
        nt = ", ".join(["view<2304xf32>, buffer, buffer"] * nnorm)
        edits.append(
            (m.start(), m.end(), f"  scf.if %gemm {{\n  {m[0].strip()}\n  }}\n")
        )
        edits.append(
            (
                r.start(),
                r.end(),
                f"    kernel.launch @g_ink_mix{n}_nq{nnorm}[%t, {q}]({args}, {norm}) : [index, index]({types}, {nt})\n",
            )
        )
    for s_, e, rep in sorted(edits, reverse=True):
        chunk = chunk[:s_] + rep + chunk[e:]
    return chunk


# ---- mvf --------------------------------------------------------------------------------------------------------
MVF = re.compile(
    r"kernel\.launch @g_mv_f32_k(4608|2304)_t4_r1\[(%\w+), %t\]\(\2, %t, %zero_i32, (%\w+), (%\w+), (%\w+)\) : "
    r"\[index, index\]\(index, index, i32, buffer, (view<\d+xi8>), buffer\)"
)


def edit_mvf(body):
    return MVF.sub(
        lambda m: f"kernel.launch @g_mvf_k{m[1]}_t4[{m[2]}, %t]({m[2]}, %t, %zero_i32, {m[2]}, {m[3]}, {m[4]}, {m[5]}) : "
        f"[index, index](index, index, i32, index, buffer, {m[6]}, buffer)",
        body,
    )


def edit_command(body, sconv=False, verify=False):
    head, *steps = re.split(r"(?=  // ==== step \d+ )", body)
    at = head.index("\n", head.index("%huge = ")) + 1
    head = head[:at] + PRO + head[at:]
    at = head.index("\n", head.index("  %z_buf = buffer.alloca")) + 1
    head = head[:at] + SCRATCH + head[at:]
    out = [head]
    for s in steps:
        step = int(re.match(r"  // ==== step (\d+)", s)[1])
        s = edit_ctrl(edit_norms(edit_proj(s, step)))
        s = edit_mix(s) if verify else s
        out.append(edit_sconv(s) if sconv else s)
    return edit_mvf("".join(out))


def main(pkg, sconv=False):
    path = Path(pkg) / "program" / "llm.loom"
    text = path.read_text()
    if MARK in text:
        sys.exit(f"{path} is already edited")
    kernels = "\n".join(
        [
            projgen.module(("g_ink_proj6_t4", "g_ink_proj8_t4")),
            rmsqgen.module("g_ink_rmsq"),
            ctrlgen.module(text),
            mixgen.module(text, rmsqgen.body("g_ink_rmsq_mixsrc")),
        ]
        + ([sconvgen.module(text)] if sconv else [])
    )
    for name in COMMANDS:
        s = text.index(f"command.def public target(@device) {name}")
        e = text.index("\ncommand.def ", s + 1)
        text = (
            text[:s]
            + edit_command(
                text[s:e], sconv and name == "@llm_verify()", name == "@llm_verify()"
            )
            + text[e:]
        )
    text = text.rstrip("\n") + f"\n\n{MARK}\n" + kernels
    path.unlink()  # the package may be a hard-link copy: never write through to the original
    path.write_text(text)


if __name__ == "__main__":
    main(sys.argv[1], "--sconv" in sys.argv[2:])
