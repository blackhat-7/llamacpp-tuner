"""Prefill edits for the Inkling-2-Mini Loom package, applied in place to PKG/program/*.loom.

usage: prefill.py PKG [--no-split]

1. prepare.loom: build drafter proposal stages only up to `rows` spans. The stages for 2..1024 spans could never
   run with rows=1 but reserved ~1.5 GiB of workspace at prefill_capacity >= 768, which kept the chunk at 512.
2. llm.loom: split-K global-layer flash attention. g_ink_attention_flash2_ts2_sk is g_ink_attention_flash2_ts2 run
   over NS key ranges (grid z), writing unnormalized partials into the listed fallback's 16-split scratch
   (fb_acc/fb_ml, indexed by token, so flash and fallback tokens never collide); g_ink_attn_combine_flash merges
   them with the sink and the output gate. Only the prefill (%gemm) launches of ts2 change.
Deterministic and idempotent: a second run changes nothing.
"""

import re
import sys
from pathlib import Path

ATTN = __import__("os").environ.get(
    "PREFILL_ATTN", "sk"
)  # sk, or pf (sk + register prefetch)
QL = int(
    __import__("os").environ.get("PREFILL_QL", "8")
)  # pf: Q fragments kept in workgroup memory
NS = int(
    __import__("os").environ.get("PREFILL_NS", "4")
)  # key splits per tile, 2..16 (the scratch holds 16)


def edit_prepare(text: str) -> str:
    old = "    %span32 = scalar.trunci %span_capacity : i64 to i32\n"
    new = (
        "    // At most `rows` spans propose in one epoch; bigger proposal stages only reserve workspace.\n"
        "    %propose_spans = scalar.minsi %span_capacity, %rows : i64\n"
        "    %span32 = scalar.trunci %propose_spans : i64 to i32\n"
    )
    if "%propose_spans" in text:
        return text
    assert text.count(old) == 1, "prepare.loom: proposal level anchor not found"
    return text.replace(old, new)


def kernel_text(lines: list[str], name: str) -> list[str]:
    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith(f'kernel.def export("{name}")')
    )
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i].startswith("kernel.def")
    )
    while not lines[end - 1].startswith("}"):
        end -= 1
    return lines[start:end]


def split_kernel(ts2: list[str]) -> list[str]:
    src = "\n".join(ts2)
    src = src.replace(
        'export("g_ink_attention_flash2_ts2") @g_ink_attention_flash2_ts2(',
        'export("g_ink_attention_flash2_ts2_sk") @g_ink_attention_flash2_ts2_sk(',
    )
    src = src.replace(
        "  kernel.launch.config workgroups(%tiles_count, %ckv, %c1)",
        f"  %cNS = index.constant {NS} : index\n"
        "  kernel.launch.config workgroups(%tiles_count, %ckv, %cNS)",
    )
    src = src.replace(
        "%arena: buffer, %out: buffer) {",
        "%arena: buffer, %part_acc: buffer, %part_ml: buffer) {",
    )
    # The split covers key tiles [kt_lo, kt_hi); partials start empty (the combine adds the sink).
    src = src.replace(
        "  %key_tiles = index.div %key_tiles_round, %ckt : index\n",
        "  %key_tiles = index.div %key_tiles_round, %ckt : index\n"
        f"  %sk_n = index.constant {NS} : index\n"
        f"  %sk_n1 = index.constant {NS - 1} : index\n"
        "  %sk_id = kernel.workgroup.id<z> : index\n"
        f"  %sk = index.assume %sk_id [range(%sk_id, 0, {NS - 1})] : index\n"
        "  %sk_round = index.add %key_tiles, %sk_n1 : index\n"
        "  %sk_per = index.div %sk_round, %sk_n : index\n"
        "  %sk_lo_raw = index.mul %sk, %sk_per : index\n"
        "  %kt_lo = index.min %sk_lo_raw, %key_tiles : index\n"
        "  %sk_hi_raw = index.add %kt_lo, %sk_per : index\n"
        "  %kt_hi = index.min %sk_hi_raw, %key_tiles : index\n",
    )
    src = src.replace(
        "  %m_init = scalar.addf %sink_raw, %zero_f : f32\n",
        "  %m_init = scalar.constant -3.0e38 : f32\n  %l_init = scalar.constant 0.0 : f32\n",
    )
    src = src.replace(
        "  %sink_raw = view.load %sinks_view[%q_token, %head] : view<[%tokens]x24xf32> -> f32\n",
        "",
    )
    src = src.replace(
        "scf.for %kt = [%c0 to %key_tiles step %c1](%m = %m_init : f32, %l = %one_init : f32,",
        "scf.for %kt = [%kt_lo to %kt_hi step %c1](%m = %m_init : f32, %l = %l_init : f32,",
    )
    # Epilogue: store O (unnormalized), m and l instead of the gated output.
    head, tail = src.split("  %one_f = scalar.constant 1.0 : f32\n")
    store = [
        "  %cst = index.constant 49152 : index",  # 24 heads x 16 splits x 128 dims per token
        "  %acc_layout = encoding.layout.strided [%c1, %cst] : encoding<layout>",
        "  %c16s = index.constant 16 : index",
        "  %th_a = index.mul %first_token, %cqh_sk : index",
        "  %th = index.add %th_a, %head : index",
        "  %ths_a = index.mul %th, %c16s : index",
        "  %ths = index.add %ths_a, %sk : index",
        "  %cdim_sk = index.constant 128 : index",
        "  %acc_first = index.mul %ths, %cdim_sk : index",
        "  %four_sk = index.constant 4 : offset",
        "  %acc_offset = index.scale %acc_first, %four_sk : index, offset -> offset",
        "  %acc_view = buffer.view %part_acc[%acc_offset] : buffer -> view<128x16xf32, %acc_layout>",
        "  %ml_view = buffer.view %part_ml[%origin] : buffer -> view<[%tokens]x24x16x2xf32>",
        "  scf.if %mine {",
    ]
    store.insert(0, "  %cqh_sk = index.constant 24 : index")
    for i, d in enumerate(
        ["%d0r", "%d16r", "%d32r", "%d48r", "%d64r", "%d80r", "%d96r", "%d112r"]
    ):
        store.append(
            f"  vector.fragment.store<result> %o{i}, %acc_view[{d}, %c0] shape [%c16, %c16] : "
            "vector<8xf32>, view<128x16xf32, %acc_layout>"
        )
    store += [
        "  %lo_half = scalar.xori %hi_half, %true_top : i1",
        "  scf.if %lo_half {",
        "    view.store %m_final, %ml_view[%q_token, %head, %sk, %c0] : f32, view<[%tokens]x24x16x2xf32>",
        "    view.store %l_final, %ml_view[%q_token, %head, %sk, %c1] : f32, view<[%tokens]x24x16x2xf32>",
        "  }",
        "  }",
        "  }",
        "  kernel.return",
        "}",
    ]
    return (head + "\n".join(store)).split("\n")


RAW = ["vector<16xi8>", "f16", "vector<16xi8>", "f16"]


def fetch(p: str, pos: str, live: str, out: str) -> str:
    """Raw kv8 codes and scales of key POS for staging pass P (zeros when not LIVE)."""
    n = [f"%{out}k_{p}", f"%{out}ks_{p}", f"%{out}v_{p}", f"%{out}vs_{p}"]
    return f"""    %{out}pos_{p} = index.add {pos}, %st_key_{p} : index
    %{out}in_{p} = index.cmp ule, %{out}pos_{p}, %pos_last : index
    %{out}ok_{p} = scalar.andi %{out}in_{p}, {live} : i1
    {", ".join(n)} = scf.if %{out}ok_{p} -> ({", ".join(RAW)}) {{
      %{out}kp_{p} = index.assume %{out}pos_{p} [range(%{out}pos_{p}, 0, 1048575)] : index
      %{out}ko_{p} = func.call @g_kv8_offset(%origins, %row, %{out}kp_{p}, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %{out}vo_{p} = index.add %{out}ko_{p}, %plane_stride : offset
      %{out}kcv_{p} = buffer.view %arena[%{out}ko_{p}] : buffer -> view<8x128xi8>
      %{out}vcv_{p} = buffer.view %arena[%{out}vo_{p}] : buffer -> view<8x128xi8>
      %{out}kc_{p} = vector.load %{out}kcv_{p}[%kv, %st_dim_{p}] : view<8x128xi8> -> vector<16xi8>
      %{out}vc_{p} = vector.load %{out}vcv_{p}[%kv, %st_dim_{p}] : view<8x128xi8> -> vector<16xi8>
      %{out}ksc_{p} = func.call @g_kv8_scale(%arena, %{out}ko_{p}, %sc_idx_{p}) : (buffer, offset, index) -> (f16)
      %{out}vsc_{p} = func.call @g_kv8_scale(%arena, %{out}vo_{p}, %sc_idx_{p}) : (buffer, offset, index) -> (f16)
      scf.yield %{out}kc_{p}, %{out}ksc_{p}, %{out}vc_{p}, %{out}vsc_{p} : {", ".join(RAW)}
    }} else {{
      scf.yield %raw_z8, %raw_zh, %raw_z8, %raw_zh : {", ".join(RAW)}
    }}
"""


def stage(p: str, ind: str) -> str:
    """Dequantize pass P's prefetched codes into K and V^T in workgroup memory (as g_kv8_load16 does)."""
    lines = [
        f"%kf_{p} = vector.sitofp %pk_{p} : vector<16xi8> to vector<16xf32>",
        f"%ksf_{p} = scalar.extf %pks_{p} : f16 to f32",
        f"%ksv_{p} = vector.splat %ksf_{p} : vector<16xf32>",
        f"%kx_{p} = vector.mulf %kf_{p}, %ksv_{p} : vector<16xf32>",
        f"%kvals_{p} = vector.fptrunc %kx_{p} : vector<16xf32> to vector<16xf16>",
        f"%vf_{p} = vector.sitofp %pv_{p} : vector<16xi8> to vector<16xf32>",
        f"%vsf_{p} = scalar.extf %pvs_{p} : f16 to f32",
        f"%vsv_{p} = vector.splat %vsf_{p} : vector<16xf32>",
        f"%vx_{p} = vector.mulf %vf_{p}, %vsv_{p} : vector<16xf32>",
        f"%vvals_{p} = vector.fptrunc %vx_{p} : vector<16xf32> to vector<16xf16>",
        f"vector.store %kvals_{p}, %k_view[%st_key_{p}, %st_dim_{p}] : vector<16xf16>, view<32x136xf16>",
        f"scf.for %j_{p} = [%c0 to %cdpt step %c1] unroll {{",
        f"  %vj_{p} = vector.extract %vvals_{p}[%j_{p}] : vector<16xf16> -> f16",
        f"  %vt_row_val_{p} = index.add %st_dim_{p}, %j_{p} : index",
        f"  %vt_row_{p} = index.assume %vt_row_val_{p} [range(%vt_row_val_{p}, 0, 127)] : index",
        f"  view.store %vj_{p}, %vt_view[%vt_row_{p}, %st_key_{p}] : f16, view<128x40xf16>",
        "}",
    ]
    return "".join(ind + line + "\n" for line in lines)


def prefetch_kernel(sk: list[str]) -> list[str]:
    """ts2_sk with the next key tile's kv8 loads issued before the current tile's math (register prefetch)."""
    src = "\n".join(sk).replace(
        '"g_ink_attention_flash2_ts2_sk") @g_ink_attention_flash2_ts2_sk(',
        '"g_ink_attention_flash2_ts2_pf") @g_ink_attention_flash2_ts2_pf(',
    )
    loop = src.index("  %m_final, %l_final, %o0,")
    body = src.index(
        "    %key_pos_p0 = index.add %tile_first, %st_key_p0 : index\n", loop
    )
    bar = "    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n"
    body_end = src.index(bar, body) + len(bar)
    # Prologue: scale indices, zero values, the first tile's raw loads.
    pro = [
        "  %raw_z8 = vector.splati 0 : vector<16xi8>",
        "  %raw_zh = scalar.constant 0.0 : f16",
        "  %c4_sc = index.constant 4 : index",
        "  %c32_sc = index.constant 32 : index",
    ]
    for p in ("p0", "p1"):
        pro += [
            f"  %sc_blk_{p} = index.div %st_dim_{p}, %c32_sc : index",
            f"  %sc_idx_{p} = index.madd %kv, %c4_sc, %sc_blk_{p} : index",
        ]
    pro += [
        "  %first_live = index.cmp ult, %kt_lo, %kt_hi : index",
        "  %first_tile = index.madd %kt_lo, %ckt, %key_low : index",
        "  %first_live_p1 = scalar.andi %first_live, %st_live_p1 : i1",
    ]
    prologue = (
        "\n".join(pro)
        + "\n"
        + fetch("p0", "%first_tile", "%first_live", "i0")
        + fetch("p1", "%first_tile", "%first_live_p1", "i0")
    )
    names = [f"%{a}_{p}" for p in ("p0", "p1") for a in ("k", "ks", "v", "vs")]
    head = src[loop:body]
    hdr_end = head.index("\n") + 1
    hdr = head[:hdr_end]
    iters = ", ".join(
        f"%p{n[1:]} = %i0{n[1:]} : {t}" for n, t in zip(names, RAW * 2, strict=True)
    )
    hdr = hdr.replace(" : vector<8xf32>) -> (", f" : vector<8xf32>, {iters}) -> (", 1)
    hdr = hdr.replace(
        "vector<8xf32>) {\n", "vector<8xf32>, " + ", ".join(RAW * 2) + ") {\n", 1
    )
    hdr = hdr.replace(
        "%o7 = scf.for",
        "%o7, " + ", ".join(f"%x{n[1:]}" for n in names) + " = scf.for",
        1,
    )
    new_body = (
        stage("p0", "    ")
        + "    scf.if %st_live_p1 {\n"
        + stage("p1", "      ")
        + "    }\n"
        + bar
        + "    %next_tile = index.add %tile_first, %ckt : index\n"
        + "    %kt_next = index.add %kt, %c1 : index\n"
        + "    %next_live = index.cmp ult, %kt_next, %kt_hi : index\n"
        + "    %next_live_p1 = scalar.andi %next_live, %st_live_p1 : i1\n"
        + fetch("p0", "%next_tile", "%next_live", "n")
        + fetch("p1", "%next_tile", "%next_live_p1", "n")
    )
    rest = src[body_end:]
    y = re.search(
        r"    scf\.yield %m_new, %l_new, %on0, %on1, %on2, %on3, %on4, %on5, %on6, %on7 : ([^\n]*)\n",
        rest,
    )
    rest = (
        rest[: y.start()]
        + "    scf.yield %m_new, %l_new, %on0, %on1, %on2, %on3, %on4, %on5, %on6, %on7, "
        + ", ".join(f"%n{n[1:]}" for n in names)
        + " : "
        + y.group(1)
        + ", "
        + ", ".join(RAW * 2)
        + "\n"
        + rest[y.end() :]
    )
    out = src[:loop] + prologue + hdr + head[hdr_end:] + new_body + rest
    # Q^T lives in workgroup memory (f16) instead of 64 VGPRs per lane; each MMA reloads its fragment.
    q_def = "  %qf7 = vector.fragment<rhs> %qh7 shape [%c16, %c16] : vector<16xf16>\n"
    cols = 16 * QL + 8  # QL fragments of 16 dims, padded against bank conflicts
    q_lds = [
        f"  %q_lds_bytes = index.constant {96 * cols * 2} : offset",  # 96 query rows (6 waves x 16)
        "  %q_lds = buffer.alloca<workgroup> align(16) %q_lds_bytes : buffer",
        f"  %q_lds_view = buffer.view %q_lds[%origin] : buffer -> view<96x{cols}xf16>",
        "  %q_row_val = index.madd %wave_all, %c16, %lane_col : index",
        "  %q_row = index.assume %q_row_val [range(%q_row_val, 0, 95)] : index",
    ]
    q_lds += [
        f"  vector.store %qh{i}, %q_lds_view[%q_row, {d}] : vector<16xf16>, view<96x{cols}xf16>"
        for i, d in enumerate(
            ["%d0", "%d1", "%d2", "%d3", "%d4", "%d5", "%d6", "%d7"][:QL]
        )
    ]
    out = out.replace(q_def, q_def + "\n".join(q_lds) + "\n", 1)
    for kb in ("0", "1"):
        for i, d in enumerate(
            ["%d0", "%d1", "%d2", "%d3", "%d4", "%d5", "%d6", "%d7"][:QL]
        ):
            mma = f"    %s{kb}_{i} = vector.mma %kl{kb}_{i}, %qf{i}, "
            assert out.count(mma) == 1, mma
            load = (
                f"    %qv{kb}_{i} = vector.load %q_lds_view[%q_row, {d}] : view<96x{cols}xf16> -> vector<16xf16>\n"
                f"    %ql{kb}_{i} = vector.fragment<rhs> %qv{kb}_{i} shape [%c16, %c16] : vector<16xf16>\n"
            )
            out = out.replace(
                mma, load + f"    %s{kb}_{i} = vector.mma %kl{kb}_{i}, %ql{kb}_{i}, "
            )
    return out.split("\n")


COMBINE = f"""kernel.def export("g_ink_attn_combine_flash") @g_ink_attn_combine_flash(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %cqh = index.constant 24 : index
  %tiles_count = index.div %tokens, %c16 : index
  kernel.launch.config workgroups(%tiles_count, %cqh, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %tiles: buffer, %part_acc: buffer, %part_ml: buffer, %sinks: buffer, %gate: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 16, 8192), multiple_of(%tokens_arg, 16)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %cS = index.constant {NS} : index
  %zero4 = vector.splatf 0.0 : vector<4xf32>
  %tile_count = index.div %tokens, %c16 : index
  %tile_id = kernel.workgroup.id<x> : index
  %tile = index.assume %tile_id [lt(%tile_id, %tile_count)] : index
  %tiles_view = buffer.view %tiles[%zero] : buffer -> view<[%tokens]xi32>
  %listed_count32 = view.load %tiles_view[%c0] : view<[%tokens]xi32> -> i32
  %listed_count = index.cast %listed_count32 : i32 to index
  %listed = index.cmp ult, %tile, %listed_count : index
  scf.if %listed {{
    %slot_value = index.add %tile, %c1 : index
    %slot = index.assume %slot_value [lt(%slot_value, %tokens)] : index
    %entry32 = view.load %tiles_view[%slot] : view<[%tokens]xi32> -> i32
    %entry_bound = scalar.assume %entry32 [range(%entry32, 0, 8176)] : i32
    %entry = index.cast %entry_bound : i32 to index
    %hd_id = kernel.workgroup.id<y> : index
    %hd = index.assume %hd_id [range(%hd_id, 0, 23)] : index
    %lane_raw = kernel.workitem.id<x> : index
    %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
    %ml_v = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x24x16x2xf32>
    %acc_v = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x24x16x32x4xf32>
    %sv = buffer.view %sinks[%zero] : buffer -> view<[%tokens]x24xf32>
    %gv = buffer.view %gate[%zero] : buffer -> view<[%tokens]x24x32x4xf32>
    %ov = buffer.view %out[%zero] : buffer -> view<[%tokens]x24x32x4xf32>
    scf.for %i = [%c0 to %c16 step %c1] {{
      %token_value = index.add %entry, %i : index
      %token = index.assume %token_value [lt(%token_value, %tokens)] : index
      %sink = view.load %sv[%token, %hd] : view<[%tokens]x24xf32> -> f32
      %mx = scf.for %s = [%c0 to %cS step %c1](%m = %sink : f32) -> (f32) {{
        %sb = index.assume %s [range(%s, 0, 15)] : index
        %ms = view.load %ml_v[%token, %hd, %sb, 0] : view<[%tokens]x24x16x2xf32> -> f32
        %nm = scalar.maxnumf %m, %ms : f32
        scf.yield %nm : f32
      }}
      %s_sh = scalar.subf %sink, %mx : f32
      %d0 = scalar.expf<afn> %s_sh : f32
      %df, %af = scf.for %s = [%c0 to %cS step %c1](%d = %d0 : f32, %a = %zero4 : vector<4xf32>) -> (f32, vector<4xf32>) {{
        %sb = index.assume %s [range(%s, 0, 15)] : index
        %ms = view.load %ml_v[%token, %hd, %sb, 0] : view<[%tokens]x24x16x2xf32> -> f32
        %ls = view.load %ml_v[%token, %hd, %sb, 1] : view<[%tokens]x24x16x2xf32> -> f32
        %os = vector.load %acc_v[%token, %hd, %sb, %lane, %c0] : view<[%tokens]x24x16x32x4xf32> -> vector<4xf32>
        %sh = scalar.subf %ms, %mx : f32
        %w = scalar.expf<afn> %sh : f32
        %nd = scalar.fmaf %ls, %w, %d : f32
        %wv = vector.splat %w : vector<4xf32>
        %na = vector.fmaf %os, %wv, %a : vector<4xf32>
        scf.yield %nd, %na : f32, vector<4xf32>
      }}
      %one_f = scalar.constant 1.0 : f32
      %inv_d = scalar.divf %one_f, %df : f32
      %idv = vector.splat %inv_d : vector<4xf32>
      %att = vector.mulf %af, %idv : vector<4xf32>
      %g4 = vector.load %gv[%token, %hd, %lane, %c0] : view<[%tokens]x24x32x4xf32> -> vector<4xf32>
      %sig = vector.logisticf<afn> %g4 : vector<4xf32>
      %res = vector.mulf %att, %sig : vector<4xf32>
      vector.store %res, %ov[%token, %hd, %lane, %c0] : vector<4xf32>, view<[%tokens]x24x32x4xf32>
    }}
  }}
  kernel.return
}}"""

LAUNCH = re.compile(
    r"^(\s*)kernel\.launch @g_ink_attention_flash2_ts2\[%t\]\((%t, [^)]*), %arena, %att\) : "
    r"\[index\]\((.*), buffer\)$"
)


def edit_llm(text: str) -> str:
    if "g_ink_attention_flash2_ts2_sk" in text:
        return text
    lines = text.split("\n")
    out, n = [], 0
    for line in lines:
        m = LAUNCH.match(line)
        if not m:
            out.append(line)
            continue
        ind, args, types = m.groups()
        out.append(
            f"{ind}kernel.launch @g_ink_attention_flash2_ts2_{ATTN}[%t]({args}, %arena, %fb_acc, %fb_ml) : "
            f"[index]({types}, buffer, buffer)"
        )
        out.append(
            f"{ind}kernel.launch @g_ink_attn_combine_flash[%t](%t, %flash_tiles, %fb_acc, %fb_ml, %sink_buf, "
            "%gate, %att) : [index](index, buffer, buffer, buffer, buffer, buffer, buffer)"
        )
        n += 1
    assert n == 16, (
        f"expected 16 ts2 launches (8 global visits x advance/verify), found {n}"
    )
    ts2 = kernel_text(lines, "g_ink_attention_flash2_ts2")
    at = out.index(ts2[-1], out.index(ts2[0])) + 1
    sk = split_kernel(ts2)
    return "\n".join(
        out[:at]
        + [""]
        + sk
        + [""]
        + prefetch_kernel(sk)
        + [""]
        + COMBINE.split("\n")
        + out[at:]
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    program = Path(args[0]) / "program"
    edits = [("prepare.loom", edit_prepare)]
    if "--no-split" not in sys.argv:
        edits.append(("llm.loom", edit_llm))
    for name, edit in edits:
        path = program / name
        old = path.read_text()
        new = edit(old)
        if new != old:
            path.unlink()  # never write through a hard link into another package
            path.write_text(new)
            print(f"edited {path}")


main()
