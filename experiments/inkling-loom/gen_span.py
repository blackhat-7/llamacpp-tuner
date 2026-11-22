#!/usr/bin/env python3
"""Emit @g_ink_attn_split_span_v2: a drop-in replacement for @g_ink_attn_split_span (verify spans: same arguments,
grid (spans, 8 KV heads, 64 splits), the same per-group split ranges, causal masks and relative bias, and the same
pm/pl/po partials, so @g_ink_attn_combine is unchanged).

A workgroup has 3 x P waves: wave w serves query head w % 3 and page group w / 3. Tokens of a span are taken in groups
of TG; a group's split range is read once for its 4 tokens. A lane holds 16 dims of one key of a block of 4U keys
(lane = 8 * quad + d), so each load is 32 bytes (f16) or 16 bytes plus a scale (kv8). Scores are reduced over the
8 lanes of a quad; the online softmax runs once per block and token. Waves merge through LDS per group.

usage: [FMT=f16|kv8] [U=2] [P=1] [TG=4] [ALLH=0|1] [DL=16|8] gen_span.py > kernel.loom
"""
import os, sys

KV8 = os.environ.get("FMT", "f16") == "kv8"
U = int(os.environ.get("U", 2))
P = int(os.environ.get("P", 1))
ALLH = os.environ.get("ALLH", "0") == "1"  # each wave serves all 3 query heads (keys loaded once per workgroup)
NW = P if ALLH else 3 * P
# queries of a wave: (token j, head name, head index SSA): one head %h per wave, or all three
HS = [(f"_h{h}", f"%hc{h}", f"%hdq{h}") for h in range(3)] if ALLH else [("", "%h", "%hd")]
TG = int(os.environ.get("TG", 4))  # span tokens per pass over the keys
DL = int(os.environ.get("DL", 16))  # dims of a key per lane: LPK = 128 / DL lanes per key, KW = 32 / LPK keys per wave step
LPK, KW = 128 // DL, 32 // (128 // DL)
POS_BYTES = 1088 if KV8 else 2048
OFFSET_FN = "g_kv8_offset" if KV8 else "g_kv_offset"
L = []
e = L.append


def load_kv(u):
    """K and V of key u as vector<{DL}xf32> %k{u} / %v{u} from byte offsets %ko{u} / %vo{u}. KV8 leaves the codes
    unscaled: the lane's block scales %ks{u} / %vs{u} multiply its dot partial and its softmax weights."""
    if not KV8:
        return f"""        %kvw{u} = buffer.view %arena[%ko{u}] : buffer -> view<{DL}xf16>
        %kh16_{u} = vector.load %kvw{u}[%c0] : view<{DL}xf16> -> vector<{DL}xf16>
        %vvw{u} = buffer.view %arena[%vo{u}] : buffer -> view<{DL}xf16>
        %vh16_{u} = vector.load %vvw{u}[%c0] : view<{DL}xf16> -> vector<{DL}xf16>
        %k{u} = vector.extf %kh16_{u} : vector<{DL}xf16> to vector<{DL}xf32>
        %v{u} = vector.extf %vh16_{u} : vector<{DL}xf16> to vector<{DL}xf32>
"""
    out = f"""        %sb{u} = index.add %sb, %jo{u} : offset
        %svb{u} = index.add %sb{u}, %plane_stride : offset
"""
    for x, base, so in (("k", f"%ko{u}", f"%sb{u}"), ("v", f"%vo{u}", f"%svb{u}")):
        out += f"""        %{x}cw{u} = buffer.view %arena[{base}] : buffer -> view<{DL}xi8>
        %{x}i8_{u} = vector.load %{x}cw{u}[%c0] : view<{DL}xi8> -> vector<{DL}xi8>
        %{x}{u} = vector.sitofp %{x}i8_{u} : vector<{DL}xi8> to vector<{DL}xf32>
        %{x}sw{u} = buffer.view %arena[{so}] : buffer -> view<1xf16>
        %{x}s16_{u} = view.load %{x}sw{u}[%c0] : view<1xf16> -> f16
        %{x}s{u} = scalar.extf %{x}s16_{u} : f16 to f32
"""
    return out


J = range(TG)
Q = [(j, hn, hi, hdn) for j in range(TG) for (hn, hi, hdn) in HS]
e(f"""// Verify-span split-K attention, v2: {NW} waves (3 query heads x {P} page groups) per (span, KV head, split); a group
// of 4 span tokens reads its split's keys once (16 dims of one key per lane, {4 * U}-key blocks), the online softmax runs
// once per block and token, and the waves merge through LDS. Same arguments, split ranges, masks and pm/pl/po
// partials as @g_ink_attn_split_span, so @g_ink_attn_combine is unchanged.
kernel.def export("g_ink_attn_split_span_v2") @g_ink_attn_split_span_v2(%spans: index) {{
  %c1 = index.constant 1 : index
  %cKH = index.constant 8 : index
  %cS = index.constant 64 : index
  %lanes = index.constant {32 * NW} : index
  kernel.launch.config workgroups(%spans, %cKH, %cS) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %blocks_arg: index, %pool_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %q_in: buffer, %r_in: buffer, %rel_bias: buffer, %arena: buffer, %pm: buffer, %pl: buffer, %po: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 131072)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 1, 2147483584)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, 7)] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %kh_id = kernel.workgroup.id<y> : index
  %kh = index.assume %kh_id [range(%kh_id, 0, 7)] : index
  %sp_id = kernel.workgroup.id<z> : index
  %sp = index.assume %sp_id [range(%sp_id, 0, 63)] : index
  %wave_raw = kernel.subgroup.id : index
  %wave = index.assume %wave_raw [range(%wave_raw, 0, {NW - 1})] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %tid_raw = kernel.workitem.id<x> : index
  %tid = index.assume %tid_raw [range(%tid_raw, 0, {32 * NW - 1})] : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c7 = index.constant 7 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c128 = index.constant 128 : index
  %cB = index.constant {KW * U} : index
  %cP = index.constant {P} : index
  %cKWm = index.constant {KW} : index
  %cT = index.constant {32 * NW} : index
  %cTG = index.constant {TG} : index
  %cOUT = index.constant {3 * TG * 128} : index
  %cTG128 = index.constant {TG * 128} : index
  %cS_l = index.constant 64 : index
  %cEXT = index.constant 1024 : index
  %cE1 = index.constant 1023 : index
  %zero = index.constant 0 : offset
  %i1 = scalar.constant 1 : i32
  %i2 = scalar.constant 2 : i32
  %i4 = scalar.constant 4 : i32
  %i8 = scalar.constant 8 : i32
  %i16 = scalar.constant 16 : i32
  %i32 = scalar.constant 32 : i32
  %iLPK = scalar.constant {LPK} : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_inf = scalar.constant -3.40282347e+38 : f32
  %zero16 = vector.splatf 0.0 : vector<{DL}xf32>
  %scale = scalar.constant 0.08838834764831843 : f32
  %scale16 = vector.splat %scale : vector<{DL}xf32>
  %pos_bytes = index.constant {POS_BYTES} : offset
""" + ("""  %pgr = index.add %wave, %c0 : index
""" + "".join(f"  %hc{h} = index.constant {h} : index\n" for h in range(3)) if ALLH else """  %h_v = index.rem %wave, %c3 : index
  %h = index.assume %h_v [range(%h_v, 0, 2)] : index
  %pgr_v = index.div %wave, %c3 : index
  %pgr = index.assume %pgr_v [range(%pgr_v, 0, PM1)] : index
""".replace("PM1", str(P - 1))) + f"""  %cLPK = index.constant {LPK} : index
  %quad_v = index.div %lane, %cLPK : index
  %quad = index.assume %quad_v [range(%quad_v, 0, {KW - 1})] : index
  %d_v = index.rem %lane, %cLPK : index
  %d = index.assume %d_v [range(%d_v, 0, {LPK - 1})] : index
  %is_d0 = index.cmp eq, %d, %c0 : index
  %is_l0 = index.cmp eq, %lane, %c0 : index
  %hd_base = index.mul %kh, %c3 : index
""" + ("".join(f"  %hdq{h}_v = index.add %hd_base, %hc{h} : index\n  %hdq{h} = index.assume %hdq{h}_v [range(%hdq{h}_v, 0, 23)] : index\n" for h in range(3)) if ALLH else """  %hd_v = index.add %hd_base, %h : index
  %hd = index.assume %hd_v [range(%hd_v, 0, 23)] : index
""") + f"""  %acc_bytes = index.constant {P * KW * 3 * TG * 128 * 4} : offset
  %accs = buffer.alloca<workgroup> align(16) %acc_bytes : buffer
  %acc_l = buffer.view %accs[%zero] : buffer -> view<{P}x{KW}x3x{TG}x{LPK}x{DL}xf32>
  %acc_r = buffer.view %accs[%zero] : buffer -> view<{P}x{KW}x3x{TG}x128xf32>
  %ml_bytes = index.constant {P * (KW + 1) * 3 * TG * 4} : offset
  %mls = buffer.alloca<workgroup> align(16) %ml_bytes : buffer
  %m_l = buffer.view %mls[%zero] : buffer -> view<{P}x{KW + 1}x3x{TG}xf32>
  %kh_off = index.constant {256 // (2 if KV8 else 1)} : offset
  %d_off = index.constant {DL * (1 if KV8 else 2)} : offset
  %lo_a = index.scale %kh, %kh_off : index, offset -> offset
  %lo_b = index.scale %d, %d_off : index, offset -> offset
  %lane_off = index.add %lo_a, %lo_b : offset
""")
if KV8:
    e(f"""  %cdb = index.constant {32 // DL} : index
  %blk_v = index.div %d, %cdb : index
  %blk = index.madd %kh, %c4, %blk_v : index
  %sc_base = index.constant 1024 : offset
  %sc_two = index.constant 2 : offset
  %sc_b = index.scale %blk, %sc_two : index, offset -> offset
  %sc_off = index.add %sc_base, %sc_b : offset
""")
e(f"""  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {{
    %start_b = scalar.assume %start32 [range(%start32, 0, 1048575)] : i32
    %start = index.cast %start_b : i32 to index
    %k_plane = index.mul %layer, %c2 : index
    %plane_stride = index.scale %pool, %pos_bytes : index, offset -> offset
    %qv = buffer.view %q_in[%zero] : buffer -> view<[%tokens]x24x{LPK}x{DL}xf32>
    %rv = buffer.view %r_in[%zero] : buffer -> view<[%tokens]x24x16xf32>
    %rbv = buffer.view %rel_bias[%zero] : buffer -> view<1024x16xf32>
    %pm_v = buffer.view %pm[%zero] : buffer -> view<[%tokens]x24x64xf32>
    %pl_v = buffer.view %pl[%zero] : buffer -> view<[%tokens]x24x64xf32>
    %po_v = buffer.view %po[%zero] : buffer -> view<[%tokens]x24x64x128xf32>
    scf.for %cb = [%c0 to %length step %cTG] {{
""")
# tokens of the group: missing ones repeat token 0 (not written)
for j in J:
    if j == 0:
        e(f"""      %has0 = index.cmp ult, %cb, %length : index
      %tk0_v = index.add %first, %cb : index
      %tk0 = index.assume %tk0_v [range(%tk0_v, 0, 8191), lt(%tk0_v, %tokens)] : index
      %ps0_v = index.add %start, %cb : index
      %ps0 = index.assume %ps0_v [range(%ps0_v, 0, 1048575)] : index
""")
    else:
        e(f"""      %cj{j} = index.constant {j} : index
      %cbj{j} = index.add %cb, %cj{j} : index
      %has{j} = index.cmp ult, %cbj{j}, %length : index
      %tkj{j} = index.add %first, %cbj{j} : index
      %tks{j} = scf.select %has{j}, %tkj{j}, %tk0 : index
      %tk{j} = index.assume %tks{j} [range(%tks{j}, 0, 8191), lt(%tks{j}, %tokens)] : index
      %psj{j} = index.add %start, %cbj{j} : index
      %pss{j} = scf.select %has{j}, %psj{j}, %ps0 : index
      %ps{j} = index.assume %pss{j} [range(%pss{j}, 0, 1048575)] : index
""")
    for hn, hi, hdn in HS:
        e(f"""      %qr{j}{hn} = vector.load %qv[%tk{j}, {hdn}, %d, %c0] : view<[%tokens]x24x{LPK}x{DL}xf32> -> vector<{DL}xf32>
      %q{j}{hn} = vector.mulf %qr{j}{hn}, %scale16 : vector<{DL}xf32>
""")
prev = "%ps0"
for j in range(1, TG):
    e(f"      %pmx{j} = index.max {prev}, %ps{j} : index\n")
    prev = f"%pmx{j}"
e(f"      %pos_last = index.add {prev}, %c0 : index\n")
e(f"""      %nk_min_raw = index.add %ps0, %c1 : index
      %nk_min = index.assume %nk_min_raw [range(%nk_min_raw, 1, 1048576)] : index
      %nk7 = index.add %nk_min, %c7 : index
      %nk7_b = index.assume %nk7 [range(%nk7, 8, 1048583)] : index
      %s_need = index.div %nk7_b, %c8 : index
      %s_eff0 = index.min %s_need, %cS_l : index
      %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, 64)] : index
      %sp_active = index.cmp ult, %sp, %s_eff : index
      scf.if %sp_active {{
        %n_keys_raw = index.add %pos_last, %c1 : index
        %n_keys = index.assume %n_keys_raw [range(%n_keys_raw, 1, 1048576)] : index
        %ch0 = index.add %n_keys, %s_eff : index
        %ch1_r = index.sub %ch0, %c1 : index
        %ch1 = index.assume %ch1_r [range(%ch1_r, 1, 1048640)] : index
        %chunk = index.div %ch1, %s_eff : index
        %st0 = index.mul %sp, %chunk : index
        %st_pos = index.min %st0, %n_keys : index
        %en0 = index.add %st_pos, %chunk : index
        %en_pos = index.min %en0, %n_keys : index
        %has_keys = index.cmp ult, %st_pos, %en_pos : index
        %pg0 = index.div %st_pos, %cB : index
        %en_m1_v = index.sub %en_pos, %c1 : index
        %en_m1_s = scf.select %has_keys, %en_m1_v, %st_pos : index
        %en_m1 = index.assume %en_m1_s [range(%en_m1_s, 0, 1048575)] : index
        %pg1_0 = index.div %en_m1, %cB : index
        %pg1_1 = index.add %pg1_0, %c1 : index
        %pg1 = scf.select %has_keys, %pg1_1, %pg0 : index
        %npg = index.sub %pg1, %pg0 : index
        %npa0 = index.min %npg, %cP : index
        %npa = index.max %npa0, %c1 : index
        %pgw = index.add %pg0, %pgr : index
        %pgs = index.min %pgw, %pg1 : index
""")
car = []
for j, hn, _, _ in Q:
    car += [f"%m{j}{hn} = %neg_inf : f32", f"%l{j}{hn} = %zero_f : f32", f"%a{j}{hn} = %zero16 : vector<{DL}xf32>"]
cty = ", ".join(["f32", "f32", "vector<{DL}xf32>"] * len(Q))
res = ", ".join(f"%fm{j}{hn}, %fl{j}{hn}, %fa{j}{hn}" for j, hn, _, _ in Q)
e(f"""        {res} = scf.for %pg = [%pgs to %pg1 step %cP]({", ".join(car)}) -> ({cty}) {{
          %p_base = index.mul %pg, %cB : index
          %lo_pos = index.max %p_base, %st_pos : index
          %p_next = index.add %p_base, %cB : index
          %hi_pos = index.min %p_next, %en_pos : index
          %hi_m1 = index.sub %hi_pos, %c1 : index
          %ko_base = func.call @{OFFSET_FN}(%origins, %row, %p_base, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
          %kb = index.add %ko_base, %lane_off : offset
""")
if KV8:
    e("          %sb = index.add %ko_base, %sc_off : offset\n")
for u in range(U):
    e(f"""        %ku{u} = index.constant {KW * u} : index
        %kq{u} = index.add %p_base, %ku{u} : index
        %kp{u} = index.add %kq{u}, %quad : index
        %kx{u} = index.max %kp{u}, %lo_pos : index
        %kc{u} = index.min %kx{u}, %hi_m1 : index
        %in{u} = index.cmp eq, %kp{u}, %kc{u} : index
        %j{u} = index.sub %kc{u}, %p_base : index
        %jo{u} = index.scale %j{u}, %pos_bytes : index, offset -> offset
        %ko{u} = index.add %kb, %jo{u} : offset
        %vo{u} = index.add %ko{u}, %plane_stride : offset
""")
    e(load_kv(u))
    for j in J:
        e(f"""        %cz{j}_{u} = index.cmp ule, %kc{u}, %ps{j} : index
        %ok{j}_{u} = scalar.andi %in{u}, %cz{j}_{u} : i1
""")
for j, hn, _, _ in Q:
    for u in range(U):
        n = f"{j}{hn}_{u}"
        dot = f"%pd{n}" if KV8 else f"%p{n}"
        e(f"        {dot} = vector.dotf<reassoc|contract> %q{j}{hn}, %k{u}, %zero_f : vector<{DL}xf32>, vector<{DL}xf32>, f32\n")
        if KV8:
            e(f"        %p{n} = scalar.mulf %pd{n}, %ks{u} : f32\n")
        # xor butterfly over the LPK lanes of a key: every lane ends with the same sum
        prev, x = f"%p{n}", 1
        while x < LPK:
            nxt = f"%sq{n}" if 2 * x == LPK else f"%r{x}_{n}"
            e(f"""        %x{x}_{n}, %y{x}_{n} = kernel.subgroup.shuffle<xor> {prev}, %i{x}, %iLPK : f32, i32, i32
        {nxt} = scalar.addf {prev}, %x{x}_{n} : f32
""")
            prev, x = nxt, 2 * x
# relative bias of keys within 1024 of a token (keys past token 0 are near for every later token too)
bn = [f"%b{j}{hn}_{u}" for j, hn, _, _ in Q for u in range(U)]
e(f"""          %past0 = index.cmp ugt, %hi_m1, %ps0 : index
          %p0s = index.max %ps0, %hi_m1 : index
          %mind = index.sub %p0s, %hi_m1 : index
          %close = index.cmp ult, %mind, %cEXT : index
          %near = scalar.ori %past0, %close : i1
          {", ".join(bn)} = scf.if %near -> ({", ".join(["f32"] * len(bn))}) {{
""")
for j in J:
    for u in range(U):
        n = f"{j}_{u}"
        e(f"""            %dp{n} = index.max %ps{j}, %kc{u} : index
            %dist{n} = index.sub %dp{n}, %kc{u} : index
            %ie{n} = index.cmp ult, %dist{n}, %cEXT : index
            %dc{n} = index.min %dist{n}, %cE1 : index
            %db{n} = index.assume %dc{n} [range(%dc{n}, 0, 1023)] : index
            %bv{n} = vector.load %rbv[%db{n}, %c0] : view<1024x16xf32> -> vector<16xf32>
""")
for j, hn, _, hdn in Q:
    e(f"            %r16_{j}{hn} = vector.load %rv[%tk{j}, {hdn}, %c0] : view<[%tokens]x24x16xf32> -> vector<16xf32>\n")
    for u in range(U):
        n = f"{j}{hn}_{u}"
        e(f"""            %rd{n} = vector.dotf<reassoc|contract> %r16_{j}{hn}, %bv{j}_{u}, %zero_f : vector<16xf32>, vector<16xf32>, f32
            %rs{n} = scf.select %ie{j}_{u}, %rd{n}, %zero_f : f32
""")
e(f"""            scf.yield {", ".join(f"%rs{j}{hn}_{u}" for j, hn, _, _ in Q for u in range(U))} : {", ".join(["f32"] * len(bn))}
          }} else {{
            scf.yield {", ".join(["%zero_f"] * len(bn))} : {", ".join(["f32"] * len(bn))}
          }}
""")
yl = []
for j0, hn, _, _ in Q:
    j = f"{j0}{hn}"
    for u in range(U):
        n = f"{j}_{u}"
        e(f"""          %sb{n} = scalar.addf %sq{n}, %b{n} : f32
          %s{n} = scf.select %ok{j0}_{u}, %sb{n}, %neg_inf : f32
""")
    prev = f"%s{j}_0"
    for u in range(1, U):
        e(f"          %bmc{j}_{u} = scalar.maxnumf {prev}, %s{j}_{u} : f32\n")
        prev = f"%bmc{j}_{u}"
    x = LPK
    while x < 32:
        nxt = f"%bm{j}" if 2 * x == 32 else f"%bmd{j}_{x}"
        e(f"""          %bx{j}_{x}, %by{j}_{x} = kernel.subgroup.shuffle<xor> {prev}, %i{x}, %i32 : f32, i32, i32
          {nxt} = scalar.maxnumf {prev}, %bx{j}_{x} : f32
""")
        prev, x = nxt, 2 * x
    e(f"""          %mn{j} = scalar.maxnumf %m{j}, %bm{j} : f32
          %ms{j} = scalar.subf %m{j}, %mn{j} : f32
          %al{j} = scalar.expf<afn> %ms{j} : f32
          %alv{j} = vector.splat %al{j} : vector<{DL}xf32>
          %as{j} = vector.mulf %a{j}, %alv{j} : vector<{DL}xf32>
          %ls{j} = scalar.mulf %l{j}, %al{j} : f32
""")
    pa, plv = f"%as{j}", f"%ls{j}"
    for u in range(U):
        n = f"{j}_{u}"
        # a block can hold no key of a token (all past it): its weights must be 0, not exp(0)
        e(f"""          %ds{n} = scalar.subf %s{n}, %mn{j} : f32
          %ex{n} = scalar.expf<afn> %ds{n} : f32
          %e{n} = scf.select %ok{j0}_{u}, %ex{n}, %zero_f : f32
          %ll{n} = scalar.addf {plv}, %e{n} : f32
""")
        w = f"%e{n}"
        if KV8:
            e(f"          %ew{n} = scalar.mulf %e{n}, %vs{u} : f32\n")
            w = f"%ew{n}"
        e(f"""          %ev{n} = vector.splat {w} : vector<{DL}xf32>
          %aa{n} = vector.fmaf %ev{n}, %v{u}, {pa} : vector<{DL}xf32>
""")
        pa, plv = f"%aa{n}", f"%ll{n}"
    yl += [f"%mn{j}", plv, pa]
e(f"""          scf.yield {", ".join(yl)} : {cty}
        }}
""")
for j, hn, hi, _ in Q:
    e(f"        vector.store %fa{j}{hn}, %acc_l[%pgr, %quad, {hi}, %cjj{j}, %d, %c0] : vector<{DL}xf32>, view<{P}x{KW}x3x{TG}x{LPK}x{DL}xf32>\n")
e("        scf.if %is_d0 {\n")
for j, hn, hi, _ in Q:
    e(f"          view.store %fl{j}{hn}, %m_l[%pgr, %quad, {hi}, %cjj{j}] : f32, view<{P}x{KW + 1}x3x{TG}xf32>\n")
e("        }\n        scf.if %is_l0 {\n")
for j, hn, hi, _ in Q:
    e(f"          view.store %fm{j}{hn}, %m_l[%pgr, %cKWm, {hi}, %cjj{j}] : f32, view<{P}x{KW + 1}x3x{TG}xf32>\n")
e(f"""        }}
        kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
        scf.for %idx = [%tid to %cOUT step %cT] {{
          %oh_v = index.div %idx, %cTG128 : index
          %oh = index.assume %oh_v [range(%oh_v, 0, 2)] : index
          %ojr = index.rem %idx, %cTG128 : index
          %oj_v = index.div %ojr, %c128 : index
          %oj = index.assume %oj_v [range(%oj_v, 0, {TG - 1})] : index
          %dim_v = index.rem %idx, %c128 : index
          %dim = index.assume %dim_v [range(%dim_v, 0, 127)] : index
          %ojx = index.add %cb, %oj : index
          %owr = index.cmp ult, %ojx, %length : index
          scf.if %owr {{
            %mm = scf.for %w = [%c0 to %npa step %c1](%mx = %neg_inf : f32) -> (f32) {{
              %wb = index.assume %w [range(%w, 0, {P - 1})] : index
              %mw = view.load %m_l[%wb, %cKWm, %oh, %oj] : view<{P}x{KW + 1}x3x{TG}xf32> -> f32
              %mxn = scalar.maxnumf %mx, %mw : f32
              scf.yield %mxn : f32
            }}
            %ls_t, %os_t = scf.for %w = [%c0 to %npa step %c1](%lt = %zero_f : f32, %ot = %zero_f : f32) -> (f32, f32) {{
              %wb = index.assume %w [range(%w, 0, {P - 1})] : index
              %mw = view.load %m_l[%wb, %cKWm, %oh, %oj] : view<{P}x{KW + 1}x3x{TG}xf32> -> f32
              %dw = scalar.subf %mw, %mm : f32
              %fw = scalar.expf<afn> %dw : f32
""")
for g in range(KW):
    e(f"""              %cg{g} = index.constant {g} : index
              %lw{g} = view.load %m_l[%wb, %cg{g}, %oh, %oj] : view<{P}x{KW + 1}x3x{TG}xf32> -> f32
              %ow{g} = view.load %acc_r[%wb, %cg{g}, %oh, %oj, %dim] : view<{P}x{KW}x3x{TG}x128xf32> -> f32
""")
e(f"""              %lq01 = scalar.addf %lw0, %lw1 : f32
              %oq01 = scalar.addf %ow0, %ow1 : f32
""" + (f"""              %lq23 = scalar.addf %lw2, %lw3 : f32
              %lq = scalar.addf %lq01, %lq23 : f32
              %oq23 = scalar.addf %ow2, %ow3 : f32
              %oq = scalar.addf %oq01, %oq23 : f32
""" if KW == 4 else """              %lq = scalar.addf %lq01, %zero_f : f32
              %oq = scalar.addf %oq01, %zero_f : f32
""") + f"""              %ltn = scalar.fmaf %lq, %fw, %lt : f32
              %otn = scalar.fmaf %oq, %fw, %ot : f32
              scf.yield %ltn, %otn : f32, f32
            }}
            %otk_v = index.add %first, %ojx : index
            %otk = index.assume %otk_v [range(%otk_v, 0, 8191), lt(%otk_v, %tokens)] : index
            %ohd_v = index.add %hd_base, %oh : index
            %ohd = index.assume %ohd_v [range(%ohd_v, 0, 23)] : index
            view.store %os_t, %po_v[%otk, %ohd, %sp, %dim] : f32, view<[%tokens]x24x64x128xf32>
            %dim0 = index.cmp eq, %dim, %c0 : index
            scf.if %dim0 {{
              view.store %mm, %pm_v[%otk, %ohd, %sp] : f32, view<[%tokens]x24x64xf32>
              view.store %ls_t, %pl_v[%otk, %ohd, %sp] : f32, view<[%tokens]x24x64xf32>
            }}
          }}
        }}
        // the next group rewrites the LDS partials
        kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
      }} else {{
        // a later token of the group can have more splits than its first (the combine reads them): neutral partials
        scf.for %idx = [%tid to %cOUT step %cT] {{
          %nh_v = index.div %idx, %cTG128 : index
          %njr = index.rem %idx, %cTG128 : index
          %nj = index.div %njr, %c128 : index
          %ndim_v = index.rem %idx, %c128 : index
          %ndim = index.assume %ndim_v [range(%ndim_v, 0, 127)] : index
          %njx = index.add %cb, %nj : index
          %nwr = index.cmp ult, %njx, %length : index
          scf.if %nwr {{
            %ntk_v = index.add %first, %njx : index
            %ntk = index.assume %ntk_v [range(%ntk_v, 0, 8191), lt(%ntk_v, %tokens)] : index
            %nhd_v = index.add %hd_base, %nh_v : index
            %nhd = index.assume %nhd_v [range(%nhd_v, 0, 23)] : index
            view.store %zero_f, %po_v[%ntk, %nhd, %sp, %ndim] : f32, view<[%tokens]x24x64x128xf32>
            %ndim0 = index.cmp eq, %ndim, %c0 : index
            scf.if %ndim0 {{
              view.store %neg_inf, %pm_v[%ntk, %nhd, %sp] : f32, view<[%tokens]x24x64xf32>
              view.store %zero_f, %pl_v[%ntk, %nhd, %sp] : f32, view<[%tokens]x24x64xf32>
            }}
          }}
        }}
      }}
    }}
  }}
  kernel.return
}}
""")
text = "".join(L).replace("{DL}", str(DL)).replace("{LPK}", str(LPK))  # plain strings above use the f-string spelling
text = text.replace("  %c0 = index.constant 0 : index\n", "  %c0 = index.constant 0 : index\n"
                    + "".join(f"  %cjj{j} = index.constant {j} : index\n" for j in J), 1)
sys.stdout.write(text)
