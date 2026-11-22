#!/usr/bin/env python3
"""Emit @g_ink_attn_split_v2: a drop-in replacement for @g_ink_attn_split (same launch arguments, grid
(tokens, 8 KV heads, 64 splits), split ranges and pm/pl/po partial layout, so @g_ink_attn_combine is unchanged).

Per workgroup (token, KV head, split): NW waves take the split's 16-key pages round-robin. A wave loads each
key once for all 3 query heads of its KV head: lane = 8 * quad + d holds dims 16d..16d+15 of key 4u + quad
(u = 0..3) of the page, as one 32-byte load. Scores are reduced over the 8 lanes of a quad, and the online
softmax runs once per page per head. The waves merge through LDS at the end.

usage: [FMT=f16|kv8] gen_v2.py NW > kernel.loom
FMT=kv8 reads the 8-bit KV of the kv8 package (engines/loom) (1088 bytes per position: 8 heads x 128 int8 codes,
then 32 f16 scales, one per head and 32-dim block; value = code x scale); only the load block differs.
"""
import os, sys

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 4
KV8 = os.environ.get("FMT", "f16") == "kv8"
POS_BYTES = 1088 if KV8 else 2048
H = 3  # query heads per KV head
U = int(os.environ.get("U", 4))  # keys per lane per block; a block of 4U keys (4 quads) lies within one 16-key page

def load_kv(u):
    """K and V of key u as vector<16xf32> %k{u} / %v{u}, from byte offsets %ko{u} / %vo{u} (the format-specific part).
    KV8 leaves the codes unscaled; the lane's block scales %ks{u} / %vs{u} multiply its dot partial and softmax weight."""
    if not KV8:
        return f"""        %kvw{u} = buffer.view %arena[%ko{u}] : buffer -> view<16xf16>
        %kh16_{u} = vector.load %kvw{u}[%c0] : view<16xf16> -> vector<16xf16>
        %vvw{u} = buffer.view %arena[%vo{u}] : buffer -> view<16xf16>
        %vh16_{u} = vector.load %vvw{u}[%c0] : view<16xf16> -> vector<16xf16>
        %k{u} = vector.extf %kh16_{u} : vector<16xf16> to vector<16xf32>
        %v{u} = vector.extf %vh16_{u} : vector<16xf16> to vector<16xf32>
"""
    out = ""
    for x, base in (("k", f"%ko{u}"), ("v", f"%vo{u}")):
        so = f"%sb{u}" if x == "k" else f"%svb{u}"
        out += f"""        %{x}cw{u} = buffer.view %arena[{base}] : buffer -> view<16xi8>
        %{x}i8_{u} = vector.load %{x}cw{u}[%c0] : view<16xi8> -> vector<16xi8>
"""
    out += f"""        %sb{u} = index.add %sb, %jo{u} : offset
        %svb{u} = index.add %sb{u}, %plane_stride : offset
"""
    for x, so in (("k", f"%sb{u}"), ("v", f"%svb{u}")):
        out += f"""        %{x}sw{u} = buffer.view %arena[{so}] : buffer -> view<1xf16>
        %{x}s16_{u} = view.load %{x}sw{u}[%c0] : view<1xf16> -> f16
        %{x}s{u} = scalar.extf %{x}s16_{u} : f16 to f32
        %{x}{u} = vector.sitofp %{x}i8_{u} : vector<16xi8> to vector<16xf32>
"""
    return out


L = []
e = L.append

e(f"""// Split-K decode attention, v2: {NW} waves per (token, KV head, split) share the split's 16-key pages; each lane
// loads 16 dims of one key once for the 3 query heads (32-byte loads), and the online softmax runs once per page.
// The merge reads only the waves that had pages, so short contexts pay for one.
// Same arguments, split ranges and pm/pl/po partials as @g_ink_attn_split, so @g_ink_attn_combine is unchanged.
kernel.def export("g_ink_attn_split_v2") @g_ink_attn_split_v2(%tokens: index) {{
  %c1 = index.constant 1 : index
  %cKH = index.constant 8 : index
  %cS = index.constant 64 : index
  %lanes = index.constant {32 * NW} : index
  kernel.launch.config workgroups(%tokens, %cKH, %cS) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %blocks_arg: index, %pool_arg: index, %layer_arg: index, %sliding_arg: i32, %info: buffer, %origins: buffer, %q_in: buffer, %r_in: buffer, %rel_bias: buffer, %arena: buffer, %pm: buffer, %pl: buffer, %po: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %blocks = index.assume %blocks_arg [range(%blocks_arg, 1, 131072)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 2097152)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, 47)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
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
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %cB = index.constant {4 * U} : index
  %c128 = index.constant 128 : index
  %c384 = index.constant 384 : index
  %cNW = index.constant {NW} : index
  %cT = index.constant {32 * NW} : index
  %cEXT = index.constant 1024 : index
  %cE1 = index.constant 1023 : index
  %cWIN1 = index.constant 1023 : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %i1 = scalar.constant 1 : i32
  %i2 = scalar.constant 2 : i32
  %i4 = scalar.constant 4 : i32
  %i8 = scalar.constant 8 : i32
  %i16 = scalar.constant 16 : i32
  %i32 = scalar.constant 32 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_inf = scalar.constant -3.40282347e+38 : f32
  %zero16 = vector.splatf 0.0 : vector<16xf32>
  %scale = scalar.constant 0.08838834764831843 : f32
  %scale16 = vector.splat %scale : vector<16xf32>
  %pos_bytes = index.constant {POS_BYTES} : offset
  %quad_v = index.div %lane, %c8 : index
  %quad = index.assume %quad_v [range(%quad_v, 0, 3)] : index
  %d_v = index.rem %lane, %c8 : index
  %d = index.assume %d_v [range(%d_v, 0, 7)] : index
  %is_d0 = index.cmp eq, %d, %c0 : index
  %is_l0 = index.cmp eq, %lane, %c0 : index
  %info_v = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_v[%tok, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %acc_bytes = index.constant {NW * 4 * H * 128 * 4} : offset
  %accs = buffer.alloca<workgroup> align(16) %acc_bytes : buffer
  %acc_l = buffer.view %accs[%zero] : buffer -> view<{NW}x4x{H}x8x16xf32>
  %acc_r = buffer.view %accs[%zero] : buffer -> view<{NW}x4x{H}x128xf32>
  %ml_bytes = index.constant {NW * 5 * H * 4} : offset
  %mls = buffer.alloca<workgroup> align(16) %ml_bytes : buffer
  %m_l = buffer.view %mls[%zero] : buffer -> view<{NW}x5x{H}xf32>
  scf.if %live {{
    %row32 = view.load %info_v[%tok, 0] : view<[%tokens]x4xi32> -> i32
    %pos32 = view.load %info_v[%tok, 1] : view<[%tokens]x4xi32> -> i32
    %row_b = scalar.assume %row32 [range(%row32, 0, 1023)] : i32
    %pos_b = scalar.assume %pos32 [range(%pos32, 0, 2097151)] : i32
    %row = index.cast %row_b : i32 to index
    %pos = index.cast %pos_b : i32 to index
    %is_swa = scalar.cmpi ne, %sliding_arg, %c0_i32 : i32
    %w_low = scf.if %is_swa -> (index) {{
      %gt = index.cmp ugt, %pos, %cWIN1 : index
      %sub = index.sub %pos, %cWIN1 : index
      %wl = scf.select %gt, %sub, %c0 : index
      scf.yield %wl : index
    }} else {{
      scf.yield %c0 : index
    }}
    %n_keys_0 = index.sub %pos, %w_low : index
    %n_keys_raw = index.add %n_keys_0, %c1 : index
    %n_keys = index.assume %n_keys_raw [range(%n_keys_raw, 1, 2097152)] : index
    %c7 = index.constant 7 : index
    %nk7 = index.add %n_keys, %c7 : index
    %nk7_b = index.assume %nk7 [range(%nk7, 8, 2097159)] : index
    %s_need = index.div %nk7_b, %c8 : index
    %s_eff0 = index.min %s_need, %cS_l : index
    %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, 64)] : index
    %sp_active = index.cmp ult, %sp, %s_eff : index
    scf.if %sp_active {{
      %chunk0 = index.add %n_keys, %s_eff : index
      %chunk1_raw = index.sub %chunk0, %c1 : index
      %chunk1 = index.assume %chunk1_raw [range(%chunk1_raw, 1, 2097216)] : index
      %chunk = index.div %chunk1, %s_eff : index
      %st0 = index.mul %sp, %chunk : index
      %st = index.min %st0, %n_keys : index
      %en0 = index.add %st, %chunk : index
      %en = index.min %en0, %n_keys : index
      %k_plane = index.mul %layer, %c2 : index
      %plane_stride = index.scale %pool, %pos_bytes : index, offset -> offset
      %st_pos_v = index.add %w_low, %st : index
      %st_pos = index.assume %st_pos_v [range(%st_pos_v, 0, 2097152)] : index
      %en_pos_v = index.add %w_low, %en : index
      %en_pos = index.assume %en_pos_v [range(%en_pos_v, 0, 2097152)] : index
      %has_keys = index.cmp ult, %st_pos, %en_pos : index
      %pg0 = index.div %st_pos, %cB : index
      %en_m1_v = index.sub %en_pos, %c1 : index
      %en_m1_s = scf.select %has_keys, %en_m1_v, %st_pos : index
      %en_m1 = index.assume %en_m1_s [range(%en_m1_s, 0, 2097151)] : index
      %pg1_0 = index.div %en_m1, %cB : index
      %pg1_1 = index.add %pg1_0, %c1 : index
      %pg1 = scf.select %has_keys, %pg1_1, %pg0 : index
      %npg = index.sub %pg1, %pg0 : index
      %nwa0 = index.min %npg, %cNW : index
      %nwa = index.max %nwa0, %c1 : index
      %pgw = index.add %pg0, %wave : index
      %pgs = index.min %pgw, %pg1 : index
      %kh_off = index.constant {256 // (2 if KV8 else 1)} : offset
      %d_off = index.constant {32 // (2 if KV8 else 1)} : offset
      %lo_a = index.scale %kh, %kh_off : index, offset -> offset
      %lo_b = index.scale %d, %d_off : index, offset -> offset
      %lane_off = index.add %lo_a, %lo_b : offset
""" + ("""      %blk_v = index.div %d, %c2 : index
      %blk = index.madd %kh, %c4, %blk_v : index
      %sc_base = index.constant 1024 : offset
      %sc_two = index.constant 2 : offset
      %sc_b = index.scale %blk, %sc_two : index, offset -> offset
      %sc_off = index.add %sc_base, %sc_b : offset
""" if KV8 else "") + f"""      %qv = buffer.view %q_in[%zero] : buffer -> view<[%tokens]x24x8x16xf32>
      %hd_base = index.mul %kh, %c3 : index
""")
for h in range(H):
    e(f"""      %hd{h}_v = index.add %hd_base, %c{h}_h : index
      %hd{h} = index.assume %hd{h}_v [range(%hd{h}_v, 0, 23)] : index
      %qr{h} = vector.load %qv[%tok, %hd{h}, %d, %c0] : view<[%tokens]x24x8x16xf32> -> vector<16xf32>
      %q{h} = vector.mulf %qr{h}, %scale16 : vector<16xf32>
""")
carried_init = ", ".join([f"%m{h} = %neg_inf : f32" for h in range(H)] + [f"%l{h} = %zero_f : f32" for h in range(H)]
                         + [f"%a{h} = %zero16 : vector<16xf32>" for h in range(H)])
carried_ty = ", ".join(["f32"] * (2 * H) + ["vector<16xf32>"] * H)
res = ", ".join([f"%fm{h}" for h in range(H)] + [f"%fl{h}" for h in range(H)] + [f"%fa{h}" for h in range(H)])
e(f"""      {res} = scf.for %pg = [%pgs to %pg1 step %cNW]({carried_init}) -> ({carried_ty}) {{
        %p_base = index.mul %pg, %cB : index
        %lo_pos = index.max %p_base, %st_pos : index
        %p_next = index.add %p_base, %cB : index
        %hi_pos = index.min %p_next, %en_pos : index
        %hi_m1 = index.sub %hi_pos, %c1 : index
        %ko_base = func.call @{"g_kv8_offset" if KV8 else "g_kv_offset"}(%origins, %row, %p_base, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
        %kb = index.add %ko_base, %lane_off : offset
""" + ("        %sb = index.add %ko_base, %sc_off : offset\n" if KV8 else "") + f"""""")
for u in range(U):
    e(f"""        %ku{u} = index.constant {4 * u} : index
        %kq{u} = index.add %p_base, %ku{u} : index
        %kp{u} = index.add %kq{u}, %quad : index
        %kx{u} = index.max %kp{u}, %lo_pos : index
        %kc{u} = index.min %kx{u}, %hi_m1 : index
        %ok{u} = index.cmp eq, %kp{u}, %kc{u} : index
        %j{u} = index.sub %kc{u}, %p_base : index
        %jo{u} = index.scale %j{u}, %pos_bytes : index, offset -> offset
        %ko{u} = index.add %kb, %jo{u} : offset
        %vo{u} = index.add %ko{u}, %plane_stride : offset
""")
    e(load_kv(u))
for h in range(H):
    for u in range(U):
        n = f"{h}_{u}"
        dot = f"%pd{n}" if KV8 else f"%p{n}"
        e(f"""        {dot} = vector.dotf<reassoc|contract> %q{h}, %k{u}, %zero_f : vector<16xf32>, vector<16xf32>, f32
""" + (f"        %p{n} = scalar.mulf %pd{n}, %ks{u} : f32\n" if KV8 else "") + f"""        %x1_{n}, %y1_{n} = kernel.subgroup.shuffle<xor> %p{n}, %i1, %i8 : f32, i32, i32
        %r1_{n} = scalar.addf %p{n}, %x1_{n} : f32
        %x2_{n}, %y2_{n} = kernel.subgroup.shuffle<xor> %r1_{n}, %i2, %i8 : f32, i32, i32
        %r2_{n} = scalar.addf %r1_{n}, %x2_{n} : f32
        %x4_{n}, %y4_{n} = kernel.subgroup.shuffle<xor> %r2_{n}, %i4, %i8 : f32, i32, i32
        %sq{n} = scalar.addf %r2_{n}, %x4_{n} : f32
""")
# relative-position bias for keys within 1024 of the query (only the last pages)
bias_names = [f"%b{h}_{u}" for h in range(H) for u in range(U)]
e(f"""        %min_dist = index.sub %pos, %hi_m1 : index
        %near = index.cmp ult, %min_dist, %cEXT : index
        {", ".join(bias_names)} = scf.if %near -> ({", ".join(["f32"] * len(bias_names))}) {{
          %rv = buffer.view %r_in[%zero] : buffer -> view<[%tokens]x24x16xf32>
          %rbv = buffer.view %rel_bias[%zero] : buffer -> view<1024x16xf32>
""")
for h in range(H):
    e(f"          %r16_{h} = vector.load %rv[%tok, %hd{h}, %c0] : view<[%tokens]x24x16xf32> -> vector<16xf32>\n")
for u in range(U):
    e(f"""          %dist{u} = index.sub %pos, %kc{u} : index
          %in{u} = index.cmp ult, %dist{u}, %cEXT : index
          %dc{u} = index.min %dist{u}, %cE1 : index
          %db{u} = index.assume %dc{u} [range(%dc{u}, 0, 1023)] : index
          %bv{u} = vector.load %rbv[%db{u}, %c0] : view<1024x16xf32> -> vector<16xf32>
""")
    for h in range(H):
        e(f"""          %rd{h}_{u} = vector.dotf<reassoc|contract> %r16_{h}, %bv{u}, %zero_f : vector<16xf32>, vector<16xf32>, f32
          %rs{h}_{u} = scf.select %in{u}, %rd{h}_{u}, %zero_f : f32
""")
e(f"""          scf.yield {", ".join(f"%rs{h}_{u}" for h in range(H) for u in range(U))} : {", ".join(["f32"] * len(bias_names))}
        }} else {{
          scf.yield {", ".join(["%zero_f"] * len(bias_names))} : {", ".join(["f32"] * len(bias_names))}
        }}
""")
for h in range(H):
    for u in range(U):
        n = f"{h}_{u}"
        e(f"""        %sb{n} = scalar.addf %sq{n}, %b{n} : f32
        %s{n} = scf.select %ok{u}, %sb{n}, %neg_inf : f32
""")
    if U == 4:
        e(f"""        %bm{h}_a = scalar.maxnumf %s{h}_0, %s{h}_1 : f32
        %bm{h}_b = scalar.maxnumf %s{h}_2, %s{h}_3 : f32
        %bm{h}_c = scalar.maxnumf %bm{h}_a, %bm{h}_b : f32
""")
    else:
        prev = f"%s{h}_0"
        for u in range(1, U):
            e(f"        %bmc{h}_{u} = scalar.maxnumf {prev}, %s{h}_{u} : f32\n")
            prev = f"%bmc{h}_{u}"
        e(f"        %bm{h}_c = scalar.addf {prev}, %zero_f : f32\n" if U == 1 else f"        %bm{h}_c = scalar.maxnumf {prev}, {prev} : f32\n")
    e(f"""        %bx{h}_8, %by{h}_8 = kernel.subgroup.shuffle<xor> %bm{h}_c, %i8, %i32 : f32, i32, i32
        %bm{h}_d = scalar.maxnumf %bm{h}_c, %bx{h}_8 : f32
        %bx{h}_16, %by{h}_16 = kernel.subgroup.shuffle<xor> %bm{h}_d, %i16, %i32 : f32, i32, i32
        %bm{h} = scalar.maxnumf %bm{h}_d, %bx{h}_16 : f32
        %mn{h} = scalar.maxnumf %m{h}, %bm{h} : f32
        %ms{h} = scalar.subf %m{h}, %mn{h} : f32
        %al{h} = scalar.expf<afn> %ms{h} : f32
        %alv{h} = vector.splat %al{h} : vector<16xf32>
        %as{h} = vector.mulf %a{h}, %alv{h} : vector<16xf32>
        %ls{h} = scalar.mulf %l{h}, %al{h} : f32
""")
    prev_a, prev_l = f"%as{h}", f"%ls{h}"
    for u in range(U):
        n = f"{h}_{u}"
        e(f"""        %ds{n} = scalar.subf %s{n}, %mn{h} : f32
        %e{n} = scalar.expf<afn> %ds{n} : f32
        %ll{n} = scalar.addf {prev_l}, %e{n} : f32
""" + (f"        %ew{n} = scalar.mulf %e{n}, %vs{u} : f32\n        %ev{n} = vector.splat %ew{n} : vector<16xf32>\n" if KV8
       else f"        %ev{n} = vector.splat %e{n} : vector<16xf32>\n") + f"""        %aa{n} = vector.fmaf %ev{n}, %v{u}, {prev_a} : vector<16xf32>
""")
        prev_a, prev_l = f"%aa{n}", f"%ll{n}"
yield_vals = [f"%mn{h}" for h in range(H)] + [f"%ll{h}_{U - 1}" for h in range(H)] + [f"%aa{h}_{U - 1}" for h in range(H)]
e(f"""        scf.yield {", ".join(yield_vals)} : {carried_ty}
      }}
""")
# wave partials to LDS: acc per (wave, quad, head, dims of d); l per (wave, quad, head) from lane d == 0; m per (wave, head)
for h in range(H):
    e(f"""      vector.store %fa{h}, %acc_l[%wave, %quad, %c{h}_h, %d, %c0] : vector<16xf32>, view<{NW}x4x{H}x8x16xf32>
""")
e("      scf.if %is_d0 {\n")
for h in range(H):
    e(f"        view.store %fl{h}, %m_l[%wave, %quad, %c{h}_h] : f32, view<{NW}x5x{H}xf32>\n")
e("      }\n      scf.if %is_l0 {\n")
for h in range(H):
    e(f"        view.store %fm{h}, %m_l[%wave, %c4, %c{h}_h] : f32, view<{NW}x5x{H}xf32>\n")
e("      }\n      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n")
# merge: thread t handles outputs idx = t, t + T, ... of 3 heads x 128 dims
e(f"""      %pm_v = buffer.view %pm[%zero] : buffer -> view<[%tokens]x24x64xf32>
      %pl_v = buffer.view %pl[%zero] : buffer -> view<[%tokens]x24x64xf32>
      %po_v = buffer.view %po[%zero] : buffer -> view<[%tokens]x24x64x128xf32>
      scf.for %idx = [%tid to %c384 step %cT] {{
        %h_v = index.div %idx, %c128 : index
        %hh = index.assume %h_v [range(%h_v, 0, {H - 1})] : index
        %dim_v = index.rem %idx, %c128 : index
        %dim = index.assume %dim_v [range(%dim_v, 0, 127)] : index
        %mm = scf.for %w = [%c0 to %nwa step %c1](%mx = %neg_inf : f32) -> (f32) {{
          %wb = index.assume %w [range(%w, 0, {NW - 1})] : index
          %mw = view.load %m_l[%wb, %c4, %hh] : view<{NW}x5x{H}xf32> -> f32
          %mxn = scalar.maxnumf %mx, %mw : f32
          scf.yield %mxn : f32
        }}
        %ls_t, %os_t = scf.for %w = [%c0 to %nwa step %c1](%lt = %zero_f : f32, %ot = %zero_f : f32) -> (f32, f32) {{
          %wb = index.assume %w [range(%w, 0, {NW - 1})] : index
          %mw = view.load %m_l[%wb, %c4, %hh] : view<{NW}x5x{H}xf32> -> f32
          %dw = scalar.subf %mw, %mm : f32
          %fw = scalar.expf<afn> %dw : f32
""")
lsum, osum = None, None
for g in range(4):
    e(f"""          %cg{g} = index.constant {g} : index
          %lw{g} = view.load %m_l[%wb, %cg{g}, %hh] : view<{NW}x5x{H}xf32> -> f32
          %ow{g} = view.load %acc_r[%wb, %cg{g}, %hh, %dim] : view<{NW}x4x{H}x128xf32> -> f32
""")
e("""          %lq01 = scalar.addf %lw0, %lw1 : f32
          %lq23 = scalar.addf %lw2, %lw3 : f32
          %lq = scalar.addf %lq01, %lq23 : f32
          %oq01 = scalar.addf %ow0, %ow1 : f32
          %oq23 = scalar.addf %ow2, %ow3 : f32
          %oq = scalar.addf %oq01, %oq23 : f32
          %ltn = scalar.fmaf %lq, %fw, %lt : f32
          %otn = scalar.fmaf %oq, %fw, %ot : f32
          scf.yield %ltn, %otn : f32, f32
        }
        %hdm_v = index.add %hd_base, %hh : index
        %hdm = index.assume %hdm_v [range(%hdm_v, 0, 23)] : index
        view.store %os_t, %po_v[%tok, %hdm, %sp, %dim] : f32, view<[%tokens]x24x64x128xf32>
        %dim0 = index.cmp eq, %dim, %c0 : index
        scf.if %dim0 {
          view.store %mm, %pm_v[%tok, %hdm, %sp] : f32, view<[%tokens]x24x64xf32>
          view.store %ls_t, %pl_v[%tok, %hdm, %sp] : f32, view<[%tokens]x24x64xf32>
        }
      }
    }
  }
  kernel.return
}
""")
text = "".join(L)
# constants used above that need defining once
text = text.replace("  %c0 = index.constant 0 : index\n", "  %c0 = index.constant 0 : index\n  %cS_l = index.constant 64 : index\n"
                    + "".join(f"  %c{h}_h = index.constant {h} : index\n" for h in range(H)), 1)
sys.stdout.write(text)
