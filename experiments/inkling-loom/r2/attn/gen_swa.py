#!/usr/bin/env python3
"""Emit @g_ink_attn_split_swa_v2: sliding-window (1,024 keys) decode and verify attention on the f16 SWA ring.

Takes the arguments of @g_ink_attn_split_swa (one token per grid row; a verify span launches it with all its tokens:
each token finds its span, row and position through info and the span metadata, as @g_ink_attn_split_swa does).
Writes the pm/pl/po partials of the old kernels, with s_eff = min(ceil(keys / G), SMAX) splits; the combine must
use the same split count.

Per workgroup (token, KV head, split): NW waves take the split's 4U-key blocks round-robin. A lane holds 16 dims of
one key (lane = 8 * quad + d) and the key is loaded once for the 3 query heads of its KV head; the online softmax
runs once per block. The waves merge through LDS. Same structure as gen_v2.py (the global-layer kernel).

usage: [NW=4] [U=4] [G=64] [SMAX=16] gen_swa.py > kernel.loom
"""
import os
import sys

NW = int(os.environ.get("NW", 4))
U = int(os.environ.get("U", 4))  # keys per lane per block: a block is 4U keys; 1040 ring slots hold whole blocks
G = int(os.environ.get("G", 64))  # keys per split (at most)
SMAX = int(os.environ.get("SMAX", 16))  # grid splits
TW = int(os.environ.get("TW", 0))  # >0: a workgroup has TW waves, one per token, each over the whole split
if TW:
    NW = TW
H = 3
assert 1040 % (4 * U) == 0 and SMAX <= 64

L = []
e = L.append
e(f"""// SWA decode/verify attention v2 (NW={NW}, U={U}, G={G}, SMAX={SMAX}): {NW} waves per (token, KV head, split) share the
// split's {4 * U}-key blocks; a lane loads 16 dims of one key once for the 3 query heads; softmax once per block.
kernel.def export("g_ink_attn_split_swa_v2") @g_ink_attn_split_swa_v2(%tokens: index) {{
  %c1 = index.constant 1 : index
  %cKH = index.constant 8 : index
  %cS = index.constant {SMAX} : index
  %lanes = index.constant {32 * NW} : index
""" + (f"""  %cTWm1 = index.constant {TW - 1} : index
  %cTWc = index.constant {TW} : index
  %gx0 = index.add %tokens, %cTWm1 : index
  %gx = index.div %gx0, %cTWc : index
  kernel.launch.config workgroups(%gx, %cKH, %cS)""" if TW else """  kernel.launch.config workgroups(%tokens, %cKH, %cS)""") + f""" workgroup_size(%lanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %swa_off_arg: index, %metadata: buffer, %info: buffer, %origins: buffer, %q_in: buffer, %r_in: buffer, %rel_bias: buffer, %arena: buffer, %pm: buffer, %pl: buffer, %po: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %swa_off_b = index.assume %swa_off_arg [range(%swa_off_arg, 0, 1073741824)] : index
  %swa_off = index.cast %swa_off_b : index to offset
""" + ("" if TW else """  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
""") + f"""  %kh_id = kernel.workgroup.id<y> : index
  %kh = index.assume %kh_id [range(%kh_id, 0, 7)] : index
  %sp_id = kernel.workgroup.id<z> : index
  %sp = index.assume %sp_id [range(%sp_id, 0, {SMAX - 1})] : index
  %wave_raw = kernel.subgroup.id : index
  %wave = index.assume %wave_raw [range(%wave_raw, 0, {NW - 1})] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %tid_raw = kernel.workitem.id<x> : index
  %tid = index.assume %tid_raw [range(%tid_raw, 0, {32 * NW - 1})] : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
""" + "".join(f"  %c{h}_h = index.constant {h} : index\n" for h in range(H)) + f"""  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %cB = index.constant {4 * U} : index
  %c128 = index.constant 128 : index
  %c384 = index.constant 384 : index
  %cNW = index.constant {NW} : index
  %cT = index.constant {32 * NW} : index
  %cG = index.constant {G} : index
  %cGm1 = index.constant {G - 1} : index
  %cWIN1 = index.constant 1023 : index
  %cRING = index.constant 1040 : index
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
  %pos_bytes = index.constant 2048 : offset
  %plane_stride = index.constant 2129920 : offset
  %quad_v = index.div %lane, %c8 : index
  %quad = index.assume %quad_v [range(%quad_v, 0, 3)] : index
  %d_v = index.rem %lane, %c8 : index
  %d = index.assume %d_v [range(%d_v, 0, 7)] : index
  %is_d0 = index.cmp eq, %d, %c0 : index
  %is_l0 = index.cmp eq, %lane, %c0 : index
""" + (f"""  %gx_id = kernel.workgroup.id<x> : index
  %cTW = index.constant {TW} : index
  %tok_r = index.madd %gx_id, %cTW, %wave : index
  %tok_in = index.cmp ult, %tok_r, %tokens : index
  %tokens_m1 = index.sub %tokens, %c1 : index
  %tok_m = index.min %tok_r, %tokens_m1 : index
  %tok = index.assume %tok_m [lt(%tok_m, %tokens)] : index
""" if TW else "") + f"""  %info_v = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_v[%tok, 2] : view<[%tokens]x4xi32> -> i32
""" + ("""  %live0 = scalar.cmpi ne, %live32, %c0_i32 : i32
  %live = scalar.andi %live0, %tok_in : i1
""" if TW else """  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
""") + ("" if TW else f"""  %acc_bytes = index.constant {NW * 4 * H * 128 * 4} : offset
  %accs = buffer.alloca<workgroup> align(16) %acc_bytes : buffer
  %acc_l = buffer.view %accs[%zero] : buffer -> view<{NW}x4x{H}x8x16xf32>
  %acc_r = buffer.view %accs[%zero] : buffer -> view<{NW}x4x{H}x128xf32>
  %ml_bytes = index.constant {NW * 5 * H * 4} : offset
  %mls = buffer.alloca<workgroup> align(16) %ml_bytes : buffer
  %m_l = buffer.view %mls[%zero] : buffer -> view<{NW}x5x{H}xf32>
""") + f"""  scf.if %live {{
    %pos32 = view.load %info_v[%tok, 1] : view<[%tokens]x4xi32> -> i32
    %pos_b = scalar.assume %pos32 [range(%pos32, 0, 1048575)] : i32
    %pos = index.cast %pos_b : i32 to index
    %gt = index.cmp ugt, %pos, %cWIN1 : index
    %sub = index.sub %pos, %cWIN1 : index
    %w_low = scf.select %gt, %sub, %c0 : index
    %n_keys_0 = index.sub %pos, %w_low : index
    %n_keys_raw = index.add %n_keys_0, %c1 : index
    %n_keys = index.assume %n_keys_raw [range(%n_keys_raw, 1, 1024)] : index
    %nkg = index.add %n_keys, %cGm1 : index
    %nkg_b = index.assume %nkg [range(%nkg, {G}, {1023 + G})] : index
    %s_need = index.div %nkg_b, %cG : index
    %s_eff0 = index.min %s_need, %cS_l : index
    %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, {SMAX})] : index
    %sp_active = index.cmp ult, %sp, %s_eff : index
    scf.if %sp_active {{
      %span32 = view.load %info_v[%tok, 3] : view<[%tokens]x4xi32> -> i32
      %span_b = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
      %span = index.cast %span_b : i32 to index
      %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
      %ring_rd, %ring_wr, %ring_fr = func.call @g_gdn_slots(%origins, %row, %start32, %swa_off) : (buffer, index, i32, offset) -> (offset, offset, i1)
      %chunk0 = index.add %n_keys, %s_eff : index
      %chunk1_raw = index.sub %chunk0, %c1 : index
      %chunk1 = index.assume %chunk1_raw [range(%chunk1_raw, 1, {1023 + SMAX})] : index
      %chunk = index.div %chunk1, %s_eff : index
      %st0 = index.mul %sp, %chunk : index
      %st = index.min %st0, %n_keys : index
      %en0 = index.add %st, %chunk : index
      %en = index.min %en0, %n_keys : index
      %st_pos_v = index.add %w_low, %st : index
      %st_pos = index.assume %st_pos_v [range(%st_pos_v, 0, 1048575)] : index
      %en_pos_v = index.add %w_low, %en : index
      %en_pos = index.assume %en_pos_v [range(%en_pos_v, 0, 1048576)] : index
      %has_keys = index.cmp ult, %st_pos, %en_pos : index
      %pg0 = index.div %st_pos, %cB : index
      %en_m1_v = index.sub %en_pos, %c1 : index
      %en_m1_s = scf.select %has_keys, %en_m1_v, %st_pos : index
      %en_m1 = index.assume %en_m1_s [range(%en_m1_s, 0, 1048575)] : index
      %pg1_0 = index.div %en_m1, %cB : index
      %pg1_1 = index.add %pg1_0, %c1 : index
      %pg1 = scf.select %has_keys, %pg1_1, %pg0 : index
      %npg = index.sub %pg1, %pg0 : index
      %nwa0 = index.min %npg, %cNW : index
      %nwa = index.max %nwa0, %c1 : index
""" + ("""      %pgs = index.add %pg0, %c0 : index
""" if TW else """      %pgw = index.add %pg0, %wave : index
      %pgs = index.min %pgw, %pg1 : index
""") + f"""      %kh_off = index.constant 256 : offset
      %d_off = index.constant 32 : offset
      %lo_a = index.scale %kh, %kh_off : index, offset -> offset
      %lo_b = index.scale %d, %d_off : index, offset -> offset
      %lane_off0 = index.add %lo_a, %lo_b : offset
      %lane_off = index.add %lane_off0, %ring_wr : offset
      %qv = buffer.view %q_in[%zero] : buffer -> view<[%tokens]x24x8x16xf32>
      %hd_base = index.mul %kh, %c3 : index
""")
for h in range(H):
    e(f"""      %hd{h}_v = index.add %hd_base, %c{h}_h : index
      %hd{h} = index.assume %hd{h}_v [range(%hd{h}_v, 0, 23)] : index
      %qr{h} = vector.load %qv[%tok, %hd{h}, %d, %c0] : view<[%tokens]x24x8x16xf32> -> vector<16xf32>
      %q{h} = vector.mulf %qr{h}, %scale16 : vector<16xf32>
""")
e("""      %rv = buffer.view %r_in[%zero] : buffer -> view<[%tokens]x24x16xf32>
      %rbv = buffer.view %rel_bias[%zero] : buffer -> view<1024x16xf32>
""")
for h in range(H):
    e(f"      %r16_{h} = vector.load %rv[%tok, %hd{h}, %c0] : view<[%tokens]x24x16xf32> -> vector<16xf32>\n")
carried_init = ", ".join([f"%m{h} = %neg_inf : f32" for h in range(H)] + [f"%l{h} = %zero_f : f32" for h in range(H)]
                         + [f"%a{h} = %zero16 : vector<16xf32>" for h in range(H)])
carried_ty = ", ".join(["f32"] * (2 * H) + ["vector<16xf32>"] * H)
res = ", ".join([f"%fm{h}" for h in range(H)] + [f"%fl{h}" for h in range(H)] + [f"%fa{h}" for h in range(H)])
e(f"""      {res} = scf.for %pg = [%pgs to %pg1 step {"%c1" if TW else "%cNW"}]({carried_init}) -> ({carried_ty}) {{
        %p_base = index.mul %pg, %cB : index
        %lo_pos = index.max %p_base, %st_pos : index
        %p_next = index.add %p_base, %cB : index
        %hi_pos = index.min %p_next, %en_pos : index
        %hi_m1 = index.sub %hi_pos, %c1 : index
        %slot = index.rem %p_base, %cRING : index
        %slot_b = index.assume %slot [range(%slot, 0, 1039)] : index
        %slot_off = index.scale %slot_b, %pos_bytes : index, offset -> offset
        %kb = index.add %slot_off, %lane_off : offset
""")
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
        %kvw{u} = buffer.view %arena[%ko{u}] : buffer -> view<16xf16>
        %kh16_{u} = vector.load %kvw{u}[%c0] : view<16xf16> -> vector<16xf16>
        %vvw{u} = buffer.view %arena[%vo{u}] : buffer -> view<16xf16>
        %vh16_{u} = vector.load %vvw{u}[%c0] : view<16xf16> -> vector<16xf16>
        %k{u} = vector.extf %kh16_{u} : vector<16xf16> to vector<16xf32>
        %v{u} = vector.extf %vh16_{u} : vector<16xf16> to vector<16xf32>
        %dist{u} = index.sub %pos, %kc{u} : index
        %db{u} = index.assume %dist{u} [range(%dist{u}, 0, 1023)] : index
        %bv{u} = vector.load %rbv[%db{u}, %c0] : view<1024x16xf32> -> vector<16xf32>
""")
for h in range(H):
    for u in range(U):
        n = f"{h}_{u}"
        e(f"""        %p{n} = vector.dotf<reassoc|contract> %q{h}, %k{u}, %zero_f : vector<16xf32>, vector<16xf32>, f32
        %x1_{n}, %y1_{n} = kernel.subgroup.shuffle<xor> %p{n}, %i1, %i8 : f32, i32, i32
        %r1_{n} = scalar.addf %p{n}, %x1_{n} : f32
        %x2_{n}, %y2_{n} = kernel.subgroup.shuffle<xor> %r1_{n}, %i2, %i8 : f32, i32, i32
        %r2_{n} = scalar.addf %r1_{n}, %x2_{n} : f32
        %x4_{n}, %y4_{n} = kernel.subgroup.shuffle<xor> %r2_{n}, %i4, %i8 : f32, i32, i32
        %sq{n} = scalar.addf %r2_{n}, %x4_{n} : f32
        %b{n} = vector.dotf<reassoc|contract> %r16_{h}, %bv{u}, %zero_f : vector<16xf32>, vector<16xf32>, f32
        %sb{n} = scalar.addf %sq{n}, %b{n} : f32
        %s{n} = scf.select %ok{u}, %sb{n}, %neg_inf : f32
""")
for h in range(H):
    prev = f"%s{h}_0"
    for u in range(1, U):
        e(f"        %bmc{h}_{u} = scalar.maxnumf {prev}, %s{h}_{u} : f32\n")
        prev = f"%bmc{h}_{u}"
    e(f"""        %bx{h}_8, %by{h}_8 = kernel.subgroup.shuffle<xor> {prev}, %i8, %i32 : f32, i32, i32
        %bm{h}_d = scalar.maxnumf {prev}, %bx{h}_8 : f32
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
        %ev{n} = vector.splat %e{n} : vector<16xf32>
        %aa{n} = vector.fmaf %ev{n}, %v{u}, {prev_a} : vector<16xf32>
""")
        prev_a, prev_l = f"%aa{n}", f"%ll{n}"
yield_vals = [f"%mn{h}" for h in range(H)] + [f"%ll{h}_{U - 1}" for h in range(H)] + [f"%aa{h}_{U - 1}" for h in range(H)]
e(f"""        scf.yield {", ".join(yield_vals)} : {carried_ty}
      }}
""")
def tw_tail():
    e("""      %pm_v = buffer.view %pm[%zero] : buffer -> view<[%tokens]x24x64xf32>
      %pl_v = buffer.view %pl[%zero] : buffer -> view<[%tokens]x24x64xf32>
      %po_v = buffer.view %po[%zero] : buffer -> view<[%tokens]x24x64x8x16xf32>
      %is_q0 = index.cmp eq, %quad, %c0 : index
""")
    for h in range(H):
        e(f"""      %lx8_{h}, %ly8_{h} = kernel.subgroup.shuffle<xor> %fl{h}, %i8, %i32 : f32, i32, i32
      %lq8_{h} = scalar.addf %fl{h}, %lx8_{h} : f32
      %lx16_{h}, %ly16_{h} = kernel.subgroup.shuffle<xor> %lq8_{h}, %i16, %i32 : f32, i32, i32
      %lt{h} = scalar.addf %lq8_{h}, %lx16_{h} : f32
""")
        el = []
        for x in range(16):
            e(f"""      %ae{h}_{x} = vector.extract %fa{h}[{x}] : vector<16xf32> -> f32
      %ax8_{h}_{x}, %ay8_{h}_{x} = kernel.subgroup.shuffle<xor> %ae{h}_{x}, %i8, %i32 : f32, i32, i32
      %aq8_{h}_{x} = scalar.addf %ae{h}_{x}, %ax8_{h}_{x} : f32
      %ax16_{h}_{x}, %ay16_{h}_{x} = kernel.subgroup.shuffle<xor> %aq8_{h}_{x}, %i16, %i32 : f32, i32, i32
      %at{h}_{x} = scalar.addf %aq8_{h}_{x}, %ax16_{h}_{x} : f32
""")
            el.append(f"%at{h}_{x}")
        e(f"""      %av{h} = vector.from_elements {", ".join(el)} : vector<16xf32>
      scf.if %is_q0 {{
        vector.store %av{h}, %po_v[%tok, %hd{h}, %sp, %d, %c0] : vector<16xf32>, view<[%tokens]x24x64x8x16xf32>
      }}
      scf.if %is_l0 {{
        view.store %fm{h}, %pm_v[%tok, %hd{h}, %sp] : f32, view<[%tokens]x24x64xf32>
        view.store %lt{h}, %pl_v[%tok, %hd{h}, %sp] : f32, view<[%tokens]x24x64xf32>
      }}
""")
    e("""    }
  }
  kernel.return
}
""")


mark = len(L)
for h in range(H):
    e(f"      vector.store %fa{h}, %acc_l[%wave, %quad, %c{h}_h, %d, %c0] : vector<16xf32>, view<{NW}x4x{H}x8x16xf32>\n")
e("      scf.if %is_d0 {\n")
for h in range(H):
    e(f"        view.store %fl{h}, %m_l[%wave, %quad, %c{h}_h] : f32, view<{NW}x5x{H}xf32>\n")
e("      }\n      scf.if %is_l0 {\n")
for h in range(H):
    e(f"        view.store %fm{h}, %m_l[%wave, %c4, %c{h}_h] : f32, view<{NW}x5x{H}xf32>\n")
e("      }\n      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n")
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

if TW:
    del L[mark:]
    tw_tail()
text = "".join(L).replace("  %c0 = index.constant 0 : index\n", f"  %c0 = index.constant 0 : index\n  %cS_l = index.constant {SMAX} : index\n", 1)
sys.stdout.write(text)
