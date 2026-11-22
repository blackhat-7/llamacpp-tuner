#!/usr/bin/env python3
"""Emit @g_ink_attn_combine: the same arguments and output (gated attention row as q8 codes + scale and code sum)
as the shipped combine, but the split reductions run across lanes instead of one split at a time.

Split counts: sliding layers s_eff = min(ceil(keys / GS), SMAXS); global layers min(ceil(keys / GG), 64).
They must match the split kernels (gen_swa.py G/SMAX; the v2 kernel's ceil(keys / 8)).

One wave per (token, query head); lane l owns dims 4l..4l+3 and the weights of splits l and l + 32 (to LDS).
The weighted sum over splits is unrolled by UN with independent loads.

usage: [GS=64] [SMAXS=16] [GG=8] [UN=8] gen_combine.py > kernel.loom
"""
import os
import sys

GS = int(os.environ.get("GS", 64))
SMAXS = int(os.environ.get("SMAXS", 16))
GG = int(os.environ.get("GG", 8))
UN = int(os.environ.get("UN", 8))

L = []
e = L.append
e(f"""// Combine split partials (GS={GS}, SMAXS={SMAXS}, GG={GG}): max and denominator across lanes, weighted sum unrolled
// by {UN}; then sinks, sigmoid gate and the q8 row of the shipped combine.
kernel.def export("g_ink_attn_combine") @g_ink_attn_combine(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %cQH = index.constant 24 : index
  kernel.launch.config workgroups(%tokens, %cQH, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %sliding_arg: i32, %info: buffer, %pm: buffer, %pl: buffer, %po: buffer, %sinks: buffer, %gate: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %hd_id = kernel.workgroup.id<y> : index
  %hd = index.assume %hd_id [range(%hd_id, 0, 23)] : index
  %lane_raw = kernel.workitem.id<x> : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %c32 = index.constant 32 : index
  %cUN = index.constant {UN} : index
  %cGS = index.constant {GS - 1} : index
  %cGG = index.constant {GG - 1} : index
  %dGS = index.constant {GS} : index
  %dGG = index.constant {GG} : index
  %cSS = index.constant {SMAXS} : index
  %cS = index.constant 64 : index
  %cWIN1 = index.constant 1023 : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_inf = scalar.constant -3.40282347e+38 : f32
  %zero4 = vector.splatf 0.0 : vector<4xf32>
  %c4 = index.constant 4 : index
  %c127_f = scalar.constant 127.0 : f32
  %c1_i32 = scalar.constant 1 : i32
  %c2_i32 = scalar.constant 2 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c8_i32 = scalar.constant 8 : i32
  %zero1i = vector.splati 0 : vector<1xi32>
  %ones4 = vector.splati 1 : vector<4xi8>
  %w_bytes = index.constant 256 : offset
  %w_lds = buffer.alloca<workgroup> align(16) %w_bytes : buffer
  %wl = buffer.view %w_lds[%zero] : buffer -> view<64xf32>
  %info_v = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %qg_in_hd = index.div %lane, %c8 : index
  %sub_l = index.rem %lane, %c8 : index
  %g_base = index.mul %hd, %c4 : index
  %g_v = index.add %g_base, %qg_in_hd : index
  %g = index.assume %g_v [range(%g_v, 0, 95)] : index
  %qv = buffer.view %out[%zero] : buffer -> view<[%tokens]x96x8xi32>
  %m_bytes = index.constant 3072 : offset
  %m_off = index.scale %tokens, %m_bytes : index, offset -> offset
  %mv = buffer.view %out[%m_off] : buffer -> view<[%tokens]x96x2xf32>
  %is_sub0 = index.cmp eq, %sub_l, %c0 : index
  %live32 = view.load %info_v[%tok, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %res = scf.if %live -> (vector<4xf32>) {{
    %pos32 = view.load %info_v[%tok, 1] : view<[%tokens]x4xi32> -> i32
    %pos_b = scalar.assume %pos32 [range(%pos32, 0, 2097151)] : i32
    %pos = index.cast %pos_b : i32 to index
    %is_swa = scalar.cmpi ne, %sliding_arg, %c0_i32 : i32
    %w_low, %gm1, %gdiv, %smax = scf.if %is_swa -> (index, index, index, index) {{
      %gt = index.cmp ugt, %pos, %cWIN1 : index
      %sub = index.sub %pos, %cWIN1 : index
      %wlo = scf.select %gt, %sub, %c0 : index
      scf.yield %wlo, %cGS, %dGS, %cSS : index, index, index, index
    }} else {{
      scf.yield %c0, %cGG, %dGG, %cS : index, index, index, index
    }}
    %nk0 = index.sub %pos, %w_low : index
    %nk_raw = index.add %nk0, %c1 : index
    %nk = index.assume %nk_raw [range(%nk_raw, 1, 2097152)] : index
    %nkg = index.add %nk, %gm1 : index
    %nkg_b = index.assume %nkg [range(%nkg, 1, 2097216)] : index
    %gdiv_b = index.assume %gdiv [range(%gdiv, 1, 1024)] : index
    %s_need = index.div %nkg_b, %gdiv_b : index
    %s_eff0 = index.min %s_need, %smax : index
    %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, 64)] : index
    %pm_v = buffer.view %pm[%zero] : buffer -> view<[%tokens]x24x64xf32>
    %pl_v = buffer.view %pl[%zero] : buffer -> view<[%tokens]x24x64xf32>
    %po_v = buffer.view %po[%zero] : buffer -> view<[%tokens]x24x64x32x4xf32>
    %sv = buffer.view %sinks[%zero] : buffer -> view<[%tokens]x24xf32>
    %gv = buffer.view %gate[%zero] : buffer -> view<[%tokens]x24x32x4xf32>
    %sink = view.load %sv[%tok, %hd] : view<[%tokens]x24xf32> -> f32
    %s_hi_v = index.add %lane, %c32 : index
    %s_hi = index.assume %s_hi_v [range(%s_hi_v, 32, 63)] : index
    %in_a = index.cmp ult, %lane, %s_eff : index
    %in_b = index.cmp ult, %s_hi, %s_eff : index
    %ma, %la = scf.if %in_a -> (f32, f32) {{
      %m = view.load %pm_v[%tok, %hd, %lane] : view<[%tokens]x24x64xf32> -> f32
      %l = view.load %pl_v[%tok, %hd, %lane] : view<[%tokens]x24x64xf32> -> f32
      scf.yield %m, %l : f32, f32
    }} else {{
      scf.yield %neg_inf, %zero_f : f32, f32
    }}
    %mb, %lb = scf.if %in_b -> (f32, f32) {{
      %m = view.load %pm_v[%tok, %hd, %s_hi] : view<[%tokens]x24x64xf32> -> f32
      %l = view.load %pl_v[%tok, %hd, %s_hi] : view<[%tokens]x24x64xf32> -> f32
      scf.yield %m, %l : f32, f32
    }} else {{
      scf.yield %neg_inf, %zero_f : f32, f32
    }}
    %mab = scalar.maxnumf %ma, %mb : f32
    %mlane = scalar.maxnumf %mab, %sink : f32
    %mx = kernel.subgroup.reduce<maxnumf> %mlane : f32
    %dma = scalar.subf %ma, %mx : f32
    %wa0 = scalar.expf<afn> %dma : f32
    %wa = scf.select %in_a, %wa0, %zero_f : f32
    %dmb = scalar.subf %mb, %mx : f32
    %wb0 = scalar.expf<afn> %dmb : f32
    %wb = scf.select %in_b, %wb0, %zero_f : f32
    view.store %wa, %wl[%lane] : f32, view<64xf32>
    view.store %wb, %wl[%s_hi] : f32, view<64xf32>
    %lwa = scalar.mulf %la, %wa : f32
    %lw = scalar.fmaf %lb, %wb, %lwa : f32
    %lsum = kernel.subgroup.reduce<addf> %lw : f32
    %s_sh = scalar.subf %sink, %mx : f32
    %d0 = scalar.expf<afn> %s_sh : f32
    %df = scalar.addf %lsum, %d0 : f32
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %s_last = index.sub %s_eff, %c1 : index
    %af = scf.for %s0 = [%c0 to %s_eff step %cUN](%a = %zero4 : vector<4xf32>) -> (vector<4xf32>) {{
""")
prev = "%a"
for u in range(UN):
    e(f"""      %su{u} = index.constant {u} : index
      %sr{u} = index.add %s0, %su{u} : index
      %sc{u} = index.min %sr{u}, %s_last : index
      %sb{u} = index.assume %sc{u} [range(%sc{u}, 0, 63)] : index
      %os{u} = vector.load %po_v[%tok, %hd, %sb{u}, %lane, %c0] : view<[%tokens]x24x64x32x4xf32> -> vector<4xf32>
      %ok{u} = index.cmp ult, %sr{u}, %s_eff : index
      %ws{u} = view.load %wl[%sb{u}] : view<64xf32> -> f32
      %w{u} = scf.select %ok{u}, %ws{u}, %zero_f : f32
""")
for u in range(UN):
    e(f"""      %wv{u} = vector.splat %w{u} : vector<4xf32>
      %na{u} = vector.fmaf %os{u}, %wv{u}, {prev} : vector<4xf32>
""")
    prev = f"%na{u}"
e(f"""      scf.yield {prev} : vector<4xf32>
    }}
    %one_f = scalar.constant 1.0 : f32
    %inv_d = scalar.divf %one_f, %df : f32
    %idv = vector.splat %inv_d : vector<4xf32>
    %att = vector.mulf %af, %idv : vector<4xf32>
    %g4 = vector.load %gv[%tok, %hd, %lane, %c0] : view<[%tokens]x24x32x4xf32> -> vector<4xf32>
    %sig = vector.logisticf<afn> %g4 : vector<4xf32>
    %res_live = vector.mulf %att, %sig : vector<4xf32>
    scf.yield %res_live : vector<4xf32>
  }} else {{
    scf.yield %zero4 : vector<4xf32>
  }}
""")
# The q8 epilogue is the shipped kernel's, verbatim.
e("""  %av = vector.absf %res : vector<4xf32>
  %am0 = vector.reduce<maxnumf> %av, %zero_f : vector<4xf32>, f32
  %am1p, %am1u = kernel.subgroup.shuffle<xor> %am0, %c1_i32, %c8_i32 : f32, i32, i32
  %am1 = scalar.maxnumf %am0, %am1p : f32
  %am2p, %am2u = kernel.subgroup.shuffle<xor> %am1, %c2_i32, %c8_i32 : f32, i32, i32
  %am2 = scalar.maxnumf %am1, %am2p : f32
  %am3p, %am3u = kernel.subgroup.shuffle<xor> %am2, %c4_i32, %c8_i32 : f32, i32, i32
  %amax = scalar.maxnumf %am2, %am3p : f32
  %d = scalar.divf %amax, %c127_f : f32
  %nz = scalar.cmpf one, %amax, %zero_f : f32
  %dv = vector.splat %d : vector<4xf32>
  %sc = vector.divf %res, %dv : vector<4xf32>
  %rd = vector.roundf %sc : vector<4xf32>
  %nzv = vector.splat %nz : vector<4xi1>
  %rs = vector.select %nzv, %rd, %zero4 : vector<4xf32>
  %q4 = vector.fptosi %rs : vector<4xf32> to vector<4xi8>
  %qw1 = vector.bitcast %q4 : vector<4xi8> to vector<1xi32>
  %qw = vector.extract %qw1[0] : vector<1xi32> -> i32
  view.store %qw, %qv[%tok, %g, %sub_l] : i32, view<[%tokens]x96x8xi32>
  %qs1 = vector.dot4i<s8s8> %q4, %ones4, %zero1i : vector<4xi8>, vector<4xi8>, vector<1xi32>
  %q0 = vector.extract %qs1[0] : vector<1xi32> -> i32
  %q1p, %q1u = kernel.subgroup.shuffle<xor> %q0, %c1_i32, %c8_i32 : i32, i32, i32
  %q1 = scalar.addi %q0, %q1p : i32
  %q2p, %q2u = kernel.subgroup.shuffle<xor> %q1, %c2_i32, %c8_i32 : i32, i32, i32
  %q2 = scalar.addi %q1, %q2p : i32
  %q3p, %q3u = kernel.subgroup.shuffle<xor> %q2, %c4_i32, %c8_i32 : i32, i32, i32
  %qtot = scalar.addi %q2, %q3p : i32
  scf.if %is_sub0 {
    %qsf = scalar.sitofp %qtot : i32 to f32
    %dh = scalar.fptrunc %d : f32 to f16
    %da = scalar.extf %dh : f16 to f32
    %meta = vector.from_elements %da, %qsf : vector<2xf32>
    vector.store %meta, %mv[%tok, %g, %c0] : vector<2xf32>, view<[%tokens]x96x2xf32>
  }
  kernel.return
}
""")
sys.stdout.write("".join(L))
