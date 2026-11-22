"""RMSNorm + planar Q8 activation in one launch: g_ink_rms_norm_plane then g_quant_q8(width 2304), bit for bit.

The norm keeps g_ink_rms_norm_plane's layout (256 threads, vec4 channels tid, tid+256, tid+512) so its sums are
the same; each 32-value Q8 group is 8 consecutive threads, combined with xor shuffles (max and integer sums are
order-free). P = 1 or 3 norms per launch (grid y picks the norm)."""


def body(fn):
    return f"""func.def inline @{fn}(%tokens: index, %pl: index, %pi: index, %x: buffer, %w: buffer, %out: buffer, %outq: buffer) {{
  %tok_id = kernel.workgroup.id<x> : index
  %tok = index.assume %tok_id [lt(%tok_id, %tokens)] : index
  %tid_raw = kernel.workitem.id<x> : index
  %tid = index.assume %tid_raw [range(%tid_raw, 0, 255)] : index
  %c0 = index.constant 0 : index
  %c8 = index.constant 8 : index
  %c256 = index.constant 256 : index
  %cH4 = index.constant 576 : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %zero4 = vector.splatf 0.0 : vector<4xf32>
  %eps = scalar.constant 9.999999974752427e-07 : f32
  %inv_H = scalar.constant 0.00043402777777777775 : f32
  %c127 = scalar.constant 127.0 : f32
  %zero_i = scalar.constant 0 : i32
  %bw32 = scalar.constant 32 : i32
  %s1 = scalar.constant 1 : i32
  %s2 = scalar.constant 2 : i32
  %s4 = scalar.constant 4 : i32
  %ones4 = vector.splati 1 : vector<4xi8>
  %zero1i = vector.splati 0 : vector<1xi32>
  %xv = buffer.view %x[%zero] : buffer -> view<[%pl]x[%tokens]x576x4xf32>
  %wv = buffer.view %w[%zero] : buffer -> view<576x4xf32>
  %ov = buffer.view %out[%zero] : buffer -> view<[%tokens]x576x4xf32>
  %qv = buffer.view %outq[%zero] : buffer -> view<[%tokens]x576xi32>
  %m_bytes = index.constant 2304 : offset
  %m_off = index.scale %tokens, %m_bytes : index, offset -> offset
  %mv = buffer.view %outq[%m_off] : buffer -> view<[%tokens]x72x2xf32>
  %sq = scf.for %i = [%tid to %cH4 step %c256](%a = %zero_f : f32) -> (f32) {{
    %ch = index.assume %i [range(%i, 0, 575)] : index
    %v = vector.load %xv[%pi, %tok, %ch, %c0] : view<[%pl]x[%tokens]x576x4xf32> -> vector<4xf32>
    %na = vector.dotf<reassoc|contract> %v, %v, %a : vector<4xf32>, vector<4xf32>, f32
    scf.yield %na : f32
  }}
  %tot = kernel.workgroup.reduce<addf> %sq : f32
  %mn = scalar.mulf %tot, %inv_H : f32
  %vr = scalar.addf %mn, %eps : f32
  %ir = scalar.rsqrtf %vr : f32
  %ir4 = vector.splat %ir : vector<4xf32>
  %sub = index.rem %tid, %c8 : index
  %lead = index.cmp eq, %sub, %c0 : index
  scf.for %i = [%tid to %cH4 step %c256] {{
    %ch = index.assume %i [range(%i, 0, 575)] : index
    %v = vector.load %xv[%pi, %tok, %ch, %c0] : view<[%pl]x[%tokens]x576x4xf32> -> vector<4xf32>
    %gw = vector.load %wv[%ch, %c0] : view<576x4xf32> -> vector<4xf32>
    %nv = vector.mulf %v, %ir4 : vector<4xf32>
    %y = vector.mulf %nv, %gw : vector<4xf32>
    vector.store %y, %ov[%tok, %ch, %c0] : vector<4xf32>, view<[%tokens]x576x4xf32>
    %ab = vector.absf %y : vector<4xf32>
    %am0 = vector.reduce<maxnumf> %ab, %zero_f : vector<4xf32>, f32
    %am1r, %am1u = kernel.subgroup.shuffle<xor> %am0, %s1, %bw32 : f32, i32, i32
    %am1 = scalar.maxnumf %am0, %am1r : f32
    %am2r, %am2u = kernel.subgroup.shuffle<xor> %am1, %s2, %bw32 : f32, i32, i32
    %am2 = scalar.maxnumf %am1, %am2r : f32
    %am4r, %am4u = kernel.subgroup.shuffle<xor> %am2, %s4, %bw32 : f32, i32, i32
    %amax = scalar.maxnumf %am2, %am4r : f32
    %d = scalar.divf %amax, %c127 : f32
    %nz = scalar.cmpf one, %amax, %zero_f : f32
    %dv = vector.splat %d : vector<4xf32>
    %sc = vector.divf %y, %dv : vector<4xf32>
    %rd = vector.roundf %sc : vector<4xf32>
    %nzv = vector.splat %nz : vector<4xi1>
    %rs = vector.select %nzv, %rd, %zero4 : vector<4xf32>
    %q = vector.fptosi %rs : vector<4xf32> to vector<4xi8>
    %qw = vector.bitcast %q : vector<4xi8> to vector<1xi32>
    %qi = vector.extract %qw[0] : vector<1xi32> -> i32
    view.store %qi, %qv[%tok, %ch] : i32, view<[%tokens]x576xi32>
    %qs1 = vector.dot4i<s8s8> %q, %ones4, %zero1i : vector<4xi8>, vector<4xi8>, vector<1xi32>
    %qs0 = vector.extract %qs1[0] : vector<1xi32> -> i32
    %qs1r, %qs1u = kernel.subgroup.shuffle<xor> %qs0, %s1, %bw32 : i32, i32, i32
    %qa1 = scalar.addi %qs0, %qs1r : i32
    %qs2r, %qs2u = kernel.subgroup.shuffle<xor> %qa1, %s2, %bw32 : i32, i32, i32
    %qa2 = scalar.addi %qa1, %qs2r : i32
    %qs4r, %qs4u = kernel.subgroup.shuffle<xor> %qa2, %s4, %bw32 : i32, i32, i32
    %qsi = scalar.addi %qa2, %qs4r : i32
    scf.if %lead {{
      %g_v = index.div %ch, %c8 : index
      %g = index.assume %g_v [range(%g_v, 0, 71)] : index
      %qsf = scalar.sitofp %qsi : i32 to f32
      %dh = scalar.fptrunc %d : f32 to f16
      %da = scalar.extf %dh : f16 to f32
      %meta = vector.from_elements %da, %qsf : vector<2xf32>
      vector.store %meta, %mv[%tok, %g, %c0] : vector<2xf32>, view<[%tokens]x72x2xf32>
    }}
  }}
  func.return
}}
"""


def kernel(name, fn, nnorm):
    args = ", ".join(
        f"%pl{p}_arg: index, %pi{p}_arg: index, %x{p}: buffer, %w{p}: buffer, %out{p}: buffer, %outq{p}: buffer"
        for p in range(nnorm)
    )
    lines = [
        f"""kernel.def export("{name}") @{name}(%tokens: index) {{
  %c1 = index.constant 1 : index
  %cP = index.constant {nnorm} : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %cP, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, {args}) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %py = kernel.workgroup.id<y> : index"""
    ]
    for p in range(nnorm):
        lines.append(f"""  %pl{p} = index.assume %pl{p}_arg [range(%pl{p}_arg, 1, 4)] : index
  %pi{p} = index.assume %pi{p}_arg [range(%pi{p}_arg, 0, 3), lt(%pi{p}_arg, %pl{p})] : index""")

    def call(p):
        return f"func.call @{fn}(%tokens, %pl{p}, %pi{p}, %x{p}, %w{p}, %out{p}, %outq{p}) : (index, index, index, buffer, buffer, buffer, buffer)\n"

    def chain(p):
        if p == nnorm - 1:
            return call(p)
        return f"%isp{p} = index.cmp eq, %py, %cp{p} : index\nscf.if %isp{p} {{\n{call(p)}}} else {{\n{chain(p + 1)}}}\n"

    for p in range(nnorm - 1):
        lines.append(f"  %cp{p} = index.constant {p} : index")
    lines.append(chain(0) + "kernel.return\n}\n")
    return "\n".join(lines)


def module(prefix):
    return (
        body(f"{prefix}_body")
        + kernel(f"{prefix}1", f"{prefix}_body", 1)
        + kernel(f"{prefix}3", f"{prefix}_body", 3)
    )
