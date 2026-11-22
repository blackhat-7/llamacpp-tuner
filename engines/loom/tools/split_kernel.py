"""Emit the KV8 decode split kernel @g_ink_attn_split (same launch signature and partial-buffer layout as the f16 one)."""


def kernel() -> str:
    parts = []
    a = parts.append
    a("""// Split-K causal GQA decode attention over the KV8 planes, relative-position bias (dist < 1024), optional SWA.
// One wave per query head; each split owns whole 16-key pages. Each lane holds 4 dims of q; a page's 16 partial
// dot products are summed in one butterfly (16 shuffles), after which lane (lane & 15) owns key (lane & 15): it
// applies that key's bias. Scales are per block of 32 dims (8 lanes), so each lane scales its own partials and V codes.
// Keys past the visible range read stale codes (finite int8) whose scales are never used: selects drop them.
kernel.def export("g_ink_attn_split") @g_ink_attn_split(%tokens: index) {
  %c1 = index.constant 1 : index
  %cKH = index.constant 8 : index
  %cS = index.constant 64 : index
  %lanes = index.constant 96 : index
  kernel.launch.config workgroups(%tokens, %cKH, %cS) workgroup_size(%lanes, %c1, %c1) : index
} launch(%tokens_arg: index, %blocks_arg: index, %pool_arg: index, %layer_arg: index, %sliding_arg: i32, %info: buffer, %origins: buffer, %q_in: buffer, %r_in: buffer, %rel_bias: buffer, %arena: buffer, %pm: buffer, %pl: buffer, %po: buffer) {
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
  %wave = index.assume %wave_raw [range(%wave_raw, 0, 2)] : index
  %lane_raw = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_raw [range(%lane_raw, 0, 31)] : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %cG = index.constant 3 : index
  %cS = index.constant 64 : index
  %cEXT = index.constant 1024 : index
  %cWIN1 = index.constant 1023 : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %w32 = scalar.constant 32 : i32
  %x1 = scalar.constant 1 : i32
  %x2 = scalar.constant 2 : i32
  %x4 = scalar.constant 4 : i32
  %x8 = scalar.constant 8 : i32
  %x16 = scalar.constant 16 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_inf = scalar.constant -3.40282347e+38 : f32
  %zero4 = vector.splatf 0.0 : vector<4xf32>
  %scale = scalar.constant 0.08838834764831843 : f32
  %scale4 = vector.splat %scale : vector<4xf32>
  %hd_base = index.mul %kh, %cG : index
  %hd_v = index.add %hd_base, %wave : index
  %hd = index.assume %hd_v [range(%hd_v, 0, 23)] : index
  %info_v = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_v[%tok, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %pm_v = buffer.view %pm[%zero] : buffer -> view<[%tokens]x24x64xf32>
  %pl_v = buffer.view %pl[%zero] : buffer -> view<[%tokens]x24x64xf32>
  %po_v = buffer.view %po[%zero] : buffer -> view<[%tokens]x24x64x32x4xf32>
  %is_l0 = index.cmp eq, %lane, %c0 : index
  %kl = index.rem %lane, %c16 : index
  %kl_b = index.assume %kl [range(%kl, 0, 15)] : index
  %lane_blk = index.div %lane, %c8 : index
  %sidx = index.madd %kh, %c4, %lane_blk : index""")
    for k, d in enumerate((1, 2, 4, 8)):
        src = "%lane" if d == 1 else f"%bq{k}"
        if d > 1:
            a(f"  %bq{k} = index.div %lane, %c{d} : index")
        a(f"""  %bm{k} = index.rem {src}, %c2 : index
  %bit{k} = index.cmp ne, %bm{k}, %c0 : index""")
    a("""  scf.if %live {
    %row32 = view.load %info_v[%tok, 0] : view<[%tokens]x4xi32> -> i32
    %pos32 = view.load %info_v[%tok, 1] : view<[%tokens]x4xi32> -> i32
    %row_b = scalar.assume %row32 [range(%row32, 0, 1023)] : i32
    %pos_b = scalar.assume %pos32 [range(%pos32, 0, 2097151)] : i32
    %row = index.cast %row_b : i32 to index
    %pos = index.cast %pos_b : i32 to index
    %is_swa = scalar.cmpi ne, %sliding_arg, %c0_i32 : i32
    %w_low = scf.if %is_swa -> (index) {
      %gt = index.cmp ugt, %pos, %cWIN1 : index
      %sub = index.sub %pos, %cWIN1 : index
      %wl = scf.select %gt, %sub, %c0 : index
      scf.yield %wl : index
    } else {
      scf.yield %c0 : index
    }
    %n_keys_0 = index.sub %pos, %w_low : index
    %n_keys_raw = index.add %n_keys_0, %c1 : index
    %n_keys = index.assume %n_keys_raw [range(%n_keys_raw, 1, 2097152)] : index
    // The combine kernel reads splits [0, s_eff): s_eff = min(S, ceil(n_keys / 8)) must match it.
    %c7 = index.constant 7 : index
    %nk7 = index.add %n_keys, %c7 : index
    %nk7_b = index.assume %nk7 [range(%nk7, 8, 2097159)] : index
    %s_need = index.div %nk7_b, %c8 : index
    %s_eff0 = index.min %s_need, %cS : index
    %s_eff = index.assume %s_eff0 [range(%s_eff0, 1, 64)] : index
    %sp_active = index.cmp ult, %sp, %s_eff : index
    scf.if %sp_active {
      %pg_first = index.div %w_low, %c16 : index
      %pg_last = index.div %pos, %c16 : index
      %n_pg_0 = index.sub %pg_last, %pg_first : index
      %n_pg_raw = index.add %n_pg_0, %c1 : index
      %n_pg = index.assume %n_pg_raw [range(%n_pg_raw, 1, 131072)] : index
      %cpp_0 = index.add %n_pg, %s_eff : index
      %cpp_1 = index.sub %cpp_0, %c1 : index
      %cpp = index.div %cpp_1, %s_eff : index
      %lo_rel_0 = index.mul %sp, %cpp : index
      %lo_rel = index.min %lo_rel_0, %n_pg : index
      %hi_rel_0 = index.add %lo_rel, %cpp : index
      %hi_rel = index.min %hi_rel_0, %n_pg : index
      %pg_lo = index.add %pg_first, %lo_rel : index
      %pg_hi = index.add %pg_first, %hi_rel : index
      %k_plane = index.mul %layer, %c2 : index
      %pos_bytes = index.constant 1088 : offset
      %plane_stride = index.scale %pool, %pos_bytes : index, offset -> offset
      %qv = buffer.view %q_in[%zero] : buffer -> view<[%tokens]x24x32x4xf32>
      %q_raw = vector.load %qv[%tok, %hd, %lane, %c0] : view<[%tokens]x24x32x4xf32> -> vector<4xf32>
      %q4 = vector.mulf %q_raw, %scale4 : vector<4xf32>
      %rv = buffer.view %r_in[%zero] : buffer -> view<[%tokens]x24x16xf32>
      %r16 = vector.load %rv[%tok, %hd, %c0] : view<[%tokens]x24x16xf32> -> vector<16xf32>
      %rbv = buffer.view %rel_bias[%zero] : buffer -> view<1024x16xf32>
      %mf, %lf, %af = scf.for %pg = [%pg_lo to %pg_hi step %c1](%m = %neg_inf : f32, %l = %zero_f : f32, %acc = %zero4 : vector<4xf32>) -> (f32, f32, vector<4xf32>) {
        %p_base = index.mul %pg, %c16 : index
        %ko_base = func.call @g_kv8_offset(%origins, %row, %p_base, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
        %vo_base = index.add %ko_base, %plane_stride : offset""")
    for j in range(16):
        a(f"""        %ko{j} = index.add %ko_base, %o{j} : offset
        %kc{j} = buffer.view %arena[%ko{j}] : buffer -> view<8x32x4xi8>
        %ki{j} = vector.load %kc{j}[%kh, %lane, %c0] : view<8x32x4xi8> -> vector<4xi8>
        %vo{j} = index.add %vo_base, %o{j} : offset
        %vc{j} = buffer.view %arena[%vo{j}] : buffer -> view<8x32x4xi8>
        %vi{j} = vector.load %vc{j}[%kh, %lane, %c0] : view<8x32x4xi8> -> vector<4xi8>""")
    for j in range(16):
        a(f"""        %ksh{j} = func.call @g_kv8_scale(%arena, %ko{j}, %sidx) : (buffer, offset, index) -> (f16)
        %vsh{j} = func.call @g_kv8_scale(%arena, %vo{j}, %sidx) : (buffer, offset, index) -> (f16)
        %kf{j} = vector.sitofp %ki{j} : vector<4xi8> to vector<4xf32>
        %dr{j} = vector.dotf<reassoc|contract> %q4, %kf{j}, %zero_f : vector<4xf32>, vector<4xf32>, f32
        %ksf{j} = scalar.extf %ksh{j} : f16 to f32
        %d0_{j} = scalar.mulf %dr{j}, %ksf{j} : f32""")
    # butterfly: stage k halves the value list; lane keeps the keys whose bit k equals its own lane bit k
    vals = [f"%d0_{j}" for j in range(16)]
    for k, x in enumerate(("%x1", "%x2", "%x4", "%x8")):
        nxt = []
        for i in range(len(vals) // 2):
            lo, hi = vals[2 * i], vals[2 * i + 1]
            n = f"{k + 1}_{i}"
            a(f"""        %give{n} = scf.select %bit{k}, {lo}, {hi} : f32
        %keep{n} = scf.select %bit{k}, {hi}, {lo} : f32
        %got{n}, %gu{n} = kernel.subgroup.shuffle<xor> %give{n}, {x}, %w32 : f32, i32, i32
        %d{n} = scalar.addf %keep{n}, %got{n} : f32""")
            nxt.append(f"%d{n}")
        vals = nxt
    a(f"""        %got5, %gu5 = kernel.subgroup.shuffle<xor> {vals[0]}, %x16, %w32 : f32, i32, i32
        %s_raw = scalar.addf {vals[0]}, %got5 : f32
        %kp = index.add %p_base, %kl_b : index
        %in_lo = index.cmp uge, %kp, %w_low : index
        %in_hi = index.cmp ule, %kp, %pos : index
        %valid = scalar.andi %in_lo, %in_hi : i1
        %s_qk = scalar.addf %s_raw, %zero_f : f32
        %pg_end = index.add %p_base, %c16 : index
        %min_dist = index.sub %pos, %pg_end : index
        %far_pg = index.cmp uge, %pos, %pg_end : index
        %far_dist = index.cmp uge, %min_dist, %cEXT : index
        %no_bias = scalar.andi %far_pg, %far_dist : i1
        %s = scf.if %no_bias -> (f32) {{
          scf.yield %s_qk : f32
        }} else {{
          %dist = index.sub %pos, %kp : index
          %in_ext = index.cmp ult, %dist, %cEXT : index
          %biased = scalar.andi %valid, %in_ext : i1
          %rb = scf.if %biased -> (f32) {{
            %db = index.assume %dist [range(%dist, 0, 1023)] : index
            %bv = vector.load %rbv[%db, %c0] : view<1024x16xf32> -> vector<16xf32>
            %rd = vector.dotf<reassoc|contract> %r16, %bv, %zero_f : vector<16xf32>, vector<16xf32>, f32
            scf.yield %rd : f32
          }} else {{
            scf.yield %zero_f : f32
          }}
          %sb = scalar.addf %s_qk, %rb : f32
          scf.yield %sb : f32
        }}
        %sm = scf.select %valid, %s, %neg_inf : f32""")
    prev = "%sm"
    for k, x in enumerate(("%x1", "%x2", "%x4", "%x8")):
        a(f"""        %mx{k}r, %mx{k}u = kernel.subgroup.shuffle<xor> {prev}, {x}, %w32 : f32, i32, i32
        %mx{k} = scalar.maxnumf {prev}, %mx{k}r : f32""")
        prev = f"%mx{k}"
    a(f"""        %nm = scalar.maxnumf %m, {prev} : f32
        %ms = scalar.subf %m, %nm : f32
        %al = scalar.expf<afn> %ms : f32
        %ds = scalar.subf %sm, %nm : f32
        %p = scalar.expf<afn> %ds : f32""")
    prev = "%p"
    for k, x in enumerate(("%x1", "%x2", "%x4", "%x8")):
        a(f"""        %ps{k}r, %ps{k}u = kernel.subgroup.shuffle<xor> {prev}, {x}, %w32 : f32, i32, i32
        %ps{k} = scalar.addf {prev}, %ps{k}r : f32""")
        prev = f"%ps{k}"
    a(f"""        %ls = scalar.mulf %l, %al : f32
        %nl = scalar.addf %ls, {prev} : f32
        %pv = scf.select %valid, %p, %zero_f : f32
        %alv = vector.splat %al : vector<4xf32>
        %a_in = vector.mulf %acc, %alv : vector<4xf32>""")
    acc = "%a_in"
    for j in range(16):
        a(f"""        %w{j} = kernel.subgroup.broadcast %pv from %l{j} : f32, i32
        %vf{j} = vector.sitofp %vi{j} : vector<4xi8> to vector<4xf32>
        %vsf{j} = scalar.extf %vsh{j} : f16 to f32
        %kpj{j} = index.add %p_base, %c{j}k : index
        %vlo{j} = index.cmp uge, %kpj{j}, %w_low : index
        %vhi{j} = index.cmp ule, %kpj{j}, %pos : index
        %vok{j} = scalar.andi %vlo{j}, %vhi{j} : i1
        %wraw{j} = scalar.mulf %w{j}, %vsf{j} : f32
        %wsc{j} = scf.select %vok{j}, %wraw{j}, %zero_f : f32
        %wv{j} = vector.splat %wsc{j} : vector<4xf32>
        %a{j} = vector.fmaf %wv{j}, %vf{j}, {acc} : vector<4xf32>""")
        acc = f"%a{j}"
    a(f"""        scf.yield %nm, %nl, {acc} : f32, f32, vector<4xf32>
      }}
      scf.if %is_l0 {{
        view.store %mf, %pm_v[%tok, %hd, %sp] : f32, view<[%tokens]x24x64xf32>
        view.store %lf, %pl_v[%tok, %hd, %sp] : f32, view<[%tokens]x24x64xf32>
      }}
      vector.store %af, %po_v[%tok, %hd, %sp, %lane, %c0] : vector<4xf32>, view<[%tokens]x24x64x32x4xf32>
    }}
  }}
  kernel.return
}}
""")
    text = "\n".join(parts)
    # per-key byte offsets and lane ids, declared once at the top of the launch body
    decl = "\n".join(
        f"  %o{j} = index.constant {j * 1088} : offset\n  %l{j} = scalar.constant {j} : i32\n  %c{j}k = index.constant {j} : index"
        for j in range(16)
    )
    return text.replace(
        "  %c0 = index.constant 0 : index\n",
        "  %c0 = index.constant 0 : index\n" + decl + "\n",
        1,
    )
