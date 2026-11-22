"""F32-weight matvec for up to 4 tokens with the math of g_mv_f32_kK_t4_r1 (one row per workgroup, one lane per
32-value group, per-wave reduce, waves summed in order), bit for bit, but with every load issued up front.
Pair with mvgen.kernel(..., lanes=32 * waves, part_bytes=16 * waves) for single or multi-segment launches."""


def waves(k):
    return -(-(k // 32) // 32)


def body(k, fn):
    groups = k // 32
    nwaves = waves(k)
    lines = [
        f"""func.def inline @{fn}(%x: buffer, %w: buffer, %y: buffer, %parts: buffer, %row: index, %n: index, %tokens: index, %token0: index, %accumulate: i32) {{
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cW = index.constant {nwaves} : index
  %tg = index.constant 4 : index
  %zero_f = scalar.constant 0.0 : f32
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {32 * nwaves - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {nwaves - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %first = index.cmp eq, %lane, %c0 : index
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>
  %groups = index.constant {groups} : index
  %ok = index.cmp ult, %workitem, %groups : index
  %g_last = index.constant {groups - 1} : index
  %g_c = index.min %workitem, %g_last : index
  %g = index.assume %g_c [range(%g_c, 0, {groups - 1})] : index
  %row_bytes = index.constant {k * 4} : offset
  %block_bytes = index.constant 128 : offset
  %ba = index.scale %g, %block_bytes : index, offset -> offset
  %ro = index.scale %row, %row_bytes : index, offset -> offset
  %o = index.add %ro, %ba : offset
  %wv0 = func.call @g_f32_values(%w, %o, %c0) : (buffer, offset, index) -> (vector<32xf32>)
  %t_last = index.sub %tokens, %c1 : index
  %pv = buffer.view %parts[%zero] : buffer -> view<{nwaves}x4xf32>"""
    ]
    a = lines.append
    for j in range(4):
        a(f"""  %tj{j} = index.constant {j} : index
  %tok{j}_raw = index.add %token0, %tj{j} : index
  %live{j} = index.cmp ult, %tok{j}_raw, %tokens : index
  %use{j} = scalar.andi %live{j}, %ok : i1
  %tkc{j} = index.min %tok{j}_raw, %t_last : index
  %tk{j} = index.assume %tkc{j} [lt(%tkc{j}, %tokens)] : index
  %x{j} = vector.load %x_view[%tk{j}, %g, %c0] : view<[%tokens]x{groups}x32xf32> -> vector<32xf32>""")
    for j in range(4):
        a(f"""  %pr{j} = vector.dotf<reassoc|contract> %wv0, %x{j}, %zero_f : vector<32xf32>, vector<32xf32>, f32
  %pa{j} = scf.select %use{j}, %pr{j}, %zero_f : f32
  %s{j} = kernel.subgroup.reduce<addf> %pa{j} : f32
  scf.if %first {{
    view.store %s{j}, %pv[%wave, {j}] : f32, view<{nwaves}x4xf32>
  }}""")
    a(f"""  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %fin = index.cmp ult, %workitem, %tg : index
  scf.if %fin {{
    %jj = index.assume %workitem [range(%workitem, 0, 3)] : index
    %tok_raw = index.add %token0, %jj : index
    %tok_live = index.cmp ult, %tok_raw, %tokens : index
    scf.if %tok_live {{
      %t0 = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
        %bw = index.assume %ww [range(%ww, 0, {nwaves - 1})] : index
        %p = view.load %pv[%bw, %jj] : view<{nwaves}x4xf32> -> f32
        %nx = scalar.addf %acc, %p : f32
        scf.yield %nx : f32
      }}
      %value = scalar.addf %t0, %zero_f : f32
      func.call @g_store_row(%y, %tokens, %n, %tok_raw, %row, %value, %accumulate) : (buffer, index, index, index, index, f32, i32)
    }}
  }}
  func.return
}}
""")
    return "\n".join(lines)
