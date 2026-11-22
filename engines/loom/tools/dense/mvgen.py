"""Q8_0 x planar-q8 matvec for up to 4 tokens, one row per 96-lane workgroup (the math of g_mv_q8_0_kK_t4_q8_r1,
bit for bit), with every load issued unconditionally so the weight and the 4 activation loads overlap, and with
several projections ("segments") in one launch: rows [0, n0) use (x0, w0, y0), the next n1 rows (x1, w1, y1)...

body(K, name) -> the inline function; kernel(K, name, xs) -> the kernel, xs = number of activation buffers and
segment s reads x[XMAP[s]] (XMAP given per kernel)."""

TG = 4


def body(k, fn, nrows=1):
    groups = k // 32
    rb = groups * 34
    lines = []
    a = lines.append
    a(
        f"func.def inline @{fn}(%x: buffer, %w: buffer, %y: buffer, %parts: buffer, %row: index, %n: index, %tokens: index, %token0: index, %accumulate: i32) {{"
    )
    a(f"""  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cW = index.constant 3 : index
  %tg = index.constant 4 : index
  %groups = index.constant {groups} : index
  %row_bytes = index.constant {rb} : offset
  %block_bytes = index.constant 34 : offset
  %zero_f = scalar.constant 0.0 : f32
  %zero1i = vector.splati 0 : vector<1xi32>
  %bw32 = scalar.constant 32 : i32
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 95)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 2)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %xq_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{groups}x8xi32>
  %xm_k = index.constant {k} : offset
  %xm_off = index.scale %tokens, %xm_k : index, offset -> offset
  %xm_view = buffer.view %x[%xm_off] : buffer -> view<[%tokens]x{groups}x2xf32>
  %ok = index.cmp ult, %workitem, %groups : index
  %g_last = index.constant {groups - 1} : index
  %g_c = index.min %workitem, %g_last : index
  %g = index.assume %g_c [range(%g_c, 0, {groups - 1})] : index
  %ba = index.scale %g, %block_bytes : index, offset -> offset
  %t_last = index.sub %tokens, %c1 : index
  %n_last = index.sub %n, %c1 : index""")
    for r in range(nrows):
        a(f"""  %rr{r}_c = index.constant {r} : index
  %rowr{r} = index.add %row, %rr{r}_c : index
  %rowc{r} = index.min %rowr{r}, %n_last : index
  %ro{r} = index.scale %rowc{r}, %row_bytes : index, offset -> offset
  %o{r} = index.add %ro{r}, %ba : offset
  %wq{r}, %sc{r} = func.call @g_q8_0_icodes(%w, %o{r}, %c0) : (buffer, offset, index) -> (vector<32xi8>, f32)""")
    for j in range(TG):
        a(f"""  %tj{j} = index.constant {j} : index
  %tok{j}_raw = index.add %token0, %tj{j} : index
  %live{j} = index.cmp ult, %tok{j}_raw, %tokens : index
  %use{j} = scalar.andi %live{j}, %ok : i1
  %tkc{j} = index.min %tok{j}_raw, %t_last : index
  %tk{j} = index.assume %tkc{j} [lt(%tkc{j}, %tokens)] : index
  %cw{j} = vector.load %xq_view[%tk{j}, %g, %c0] : view<[%tokens]x{groups}x8xi32> -> vector<8xi32>
  %q{j} = vector.bitcast %cw{j} : vector<8xi32> to vector<32xi8>
  %mt{j} = vector.load %xm_view[%tk{j}, %g, %c0] : view<[%tokens]x{groups}x2xf32> -> vector<2xf32>
  %da{j} = vector.extract %mt{j}[0] : vector<2xf32> -> f32""")
    for j in range(TG):
        for i in range(8):
            a(
                f"  %qk{j}_{i} = vector.slice %q{j}[{4 * i}] : vector<32xi8> -> vector<4xi8>"
            )
    for r in range(nrows):
        for i in range(8):
            a(
                f"  %wk{r}_{i} = vector.slice %wq{r}[{4 * i}] : vector<32xi8> -> vector<4xi8>"
            )
        for j in range(TG):
            prev = "%zero1i"
            for i in range(8):
                a(
                    f"  %ac{r}{j}_{i} = vector.dot4i<s8s8> %wk{r}_{i}, %qk{j}_{i}, {prev} : vector<4xi8>, vector<4xi8>, vector<1xi32>"
                )
                prev = f"%ac{r}{j}_{i}"
            a(f"""  %ie{r}{j} = vector.extract %ac{r}{j}_7[0] : vector<1xi32> -> i32
  %df{r}{j} = scalar.sitofp %ie{r}{j} : i32 to f32
  %wd{r}{j} = scalar.mulf %sc{r}, %da{j} : f32
  %pr{r}{j} = scalar.mulf %wd{r}{j}, %df{r}{j} : f32
  %pa{r}{j} = scf.select %use{j}, %pr{r}{j}, %zero_f : f32""")
    # The original kernel's transpose-reduce: lane bits 16 and 8 pick the token, then a plain xor reduce.
    a("""  %bc2 = index.constant 2 : index
  %bd16 = scalar.constant 16 : i32
  %bdi16 = index.constant 16 : index
  %bq16 = index.div %lane, %bdi16 : index
  %br16 = index.rem %bq16, %bc2 : index
  %bb16 = index.cmp ne, %br16, %c0 : index
  %bh16 = index.constant 2 : index
  %bo16 = scf.select %bb16, %bh16, %c0 : index
  %bd8 = scalar.constant 8 : i32
  %bdi8 = index.constant 8 : index
  %bq8 = index.div %lane, %bdi8 : index
  %br8 = index.rem %bq8, %bc2 : index
  %bb8 = index.cmp ne, %br8, %c0 : index
  %bh8 = index.constant 1 : index
  %bo8 = scf.select %bb8, %bh8, %c0 : index
  %bd4 = scalar.constant 4 : i32
  %bdi4 = index.constant 4 : index
  %bq4 = index.div %lane, %bdi4 : index
  %br4 = index.rem %bq4, %bc2 : index
  %bb4 = index.cmp ne, %br4, %c0 : index
  %bd2 = scalar.constant 2 : i32
  %bdi2 = index.constant 2 : index
  %bq2 = index.div %lane, %bdi2 : index
  %br2 = index.rem %bq2, %bc2 : index
  %bb2 = index.cmp ne, %br2, %c0 : index
  %bd1 = scalar.constant 1 : i32
  %bdi1 = index.constant 1 : index
  %bq1 = index.div %lane, %bdi1 : index
  %br1 = index.rem %bq1, %bc2 : index
  %bb1 = index.cmp ne, %br1, %c0 : index
  %bi0 = index.add %c0, %bo16 : index
  %bt_idx_v = index.add %bi0, %bo8 : index
  %bt_idx = index.assume %bt_idx_v [range(%bt_idx_v, 0, 3)] : index
  %bt_true = scalar.constant true : i1
  %bn0 = scalar.xori %bb4, %bt_true : i1
  %bn1 = scalar.xori %bb2, %bt_true : i1
  %bs1 = scalar.andi %bn0, %bn1 : i1
  %bn2 = scalar.xori %bb1, %bt_true : i1
  %bt_st = scalar.andi %bs1, %bn2 : i1
""")
    for r in range(nrows):
        a(f"""  %b{r}_16_0s = scf.select %bb16, %pa{r}0, %pa{r}2 : f32
  %b{r}_16_0k = scf.select %bb16, %pa{r}2, %pa{r}0 : f32
  %b{r}_16_0r, %b{r}_16_0u = kernel.subgroup.shuffle<xor> %b{r}_16_0s, %bd16, %bw32 : f32, i32, i32
  %b{r}_16_0 = scalar.addf %b{r}_16_0k, %b{r}_16_0r : f32
  %b{r}_16_1s = scf.select %bb16, %pa{r}1, %pa{r}3 : f32
  %b{r}_16_1k = scf.select %bb16, %pa{r}3, %pa{r}1 : f32
  %b{r}_16_1r, %b{r}_16_1u = kernel.subgroup.shuffle<xor> %b{r}_16_1s, %bd16, %bw32 : f32, i32, i32
  %b{r}_16_1 = scalar.addf %b{r}_16_1k, %b{r}_16_1r : f32
  %b{r}_8_0s = scf.select %bb8, %b{r}_16_0, %b{r}_16_1 : f32
  %b{r}_8_0k = scf.select %bb8, %b{r}_16_1, %b{r}_16_0 : f32
  %b{r}_8_0r, %b{r}_8_0u = kernel.subgroup.shuffle<xor> %b{r}_8_0s, %bd8, %bw32 : f32, i32, i32
  %b{r}_8_0 = scalar.addf %b{r}_8_0k, %b{r}_8_0r : f32
  %b{r}_4r, %b{r}_4u = kernel.subgroup.shuffle<xor> %b{r}_8_0, %bd4, %bw32 : f32, i32, i32
  %b{r}_4 = scalar.addf %b{r}_8_0, %b{r}_4r : f32
  %b{r}_2r, %b{r}_2u = kernel.subgroup.shuffle<xor> %b{r}_4, %bd2, %bw32 : f32, i32, i32
  %b{r}_2 = scalar.addf %b{r}_4, %b{r}_2r : f32
  %b{r}_1r, %b{r}_1u = kernel.subgroup.shuffle<xor> %b{r}_2, %bd1, %bw32 : f32, i32, i32
  %b{r}_1 = scalar.addf %b{r}_2, %b{r}_1r : f32
""")
    a(f"""  %pv = buffer.view %parts[%zero] : buffer -> view<{nrows}x3x4xf32>
  scf.if %bt_st {{""")
    for r in range(nrows):
        a(
            f"    view.store %b{r}_1, %pv[{r}, %wave, %bt_idx] : f32, view<{nrows}x3x4xf32>"
        )
    a(f"""  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %nout = index.constant {4 * nrows} : index
  %fin = index.cmp ult, %workitem, %nout : index
  scf.if %fin {{
    %wr = index.assume %workitem [range(%workitem, 0, {4 * nrows - 1})] : index
    %rr_v = index.div %wr, %tg : index
    %rr = index.assume %rr_v [range(%rr_v, 0, {nrows - 1})] : index
    %jj_v = index.rem %wr, %tg : index
    %jj = index.assume %jj_v [range(%jj_v, 0, 3)] : index
    %tok_raw = index.add %token0, %jj : index
    %tok_l = index.cmp ult, %tok_raw, %tokens : index
    %orow = index.add %row, %rr : index
    %row_l = index.cmp ult, %orow, %n : index
    %tok_live = scalar.andi %tok_l, %row_l : i1
    scf.if %tok_live {{
      %t0 = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
        %bw = index.assume %ww [range(%ww, 0, 2)] : index
        %p = view.load %pv[%rr, %bw, %jj] : view<{nrows}x3x4xf32> -> f32
        %nx = scalar.addf %acc, %p : f32
        scf.yield %nx : f32
      }}
      %value = scalar.addf %t0, %zero_f : f32
      func.call @g_store_row(%y, %tokens, %n, %tok_raw, %orow, %value, %accumulate) : (buffer, index, index, index, index, f32, i32)
    }}
  }}
  func.return
}}
""")
    return "\n".join(lines)


def kernel(k, name, fn, xmap, nrows=1, lanes=96, part_bytes=None):
    """xmap[s] = index of the activation buffer segment s reads; len(xmap) = segments."""
    nseg, nx = len(xmap), max(xmap) + 1
    ns = ", ".join(f"%n{s}_arg: index" for s in range(nseg))
    xs = ", ".join(f"%x{i}: buffer" for i in range(nx))
    ws = ", ".join(f"%w{s}: buffer" for s in range(nseg))
    ys = ", ".join(f"%y{s}: buffer" for s in range(nseg))
    lines = [
        f"""kernel.def export("{name}") @{name}(%n: index, %tokens: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant {lanes} : index
  %tg = index.constant 4 : index
  %tg_less = index.constant 3 : index
  %tg_pad = index.add %tokens, %tg_less : index
  %token_groups = index.div %tg_pad, %tg : index
  kernel.launch.config workgroups(%token_groups, %n, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%n_arg: index, %tokens_arg: index, %accumulate: i32, {ns}, {xs}, {ws}, {ys}) {{
  %n = index.assume %n_arg [range(%n_arg, 1, 262144)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %zero = index.constant 0 : offset
  %c4 = index.constant 4 : index
  %group_id = kernel.workgroup.id<y> : index
  %tgroup_id = kernel.workgroup.id<x> : index
  %token0 = index.mul %tgroup_id, %c4 : index
  %part_bytes = index.constant {part_bytes or 48 * nrows} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer
  %lo0 = index.constant 0 : index
  %rc = index.constant {nrows} : index
  %rl = index.constant {nrows - 1} : index"""
    ]
    a = lines.append
    for s in range(nseg):
        a(f"  %n{s} = index.assume %n{s}_arg [range(%n{s}_arg, 1, 262144)] : index")
        a(f"  %np{s} = index.add %n{s}, %rl : index")
        a(f"  %nw{s} = index.div %np{s}, %rc : index")
        a(f"  %hi{s} = index.add %lo{s}, %nw{s} : index")
        if s + 1 < nseg:
            a(f"  %lo{s + 1} = index.add %hi{s}, %lo0 : index")

    def chain(s, ind):
        p = " " * ind
        row = f"%r{s}"
        inner = (
            f"{p}%rw{s} = index.sub %group_id, %lo{s} : index\n"
            f"{p}%rv{s} = index.mul %rw{s}, %rc : index\n"
            f"{p}{row} = index.assume %rv{s} [lt(%rv{s}, %n{s})] : index\n"
            f"{p}func.call @{fn}(%x{xmap[s]}, %w{s}, %y{s}, %parts, {row}, %n{s}, %tokens, %token0, %accumulate) : "
            f"(buffer, buffer, buffer, buffer, index, index, index, index, i32)\n"
        )
        if s == nseg - 1:
            return inner
        return (
            f"{p}%in{s} = index.cmp ult, %group_id, %hi{s} : index\n{p}scf.if %in{s} {{\n"
            + inner.replace(p, p + "  ")
            + f"{p}}} else {{\n"
            + chain(s + 1, ind + 2)
            + f"{p}}}\n"
        )

    a(chain(0, 2) + "  kernel.return\n}\n")
    return "\n".join(lines)
