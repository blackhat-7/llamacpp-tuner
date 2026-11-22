"""One launch for every projection of an Inkling layer visit (t = 2..4 tokens):
bf16 rows (attn_sink_proj from h; on global visits also attn_k/v_conv_proj from hk/hv), exactly as g_ink_bf16_proj,
then the Q8_0 rows q, gate, k, v, rel exactly as g_mv_q8_0_k2304_t4_q8_r1 (mvgen), R rows per workgroup.
Workgroup ranges: bf16 segments first (their serial reductions start early), then q8 segments."""

import mvgen

R = 2
Q8 = [
    ("q", 3072, 0),
    ("g", 3072, 0),
    ("k", 1024, 1),
    ("v", 1024, 2),
    ("r", 384, 0),
]  # name, rows, x index
BF = {6: [("s", 24, 0)], 8: [("s", 24, 0), ("dk", 32, 1), ("dv", 32, 2)]}


def bf_body(fn):
    """R rows of g_ink_bf16_proj for tokens [0, tokens) (tokens <= 4), same per-row math and reduction."""
    rows = "".join(
        f"""
  %rr{r}_c = index.constant {r} : index
  %rowr{r} = index.add %row, %rr{r}_c : index
  %rl{r} = index.cmp ult, %rowr{r}, %n : index
  %rowc{r} = index.min %rowr{r}, %n_last : index
  %rw{r} = index.assume %rowc{r} [lt(%rowc{r}, %n)] : index
  %wf{r} = scf.if %ok -> (vector<32xf32>) {{
    %g = index.assume %tid [range(%tid, 0, 71)] : index
    %wb = vector.load %wv[%rw{r}, %g, %c0] : view<[%n]x72x32xbf16> -> vector<32xbf16>
    %wfv = vector.extf %wb : vector<32xbf16> to vector<32xf32>
    scf.yield %wfv : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  scf.for %dt = [%c0 to %tokens step %c1] {{
    %tok = index.assume %dt [lt(%dt, %tokens)] : index
    %lv32 = view.load %iv[%tok, 2] : view<[%tokens]x4xi32> -> i32
    %live = scalar.cmpi ne, %lv32, %c0_i32 : i32
    scf.if %live {{
      %part = scf.if %ok -> (f32) {{
        %g = index.assume %tid [range(%tid, 0, 71)] : index
        %xg = vector.load %xv[%tok, %g, %c0] : view<[%tokens]x72x32xf32> -> vector<32xf32>
        %d = vector.dotf<reassoc|contract> %wf{r}, %xg, %zero_f : vector<32xf32>, vector<32xf32>, f32
        scf.yield %d : f32
      }} else {{
        scf.yield %zero_f : f32
      }}
      %tot = kernel.workgroup.reduce<addf> %part : f32
      %st = scalar.andi %is_t0, %rl{r} : i1
      scf.if %st {{
        view.store %tot, %ov[%tok, %rw{r}] : f32, view<[%tokens]x[%n]xf32>
      }}
    }}
  }}"""
        for r in range(R)
    )
    return f"""func.def inline @{fn}(%info: buffer, %x: buffer, %w: buffer, %out: buffer, %row: index, %n: index, %tokens: index) {{
  %tid_raw = kernel.workitem.id<x> : index
  %tid = index.assume %tid_raw [range(%tid_raw, 0, 95)] : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cHG = index.constant 72 : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.splatf 0.0 : vector<32xf32>
  %ok = index.cmp ult, %tid, %cHG : index
  %is_t0 = index.cmp eq, %tid, %c0 : index
  %n_last = index.sub %n, %c1 : index
  %wv = buffer.view %w[%zero] : buffer -> view<[%n]x72x32xbf16>
  %iv = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %xv = buffer.view %x[%zero] : buffer -> view<[%tokens]x72x32xf32>
  %ov = buffer.view %out[%zero] : buffer -> view<[%tokens]x[%n]xf32>{rows}
  func.return
}}
"""


def kernel(name, nseg):
    bf = BF[nseg]
    segs = [(n, rows, x, "bf") for n, rows, x in bf] + [
        (n, rows, x, "q8") for n, rows, x in Q8
    ]
    ws = ", ".join(f"%w_{n}: buffer" for n, *_ in segs)
    ys = ", ".join(f"%y_{n}: buffer" for n, *_ in segs)
    lines = [
        f"""kernel.def export("{name}") @{name}(%n: index, %tokens: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant 96 : index
  kernel.launch.config workgroups(%c1, %n, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%n_arg: index, %tokens_arg: index, %x0: buffer, %x1: buffer, %x2: buffer, %h0: buffer, %h1: buffer, %h2: buffer, %info: buffer, {ws}, {ys}) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 4)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %zero_i32 = scalar.constant 0 : i32
  %rc = index.constant {R} : index
  %group_id = kernel.workgroup.id<y> : index
  %part_bytes = index.constant {48 * R} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer"""
    ]
    a = lines.append
    lo = 0
    bounds = []
    for n, rows, x, kind in segs:
        nw = -(-rows // R)
        bounds.append((n, rows, x, kind, lo, lo + nw))
        lo += nw
    for n, rows, _x, _kind, lo_, hi in bounds:
        a(
            f"  %n_{n} = index.constant {rows} : index\n  %lo_{n} = index.constant {lo_} : index\n  %hi_{n} = index.constant {hi} : index"
        )

    def call(n, rows, x, kind):
        s = (
            f"%rw_{n} = index.sub %group_id, %lo_{n} : index\n%rv_{n} = index.mul %rw_{n}, %rc : index\n"
            f"%r_{n} = index.assume %rv_{n} [lt(%rv_{n}, %n_{n})] : index\n"
        )
        if kind == "bf":
            return s + (
                f"func.call @{name}_bf(%info, %h{x}, %w_{n}, %y_{n}, %r_{n}, %n_{n}, %tokens) : "
                "(buffer, buffer, buffer, buffer, index, index, index)\n"
            )
        return s + (
            f"func.call @{name}_q8(%x{x}, %w_{n}, %y_{n}, %parts, %r_{n}, %n_{n}, %tokens, %c0, %zero_i32) : "
            "(buffer, buffer, buffer, buffer, index, index, index, index, i32)\n"
        )

    def chain(i):
        n, rows, x, kind, lo_, hi = bounds[i]
        if i == len(bounds) - 1:
            return call(n, rows, x, kind)
        return (
            f"%in_{n} = index.cmp ult, %group_id, %hi_{n} : index\nscf.if %in_{n} {{\n"
            + call(n, rows, x, kind)
            + "} else {\n"
            + chain(i + 1)
            + "}\n"
        )

    a(chain(0) + "kernel.return\n}\n")
    return "\n".join(lines), lo


def module(names):
    """Kernel text for the two variants: names = (six_segment_name, eight_segment_name)."""
    out = []
    for name, nseg in zip(names, (6, 8), strict=True):
        k, _ = kernel(name, nseg)
        out += [mvgen.body(2304, f"{name}_q8", R), bf_body(f"{name}_bf"), k]
    return "\n".join(out)


def workgroups(nseg):
    return kernel("x", nseg)[1]
