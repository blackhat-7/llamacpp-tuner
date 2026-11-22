"""Rewrite llm.loom so the 16 global-attention KV planes hold int8 codes + f16 scales, decoded by attention v2.

Position layout (1088 bytes): 8 heads x 128 int8 codes, then 32 f16 scales (one per head and 32-dim block).
Decode and verify spans launch @g_ink_attn_split_v2 from ../kernels/g_ink_attn_split_v2_kv8.loom.
Usage: python transform.py SRC_llm.loom DST_llm.loom
"""

import re
import sys
from pathlib import Path

import split_kernel

V2_KERNEL = (
    Path(__file__).resolve().parent.parent / "kernels" / "g_ink_attn_split_v2_kv8.loom"
)

KERNELS = [
    "g_ink_sconv1_dyn_kv_prep_store",
    "g_ink_kv_prep_store",
    "g_ink_kv_prep_store_lv",
    "g_ink_attn_split",
    "g_ink_attn_direct",
    "g_ink_attn_split_span",
    "g_ink_attention_flash2_ts2",
    "g_ink_attention_flash2_partial",
    "g16_ink_attn_split_listed",
]

HELPERS = r"""
// KV8: a global-attention KV position is 1088 bytes: 8 KV heads x 128 int8 codes, then 32 f16 scales (head x 4 blocks of 32 dims).
func.def inline @g_kv8_offset(%origins: buffer, %row: index, %position: index, %blocks: index, %plane: index, %pool: index) -> (offset) {
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %record_stride = index.constant 40 : offset
  %position_bytes = index.constant 1088 : offset
  %bounded_row = index.assume %row [range(%row, 0, 2147483646)] : index
  %bounded_position = index.assume %position [range(%position, 0, 1048575)] : index
  %record_offset = index.scale %bounded_row, %record_stride : index, offset -> offset
  %records = buffer.view %origins[%record_offset] : buffer -> view<5xi64>
  %map64 = view.load %records[4] : view<5xi64> -> i64
  %map_bound = scalar.assume %map64 [range(%map64, 0, 9223372036854775807)] : i64
  %map_origin = index.cast %map_bound : i64 to offset
  %page_value = index.div %bounded_position, %c16 : index
  %last = index.sub %blocks, %c1 : index
  %page = index.assume %page_value [range(%page_value, 0, %last), lt(%page_value, %blocks)] : index
  %map = buffer.view %origins[%map_origin] : buffer -> view<[%blocks]xi32>
  %physical32 = view.load %map[%page] : view<[%blocks]xi32> -> i32
  %bounded_physical = scalar.assume %physical32 [range(%physical32, 0, 134217727)] : i32
  %physical = index.cast %bounded_physical : i32 to index
  %page_base = index.mul %physical, %c16 : index
  %within = index.rem %bounded_position, %c16 : index
  %slot = index.add %page_base, %within : index
  %plane_slot = index.madd %plane, %pool, %slot : index
  %bytes = index.scale %plane_slot, %position_bytes : index, offset -> offset
  func.return %bytes : offset
}

func.def inline @g_kv8_scale(%arena: buffer, %at: offset, %index: index) -> (f16) {
  %scale_at = index.constant 1024 : offset
  %scales_at = index.add %at, %scale_at : offset
  %scales = buffer.view %arena[%scales_at] : buffer -> view<32xf16>
  %bounded = index.assume %index [range(%index, 0, 31)] : index
  %scale = view.load %scales[%bounded] : view<32xf16> -> f16
  func.return %scale : f16
}

// Four dims of one head per lane (lane x 4 + i), as the f16 stores wrote them.
func.def inline @g_kv8_load4(%arena: buffer, %at: offset, %head: index, %lane: index) -> (vector<4xf16>) {
  %c0 = index.constant 0 : index
  %codes_view = buffer.view %arena[%at] : buffer -> view<8x32x4xi8>
  %codes = vector.load %codes_view[%head, %lane, %c0] : view<8x32x4xi8> -> vector<4xi8>
  %c8 = index.constant 8 : index
  %c4 = index.constant 4 : index
  %block = index.div %lane, %c8 : index
  %index = index.madd %head, %c4, %block : index
  %scale16 = func.call @g_kv8_scale(%arena, %at, %index) : (buffer, offset, index) -> (f16)
  %scale = scalar.extf %scale16 : f16 to f32
  %values = vector.sitofp %codes : vector<4xi8> to vector<4xf32>
  %scales = vector.splat %scale : vector<4xf32>
  %scaled = vector.mulf %values, %scales : vector<4xf32>
  %result = vector.fptrunc %scaled : vector<4xf32> to vector<4xf16>
  func.return %result : vector<4xf16>
}

// Sixteen consecutive dims of one head starting at %dim.
func.def inline @g_kv8_load16(%arena: buffer, %at: offset, %head: index, %dim: index) -> (vector<16xf16>) {
  %codes_view = buffer.view %arena[%at] : buffer -> view<8x128xi8>
  %codes = vector.load %codes_view[%head, %dim] : view<8x128xi8> -> vector<16xi8>
  %c32 = index.constant 32 : index
  %c4 = index.constant 4 : index
  %block = index.div %dim, %c32 : index
  %index = index.madd %head, %c4, %block : index
  %scale16 = func.call @g_kv8_scale(%arena, %at, %index) : (buffer, offset, index) -> (f16)
  %scale = scalar.extf %scale16 : f16 to f32
  %values = vector.sitofp %codes : vector<16xi8> to vector<16xf32>
  %scales = vector.splat %scale : vector<16xf32>
  %scaled = vector.mulf %values, %scales : vector<16xf32>
  %result = vector.fptrunc %scaled : vector<16xf32> to vector<16xf16>
  func.return %result : vector<16xf16>
}

// One wave (32 lanes x 4 dims) quantizes one head in 4 blocks of 8 lanes: symmetric absmax, the f16-rounded scale is stored.
func.def inline @g_kv8_store4(%arena: buffer, %at: offset, %head: index, %lane: index, %x: vector<4xf32>) {
  %c0 = index.constant 0 : index
  %zero_f = scalar.constant 0.0 : f32
  %one_f = scalar.constant 1.0 : f32
  %c127 = scalar.constant 127.0 : f32
  %zero4 = vector.splatf 0.0 : vector<4xf32>
  %abs = vector.absf %x : vector<4xf32>
  %lane_max = vector.reduce<maxnumf> %abs, %zero_f : vector<4xf32>, f32
  %x1 = scalar.constant 1 : i32
  %x2 = scalar.constant 2 : i32
  %x4 = scalar.constant 4 : i32
  %w32 = scalar.constant 32 : i32
  %m1r, %m1u = kernel.subgroup.shuffle<xor> %lane_max, %x1, %w32 : f32, i32, i32
  %m1 = scalar.maxnumf %lane_max, %m1r : f32
  %m2r, %m2u = kernel.subgroup.shuffle<xor> %m1, %x2, %w32 : f32, i32, i32
  %m2 = scalar.maxnumf %m1, %m2r : f32
  %m4r, %m4u = kernel.subgroup.shuffle<xor> %m2, %x4, %w32 : f32, i32, i32
  %amax = scalar.maxnumf %m2, %m4r : f32
  %d = scalar.divf %amax, %c127 : f32
  %d16 = scalar.fptrunc %d : f32 to f16
  %dq = scalar.extf %d16 : f16 to f32
  %nz = scalar.cmpf one, %dq, %zero_f : f32
  %inv = scalar.divf %one_f, %dq : f32
  %invs = vector.splat %inv : vector<4xf32>
  %scaled = vector.mulf %x, %invs : vector<4xf32>
  %rounded = vector.roundf %scaled : vector<4xf32>
  %nzv = vector.splat %nz : vector<4xi1>
  %safe = vector.select %nzv, %rounded, %zero4 : vector<4xf32>
  %codes = vector.fptosi %safe : vector<4xf32> to vector<4xi8>
  %codes_view = buffer.view %arena[%at] : buffer -> view<8x32x4xi8>
  vector.store %codes, %codes_view[%head, %lane, %c0] : vector<4xi8>, view<8x32x4xi8>
  %c8 = index.constant 8 : index
  %c4 = index.constant 4 : index
  %in_block = index.rem %lane, %c8 : index
  %first = index.cmp eq, %in_block, %c0 : index
  scf.if %first {
    %block = index.div %lane, %c8 : index
    %index = index.madd %head, %c4, %block : index
    %bounded = index.assume %index [range(%index, 0, 31)] : index
    %scale_at = index.constant 1024 : offset
    %scales_at = index.add %at, %scale_at : offset
    %scales = buffer.view %arena[%scales_at] : buffer -> view<32xf16>
    view.store %d16, %scales[%bounded] : f16, view<32xf16>
  }
  func.return
}
"""

KDEF = re.compile(r'^kernel\.def export\("([^"]+)"\)')
VIEW = re.compile(
    r"^(\s*)(%\w+) = buffer\.view %arena\[(%\w+)\] : buffer -> view<(8x32x4|8x128)xf16>$"
)
LOAD = re.compile(
    r"^(\s*)(%\w+) = vector\.load (%\w+)\[(%\w+), (%\w+)(?:, %c0)?\] : view<(?:8x32x4|8x128)xf16> -> vector<(4|16)xf16>$"
)
STORE = re.compile(
    r"^(\s*)vector\.store (%\w+), (%\w+)\[(%\w+), (%\w+), %c0\] : vector<4xf16>, view<8x32x4xf16>$"
)
TRUNC = re.compile(
    r"^\s*(%\w+) = vector\.fptrunc (%\w+) : vector<4xf32> to vector<4xf16>$"
)
POS_BYTES = re.compile(r"%pos_bytes = index\.constant 2048 : offset$")
SPAN = re.compile(
    r"kernel\.launch @g_ink_attn_split_span\[%spans\]\(%t, %spans, %blocks, %pool, (%fslot_\d+), %metadata, %origins, "
    r"%q, %r_buf, (%rbias_\d+), %arena, %pm, %pl, %po\) : .*"
)
SPLIT_HEAD = (
    "// Split-K causal GQA attention with on-the-fly relative-position bias (dist < 1024), SWA (window=1024), "
    'per-query sinks, sigmoid gate.\nkernel.def export("g_ink_attn_split")'
)
COMBINE = 'kernel.def export("g_ink_attn_combine")'


def kv8_planes(lines: list[str]) -> list[str]:
    """Swap the f16 KV loads and stores of KERNELS for the KV8 helpers."""
    out: list[str] = []
    current = None
    views: dict[str, str] = {}
    truncs: dict[str, str] = {}
    for line in lines:
        if m := KDEF.match(line):
            current = m.group(1) if m.group(1) in KERNELS else None
            views, truncs = {}, {}
        if current is None:
            out.append(line)
            continue
        line = line.replace("@g_kv_offset(", "@g_kv8_offset(")
        if POS_BYTES.search(line):
            line = line.replace("2048", "1088")
        if m := VIEW.match(line):
            views[m.group(2)] = m.group(3)
            continue
        if m := TRUNC.match(line):
            truncs[m.group(1)] = m.group(2)
        if (m := LOAD.match(line)) and m.group(3) in views:
            ind, res, v, i1, i2, n = m.groups()
            out.append(
                f"{ind}{res} = func.call @g_kv8_load{n}(%arena, {views[v]}, {i1}, {i2})"
                f" : (buffer, offset, index, index) -> (vector<{n}xf16>)"
            )
            continue
        if (m := STORE.match(line)) and m.group(3) in views:
            ind, val, v, i1, i2 = m.groups()
            out.append(
                f"{ind}func.call @g_kv8_store4(%arena, {views[v]}, {i1}, {i2}, {truncs[val]})"
                " : (buffer, offset, index, index, vector<4xf32>)"
            )
            continue
        out.append(line)
    return out


def span_to_v2(m: re.Match[str]) -> str:
    layer = m.group(1).split("_")[1]
    return (
        f"kernel.launch @g_ink_attn_split_v2[%t](%t, %blocks, %pool, {m.group(1)}, %swa_{layer}, %info, %origins, "
        f"%q, %r_buf, {m.group(2)}, %arena, %pm, %pl, %po)"
        " : [index](index, index, index, index, i32, buffer, buffer, buffer, buffer, view<65536xi8>, buffer, buffer,"
        " buffer, buffer)"
    )


def transform(text: str) -> str:
    text = "\n".join(kv8_planes(text.split("\n")))
    a = text.index(SPLIT_HEAD)
    b = text.index(COMBINE, a)
    text = text[:a] + split_kernel.kernel() + "\n" + text[b:]
    # v2 keeps the split kernel's arguments and partial layout, so it replaces it at every decode launch site.
    b = text.index(COMBINE)
    text = text[:b] + V2_KERNEL.read_text().rstrip("\n") + "\n\n" + text[b:]
    n = text.count("kernel.launch @g_ink_attn_split[%t](")
    assert n == 16, n
    text = text.replace(
        "kernel.launch @g_ink_attn_split[%t](",
        "kernel.launch @g_ink_attn_split_v2[%t](",
    )
    # Verify spans (1 < t < 64) also take v2: it reads each token's row and position from info, as the span kernel
    # did from the span metadata, and the causal limit is the token's own position.
    text, n = SPAN.subn(span_to_v2, text)
    assert n == 16, n
    anchor = text.index("// Q4_K / Q5_K 6-bit scale")
    return text[:anchor] + HELPERS.lstrip("\n") + "\n" + text[anchor:]


if __name__ == "__main__":
    src, dst = map(Path, sys.argv[1:3])
    dst.write_text(transform(src.read_text()))
