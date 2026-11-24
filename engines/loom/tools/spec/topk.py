#!/usr/bin/env python3
"""Exact, faster sampling at temperature > 0 for Inkling-2-Mini on Loom: g_head_select's top-k without 64 passes
over the vocabulary.

At T > 0 g_head_select picks its candidates (top-k <= 64 in (logit desc, id asc) order) with one pass over all
200,058 logits per candidate: 5.5 ms a call against 2 us greedy, paid on every T > 0 verify step and decoded
token. The candidates lie in the k chunks of 128 ids whose maxima (the head kernels' partials) come first in
(maximum desc, chunk asc) order, because each of those maxima precedes every id of any other chunk. So
g_head_select ranks the 1,563 chunk maxima, then the chosen chunks' ids, both in workgroup memory: 0.125 ms, the
same candidates in the same order, so the same draws. T = 0 is untouched.

usage: topk.py PACKAGE_DIR   (after engines/loom/build.sh's steps)
Edits PACKAGE_DIR/program/llm.loom in place (unlinks it first: package copies are hard links).
"""

import pathlib
import sys

ALLOC_OLD = """  %cand_i = buffer.view %cand[%c256o] : buffer -> view<64xi32>
"""

ALLOC_NEW = """  %cand_i = buffer.view %cand[%c256o] : buffer -> view<64xi32>
  %tk_max_bytes = index.constant 6256 : offset
  %tk_max_buf = buffer.alloca<workgroup> align(16) %tk_max_bytes : buffer
  %tk_max = buffer.view %tk_max_buf[%zero] : buffer -> view<1563xf32>
  %tk_sel_bytes = index.constant 256 : offset
  %tk_sel_buf = buffer.alloca<workgroup> align(16) %tk_sel_bytes : buffer
  %tk_sel = buffer.view %tk_sel_buf[%zero] : buffer -> view<64xi32>
  %tk_vals_bytes = index.constant 32768 : offset
  %tk_vals_buf = buffer.alloca<workgroup> align(16) %tk_vals_bytes : buffer
  %tk_vals = buffer.view %tk_vals_buf[%zero] : buffer -> view<2048x4xf32>
"""

TOPK_OLD = """        %logit_view = buffer.view %logits[%zero] : buffer -> view<[%spans]x200064xf32>
        // k rounds: the largest logit strictly after the previous pick in (value desc, index asc) order.
        %last_v, %last_i = scf.for %r = [%c0 to %k step %c1](%pv = %pos_inf : f32, %pi = %c0_i32 : i32) -> (f32, i32) {
          %minus_one = scalar.constant -1 : i32
          %prev_i = scalar.addi %pi, %c0_i32 : i32
          %first_round = index.cmp eq, %r, %c0 : index
          %floor_i = scf.select %first_round, %minus_one, %prev_i : i32
          %lv, %li = scf.for %v = [%lane to %vocab step %c256](%bv = %neg_inf : f32, %bi = %big : i32) -> (f32, i32) {
            %bounded_v = index.assume %v [range(%v, 0, 200057)] : index
            %x = view.load %logit_view[%span, %bounded_v] : view<[%spans]x200064xf32> -> f32
            %v32 = index.cast %bounded_v : index to i32
            %below = scalar.cmpf olt, %x, %pv : f32
            %tie = scalar.cmpf oeq, %x, %pv : f32
            %after = scalar.cmpi sgt, %v32, %floor_i : i32
            %tie_after = scalar.andi %tie, %after : i1
            %eligible = scalar.ori %below, %tie_after : i1
            %greater = scalar.cmpf ogt, %x, %bv : f32
            %take = scalar.andi %eligible, %greater : i1
            %nv = scf.select %take, %x, %bv : f32
            %ni = scf.select %take, %v32, %bi : i32
            scf.yield %nv, %ni : f32, i32
          }
"""

TOPK_HEAD = """        // The candidates lie in the k chunks of 128 ids whose maxima (the head's partials) come first in (maximum
        // desc, chunk asc) order: each of those maxima precedes every id of any other chunk. Rank the chunk maxima,
        // then the chosen chunks' ids, both in workgroup memory, not k passes over the vocabulary (5.6 ms).
        %tk_partials = buffer.view %partials[%zero] : buffer -> view<[%spans]x1563x2xi32>
        scf.for %c = [%lane to %chunks step %c256] {
          %bounded_c = index.assume %c [range(%c, 0, 1562)] : index
          %max_bits = view.load %tk_partials[%span, %bounded_c, %c0] : view<[%spans]x1563x2xi32> -> i32
          %max_value = scalar.bitcast %max_bits : i32 to f32
          view.store %max_value, %tk_max[%bounded_c] : f32, view<1563xf32>
        }
        kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
        %tk_last_v, %tk_last_c = scf.for %r = [%c0 to %k step %c1](%pv = %pos_inf : f32, %pc = %c0_i32 : i32) -> (f32, i32) {
          %minus_one = scalar.constant -1 : i32
          %first_round = index.cmp eq, %r, %c0 : index
          %floor_c = scf.select %first_round, %minus_one, %pc : i32
          %lv, %lc = scf.for %c = [%lane to %chunks step %c256](%bv = %neg_inf : f32, %bc = %big : i32) -> (f32, i32) {
            %bounded_c = index.assume %c [range(%c, 0, 1562)] : index
            %x = view.load %tk_max[%bounded_c] : view<1563xf32> -> f32
            %c32 = index.cast %bounded_c : index to i32
            %below = scalar.cmpf olt, %x, %pv : f32
            %tie = scalar.cmpf oeq, %x, %pv : f32
            %after = scalar.cmpi sgt, %c32, %floor_c : i32
            %tie_after = scalar.andi %tie, %after : i1
            %eligible = scalar.ori %below, %tie_after : i1
            %greater = scalar.cmpf ogt, %x, %bv : f32
            %take = scalar.andi %eligible, %greater : i1
            %nv = scf.select %take, %x, %bv : f32
            %nc = scf.select %take, %c32, %bc : i32
            scf.yield %nv, %nc : f32, i32
          }
          %best = kernel.workgroup.reduce<maxnumf> %lv : f32
          %is_best = scalar.cmpf oeq, %lv, %best : f32
          %cand_c = scf.select %is_best, %lc, %big : i32
          %winner = kernel.workgroup.reduce<minsi> %cand_c : i32
          scf.if %first {
            %rr = index.assume %r [range(%r, 0, 63)] : index
            view.store %winner, %tk_sel[%rr] : i32, view<64xi32>
          }
          scf.yield %best, %winner : f32, i32
        }
        kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
        %c32_index = index.constant 32 : index
        %last_chunk = scalar.constant 1562 : i32
        %quads = index.mul %k, %c32_index : index
        %logit4 = buffer.view %logits[%zero] : buffer -> view<[%spans]x50016x4xf32>
        scf.for %q = [%lane to %quads step %c256] {
          %bq = index.assume %q [range(%q, 0, 2047)] : index
          %slot = index.div %bq, %c32_index : index
          %within = index.rem %bq, %c32_index : index
          %bslot = index.assume %slot [range(%slot, 0, 63)] : index
          %chunk_raw = view.load %tk_sel[%bslot] : view<64xi32> -> i32
          %chunk_low = scalar.minsi %chunk_raw, %last_chunk : i32
          %chunk_b = scalar.assume %chunk_low [range(%chunk_low, 0, 1562)] : i32
          %chunk = index.cast %chunk_b : i32 to index
          %gq0 = index.mul %chunk, %c32_index : index
          %gq = index.add %gq0, %within : index
          %bgq = index.assume %gq [range(%gq, 0, 50015)] : index
          %x4 = vector.load %logit4[%span, %bgq, %c0] : view<[%spans]x50016x4xf32> -> vector<4xf32>
          vector.store %x4, %tk_vals[%bq, %c0] : vector<4xf32>, view<2048x4xf32>
        }
        kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
        // k rounds: the largest logit strictly after the previous pick in (value desc, index asc) order.
        %last_v, %last_i = scf.for %r = [%c0 to %k step %c1](%pv = %pos_inf : f32, %pi = %c0_i32 : i32) -> (f32, i32) {
          %minus_one = scalar.constant -1 : i32
          %prev_i = scalar.addi %pi, %c0_i32 : i32
          %first_round = index.cmp eq, %r, %c0 : index
          %floor_i = scf.select %first_round, %minus_one, %prev_i : i32
          %lv, %li = scf.for %q = [%lane to %quads step %c256](%bv = %neg_inf : f32, %bi = %big : i32) -> (f32, i32) {
            %bq = index.assume %q [range(%q, 0, 2047)] : index
            %slot = index.div %bq, %c32_index : index
            %within = index.rem %bq, %c32_index : index
            %bslot = index.assume %slot [range(%slot, 0, 63)] : index
            %chunk_raw = view.load %tk_sel[%bslot] : view<64xi32> -> i32
            %chunk_low = scalar.minsi %chunk_raw, %last_chunk : i32
            %c128_i32 = scalar.constant 128 : i32
            %c4_i32 = scalar.constant 4 : i32
            %vocab32 = scalar.constant 200058 : i32
            %base0 = scalar.muli %chunk_low, %c128_i32 : i32
            %w32 = index.cast %within : index to i32
            %w4 = scalar.muli %w32, %c4_i32 : i32
            %base = scalar.addi %base0, %w4 : i32
            %x4 = vector.load %tk_vals[%bq, %c0] : view<2048x4xf32> -> vector<4xf32>
"""

# One id of the 4 a lane step holds. Lane order is not id order here, so ties go to the lower id explicitly.
TOPK_ELEMENT = """            %x_{e} = vector.extract %x4[{e}] : vector<4xf32> -> f32
            %o_{e} = scalar.constant {e} : i32
            %id_{e} = scalar.addi %base, %o_{e} : i32
            %live_{e} = scalar.cmpi slt, %id_{e}, %vocab32 : i32
            %below_{e} = scalar.cmpf olt, %x_{e}, %pv : f32
            %tie_{e} = scalar.cmpf oeq, %x_{e}, %pv : f32
            %after_{e} = scalar.cmpi sgt, %id_{e}, %floor_i : i32
            %tie_after_{e} = scalar.andi %tie_{e}, %after_{e} : i1
            %eligible_{e} = scalar.ori %below_{e}, %tie_after_{e} : i1
            %greater_{e} = scalar.cmpf ogt, %x_{e}, {bv} : f32
            %same_{e} = scalar.cmpf oeq, %x_{e}, {bv} : f32
            %lower_{e} = scalar.cmpi slt, %id_{e}, {bi} : i32
            %same_lower_{e} = scalar.andi %same_{e}, %lower_{e} : i1
            %better_{e} = scalar.ori %greater_{e}, %same_lower_{e} : i1
            %ok_{e} = scalar.andi %eligible_{e}, %better_{e} : i1
            %take_{e} = scalar.andi %ok_{e}, %live_{e} : i1
            %bv_{e} = scf.select %take_{e}, %x_{e}, {bv} : f32
            %bi_{e} = scf.select %take_{e}, %id_{e}, {bi} : i32
"""

TOPK_NEW = (
    TOPK_HEAD
    + "".join(
        TOPK_ELEMENT.format(
            e=e,
            bv="%bv" if e == 0 else f"%bv_{e - 1}",
            bi="%bi" if e == 0 else f"%bi_{e - 1}",
        )
        for e in range(4)
    )
    + """            scf.yield %bv_3, %bi_3 : f32, i32
          }
"""
)


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:80]!r}")
    return text.replace(old, new)


def edit_kernel(text: str, name: str, old: str, new: str) -> str:
    """Replace OLD with NEW inside kernel NAME only."""
    a = text.index(f'kernel.def export("{name}") @{name}(')
    b = text.index("\n}\n", text.index("launch(", a)) + 3
    return text[:a] + replace_once(text[a:b], old, new) + text[b:]


def transform(text: str) -> str:
    text = edit_kernel(text, "g_head_select", ALLOC_OLD, ALLOC_NEW)
    return edit_kernel(text, "g_head_select", TOPK_OLD, TOPK_NEW)


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) / "program" / "llm.loom"
    text = transform(path.read_text())
    path.unlink()  # the package may be a hard-link copy: never write through to the original
    path.write_text(text)
