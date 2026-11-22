#!/usr/bin/env python3
"""Dynamic draft span for Inkling-2-Mini on Loom: per step, verify the preset's full span (draft_depth + 1) or a
shorter one, chosen from the recent first-draft miss rate.

Applies to one-span epochs (rows = 1 decode). A shortened span of L tokens runs the verify and warm stages of
the smallest single-span packing shape that holds it (2, 4 or 8 tokens) and a width-L proposal stage, so the
drafter runs only L - 1 Markov steps. Acceptance and publication are unchanged (g_spec_accept and g_spec_rows read
each span's own length), so the output stays the target's greedy output.

Policy: miss EMA m in 0..256 (256 = every first draft rejected), m <- (m * (2^K - 1) + miss * 256) >> K after each
verified step. With --lengths L1,L2,.. (ascending) and --thresholds T1,T2,.. (descending): m > T1 -> L1, else
m > T2 -> L2, ..., else the full span. Lengths at or above the preset's span are ignored at run time.
  measured lead (d3 preset):            --lengths 2 --thresholds 90 --shift 3
  more lengths (d7 preset, once padding tokens are free), e.g. --lengths 2,3,4,6 --thresholds 110,80,50,25
  --costs 2:13.2,4:17.7,8:26.2 prints the thresholds that maximize tokens per ms for those step costs (ms per
  verify step of L tokens), assuming each draft is accepted with p = 1 - m/256; it does not edit anything.

Control state words (the 2048-byte control block; [1664, 1676) draft table, 1676 gate flag, [1700, 1828) gate
shapes are the generator's): 1684 this epoch's shape index + 1 (0 = full span), 1688 this epoch's proposal stage,
1692 the miss EMA, [1840, 1876) single-span verify shape index + 1 by token count 0..8, [1880, 1916) proposal
stage by width 0..8.

usage: dynspan.py PACKAGE_DIR --lengths .. --thresholds .. [--shift K]   |   dynspan.py --costs ..
Edits PACKAGE_DIR/program/{prepare,control}.loom in place (unlink first: package copies are hard links).
"""

import argparse
import pathlib


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:80]!r}")
    return text.replace(old, new)


def shape_tokens(length: int) -> int:
    return 1 << (length - 1).bit_length()


PREPARE_HEAD = """    // spec-dynspan: verify shape index + 1 of each single-span shape by token count, proposal stages by width
    %dsp_tab_at = index.constant 1840 : offset
    %dsp_tab = buffer.view %state[%dsp_tab_at] : buffer -> view<9xi32>
    %dsp_c8 = scalar.constant 8 : i64
    scf.for %dsp_i = [%c0 to %count step %c1] {
      %dsp_tok = view.load %shape_view[%dsp_i, 0] : view<[%count]x2xi64> -> i64
      %dsp_spn = view.load %shape_view[%dsp_i, 1] : view<[%count]x2xi64> -> i64
      %dsp_one = scalar.cmpi eq, %dsp_spn, %c1_64 : i64
      %dsp_small = scalar.cmpi ule, %dsp_tok, %dsp_c8 : i64
      %dsp_ok = scalar.andi %dsp_one, %dsp_small : i1
      scf.if %dsp_ok {
        %dsp_i32 = index.cast %dsp_i : index to i32
        %dsp_one32 = scalar.constant 1 : i32
        %dsp_tag = scalar.addi %dsp_i32, %dsp_one32 : i32
        %dsp_tb = scalar.assume %dsp_tok [range(%dsp_tok, 0, 8)] : i64
        %dsp_slot = index.cast %dsp_tb : i64 to index
        view.store %dsp_tag, %dsp_tab[%dsp_slot] : i32, view<9xi32>
      }
    }
    %dsp_ptab_at = index.constant 1880 : offset
    %dsp_ptab = buffer.view %state[%dsp_ptab_at] : buffer -> view<9xi32>
    %dsp_ptokens = scalar.constant 8 : i64
    %dsp_b0 = scalar.constant 0 : i32
    %dsp_b1 = scalar.constant 1 : i32
    %dsp_b2 = scalar.constant 2 : i32
"""

PREPARE_STAGE = """    %dsp_width_{w} = scalar.constant {w} : i64
    %dsp_ptag_{w} = scalar.constant {tag} : i64
    %dsp_pstage_{w} = func.call @stage(%propose_root, %dsp_ptag_{w}) : (buffer, i64) -> (i32)
    func.call @config_i64(%dsp_pstage_{w}, %context_key, %context) : (i32, buffer, i64)
    func.call @config_i64(%dsp_pstage_{w}, %pool_key, %pool) : (i32, buffer, i64)
    func.call @config_i64(%dsp_pstage_{w}, %tokens_key, %dsp_ptokens) : (i32, buffer, i64)
    func.call @config_i64(%dsp_pstage_{w}, %spans_key, %c1_64) : (i32, buffer, i64)
    func.call @config_i64(%dsp_pstage_{w}, %width_key, %dsp_width_{w}) : (i32, buffer, i64)
    func.call @config_i64(%dsp_pstage_{w}, %epoch_spans_key, %span_capacity) : (i32, buffer, i64)
    func.call @config_i64(%dsp_pstage_{w}, %epoch_tokens_key, %maximum) : (i32, buffer, i64)
    func.call @parameter(%dsp_pstage_{w}, %dsp_b0, %c0_source, %checkpoint, %policy) : (i32, i32, i32, i32, buffer)
    func.call @parameter(%dsp_pstage_{w}, %dsp_b1, %c0_source, %draft_source, %policy) : (i32, i32, i32, i32, buffer)
    func.call @parameter(%dsp_pstage_{w}, %dsp_b2, %c0_source, %checkpoint, %head_policy) : (i32, i32, i32, i32, buffer)
    %dsp_pslot_{w} = index.constant {w} : index
    view.store %dsp_pstage_{w}, %dsp_ptab[%dsp_pslot_{w}] : i32, view<9xi32>
"""

ENCODE_HEAD = """  view.store %flag_value, %flag_view[0] : i32, view<1xi32>
  // spec-dynspan: a single speculative span whose recent first drafts often missed verifies fewer tokens
  %ds_r0 = index.assume %c0 [lt(%c0, %plan_extent)] : index
  %ds_credit = view.load %requests[%ds_r0, 7] : view<[%plan_extent]x9xi32> -> i32
  %ds_plan = view.load %requests[%ds_r0, 0] : view<[%plan_extent]x9xi32> -> i32
  %ds_single = scalar.cmpi eq, %first_count, %c1_i32 : i32
  %ds_spec = scalar.cmpi ne, %ds_credit, %c0_i32 : i32
  %ds_state_at = index.constant 1684 : offset
  %ds_state = buffer.view %control_state[%ds_state_at] : buffer -> view<3xi32>
  %ds_m = view.load %ds_state[2] : view<3xi32> -> i32
  %ds_tab_at = index.constant 1840 : offset
  %ds_tab = buffer.view %control_state[%ds_tab_at] : buffer -> view<9xi32>
  %ds_ptab_at = index.constant 1880 : offset
  %ds_ptab = buffer.view %control_state[%ds_ptab_at] : buffer -> view<9xi32>
  %ds_len_full = scalar.constant 0 : i32
"""

ENCODE_LEVEL = """  %ds_thr_{i} = scalar.constant {thr} : i32
  %ds_hit_{i} = scalar.cmpi ugt, %ds_m, %ds_thr_{i} : i32
  %ds_lv_{i} = scalar.constant {length} : i32
  %ds_len_{i} = scf.select %ds_hit_{i}, %ds_lv_{i}, {prev} : i32
"""

ENCODE_TAIL = """  %ds_want = scalar.cmpi ne, {chosen}, %c0_i32 : i32
  %ds_shorter = scalar.cmpi ult, {chosen}, %ds_plan : i32
  %ds_lb = scalar.assume {chosen} [range({chosen}, 0, 8)] : i32
  %ds_li = index.cast %ds_lb : i32 to index
  %ds_pstage = view.load %ds_ptab[%ds_li] : view<9xi32> -> i32
  %ds_has = scalar.cmpi ne, {entry}, %c0_i32 : i32
  %ds_hasp = scalar.cmpi ne, %ds_pstage, %c0_i32 : i32
  %ds_a0 = scalar.andi %ds_single, %ds_spec : i1
  %ds_a1 = scalar.andi %ds_a0, %ds_want : i1
  %ds_a2 = scalar.andi %ds_a1, %ds_shorter : i1
  %ds_a3 = scalar.andi %ds_a2, %ds_has : i1
  %shrink = scalar.andi %ds_a3, %ds_hasp : i1
  %ds_word = scf.select %shrink, {entry}, %c0_i32 : i32
  view.store %ds_word, %ds_state[0] : i32, view<3xi32>
  view.store %ds_pstage, %ds_state[1] : i32, view<3xi32>
"""

EPOCH_PROPOSE_OLD = """    %pstage = scalar.addi %propose_first, %chosen : i32
"""
EPOCH_PROPOSE_NEW = """    %pstage_full = scalar.addi %propose_first, %chosen : i32
    %ds_pstage = view.load %ds_state[1] : view<3xi32> -> i32
    %pstage = scf.select %shrunk, %ds_pstage, %pstage_full : i32
"""

EPOCH_INDEX_OLD = """  %eff_index = scf.select %gated, %flag_shape, %shape_index : index
"""
EPOCH_INDEX_NEW = """  %eff_gate = scf.select %gated, %flag_shape, %shape_index : index
  // spec-dynspan: a shortened span runs the verify and warm stages of its own packing shape
  %ds_state_at = index.constant 1684 : offset
  %ds_state = buffer.view %control_state[%ds_state_at] : buffer -> view<3xi32>
  %ds_word = view.load %ds_state[0] : view<3xi32> -> i32
  %shrunk = scalar.cmpi ne, %ds_word, %c0 : i32
  %ds_nz = scf.select %shrunk, %ds_word, %c1 : i32
  %ds_idx = scalar.subi %ds_nz, %c1 : i32
  %ds_b = scalar.assume %ds_idx [range(%ds_idx, 0, 127)] : i32
  %ds_shape = index.cast %ds_b : i32 to index
  %eff_index = scf.select %shrunk, %ds_shape, %eff_gate : index
"""

PUBLISH_OLD = """      %verification = scalar.extui %speculative : i1 to i32
"""
PUBLISH_NEW = """      %verification = scalar.extui %speculative : i1 to i32
      // spec-dynspan: first-draft miss EMA of the single decoding span
      %ds_one = scalar.cmpi eq, %count, %c1_i32 : i32
      %ds_upd = scalar.andi %speculative, %ds_one : i1
      scf.if %ds_upd {{
        %ds_state_at = index.constant 1684 : offset
        %ds_state = buffer.view %control_state[%ds_state_at] : buffer -> view<3xi32>
        %ds_m = view.load %ds_state[2] : view<3xi32> -> i32
        %ds_miss = scalar.cmpi ule, %consumed, %c1_i32 : i32
        %ds_full = scalar.constant 256 : i32
        %ds_mv = scf.select %ds_miss, %ds_full, %c0_i32 : i32
        %ds_keep = scalar.constant {keep} : i32
        %ds_mk = scalar.muli %ds_m, %ds_keep : i32
        %ds_sum = scalar.addi %ds_mk, %ds_mv : i32
        %ds_shift = scalar.constant {shift} : i32
        %ds_next = scalar.shrui %ds_sum, %ds_shift : i32
        view.store %ds_next, %ds_state[2] : i32, view<3xi32>
      }}
"""


def thresholds_from_costs(costs: dict[int, float]) -> None:
    """Best length per miss EMA value for geometric acceptance p = 1 - m/256; print the switch points."""
    lengths = sorted(costs)
    best = []
    for m in range(257):
        p = 1 - m / 256
        rate = {n: sum(p**k for k in range(n)) / costs[n] for n in lengths}
        best.append(max(lengths, key=lambda n: (rate[n], -n)))
    full = best[0]
    pairs = []
    for n in sorted(set(best) - {full}):
        pairs.append((n, min(m for m in range(257) if best[m] == n) - 1))
    print("full span (lowest miss):", full, " used:", sorted(set(best)))
    print(
        "--lengths",
        ",".join(str(n) for n, _ in pairs),
        "--thresholds",
        ",".join(str(t) for _, t in pairs),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("package", nargs="?")
    ap.add_argument("--lengths", default="2")
    ap.add_argument("--thresholds", default="128")
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--costs")
    args = ap.parse_args()
    if args.costs:
        costs = {
            int(k): float(v) for k, v in (c.split(":") for c in args.costs.split(","))
        }
        thresholds_from_costs(
            {n: c for n, c in costs.items() if n >= 2}
        )  # length 1 (no drafting) is not offered
        return
    lengths = [int(x) for x in args.lengths.split(",")]
    thresholds = [int(x) for x in args.thresholds.split(",")]
    if (
        len(lengths) != len(thresholds)
        or lengths != sorted(set(lengths))
        or thresholds != sorted(thresholds, reverse=True)
        or not all(2 <= n <= 8 for n in lengths)
    ):
        raise SystemExit(
            "--lengths ascending in 2..8, --thresholds descending, same count"
        )
    program = pathlib.Path(args.package) / "program"

    prepare = (program / "prepare.loom").read_text()
    stages = "".join(PREPARE_STAGE.format(w=n, tag=600 + n) for n in lengths)
    prepare = replace_once(
        prepare,
        "    %draft_at = index.constant 1664 : offset\n",
        PREPARE_HEAD + stages + "    %draft_at = index.constant 1664 : offset\n",
    )

    # chosen length: the first (shortest) level whose threshold the EMA exceeds, else 0 (full span)
    levels, prev = [], "%ds_len_full"
    for i in reversed(range(len(lengths))):
        levels.append(
            ENCODE_LEVEL.format(i=i, thr=thresholds[i], length=lengths[i], prev=prev)
        )
        prev = f"%ds_len_{i}"
    chosen = prev
    # verify shape entry of the chosen length's packing shape (a select chain over the configured lengths)
    entry_lines, entry = [], "%c0_i32"
    for n in lengths:
        tokens = shape_tokens(n)
        entry_lines.append(f"""  %ds_eslot_{n} = index.constant {tokens} : index
  %ds_eval_{n} = view.load %ds_tab[%ds_eslot_{n}] : view<9xi32> -> i32
  %ds_eis_{n} = scalar.cmpi eq, {chosen}, %ds_lv_{lengths.index(n)} : i32
  %ds_esel_{n} = scf.select %ds_eis_{n}, %ds_eval_{n}, {entry} : i32
""")
        entry = f"%ds_esel_{n}"
    tail = ENCODE_TAIL.replace("{chosen}", chosen).replace("{entry}", entry)
    encode = ENCODE_HEAD + "".join(levels) + "".join(entry_lines) + tail

    control = (program / "control.loom").read_text()
    control = replace_once(
        control, "  view.store %flag_value, %flag_view[0] : i32, view<1xi32>\n", encode
    )
    # encode_epoch: both span-length reads take the shortened length; store it when gated or shortened
    control = replace_once(
        control,
        """    %length_host = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %length = scf.select %gate, %c1_i32, %length_host : i32
    scf.if %gate {
""",
        f"""    %length_plan = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %length_host = scf.select %shrink, {chosen}, %length_plan : i32
    %length = scf.select %gate, %c1_i32, %length_host : i32
    %store_length = scalar.ori %gate, %shrink : i1
    scf.if %store_length {{
""",
    )
    control = replace_once(
        control,
        """    %length32_host = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
""",
        f"""    %length32_plan = view.load %requests[%sr, 0] : view<[%plan_extent]x9xi32> -> i32
    %length32_host = scf.select %shrink, {chosen}, %length32_plan : i32
""",
    )
    control = replace_once(
        control,
        PUBLISH_OLD,
        PUBLISH_NEW.format(keep=(1 << args.shift) - 1, shift=args.shift),
    )
    control = replace_once(control, EPOCH_INDEX_OLD, EPOCH_INDEX_NEW)
    control = replace_once(control, EPOCH_PROPOSE_OLD, EPOCH_PROPOSE_NEW)

    for name, text in (("prepare.loom", prepare), ("control.loom", control)):
        path = program / name
        path.unlink()
        path.write_text(text)


main()
