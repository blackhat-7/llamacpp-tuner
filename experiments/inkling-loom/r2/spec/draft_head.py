#!/usr/bin/env python3
"""Reduced draft vocabulary for the DSpark proposal head (Inkling-2-Mini on Loom).

The one-span proposal step reads a Q4_0 head of N frequent token rows (dsp_draft_head.weight, 42 MB at N = 32768)
instead of the full 200,058-row Q4_0 copy (259 MB), and maps the winning row back to its token id
(dsp_draft_head.ids). Both tensors live in the drafter GGUF written by make_draft_head.py. Verification still runs
the target's own head, so the output is unchanged; only which drafts are proposed (and accepted) can change.
Proposal cohorts of 2+ spans (multi-row presets) keep the full head.

Every proposal stage must bind the same parameter roots ("stage does not match the token runner bindings"
otherwise): the one-span stage reads only the reduced head, so both heads come from the proposal-head root
(binding 2), which prepare.loom now points at the drafter file (make_draft_head.py copies output.weight into it).

Adds kernels g_head_mv_q8_p40_sub and g_head_select_sub (derived from g_head_mv_q8_p40 / g_head_select), two
command.parameter lines and the one-span launches in @llm_dflash_propose, and points the package's "draft"
parameter file at the new GGUF.

usage: draft_head.py PACKAGE_DIR DRAFTER_WITH_HEAD.gguf N
Edits PACKAGE_DIR/program/{llm,prepare}.loom and PACKAGE_DIR/model.json in place (unlinks first: copies are hard links);
links the GGUF into PACKAGE_DIR/parameter/.
"""
import json
import os
import pathlib
import sys


def once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:90]!r}")
    return text.replace(old, new)


def kernel_text(llm: str, name: str) -> str:
    start = llm.index(f'kernel.def export("{name}")')
    end = llm.index("\nkernel.def ", start + 1)
    return llm[start:end].rstrip() + "\n"


def sub_head(src: str, n: int) -> str:
    chunks = n // 128
    k = src.replace('export("g_head_mv_q8_p40") @g_head_mv_q8_p40(', 'export("g_head_mv_q8_p40_sub") @g_head_mv_q8_p40_sub(')
    k = once(k, "%chunks = index.constant 1563 : index", f"%chunks = index.constant {chunks} : index")
    k = once(k, "[range(%chunk_id, 0, 1562)]", f"[range(%chunk_id, 0, {chunks - 1})]")
    k = once(k, "%vocab = index.constant 200058 : index", f"%vocab = index.constant {n} : index")
    k = once(k, "%last_row = index.constant 200057 : index", f"%last_row = index.constant {n - 1} : index")
    k = once(k, "%partials: buffer, %logits: buffer) {", "%partials: buffer, %logits: buffer, %idmap: buffer) {")
    # proposals are greedy (g_dsp_prepare: temperature 0), so the per-row logits are never read
    k = once(k, """      %lrow = index.assume %row_value [range(%row_value, 0, 200063)] : index
      %stored = scf.select %in_vocab, %total, %neg_inf : f32
      view.store %stored, %logit_view[%ls, %lrow] : f32, view<[%spans]x200064xf32>
""", "")
    k = once(k, "    view.store %best_id, %partial_view[%ss, %chunk, %c1] : i32, view<[%spans]x1563x2xi32>\n", f"""    %idmap_v = buffer.view %idmap[%zero] : buffer -> view<{n}xf32>
    %best_row_b = scalar.assume %best_id [range(%best_id, 0, {n - 1})] : i32
    %best_row = index.cast %best_row_b : i32 to index
    %best_bits = view.load %idmap_v[%best_row] : view<{n}xf32> -> f32
    %best_token = scalar.bitcast %best_bits : f32 to i32
    view.store %best_token, %partial_view[%ss, %chunk, %c1] : i32, view<[%spans]x1563x2xi32>
""")
    return k


def sub_select(src: str, n: int) -> str:
    k = src.replace('export("g_head_select") @g_head_select(', 'export("g_head_select_sub") @g_head_select_sub(')
    return once(k, "%chunks = index.constant 1563 : index", f"%chunks = index.constant {n // 128} : index")


def main() -> None:
    package, gguf_path, n = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).resolve(), int(sys.argv[3])
    assert n % 128 == 0
    llm_path = package / "program/llm.loom"
    llm = llm_path.read_text()
    head_bytes = n * 72 * 18
    kernels = sub_head(kernel_text(llm, "g_head_mv_q8_p40"), n) + "\n" + sub_select(kernel_text(llm, "g_head_select"), n) + "\n"
    llm = once(llm, 'kernel.def export("g_head_mv_t4_q8_p40")', kernels + 'kernel.def export("g_head_mv_t4_q8_p40")')
    llm = once(llm, '  %head = command.parameter %w_phead, "output.weight" : view<259275168xi8>\n',
               f'''  %head = command.parameter %w_phead, "output.weight" : view<259275168xi8>
  %dh_w = command.parameter %w_phead, "dsp_draft_head.weight" : view<{head_bytes}xi8>
  %dh_ids = command.parameter %w_phead, "dsp_draft_head.ids" : view<{n}xf32>
''')
    llm = once(llm, "      kernel.launch @g_head_mv_q8_p40[%spans](%spans, %spans, %step_meta, %hq3, %head, %partials, %logits) : [index](index, index, buffer, buffer, view<259275168xi8>, buffer, buffer)\n",
               f"      kernel.launch @g_head_mv_q8_p40_sub[%spans](%spans, %spans, %step_meta, %hq3, %dh_w, %partials, %logits, %dh_ids) : [index](index, index, buffer, buffer, view<{head_bytes}xi8>, buffer, buffer, view<{n}xf32>)\n")
    llm = once(llm, "    kernel.launch @g_head_select[%spans](%spans, %step_meta, %partials, %logits, %step_pred) : [index](index, buffer, buffer, buffer, buffer)\n",
               """    scf.if %one_span {
      kernel.launch @g_head_select_sub[%spans](%spans, %step_meta, %partials, %logits, %step_pred) : [index](index, buffer, buffer, buffer, buffer)
    } else {
      kernel.launch @g_head_select[%spans](%spans, %step_meta, %partials, %logits, %step_pred) : [index](index, buffer, buffer, buffer, buffer)
    }
""")
    llm_path.unlink()
    llm_path.write_text(llm)

    prepare_path = package / "program/prepare.loom"
    prepare = prepare_path.read_text()
    head_root = "%c0_source, %checkpoint, %head_policy)"
    if head_root not in prepare:
        raise SystemExit("no proposal-head parameter binding in prepare.loom")
    prepare = prepare.replace(head_root, "%c0_source, %draft_source, %head_policy)")
    prepare_path.unlink()
    prepare_path.write_text(prepare)

    target = package / "parameter" / gguf_path.name
    if not target.exists():
        os.link(gguf_path, target)
    model_path = package / "model.json"
    model = json.loads(model_path.read_text())
    files = model["variants"]["gguf"]["services"]["chat"]["parameters"]["draft"]["files"]
    assert len(files) == 1
    files[0]["path"] = f"parameter/{gguf_path.name}"
    files[0]["origin"] = f"@checkpoint/{gguf_path.name}"
    model_path.unlink()
    model_path.write_text(json.dumps(model, indent=2) + "\n")


main()
