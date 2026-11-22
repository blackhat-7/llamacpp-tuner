"""Adds image input to a built text package: SRC (kv8-r2) -> OUT (kv8-r2v). OUT holds hard links to SRC's files;
a file this step changes is unlinked first, so SRC never changes. The text programs are not regenerated:
- control.loom: image requests render through control_inkling.loom (see there), markers become placeholder
  inputs (chat_vision.loom, generated), and epochs with images set metadata word 2 (8-word input records).
- llm.loom: g_embed takes a live input's projected row, not normalized, instead of its normalized token embedding
  when its record carries one (loom-skills gen_llm.py with a vision block; llama.cpp inkling2.cpp). Text epochs
  read metadata word 2 = 0 and stay on the old path.
- program/vision_*.loom: the image component (generated), the 'vision' parameter scope (the Q8_0 mmproj),
  services.chat.components.image in model.json.

usage: vision.py SRC OUT MMPROJ.gguf"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAM = os.path.join(HERE, "program")
COMPONENT = [
    "vision_prepare.loom",
    "vision_control.loom",
    "vision_stages.loom",
    "vision_encoder.loom",
]


def sub(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(
            f"vision.py: expected one {old[:80]!r}, found {text.count(old)}"
        )
    return text.replace(old, new)


def read(name: str) -> str:
    with open(os.path.join(PROGRAM, name), encoding="utf-8") as f:
        return f.read()


def control(text: str) -> str:
    # The kit's chat_begin, limited to the leading system messages (then the effort block), for image requests.
    begin = re.search(r"(?ms)^func\.def public @chat_begin\(.*?^}\n", text).group(0)
    lead = sub(
        begin,
        "func.def public @chat_begin(%request: buffer, %tools: buffer) -> (buffer, i64) {",
        "func.def @ink_vision_begin(%request: buffer, %tools: buffer, %lead: i64) -> (buffer) {",
    )
    lead = sub(
        lead,
        "  %records = buffer.view %elements[%origin] : buffer -> view<[%extent]x5xi64>\n",
        "  %records = buffer.view %elements[%origin] : buffer -> view<[%extent]x5xi64>\n"
        "  %lead_bound = scalar.assume %lead [range(%lead, 0, 2147483647)] : i64\n"
        "  %lead_count = index.cast %lead_bound : i64 to index\n"
        "  %lead_extent = index.min %lead_count, %extent : index\n",
    )
    lead = sub(
        lead,
        "scf.for %i = [%c0 to %extent step %c1](%acc = %tools_block",
        "scf.for %i = [%c0 to %lead_extent step %c1](%acc = %tools_block",
    )
    loop = re.search(r"(?m)^  %body, %emitted = scf\.for .*\n", lead).group(0)
    lead = sub(
        lead, loop, loop + "    %ib = index.assume %i [lt(%i, %extent)] : index\n"
    ).replace("%records[%i, ", "%records[%ib, ")
    lead = sub(
        lead,
        "  func.return %rendered, %zero64 : buffer, i64\n",
        "  func.return %rendered : buffer\n",
    )
    text = sub(text, "func.def public @chat_begin(", "func.def @format_chat_begin(")
    text = sub(text, "func.def public @chat_message(", "func.def @format_chat_message(")
    # prepare_input with image markers (loom-skills build_vision.py)
    text = sub(
        text, "func.def public @prepare_input(", "func.def @plain_prepare_input("
    )
    text += """
func.def public @prepare_input(%text: buffer, %markers: buffer, %marker_count: i64, %format: i32, %boundary: i32, %capacity: i64) -> (buffer, i64, buffer) {
  %zero = scalar.constant 0 : i64
  %plain = scalar.cmpi eq, %marker_count, %zero : i64
  %tokens, %count, %spans = scf.if %plain -> (buffer, i64, buffer) {
    %t, %n, %s = func.call @plain_prepare_input(%text, %markers, %marker_count, %format, %boundary, %capacity) : (buffer, buffer, i64, i32, i32, i64) -> (buffer, i64, buffer)
    scf.yield %t, %n, %s : buffer, i64, buffer
  } else {
    %t, %n, %s = func.call @vision_prepare_input(%text, %markers, %marker_count, %capacity) : (buffer, buffer, i64, i64) -> (buffer, i64, buffer)
    scf.yield %t, %n, %s : buffer, i64, buffer
  }
  func.return %tokens, %count, %spans : buffer, i64, buffer
}
"""
    # epochs with images: metadata word 2 = 1 (the device reads 8-word input records)
    text = sub(
        text,
        """  view.store %c0, %header[2] : i32, view<4xi32>
  view.store %c0, %header[3] : i32, view<4xi32>
  buffer.copy %plan[%origin], %metadata[%header_bytes], %plan_bytes""",
        """  %vision_media = scalar.addi %first_media_count, %next_media_count : i32
  %vision_has = scalar.cmpi ne, %vision_media, %c0 : i32
  %vision_flag = scalar.extui %vision_has : i1 to i32
  view.store %vision_flag, %header[2] : i32, view<4xi32>
  view.store %c0, %header[3] : i32, view<4xi32>
  buffer.copy %plan[%origin], %metadata[%header_bytes], %plan_bytes""",
    )
    text = sub(
        text,
        "  %compressed_bytes = scalar.constant 1048576 : i64",
        "  %compressed_bytes = scalar.constant 67108864 : i64",
    )
    return text + "\n" + lead + read("chat_vision.loom") + read("control_inkling.loom")


# g_embed of loom-skills gen_llm.py with a vision block (Inkling: Q8_0 table, 72 groups of 32, no embedding scale).
EMBED_IDS = """  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]xi32>
  %raw = view.load %ids_view[%token] : view<[%tokens]xi32> -> i32
"""
EMBED_IDS_VISION = """  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]xi32>
  %meta_view = buffer.view %meta[%zero] : buffer -> view<4xi32>
  %mixed_word = view.load %meta_view[2] : view<4xi32> -> i32
  %mixed_zero = scalar.constant 0 : i32
  %mixed = scalar.cmpi ne, %mixed_word, %mixed_zero : i32
  %stride_one = index.constant 1 : index
  %stride_eight = index.constant 8 : index
  %id_stride = scf.select %mixed, %stride_eight, %stride_one : index
  %n_ids = index.mul %tokens, %id_stride : index
  %ids_flat = buffer.view %ids[%zero] : buffer -> view<[%n_ids]xi32>
  %raw_e0 = index.mul %token, %id_stride : index
  %raw_e = index.assume %raw_e0 [lt(%raw_e0, %n_ids)] : index
  %raw = view.load %ids_flat[%raw_e] : view<[%n_ids]xi32> -> i32
  %feat_word = scf.if %mixed -> (i32) {
    %fw_e0 = index.mul %token, %id_stride : index
    %fw_e1 = index.add %fw_e0, %stride_one : index
    %fw_e = index.assume %fw_e1 [lt(%fw_e1, %n_ids)] : index
    %fw = view.load %ids_flat[%fw_e] : view<[%n_ids]xi32> -> i32
    scf.yield %fw : i32
  } else {
    scf.yield %mixed_zero : i32
  }
  // padding tokens of the epoch shape read stale records: only live inputs carry features
  %tinfo_view = buffer.view %tinfo[%zero] : buffer -> view<[%tokens]x4xi32>
  %live_word = view.load %tinfo_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %is_live = scalar.cmpi ne, %live_word, %mixed_zero : i32
  %has_feat = scalar.cmpi ne, %feat_word, %mixed_zero : i32
  %is_feat = scalar.andi %has_feat, %is_live : i1
  %feat_one_byte = index.constant 1 : offset
"""
EMBED_PICK = """    %v = func.call @g_q8_0_values(%table, %origin, %within) : (buffer, offset, index) -> (vector<32xf32>)
    scf.yield %v : vector<32xf32>
"""
EMBED_PICK_VISION = """    %v = func.call @g_q8_0_values(%table, %origin, %within) : (buffer, offset, index) -> (vector<32xf32>)
    %vv = scf.if %is_feat -> (vector<32xf32>) {
      %fb_lane = index.assume %lane [range(%lane, 0, 71)] : index
      %fo_u = scalar.maxsi %feat_word, %mixed_zero : i32
      %fo_i = index.cast %fo_u : i32 to index
      %fo = index.scale %fo_i, %feat_one_byte : index, offset -> offset
      %fview = buffer.view %ids[%fo] : buffer -> view<72x32xf32>
      %fv = vector.load %fview[%fb_lane, %c0] : view<72x32xf32> -> vector<32xf32>
      scf.yield %fv : vector<32xf32>
    } else {
      scf.yield %v : vector<32xf32>
    }
    scf.yield %vv : vector<32xf32>
"""
# Inkling's stream starts at h = rmsnorm(embedding) * token_embd_norm (copied to x after g_embed); llama.cpp does
# not normalize raw (image) embeddings, so a projected row goes to h as it is.
EMBED_NORM = """    %hv = vector.mulf %n2, %weights : vector<32xf32>
    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x72x32xf32>
"""
EMBED_NORM_VISION = """    %hn = vector.mulf %n2, %weights : vector<32xf32>
    %hv = scf.if %is_feat -> (vector<32xf32>) {
      scf.yield %normalized : vector<32xf32>
    } else {
      scf.yield %hn : vector<32xf32>
    }
    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x72x32xf32>
"""
EMBED_LAUNCH = (
    "kernel.launch @g_embed[%t](%t, %tokens, %embedding, %tok_norm, %x, %h) : [index](index, buffer, view<",
    "kernel.launch @g_embed[%t](%t, %tokens, %metadata, %info, %embedding, %tok_norm, %x, %h) : [index](index, buffer, buffer, buffer, view<",
)


def llm(text: str) -> str:
    kernel = re.search(r'(?ms)^kernel\.def export\("g_embed"\).*?^}\n', text).group(0)
    new = sub(
        kernel,
        "} launch(%tokens_arg: index, %ids: buffer, %table: buffer,",
        "} launch(%tokens_arg: index, %ids: buffer, %meta: buffer, %tinfo: buffer, %table: buffer,",
    )
    new = sub(new, EMBED_IDS, EMBED_IDS_VISION)
    new = sub(new, EMBED_PICK, EMBED_PICK_VISION)
    new = sub(new, EMBED_NORM, EMBED_NORM_VISION)
    text = sub(text, kernel, new)
    if text.count(EMBED_LAUNCH[0]) != 2:  # llm_advance and llm_verify
        raise SystemExit("vision.py: expected two g_embed launches")
    return text.replace(*EMBED_LAUNCH)


def card(model: dict, mmproj: str) -> dict:
    source = model["programs"]["source"]
    source["inputs"] += [
        {"path": "program/vision_prepare.loom"},
        {"path": "program/vision_control.loom"},
    ]
    source["libraries"] += [
        {"path": "program/vision_stages.loom"},
        {"path": "program/vision_encoder.loom"},
    ]
    model["services"]["chat"]["components"] = {"image": {"version": 1}}
    model["variants"]["gguf"]["services"]["chat"]["parameters"]["vision"] = {
        "files": [{"path": f"parameter/{os.path.basename(mmproj)}", "format": "gguf"}]
    }
    return model


def write(path: str, text: str) -> None:
    if os.path.lexists(path):
        os.unlink(path)  # a hard link to SRC: never write through it
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main(src: str, out: str, mmproj: str) -> None:
    os.makedirs(out)
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(out, rel), exist_ok=True)
        for name in files:
            os.link(os.path.join(root, name), os.path.join(out, rel, name))
    os.link(mmproj, os.path.join(out, "parameter", os.path.basename(mmproj)))
    prog = os.path.join(out, "program")
    with open(os.path.join(prog, "control.loom"), encoding="utf-8") as f:
        write(os.path.join(prog, "control.loom"), control(f.read()))
    with open(os.path.join(prog, "llm.loom"), encoding="utf-8") as f:
        write(os.path.join(prog, "llm.loom"), llm(f.read()))
    for name in COMPONENT:
        write(os.path.join(prog, name), read(name))
    with open(os.path.join(out, "model.json"), encoding="utf-8") as f:
        model = json.load(f)
    write(os.path.join(out, "model.json"), json.dumps(card(model, mmproj), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
