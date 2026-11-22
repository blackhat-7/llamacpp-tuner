"""Regenerates program/ (the checked-in image component and chat code) from loom-skills' vision generator.
Needs loom-skills at 618f89e with patches/loom-skills-inkling2v.diff applied, gguf and numpy, and LOOM_RDNA4=0
(gfx1100 only: no RDNA4 kernel twins). build.sh does not run this; it copies program/.

usage: LOOM_RDNA4=0 gen_programs.py LOOM_SKILLS_DIR MMPROJ-Q8_0.gguf"""

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(sys.argv[1], "skills/loom-port-llm/starter/vision"))
import add_vision  # noqa: E402
import encoder  # noqa: E402
import gen_chat_vision  # noqa: E402
import gen_component  # noqa: E402
import preprocess  # noqa: E402

assert os.environ.get("LOOM_RDNA4") == "0", "set LOOM_RDNA4=0"
c = encoder.config(sys.argv[2])
fam = preprocess.FAMILY[c["family"]]
f, max_rows = c["patch"] * fam["merge"], fam["tokens"][1]
side = (
    int(round(max_rows**0.5 / 2)) * f
)  # the stage cold preparation compiles: a square of the median budget
T = max_rows * fam["merge"] ** 2
rows = {
    "Tp": -(-T // 256) * 256,
    "T1p": -(-(T + 1) // 256) * 256,
    "Mp": -(-max_rows // 64) * 64,
}
scratch = (
    sum(rows[r] * w * b for r, w, b in encoder.buffer_exprs(c).values())
    + 4096 * (f * max_rows**0.5 + 64) * 3 * 2
)
begin, end = add_vision.MARKERS[c["family"]]
geometry = gen_chat_vision.geometry(
    c["family"], c["patch"], fam["merge"], fam["tokens"][0], max_rows, fam["half_even"]
)
files = {
    "vision_encoder.loom": gen_component.encoder_program(c),
    "vision_prepare.loom": gen_component.prepare_program(
        c, (side, side), c["out_row"], max_rows
    ),
    "vision_stages.loom": gen_component.stages_library(),
    "vision_control.loom": gen_component.control_program(
        c, c["out_row"], max_rows, int(scratch)
    ),
    "chat_vision.loom": gen_chat_vision.chat_lib(begin, end, geometry),
}
os.makedirs(os.path.join(here, "program"), exist_ok=True)
for name, text in files.items():
    with open(
        os.path.join(here, "program", name), "w", encoding="utf-8", newline="\n"
    ) as out:
        out.write(text)
print("wrote", ", ".join(files), "max image tokens", max_rows)
