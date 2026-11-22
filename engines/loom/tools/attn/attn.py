"""Faster decode/verify attention for Inkling-2-Mini: edit a kv8w4096 package's program/llm.loom in place.

1. SWA layers (40 visits): @g_ink_attn_split_swa_v2 (kernels/g_ink_attn_split_swa_v2.loom) replaces both
   @g_ink_attn_split_swa (one token) and @g_ink_attn_split_swa_span (verify spans) at every launch site.
2. @g_ink_attn_combine is replaced by kernels/g_ink_attn_combine.loom (same arguments and output; parallel over splits,
   split counts that match the new SWA kernel and GG below).
3. Global layers: @g_ink_attn_split_v2 uses ceil(keys / GG) splits (was ceil(keys / 8)), capped at 64.

All three skip token rows that are not live (info[t][2] == 0, the padding rows of a power-of-two verify shape: the
span packet's length is the real token count), so a span costs only its real tokens.

Usage: python attn.py PACKAGE_DIR   (run after engines/loom/build.sh's steps)
"""

import re
import sys
from pathlib import Path

KERNELS = Path(__file__).resolve().parent / "kernels"
GG = 256  # must match the combine (gen_combine.py GG)

SWA_SPAN = re.compile(
    r"kernel\.launch @g_ink_attn_split_swa_span\[%spans\]\(%t, %spans, (%swa_off_\d+), %metadata, %origins, %q, %r_buf, "
    r"(%rbias_\d+), %arena, %pm, %pl, %po\) : .*"
)
SWA_TYPE = (
    " : [index](index, index, index, buffer, buffer, buffer, buffer, buffer, view<65536xi8>, buffer, buffer, buffer,"
    " buffer)"
)


def kernel_span(text: str, name: str) -> tuple[int, int]:
    i = text.index(f'kernel.def export("{name}") @{name}(')
    return i, text.index("\n}\n", text.index("launch(", i)) + 3


def transform(text: str) -> str:
    a, b = kernel_span(text, "g_ink_attn_combine")
    new = (KERNELS / "g_ink_attn_combine.loom").read_text().rstrip("\n") + "\n\n"
    new += (KERNELS / "g_ink_attn_split_swa_v2.loom").read_text().rstrip("\n") + "\n"
    text = text[:a] + new + text[b:]

    n = text.count("kernel.launch @g_ink_attn_split_swa[%t](")
    assert n == 80, n
    text = text.replace(
        "kernel.launch @g_ink_attn_split_swa[%t](",
        "kernel.launch @g_ink_attn_split_swa_v2[%t](",
    )
    # A verify span takes the same kernel with one grid row per token: each finds its span through info.
    text, n = SWA_SPAN.subn(
        lambda m: f"kernel.launch @g_ink_attn_split_swa_v2[%t](%t, %spans, {m.group(1)}, %metadata, %info, %origins, %q,"
        f" %r_buf, {m.group(2)}, %arena, %pm, %pl, %po)" + SWA_TYPE,
        text,
    )
    assert n == 80, n

    a, b = kernel_span(text, "g_ink_attn_split_v2")
    return text[:a] + transform_v2(text[a:b]) + text[b:]


def transform_v2(v2: str) -> str:
    """@g_ink_attn_split_v2 with ceil(keys / GG) splits."""
    for old, new in (
        (
            "%c7 = index.constant 7 : index\n",
            f"%c7 = index.constant {GG - 1} : index\n    %cGG = index.constant {GG} : index\n",
        ),
        ("[range(%nk7, 8, 2097159)]", f"[range(%nk7, {GG}, {2097151 + GG})]"),
        (
            "%s_need = index.div %nk7_b, %c8 : index",
            "%s_need = index.div %nk7_b, %cGG : index",
        ),
    ):
        assert v2.count(old) == 1, old
        v2 = v2.replace(old, new)
    return v2


if __name__ == "__main__":
    path = Path(sys.argv[1]) / "program" / "llm.loom"
    text = transform(path.read_text())
    path.unlink()  # the package may be a hard-link copy: never write through to the original
    path.write_text(text)
