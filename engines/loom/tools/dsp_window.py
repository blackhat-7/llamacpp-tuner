"""Limit @g_dsp_attn_split_full in LLM_LOOM, in place, to the last W positions before the anchor.

The drafter's own 8 block keys are always kept. Verification keeps the output exact; only acceptance can change.
Usage: python dsp_window.py LLM_LOOM W
"""

import sys
from pathlib import Path

NAME = 'kernel.def export("g_dsp_attn_split_full") @g_dsp_attn_split_full('
OLD = """      %w_low = index.constant 0 : index
      %n_ctx = index.add %anchor_pos, %c0 : index
"""


def window(text: str, w: int) -> str:
    i = text.index(NAME)
    k = text.index("\n}\n", text.index("launch(", i)) + 3
    kernel = text[i:k]
    assert kernel.count(OLD) == 1
    new = f"""      // Drafter context window: only the last {w} positions before the anchor (exactness is kept by verification).
      %c_win = index.constant {w} : index
      %over = index.cmp ugt, %anchor_pos, %c_win : index
      %w_sub = index.sub %anchor_pos, %c_win : index
      %w_low = scf.select %over, %w_sub, %c0 : index
      %n_ctx = index.sub %anchor_pos, %w_low : index
"""
    return text[:i] + kernel.replace(OLD, new) + text[k:]


if __name__ == "__main__":
    path = Path(sys.argv[1])
    path.write_text(window(path.read_text(), int(sys.argv[2])))
