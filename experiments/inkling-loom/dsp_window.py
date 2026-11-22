#!/usr/bin/env python3
"""Print @g_dsp_attn_split_full from LLM_LOOM with its context limited to the last W positions before the anchor
(the drafter's own 8 block keys are always kept). Verification keeps the output exact; only acceptance can change.
usage: dsp_window.py LLM_LOOM W"""
import sys
src, w = open(sys.argv[1]).read(), int(sys.argv[2])
name = 'kernel.def export("g_dsp_attn_split_full") @g_dsp_attn_split_full('
i = src.index(name); k = src.index("\n}\n", src.index("launch(", i)) + 3
t = src[i:k]
old = """      %w_low = index.constant 0 : index
      %n_ctx = index.add %anchor_pos, %c0 : index
"""
new = f"""      // Drafter context window: only the last {w} positions before the anchor (exactness is kept by verification).
      %c_win = index.constant {w} : index
      %over = index.cmp ugt, %anchor_pos, %c_win : index
      %w_sub = index.sub %anchor_pos, %c_win : index
      %w_low = scf.select %over, %w_sub, %c0 : index
      %n_ctx = index.sub %anchor_pos, %w_low : index
"""
assert t.count(old) == 1
sys.stdout.write(t.replace(old, new))
