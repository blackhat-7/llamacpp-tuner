#!/usr/bin/env python3
"""Exact speculative sampling at temperature > 0 for Inkling-2-Mini on Loom: the drafter shares the target's noise.
Run after topk.py: the drafter then draws through g_head_select's T > 0 path once per draft.

Before, at T > 0 g_head_select drew each token by inverse CDF with one uniform per (seed, position), and the
drafter proposed its argmax (g_dsp_prepare wrote temperature 0 into its step_meta), so a draft was accepted with
probability p(draft).

Gumbel-max coupling (Daliri et al. 2024, arXiv 2408.07978):
1. g_head_select's T > 0 draw becomes a Gumbel-max over the same kept candidates (top-k <= 64 by logit, soft cap
   30, the shortest prefix reaching top_p): argmax of (v - m) / T - log(-log u), with one u per (seed, position of
   the drawn token, token id): u = ((h >> 8) + 0.5) * 2^-24, h = lowbias32(position hash + id * 0x9E3779B9), the
   position hash being the one the inverse-CDF draw used. That is an exact sample of softmax(v / T) over the kept
   set, so plain decoding samples the same distribution as before (a different draw for a given seed).
2. The drafter proposes at T > 0 through the same path on its own logits: its step_meta keeps the span's
   temperature, top_p, top_k and seed (g_dsp_prepare), and g_dsp_step_commit moves step_meta's start one position
   per Markov step, so step s keys its noise on start + s, the position of the token it proposes. The target's
   verify row for that position uses the same noise, so the draft matches whenever both draws land on one token.
Every emitted token is still the target's own draw at its position, so for a given seed the output is the one
plain decoding gives whatever the drafter proposes; only acceptance changes. T = 0 paths are untouched.

usage: gumbel.py PACKAGE_DIR   (after topk.py)
Edits PACKAGE_DIR/program/llm.loom in place (unlinks it first: package copies are hard links).
"""

import pathlib
import sys

from topk import edit_kernel, replace_once

DRAW_OLD = """        // Candidates kept: the shortest prefix whose weight reaches top_p of the total.
        %kept_total, %kept_done = scf.for %r = [%c0 to %k step %c1](%acc = %zero_f : f32, %done = %greedy : i1) -> (f32, i1) {
          %rr = index.assume %r [range(%r, 0, 63)] : index
          %l = view.load %cand_v[%rr] : view<64xf32> -> f32
          %d = scalar.subf %l, %m : f32
          %ds = scalar.mulf %d, %inv_t : f32
          %w = scalar.expf<afn> %ds : f32
          %added = scalar.addf %acc, %w : f32
          %next = scf.select %done, %acc, %added : f32
          %reached = scalar.cmpf oge, %next, %mass : f32
          %now_done = scalar.ori %done, %reached : i1
          scf.yield %next, %now_done : f32, i1
        }
        // Counter-based uniform in [0, 1) from (seed, position of the drawn token).
        %start32 = view.load %packets[%span, 1] : view<[%spans]x9xi32> -> i32
        %length32 = view.load %packets[%span, 0] : view<[%spans]x9xi32> -> i32
        %position = scalar.addi %start32, %length32 : i32
        %golden = scalar.constant -1640531527 : i32
        %mix_a = scalar.constant -2048144789 : i32
        %mix_b = scalar.constant 2146121005 : i32
        %mix_c = scalar.constant -2073254261 : i32
        %c15_i32 = scalar.constant 15 : i32
        %c16_i32 = scalar.constant 16 : i32
        %c8_i32 = scalar.constant 8 : i32
        %h0 = scalar.muli %seed, %golden : i32
        %p0 = scalar.muli %position, %mix_a : i32
        %h1 = scalar.xori %h0, %p0 : i32
        %s1 = scalar.shrui %h1, %c16_i32 : i32
        %h2 = scalar.xori %h1, %s1 : i32
        %h3 = scalar.muli %h2, %mix_b : i32
        %s2 = scalar.shrui %h3, %c15_i32 : i32
        %h4 = scalar.xori %h3, %s2 : i32
        %h5 = scalar.muli %h4, %mix_c : i32
        %s3 = scalar.shrui %h5, %c16_i32 : i32
        %h6 = scalar.xori %h5, %s3 : i32
        %u24 = scalar.shrui %h6, %c8_i32 : i32
        %uf = scalar.uitofp %u24 : i32 to f32
        %scale24 = scalar.constant 5.9604644775390625e-08 : f32
        %u = scalar.mulf %uf, %scale24 : f32
        %target = scalar.mulf %u, %kept_total : f32
        %chosen, %picked = scf.for %r = [%c0 to %k step %c1](%acc = %zero_f : f32, %pick = %big : i32) -> (f32, i32) {
          %rr = index.assume %r [range(%r, 0, 63)] : index
          %l = view.load %cand_v[%rr] : view<64xf32> -> f32
          %id = view.load %cand_i[%rr] : view<64xi32> -> i32
          %d = scalar.subf %l, %m : f32
          %ds = scalar.mulf %d, %inv_t : f32
          %w = scalar.expf<afn> %ds : f32
          %next = scalar.addf %acc, %w : f32
          %unset = scalar.cmpi eq, %pick, %big : i32
          %hit = scalar.cmpf ogt, %next, %target : f32
          %take = scalar.andi %unset, %hit : i1
          %np = scf.select %take, %id, %pick : i32
          scf.yield %next, %np : f32, i32
        }
"""

DRAW_NEW = """        // Gumbel-max over the kept candidates (the shortest prefix whose weight reaches top_p of the total):
        // argmax of (l - m) / T - log(-log u), one u per (seed, position of the drawn token, token id). An exact
        // sample of softmax(l / T) over the kept set; the drafter adds the same noise to its own logits
        // (llm_dflash_propose), so its proposal is this token whenever both draws land on it.
        %start32 = view.load %packets[%span, 1] : view<[%spans]x9xi32> -> i32
        %length32 = view.load %packets[%span, 0] : view<[%spans]x9xi32> -> i32
        %position = scalar.addi %start32, %length32 : i32
        %golden = scalar.constant -1640531527 : i32
        %mix_a = scalar.constant -2048144789 : i32
        %mix_b = scalar.constant 2146121005 : i32
        %mix_c = scalar.constant -2073254261 : i32
        %c15_i32 = scalar.constant 15 : i32
        %c16_i32 = scalar.constant 16 : i32
        %c8_i32 = scalar.constant 8 : i32
        %h0 = scalar.muli %seed, %golden : i32
        %p0 = scalar.muli %position, %mix_a : i32
        %h1 = scalar.xori %h0, %p0 : i32
        %s1 = scalar.shrui %h1, %c16_i32 : i32
        %h2 = scalar.xori %h1, %s1 : i32
        %h3 = scalar.muli %h2, %mix_b : i32
        %s2 = scalar.shrui %h3, %c15_i32 : i32
        %h4 = scalar.xori %h3, %s2 : i32
        %h5 = scalar.muli %h4, %mix_c : i32
        %s3 = scalar.shrui %h5, %c16_i32 : i32
        %h6 = scalar.xori %h5, %s3 : i32
        %scale24 = scalar.constant 5.9604644775390625e-08 : f32
        %half = scalar.constant 0.5 : f32
        // -log(u) >= its value at the largest u: (u24 + 0.5) can round to 2^24, and the fast log is loose near 1.
        %e_floor = scalar.constant 5.9604644775390625e-08 : f32
        %true = scalar.constant true : i1
        %kept_total, %kept_done, %gb_best, %picked = scf.for %r = [%c0 to %k step %c1](%acc = %zero_f : f32, %done = %greedy : i1, %bs = %neg_inf : f32, %pick = %big : i32) -> (f32, i1, f32, i32) {
          %rr = index.assume %r [range(%r, 0, 63)] : index
          %l = view.load %cand_v[%rr] : view<64xf32> -> f32
          %id = view.load %cand_i[%rr] : view<64xi32> -> i32
          %d = scalar.subf %l, %m : f32
          %ds = scalar.mulf %d, %inv_t : f32
          %w = scalar.expf<afn> %ds : f32
          %t0 = scalar.muli %id, %golden : i32
          %t1 = scalar.addi %h6, %t0 : i32
          %t2 = scalar.shrui %t1, %c16_i32 : i32
          %t3 = scalar.xori %t1, %t2 : i32
          %t4 = scalar.muli %t3, %mix_b : i32
          %t5 = scalar.shrui %t4, %c15_i32 : i32
          %t6 = scalar.xori %t4, %t5 : i32
          %t7 = scalar.muli %t6, %mix_c : i32
          %t8 = scalar.shrui %t7, %c16_i32 : i32
          %t9 = scalar.xori %t7, %t8 : i32
          %u24 = scalar.shrui %t9, %c8_i32 : i32
          %uf = scalar.uitofp %u24 : i32 to f32
          %uh = scalar.addf %uf, %half : f32
          %u = scalar.mulf %uh, %scale24 : f32
          %log_u = scalar.logf<afn> %u : f32
          %e_raw = scalar.negf %log_u : f32
          %e = scalar.maxnumf %e_raw, %e_floor : f32
          %log_e = scalar.logf<afn> %e : f32
          %score = scalar.subf %ds, %log_e : f32
          %kept = scalar.xori %done, %true : i1
          %better = scalar.cmpf ogt, %score, %bs : f32
          %take = scalar.andi %kept, %better : i1
          %nbs = scf.select %take, %score, %bs : f32
          %np = scf.select %take, %id, %pick : i32
          %added = scalar.addf %acc, %w : f32
          %next = scf.select %done, %acc, %added : f32
          %reached = scalar.cmpf oge, %next, %mass : f32
          %now_done = scalar.ori %done, %reached : i1
          scf.yield %next, %now_done, %nbs, %np : f32, i1, f32, i32
        }
"""

PREPARE_OLD = """      // step_meta: greedy (temp=0, flags=1, length=1, cursor=k)
      %k32 = index.cast %kb : index to i32
      view.store %c1_i32, %sm[%kb, 0] : i32, view<[%blocks]x9xi32>
      view.store %c0_i32, %sm[%kb, 2] : i32, view<[%blocks]x9xi32>
      view.store %k32, %sm[%kb, 4] : i32, view<[%blocks]x9xi32>
      view.store %c1_i32, %sm[%kb, 6] : i32, view<[%blocks]x9xi32>
      view.store %c0_i32, %sm[%kb, 7] : i32, view<[%blocks]x9xi32>
"""

PREPARE_NEW = """      // step_meta: one selected row (flags=1, length=1, cursor=k). It keeps the span's temperature, top_p, top_k
      // and seed: at T > 0 the drafter draws with the noise the target's verify rows use (g_head_select).
      %k32 = index.cast %kb : index to i32
      view.store %c1_i32, %sm[%kb, 0] : i32, view<[%blocks]x9xi32>
      view.store %k32, %sm[%kb, 4] : i32, view<[%blocks]x9xi32>
      view.store %c1_i32, %sm[%kb, 6] : i32, view<[%blocks]x9xi32>
"""

COMMIT_ARGS_OLD = "%prev_tok: buffer, %target_tokens: buffer) {\n"
COMMIT_ARGS_NEW = "%prev_tok: buffer, %target_tokens: buffer, %step_meta: buffer) {\n"

COMMIT_LOOP_OLD = """  scf.for %b = [%c0 to %cnt step %c1] {
    %bb = index.assume %b [lt(%b, %blocks)] : index
    %tok_id = view.load %predv[%bb] : view<[%blocks]xi32> -> i32
    view.store %tok_id, %prevv[%bb] : i32, view<[%blocks]xi32>
"""

COMMIT_LOOP_NEW = """  %c16o = index.constant 16 : offset
  %c1_i32 = scalar.constant 1 : i32
  %smv = buffer.view %step_meta[%c16o] : buffer -> view<[%blocks]x9xi32>
  scf.for %b = [%c0 to %cnt step %c1] {
    %bb = index.assume %b [lt(%b, %blocks)] : index
    %tok_id = view.load %predv[%bb] : view<[%blocks]xi32> -> i32
    view.store %tok_id, %prevv[%bb] : i32, view<[%blocks]xi32>
    // The next step proposes the following position: g_head_select keys its noise on start + length.
    %sm_start = view.load %smv[%bb, 1] : view<[%blocks]x9xi32> -> i32
    %sm_next = scalar.addi %sm_start, %c1_i32 : i32
    view.store %sm_next, %smv[%bb, 1] : i32, view<[%blocks]x9xi32>
"""

LAUNCH_OLD = (
    "kernel.launch @g_dsp_step_commit(%spans, %epoch_tokens, %step, %epoch, %metadata, %slots, %step_pred, %prev_tok,"
    " %target_tokens) : (index, index, index, buffer, buffer, buffer, buffer, buffer, buffer)\n"
)
LAUNCH_NEW = (
    "kernel.launch @g_dsp_step_commit(%spans, %epoch_tokens, %step, %epoch, %metadata, %slots, %step_pred, %prev_tok,"
    " %target_tokens, %step_meta) : (index, index, index, buffer, buffer, buffer, buffer, buffer, buffer, buffer)\n"
)


def transform(text: str) -> str:
    if "%tk_vals" not in text:
        raise SystemExit(
            "run tools/spec/topk.py first: the drafter would pay the 5.5 ms top-k once per draft"
        )
    text = edit_kernel(text, "g_head_select", DRAW_OLD, DRAW_NEW)
    text = edit_kernel(text, "g_dsp_prepare", PREPARE_OLD, PREPARE_NEW)
    text = edit_kernel(text, "g_dsp_step_commit", COMMIT_ARGS_OLD, COMMIT_ARGS_NEW)
    text = edit_kernel(text, "g_dsp_step_commit", COMMIT_LOOP_OLD, COMMIT_LOOP_NEW)
    return replace_once(text, LAUNCH_OLD, LAUNCH_NEW)


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) / "program" / "llm.loom"
    text = transform(path.read_text())
    path.unlink()  # the package may be a hard-link copy: never write through to the original
    path.write_text(text)
