# Sampling at temperature > 0 (2026-11-24)

Why temperature 1 decoded ~40% slower than greedy, and what was tried. Adopted: `engines/loom/tools/spec/topk.py`
(in `engines/loom/build.sh`, package `kv8-r3`): `g_head_select` found its 64 candidates with 64 passes over the
200,058 logits (5.5 ms a call); it now ranks the 64 best chunks of 128 ids by their maxima, then only their ids
(0.125 ms). Same candidates in the same order, so the same draws. T=1 decode +28-45% (short code 131.6 -> 180.8,
short prose 89.7 -> 130.4, 66K code 106.1 -> 135.8, 66K prose 77.8 -> 104.0 t/s).

Not adopted: `gumbel.py` (the drafter proposes with the sampler's own per-position Gumbel noise). Exact, but the
DSpark drafter's distribution matches the target too loosely: acceptance +6-21%, decode -0.6% to +2.1%.

- `coupling_sim.py`: CPU check that coupled drafting equals plain sampling (and its negative controls).
- `gen_select_test.py OUT NEW_LLM OLD_LLM`, then `kernel_tests.sh OUT`: GPU chi-square of the new draw against
  softmax over the kept set, draws equal to the old kernel, negative controls, and kernel timing.
