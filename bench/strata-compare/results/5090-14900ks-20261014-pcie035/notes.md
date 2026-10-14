# RTX 5090 + i9-14900KS, 2026-10-14, `--pcie-frac 0.35`

Same as `../5090-14900ks-20261014/` (see its `notes.md`) with one change: `--pcie-frac 0.35` (the 7900 XTX config's
value) instead of this machine's `0.90`. Only the three speed runs. Run right after the rerun, same session.

Decode tok/s, 0.35 against 0.90 (first run / rerun): speed_code 98.8 vs 116.2 / 122.5, speed_edit 67.9 vs 81.2 / 85.2,
speed_prose 116.1 vs 138.8 / 138.1. On this machine 0.90 is faster on all three: with 0.35 more missed experts go to
the CPU pool (edit: pool 62.7 ms per round against 33.2 with 0.90).
