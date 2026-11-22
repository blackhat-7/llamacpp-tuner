# Loom: the Inkling-2-Mini package

`build.sh` makes `$LCT_HOME/loom/kv8-r2`, the package that profile `inkling` serves (preset `xtx-r2`). The previous
build, `kv8w4096` (changes 1-5 only, preset `xtx-kv8-196k`), is the rollback: check out commit `d8fe34f` and build.
Numbers below are from `docs/inkling-loom-research.md` (7900 XTX, decode t/s).

## Upstream

- Engine: `hrx-loom` 0.2.0 from PyPI, with the `hf` extra (source ROCm/hrx-system, Apache-2.0), Python 3.12.
  `build.sh` installs it in `$LCT_HOME/loom/venv`. lct runs `loom-serve` from that venv.
- Package: `loom-fun/unsloth-Inkling-2-Mini-loom` at revision `277f099b5003b725e099d48caa1a2c0db1f9b650`.
  `build.sh` fetches it to `$LCT_HOME/loom/Inkling-2-Mini` (about 18 GB).
- The package card pins the weights: `unsloth/Inkling-2-Mini-GGUF` at revision
  `ea0c79b2ecd73669e774c8519b79bfabfddbc027` (UD-Q4_K_XL and the DSpark Q8_0 drafter). Its `model.json.lock` holds
  their sha256, and `loom-fetch` checks them.

## Our changes

1. 8-bit global KV cache: `tools/transform.py` and `patches/storage-kv8.diff`. Int8 codes plus one f16 scale per
   32 dims, 1,088 bytes a position instead of 2,048. At 195K with the drafter off: 44 to 73 t/s. It frees the
   memory for the drafter and the prefix cache at 196K.
2. Decode attention kernel v2: `kernels/g_ink_attn_split_v2_kv8.loom` (from
   `FMT=kv8 U=2 experiments/inkling-loom/gen_v2.py 8`), launched by `transform.py`. 541 to 855 GB/s at 195K (f16);
   64K end to end 70 to 77.6 t/s.
3. Verify spans through v2: `transform.py`. 64K code with the drafter on: 73 to 102 t/s.
4. Drafter window of 4,096 tokens: `tools/dsp_window.py`. Drafter attention at 195K: 13.45 to 0.19 ms a pass.
   64K with the drafter on: 98.5 / 86.4 to 114 / 104 t/s (code / prose). Output stays exact.
5. Presets: `presets.json`. `xtx-kv8-196k` (196,608 context, drafter, prefix cache, `max_tokens` 32,768) is the one
   profile `inkling` used until round 2. `xtx-nodraft` is the same without the drafter. `xtx` (32K) is the older f16
   preset.

Round 2 (2026-11-22; same-session numbers vs `kv8w4096` in `docs/inkling-loom-research.md` "Round 2"). Each tool
edits the package in place; `build.sh` runs them in this order, because each anchors on the text the ones before
it leave:

6. `tools/attn/attn.py`: one SWA kernel for single tokens and verify spans (each lane loads a key once for all
   3 query heads), a parallel combine, and fewer global splits at short context. Short decode +5 to +12%.
7. `tools/prefill/prefill.py`: `prepare.loom` builds drafter proposal stages only up to `rows` spans (the others
   reserved ~1.5 GiB and kept the prompt chunk at 512), and split-K global flash attention for prompt chunks.
   With chunk 1,024: 64K read 28.8 to 22.0 s, 195K 138 to 100 s.
8. `tools/dense/dense_edit.py`: one launch for all projections of a layer visit, RMS norm fused with q8
   quantization, fused loop controllers and mixes. Bit-identical output; short decode +7 to +8%.
   `--sconv` (fused short-conv kernels) passes its kernel tests but is not on: no end-to-end run yet.
9. `tools/moe/moe_opt.py`: one-wave router, 2-row gate/up, and padding tokens of a verify shape skip the experts.
   Bit-identical for live tokens; ~1% (more once odd span lengths are used).
10. `tools/spec/dynspan.py --lengths 2 --thresholds 90 --shift 3`: with draft depth 3, a step verifies 2 tokens
    instead of 4 while the recent first-draft miss rate is high (prose), else the full span (code, edits).
11. Preset `xtx-r2`: `xtx-kv8-196k` with draft depth 3 and prompt chunk 1,024.

All together, short code / prose / edit 141 / 121 / 133 to 193 / 146 / 176 t/s; 64K code / prose 114 / 104 to
148 / 108; 195K 90 / 75 to 104 / 82; 195K recall 3/3.

`tools/split_kernel.py` writes an 8-bit version of the old decode kernel. `transform.py` keeps it in the program,
but v2 takes over all its launches.

## Toolchain

bash, `uv`, GNU `patch` and `diff`. The tools use only the Python standard library (`tools/dense` imports its
kernel generators from its own directory); `build.sh` runs them with the
venv's Python. The network is needed only for the first venv and the first fetch.

## Build and check

```bash
engines/loom/build.sh                       # LCT_HOME defaults to <repo>/tmp
engines/loom/build.sh --output DIR --force  # another place; replace an output that differs
```

- The `.gguf` files are hard links to the upstream package, so the output must be on the same file system.
- A second run with no change prints "up to date". An existing output that differs needs `--force`.
- Check without a GPU: `diff -r` against a known-good package, and compile a changed kernel:
  `$LCT_HOME/loom/venv/bin/loom-compile OUT/program/llm.loom --target=amdgpu:gfx1100 --root=g_ink_attn_split_v2 --output=/dev/null`.
- Check on the GPU: `lct serve --backend loom` and one short request. A program change that compiles can still
  fail registration ("stage does not match the token runner bindings"): load it once before any long benchmark
  (`experiments/inkling-loom/r2/window/regcheck.sh PKG PRESET`).

## Adding the next change (for example span-v2)

1. Develop and test it in `experiments/inkling-loom/` (`gen_span.py`, `gen_spanbench.py`; round 2's in `r2/`).
2. Put the kernel in `kernels/` or next to its tool. Add a tool in `tools/` and call it in `build.sh` after the others.
3. Give the output a new name (for example `kv8-r3`), so the old package stays for rollback.
   Point the profile at it.
4. Add one line above: what it does, the measured effect, the research doc.
