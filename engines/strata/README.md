# Strata: our engine

This folder rebuilds the Strata engine that the `strata-swift-*` profiles serve. It holds the upstream pin, our
changes as patches, the build script, the run configs as templates, and the tools that make the model packs.

## Upstream and pin

- Upstream: <https://github.com/Niko1221/Strata.git>, tag `v0.2.9` = commit `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`.
- llama.cpp (ggml for the engine, mtmd for the image reader): commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
  This is upstream's own pin (`setup.py` `LLAMA_CPP_COMMIT`, `CMakeLists.txt`).
- Our changes: the 33 patches in `patches/`, made with `git format-patch` from branch `opt13/psum` (`b62f6577`).
  Applied to `v0.2.9`, they give source tree `868a1763b3da3d2ee7fc88d2aa1ac39f18c27e30`.

## Our changes

All are for the RX 7900 XTX (gfx1100) and the RTX 4070 helper. Numbers come from the research docs:
[flash-next](../../docs/flash-next-speed-research.md) (one GPU) and [dual-gpu](../../docs/dual-gpu-research.md) (two).
"Bit-exact" means the same tokens as without the change.

Decode kernels and tools on gfx1100, mostly ported from our 0.1.43 engine (patches 1-10; together decode equals the 0.1.43 engine and
prompt reads are 2-9% faster than stock v0.2.9, which decodes 20-26% slower):

1. Hyper-connection read: pipelined staged down (`STRATA_GR_PIPE`) and one kernel (`STRATA_GR_ONE`). Read 1.16-1.40x
   faster, decode +1.5-2.5%, bit-exact.
2. Router and launch cuts: DPP top-10, router tail, GDN a/b ILP, the gfx1100 defaults table, fewer graph nodes a
   window. 4-token window 1,674 -> 1,193 nodes, bit-exact.
3. Router fix: build the warp-32 argmax kernels on wave32 HIP.
4. i-quant signs and IQ4 table lookups with `v_perm`. Expert kernels -12 to -16%, bit-exact.
5. Dense GEMVs: a row's loads first at 1-4 tokens (g11), four rows a wave at 1-6 tokens (R4). Edit +6%, bit-exact.
6. MTP drafting: cumulative draft stop, draft-head context rows, a leaner draft kernel, no residual copy, fused final
   mixer. Draft 1.92 -> 1.78 ms a round.
7. QSA decode attention: upstream PR #540's chunk kernel and the bounded merge. Attention 0.59 -> 0.37 ms a window,
   code +3.4%, bit-exact.
8. CPU expert pool: UMWAIT parking (`STRATA_UMWAIT`) and guided task sizes (`STRATA_POOL_GUIDED`).
9. QSA prompt attention v2 on gfx11 (`STRATA_PA_V2`). 603 -> 466 ms per 15K prompt, bit-exact.
10. Tools: `STRATA_BENCH_COPIES` in `native_expert_bench`, `tools/mtp_q2b.py`.

Round 11 (patches 11-22):

11. n-gram (PLE) table: `MADV_WILLNEED` prefetch of a window's rows (`--ple-io mmap`).
12. `docs/AMD_HIP.md`: which gfx1100 switches are default and which the run config sets.
13. Decode block selection: register top-k by rows (`STRATA_TOPK_ROWS`), DPP16 scorer (`STRATA_SCORES_DPP`).
14. `docs/AMD_HIP.md`: the two switches above.
15. Conversation cache disk tier (`--conversation-cache-disk-mib`, `--conversation-cache-disk-dir`). A 200K chat
    restores in 3.9 s instead of 70 s, token-identical.
16. Disk tier cleanups.
17. `STRATA_HOST_LOG`: per-window host time and faults (measurement only).
18. `arch_defaults_test` follows this branch's gfx1100 table.
19. `STRATA_QSA_NR3`: a QSA layer's three norm+ropes in one launch.
20. `STRATA_Z_CONVAB`: a GDN layer's conv/ab launch inside its z GEMV.
21. `STRATA_QSA_KV_ROLE`: a QSA layer's k and v GEMVs inside its q GEMV.
22. `STRATA_WAIT_PF`: GPU waits read the next layer's dense weights into the Infinity Cache. With 19-21: code
    160.6 -> 165.9, prose 115.0 -> 122.4, edit 138.5 -> 144.7 t/s, bit-exact.

The RTX 4070 as expert helper (rounds 12-13, patches 23-33):

23. NVIDIA helper: a CUDA plugin (`libstrata_nvhelper.so`, `STRATA_NV_HELPER`) serves `--expert-cache-device1` for
    the HIP engine. Adaptive tiers on: code +4%, prose +3%, edit +26%.
24. `STRATA_NV_PRE_DRAIN=0` switch; the arena pin check samples each layer's ends.
25. `STRATA_NV_RESERVE_MIB` (default 1536) stays free on the 4070.
26. Tail skip beside helper GPUs (`STRATA_ROUTE_TAIL_SKIP=7`); per-layer CPU expert counts. Code +3.3%, prose +2.8%.
27. `STRATA_NV_EXPDEC=1` turns on EXPDEC in the plugin only.
28. Helper review fixes.
29. `STRATA_ADAPT_PROTECT` (protected core); generate mode honours `--adapt-min-gain`; causal-profiling probes. With
    `--adapt-swaps 384 --adapt-min-gain 3`: edit +24%, code and prose flat.
30. Server logs each request's n-gram row gather time.
31. `STRATA_NV_EARLY`: launch the helper at its claim (default on).
32. `STRATA_NV_EARLY` back to opt-in: the server A/B showed no gain.
33. `STRATA_NV_PSUM`: the helper returns one weighted sum per token. Server code +1.2%, edit +1.1%, rounding-level.

## Toolchain (as built here)

- ROCm 7.2.4 in `/opt/rocm` (HIP clang from `/opt/rocm/llvm`). Host C++: GCC 16.2.
- CUDA 13.4 (nvcc V13.4.92) with the system `/usr/bin/g++` 16.2.1. A Nix gcc cannot find `libcuda.so.1`.
- CMake 4.5 (system; the venv's 4.4.3 fills in), Ninja 1.13.2 from the venv (`/usr/bin/ninja` is missing here).
- Python 3.14 for the `.venv`; packages from upstream's `requirements.txt` and `requirements-build.txt`.

## Build

```bash
engines/strata/build.sh            # NVCC=~/cuda-13.4/opt/cuda/bin/nvcc when nvcc is not on PATH
```

- Source checkout: `$STRATA_SRC` (default `$LCT_HOME/strata/src`; `$LCT_HOME` defaults to `<repo>/tmp`). The script
  clones upstream if it is missing and refuses a checkout with uncommitted changes. Branch `lct` is `v0.2.9` plus
  `patches/`. The script recreates it only when the patches give a different tree, and prints the old commit.
- Outputs, in `$LCT_HOME/strata/data/engines/` (`$STRATA_DATA`), named by the first 12 hex of the source tree:
  `strata-0.2.9-gfx1100-lct-<id>` (HIP engine), `strata-vision-lct-<id>` (image reader, CPU) and
  `libstrata_nvhelper-lct-<id>.so` (only when nvcc is found). It also copies the two small seed files (below).
- Build dirs `build-lct-hip`, `build-lct-vision`, `build-lct-cuda` stay in the checkout, so a rerun builds nothing.
- Options: `HIP_ARCH` (gfx1100), `CUDA_ARCH` (89-real), `CUDAHOSTCXX` / `CUDAHOSTCC` (`/usr/bin/g++` / `gcc`),
  `ROCM_PATH`, `JOBS`, `PYTHON`. `--force` replaces outputs that differ; without it the script stops.
- Last step: `render.py` writes the configs to `$LCT_HOME/strata/configs/`, with `"cwd"` = the checkout. It keeps a
  file that differs (`--force` replaces it) and always keeps a symlink (never writing through it), and says so.
  Set `ALLOWED_HOST` (default `localhost`; this machine uses `pc`).
- The ROCr fix in `lib_dirs` is `$LCT_HOME/rocr-7.2.4-fix/lib`, made by `engines/rocr/`.
- Check a build: `strata-0.2.9-gfx1100-lct-<id> --help` prints the options without a GPU.
- First build here (8 jobs, nice 10): ~12.5 min (HIP engine and image reader ~9 min, CUDA plugin ~4 min). A rerun
  takes 0.4 s. Disk: ~1.1 GB (checkout 136 MB, venv 203 MB, llama.cpp 211 MB, build dirs 325 MB).

## Run configs

`configs/` holds the live configs with paths as `${LCT_HOME}`, `${STRATA_SRC}`, `${STRATA_DATA}`, `${ENGINE}`,
`${VISION}`, `${NVHELPER}`, `${ROCM_PATH}` and `${ALLOWED_HOST}`. All of them now point at this build. Before, the
`-iq3_xxs-dual`, `-mixd-dual` and `-mixb-base13` configs ran engine `ac085da1` (patch 28) and `-iq3_xxs-tuned` ran
`b3dcc3d4` (patch 22); the later patches are opt-in switches, except tail skip, which only the dual configs set.

| Config | Use |
|---|---|
| `strata-swift-mixb-dual.json` | default `local-flash`: mixB experts, XTX + 4070 |
| `strata-swift-mixd-dual.json` | mixD experts, XTX + 4070 |
| `strata-swift-iq3_xxs-dual.json` | the abliterated IQ3_XXS experts, XTX + 4070 |
| `strata-swift-iq3_xxs-tuned.json` | the XTX alone |
| `strata-swift-mixb-base13.json` | bench copy of mixB: frozen profile, own save file and disk dir |

## Model files

Downloads go to `$LCT_HOME/models/`. Packs, profiles and the drafter go to `$STRATA_DATA`.

1. The abliterated Swift 1.5 (Hub revision `43b4c1a8`; shard sha256 `e66b543e…`, `d2196819…`):

   ```bash
   lct pull huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF --no-mmproj \
     --file swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
   ```

2. Swift 1.5 IQ4_XS, the 4-bit experts of mixB, mixD and the hybrid (91 GiB, 3 shards; check each sha256 against
   the Hub's LFS oid; hf_xet stalls on files this big):

   ```bash
   HF_HUB_DISABLE_XET=1 lct pull ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF --quant IQ4_XS --no-mmproj
   ```

3. The image projector (sha256 `cd1140f4…`, upstream's pin `99bb8f7f`):

   ```bash
   uv run hf download ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf \
     --revision 99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d --local-dir "$STRATA_DATA/models"
   ```

Packs. `tools/iq_pack.py` is upstream's (run it with the checkout's `.venv` python). `M=$LCT_HOME/models`,
`P=$STRATA_DATA/packs`, `ABL=$M/huihui-ai%2FHuihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF/swift-IQ3_XXS`,
`IQ4=$M/ukisai%2FSwift-1.5-Qwen3.8-Flash-Next-GGUF/IQ4_XS`, `L=1,3,5-27,33,36,40,41,45` (the 30 layers whose `down`
is Q2_0).

| Artifact | How it is made |
|---|---|
| `packs/swift-abl-iq3_xxs` | `iq_pack.py --gguf $ABL/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --out $P/swift-abl-iq3_xxs` (made 2026-10-19 with Strata 0.1.43's tool; not rerun with v0.2.9) |
| IQ4_XS pack (scratch) | `iq_pack.py --gguf $IQ4/Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf --compat-bf16 --out $P/swift-iq4_xs-ukisai` |
| `packs/swift-abl-hybrid-iq4_xs` + `models/hybrid-swift-abl-iq4_xs/` | `tools/make_hybrid.py $P/swift-abl-iq3_xxs $P/swift-iq4_xs-ukisai $IQ4/Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf $ABL/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf $P/swift-abl-hybrid-iq4_xs $M/hybrid-swift-abl-iq4_xs` |
| `packs/swift-abl-mixD` | `tools/mix_down.py $P/swift-abl-iq3_xxs $P/swift-abl-hybrid-iq4_xs $P/swift-abl-mixD $L` |
| `packs/swift-abl-mixB` | `tools/mix_down.py $P/swift-abl-iq3_xxs $P/swift-abl-hybrid-iq4_xs $P/swift-abl-mixB "" $L` |

The last three were checked: they rebuild the live `native_experts.txt` byte for byte.

Profiles and vocabulary:

- `profiles/swift-1.5-iq3_xxs.bin` (Swift expert profile) and `draft_vocab_agent.bin` (agent draft vocabulary): in
  git under `bench/strata-compare/share/`; `build.sh` copies them. Made with upstream's `tools/make_profile_mix.py`
  and `tools/draft_vocab.py` from routing traces and transcripts that were never kept.
- `profiles/swift-1.5-{iq3_xxs,mixb,mixd}-learned.bin`: the engine writes them (`expert_profile_save`). Start each
  as a copy of `swift-1.5-iq3_xxs.bin`.
- `profiles/swift-1.5-mixb-learned-frozen13.bin`: a 2026-11-17 snapshot of the mixB learned profile, for the bench
  config: in git under `bench/strata-compare/share/`; `build.sh` copies it.
- `mtp/rt-sd2-swift` (787 MB drafter): Basalt's MTP drafter with its dense part refit to Swift by self-distillation
  (round 9, `docs/flash-next-speed-research.md`; the fit scripts are in `experiments/flash-next/mtpfit/`).
  <!-- drafter-source -->Download: `uv run hf download illusion7/strata-flash-next-mtp-sd2 dense.bin dense.txt draft_vocab.bin experts.bin --revision 285266844f78118e26ba1dd489639ad323510d05 --local-dir "$LCT_HOME/strata/data/mtp/rt-sd2-swift"`.<!-- /drafter-source -->

## Make a new change

1. Build once, so `$STRATA_SRC` exists.
2. In the checkout: `git switch -c my-change lct`, edit, commit. Build and test in your own build dir with the
   flags from `build.sh`.
3. Measure. Record the numbers in a `docs/` research file.
4. Adopt: `engines/strata/export.sh my-change` rewrites `patches/`. Run `build.sh`: it rebuilds `lct` from the new
   patches and installs the new tree's binaries beside the old ones. Point the profile's config at them.
