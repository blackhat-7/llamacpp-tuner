# ROCr 7.2.4 with the BlitSdma fix

ROCm's runtime (`libhsa-runtime64`) from `ROCm/rocm-systems` tag `rocm-7.2.4` (commit `97f5574fe2fd`), the
version Arch ships as `hsa-rocr 7.2.4-1`, plus one change.

- **The bug:** `BlitSdma::PendingBytes()` reads one slot past `bytes_written_` when the copy engine's read pointer
  sits in the ring's last 7 bytes. Arch builds with `-D_GLIBCXX_ASSERTIONS`, so the over-read aborts any program
  doing millions of async copies (Strata: about 1 crash per 90 requests on the 7900 XTX).
- **The fix:** `patches/rocr-7.2.4-blit-sdma-oob.patch`, one spare slot. Cause, backtrace and upstream issue draft:
  `ISSUE.md`.
- **Reproducer:** `tools/sdma_stress2.cpp` (`hipcc -O2 tools/sdma_stress2.cpp -o s && ./s 60000000`). Stock 7.2.4
  aborts after about 8-16 M copies; the fixed library ran 40 M clean. It loads the GPU for minutes: run it only when
  asked.

## Build

`engines/rocr/build.sh [--force]` fetches the pinned source (sparse, about 20 MB) into `$LCT_HOME/rocr-7.2.4-fix.src`,
applies the patch, builds with Arch's flags and installs to `$LCT_HOME/rocr-7.2.4-fix/`. Needs cmake, ninja and ROCm
7.2.4 in `/opt/rocm` (`ROCM_PATH`). It never replaces an existing install without `--force`.

## Use

Put its `lib` first; the rest of ROCm still comes from `/opt/rocm`:

- a program: `LD_LIBRARY_PATH=$LCT_HOME/rocr-7.2.4-fix/lib:/opt/rocm/lib`
- an lct profile: `--env LD_LIBRARY_PATH=${LCT_HOME}/rocr-7.2.4-fix/lib:/opt/rocm/lib`
- a Strata run config: add `$LCT_HOME/rocr-7.2.4-fix/lib` to `lib_dirs`

Only for ROCm 7.2.4. After a ROCm upgrade, check whether upstream fixed it before rebuilding against the new tag.
