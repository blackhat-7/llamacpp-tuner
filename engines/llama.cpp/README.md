# llama.cpp named builds

`engines/llama.cpp/build.sh NAME [--force]` recreates one build at the path its profiles use. No patches of ours.

| NAME | Output | Pin | Why a profile needs it |
|---|---|---|---|
| `b12415-vk` | `$LCT_HOME/llama.cpp-b12415-vk/llama-server` | release b12415 Ubuntu Vulkan x64 (commit `4e5116f47`, asset sha256 checked) | `embed` (CPU only) and `ornith-vk`: a fixed Vulkan build while lct's own build is missing |
| `h2o` | `$LCT_HOME/llama.cpp-h2o/build-hip/bin/llama-server` | `pseudotensor/llama.cpp` `6245df86208e` (PR #30361's branch merged with master), web UI b11713 | `h2o-lightning`: H2O-Lightning-4B needs the PR's model code |
| `hip` | `$LCT_HOME/llama.cpp/build-hip/bin/llama-server` | `ggml-org/llama.cpp` `17197474510622` | `ornith`: HIP on the XTX |

- **`hip` pin:** the binary the profile runs was built on 2026-09-21 from `17197474` (its `build-info.cpp` and
  libraries say so). The managed checkout `$LCT_HOME/llama.cpp` was later moved to `aa94f2086`, so its HEAD is not
  the build's commit.
- **Shared checkout:** `hip` builds in lct's managed checkout (`lct setup` uses its `build/`); the script only owns
  `build-hip/`. It clones (shallow) when the checkout is missing, and moves an existing checkout to the pin only with
  `--force`.
- **Toolchain:** ROCm 7.2.4 in `/opt/rocm` (`ROCM_PATH`), cmake, a C++ compiler, `HIP_ARCH` (default `gfx1100`).
  Flags: `-DGGML_HIP=ON -DGPU_TARGETS=$HIP_ARCH`, Release, target `llama-server` only (`JOBS`, default 8).
- **Up to date:** a rerun compares the output's commit (`llama-server --version` for the release, `build-info.cpp`
  for HIP) with the pin and stops. A different output is replaced only with `--force`.
- **Check a build:** `CONFIGURE_ONLY=1 build.sh h2o` stops after cmake's configure. A full `h2o` compile of
  `llama-server` took 4 minutes with `nice -n 10`, 8 jobs.
