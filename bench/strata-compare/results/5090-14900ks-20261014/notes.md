# RTX 5090 + i9-14900KS, 2026-10-14

## Strata
- Version v0.1.44.1, commit 296b59d2 (branch f-1441 merge). Tree has untracked files only (bench outputs, no source changes).
- Built from source on NixOS with CUDA 13.0 (nixpkgs `cuda-merged-13.0`), `engine/strata` built 2026-10-14.
- Config: `strata-swift-iq3_xxs.json` (the server's config; copied into `system.json`).
- Non-default args against setup's output: `--vram-reserve-mib 832`, `--pcie-frac 0.90`, `--spec-min-p 0.70`, `--max-context 262144`, `--kv int8`, `--vision`. No `env` entries in the config.

## Model
- Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS (2-part GGUF), pack `packs/swift-iq3_xxs`, MTP draft `mtp/rt`.

## Machine
- OS: NixOS 26.11, Linux 6.18.55.
- CPU: Intel Core i9-14900KS (no AVX-512).
- RAM: 62.5 GiB usable (64 GB). Speed not read (dmidecode needs root).
- GPU: RTX 5090 32 GB, driver 610.43.02, power limit 450 W, PCIe gen 5 x16 max.
- Nsight Systems 2025.1.3.140 (from nixpkgs, `cudaPackages.nsight_systems`).

## How it was run
- The engine's RPATH reaches the CUDA stub `libcuda` first on NixOS, so every run failed with `CUDA driver is a stub library` at first.
  Rerun with `LD_LIBRARY_PATH=/run/opengl-driver/lib` (the real driver), which is how the server runs it too. No script or repo change.
- Strata server was stopped for all runs.

## Also running during the runs
- On the GPU: desktop (Wayland compositor bars), Discord (191 MiB, GPU process) and mpv (162 MiB). Not closed.
- CPU: normal desktop session; other load not measured.

## Server startup log (last start before the runs, 17:49:54)
```
[strata] 2026-10-14 17:49:54 engine started: /home/servius/Projects/Strata/engine/strata --serve --pack /volumes/media/strata/packs/swift-iq3_xxs --native /volumes/media/strata/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --ple-gguf /volumes/media/strata/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf --expert-profile /home/servius/Projects/Strata/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --mtp /volumes/media/strata/mtp/rt --max-context 262144 --kv int8 --vision --vram-reserve-mib 832 --pcie-frac 0.90 --spec-min-p 0.70
strata generate: memory guard: limit 62.5 GiB (source: meminfo), current 7.2 GiB, reclaimable cache credited 0.0 GiB; available 55.3 GiB (MemAvailable 55.3 GiB), floor 2560 MiB
strata generate: this CPU has no AVX-512: the expert kernels run on AVX-2 (multi-token for the i-quant gate/up rows)
strata generate: native pack: /volumes/media/strata/packs/swift-iq3_xxs experts (largest blob 2.33 MB), token embedding IQ3_S in mapped host memory (260 MiB, 0.3 s)
strata generate: 1466 MiB of weights loaded from /volumes/media/strata/packs/swift-iq3_xxs in 0.5 s (302 canonical tensors skipped: served natively)
strata generate: 300 native projection matrices, 1781.11 MiB of weights, in 0.6 s
strata generate: PLE direct read probe: 69084 rows/s (fine)
strata generate: the SSD is kept awake while rows are read: one page of the table after 100 ms without a read, until 60 s after the last request (STRATA_SSD_KEEPALIVE=0 turns it off)
strata generate: profile /home/servius/Projects/Strata/data/expert-profile.bin: 24576 ranked pairs, built for 24576 slots
strata generate: PLE on, table 320001536 rows of /volumes/media/strata/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
strata mtp: draft layer loaded, 1098 MiB of VRAM (experts 675, dense 111), files read in 0.29 s (2663 MiB/s)
strata generate: experimental native Q5_K head, 437043200 bytes, in 0.1 s
strata generate: GPU 0: NVIDIA GeForce RTX 5090, compute capability 12.0
strata generate: expert arena read through the file cache (buffered (not Windows))
strata generate: expert arena: cudaHostRegister PORTABLE ok; MAP_HUGETLB unavailable (needed 20464 2 MiB pages, vm.nr_hugepages=0); transparent huge pages requested (MADV_HUGEPAGE)
strata generate: loaded 39.97 GiB at 4.03 GiB/s
strata generate: GPU host-callback thread started before host-core pinning
strata generate: CPU pool tasks/phase: 72 (automatic), participating threads: 24
strata generate: 23 pool workers on logical processors 2,4,6,8,10,12,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, the host thread on 0 (draining too) (--host-core first)
strata aux cpus: threads that are neither pool workers nor the host go to CPUs 1 (off at start; a request's aux_cpus=1 turns it on)
strata generate: expert cache auto: 16.88 GiB free, 832 MiB reserved (+184 MiB for the draft head) -> 7321 slots
strata generate: expert cache 9801 slots, 15.88 GiB of VRAM; policy is
strata generate: the GPU computes the experts in the cache; it rounds differently from the CPU,
                 so a reply can differ slightly from a run without the cache (same quality:
                 bench/results/2026-09-27-cache-parity).
                 PROFILE, ranked by routing frequency, no eviction.
strata generate: pre-filled 9801 of 9801 slots from the profile in 0.4 s (38698 MB/s); slot 0 verified
strata generate: R4 hit path ON - resident experts are computed on the GPU
strata generate: 23 expert-pool workers + the host thread
strata generate: session is up (engine 0.1.44.1)
```
