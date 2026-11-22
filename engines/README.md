# Engines

Each folder rebuilds one engine with every change we adopted. The pattern and the rules are in
`docs/architecture.md` ("Engines: the pattern"). Outputs go under `$LCT_HOME` (`tmp/` in a checkout), where
`profiles.toml` points with `${LCT_HOME}/...`.

| Folder | Builds | Used by profiles |
|---|---|---|
| `strata/` | Strata v0.2.9 + our patches: the HIP engine, the CUDA helper for a second GPU, run configs, model packs | `strata-swift-flash-next*` (stacks `local-flash*`) |
| `loom/` | the Inkling-2-Mini Loom package with 8-bit KV, attention kernel v2 and a 4,096-token drafter window | `inkling` (stack `local-inkling`) |
| `llama.cpp/` | named llama.cpp builds: a pinned Vulkan release, PR and HIP builds | `embed`, `ornith-vk`, `ornith`, `h2o-lightning` |
| `rocr/` | ROCm's runtime with our fix for a copy-engine over-read | optional, for Strata and HIP builds |

`lct setup --backend strata` and `lct setup --backend loom` run the matching `build.sh`. The others run directly,
for example `engines/llama.cpp/build.sh b12415-vk`. Each folder's `README.md` lists its pin, its changes and how
to check a build.
