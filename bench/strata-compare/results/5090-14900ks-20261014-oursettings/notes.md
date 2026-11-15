# RTX 5090 + i9-14900KS, 2026-10-14, with the shared settings

Same machine, build (v0.1.44.1, 296b59d2) and model as `../5090-14900ks-20261014/` (see its `notes.md`), same
`LD_LIBRARY_PATH=/run/opengl-driver/lib`, Strata server stopped. Nsight Systems not on PATH, so no `nsys_code` run.

Config: the server's config with the changes from `share/README.md`:
- `--expert-profile share/swift-1.5-iq3_xxs.bin` (was `data/expert-profile.bin`)
- `--mtp-draft-vocab share/draft_vocab_agent.bin` (new)
- `--prefill auto:16384` (was `auto`)
- `--mtp-window 8192` (new)
Everything else as before, including `--pcie-frac 0.90`.

Decode tok/s, before (first run / rerun) -> with these settings:
- speed_code   116.2 / 122.5 -> 168.2  (+37 to +45%)
- speed_edit    81.2 /  85.2 -> 113.4  (+33 to +40%)
- speed_prose  138.8 / 138.1 -> 114.1  (-17%)
- prof_code    117.7 / 118.9 -> 160.4
- prof_prose   125.8 / 132.3 -> 104.7
- noskip_code  126.0 / 125.4 -> 168.0
- noskip_prose 139.1 / 140.0 -> 107.9
Prefill tok/s: prompt_15k 4372 / 4580 -> 4076, prompt_35k 5051 / 5048 -> 4820.

Why prose is slower: the drafts are unchanged (0.749 accepted against 0.748), but the profile keeps fewer prose
experts in VRAM. speed_prose: expert-cache hits 95.0% -> 93.2%, experts read over PCIe 1.31 -> 2.34 per layer,
wait for rings 8.5 -> 10.7 ms per round. Code goes the other way: hits 90.3% -> 96.4%, PCIe 5.5 -> 2.1 per layer,
wait 16.7 -> 11.1 ms. The draft vocabulary cut drafting from 0.84 to 0.75 ms per round on prose and from 1.13 to 0.97
on code. Each setting was not measured on its own: the four changes were run together.
