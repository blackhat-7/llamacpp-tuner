# Our Strata settings to try on the RTX 5090

Our engine patches are for AMD gfx1100 only and sit on Strata 0.1.43, so they do nothing on an RTX 5090 running
0.1.44.1. These files and flags do not need a rebuild. Measure them with the same `run.py` before keeping them.

## The files

- `swift-1.5-iq3_xxs.bin`: an expert profile built from Swift 1.5 routing traces. The shipped
  `data/expert-profile.bin` ranks the base model's experts. On our RX 7900 XTX, same inputs and cache size:
  code 92 -> 131 t/s, edit 96 -> 117, prose 85 -> 92. The gain is smaller with `--pcie-frac 0.90`, since most misses
  go over PCIe there, but fewer misses still means less PCIe traffic.
- `draft_vocab_agent.bin`: a smaller draft-head vocabulary (46K rows instead of 106K), for English and agent text.
  Drafts only, so the output is the same. On ours the draft took 2.8 -> 2.46 ms a round (+3% decode).

## For the agent on the RTX 5090 machine

1. Copy the server's Strata config to a new file and change its `args`:
   - `--expert-profile` -> the full path of `share/swift-1.5-iq3_xxs.bin`
   - add `--mtp-draft-vocab` with the full path of `share/draft_vocab_agent.bin`
   - add `--prefill auto:16384` (replace `--prefill auto`): bigger prompt chunks
   - add `--mtp-window 8192`: the draft layer reads the last 8K cells only (long contexts)
2. Stop the Strata server, then run the benchmark on the new config:

   ```
   python3 bench/strata-compare/run.py --config /path/to/the-new.json \
     --out bench/strata-compare/results/5090-14900ks-<yyyymmdd>-oursettings
   ```

3. Optional: the CPU expert pool read only ~18 GB/s with 23 workers (ours does 42 GB/s with 13). Try
   `--pool-workers 15` in the new config, then rerun with
   `--only speed_code,speed_edit,speed_prose --out ...-oursettings-w15`.
4. Commit the results folders and push to this branch. Keep the server's own config unless the numbers say otherwise.

To keep what the expert cache learns across restarts, add `"expert_profile_save": "expert-profile-learned.bin"` to the
server's config (upstream feature since 0.1.36).
