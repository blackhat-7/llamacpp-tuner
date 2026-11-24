# Several agents taking turns on one Loom server (2026-11-24)

Checks for lct's chat parking (`llamacpp_tuner/parking.py`, preset `xtx-park`). Results and gotchas:
`docs/inkling-loom-research.md`, "Several agents taking turns". Run each under the GPU lock against a test server,
for example `lct serve $LCT_HOME/loom/kv8-r3v --backend loom --port 6890 --extra-args '--preset=xtx-park
--device=amdgpu --pool_backing=elastic --weight_slab_bytes=32mib'` (an absolute package path).

- `agents.py BASE --mode alone|turns|concurrent [--long] --out FILE`: 4 agents (20K-151K) through `/v1/messages`,
  greedy; every mode must give the same text per agent and turn. `--long` adds ~5K-token follow-ups, so each
  restore reads many prompt chunks.
- `suffix3.py LOOM_URL VARIANT [CHARS]`: straight to Loom. A 30K chat, a parking step (`eager-live`,
  `eager-rewind`, `eager-fork`, `pin-live`, `release-live`), then a follow-up of CHARS characters, scored against a
  fresh read. With prompt chunk 1,024, `eager-rewind 6000` gives symbols; with 512 every variant matches.
- `worst.py BASE FRACTION`: near the 204,800 limit: a ~190K chat parked and restored, an image at full depth, a
  request over the limit.
- `claude.sh [N]`: N real `claude-local --offline -p` sessions at once on :6890, read-only tools.
- `capture.py OUTDIR`: a proxy on :6891 that saves each `/v1/messages` request and reply, to check replies for
  garbage or compare what a client sends back with what the model wrote.
