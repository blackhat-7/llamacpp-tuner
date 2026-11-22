#!/bin/bash
# Draft-head A/B for the coordinator's build: registration check, then short code/prose/edit, bis-2 vs bis-2 + draft head
# (preset r3: depth 3, chunk 1024). Everything waits on notes/gpu-run (hold file + lock).
L=/home/illusion/Documents/projects/llamacpp-tuner/tmp
G=$L/loom/notes/gpu-run
while [[ -e /tmp/xtx-gpu.hold ]]; do sleep 20; done  # regcheck.sh takes the lock itself
$L/loom/notes/regcheck.sh $L/loom/spec-bis2-dh r3 | tee /dev/stderr | grep -q ": ready" || exit 1
cd /home/illusion/Documents/projects/llamacpp-tuner-inkling
export LCT_HOME=$L
for pkg in bis-2 spec-bis2-dh bis-2 spec-bis2-dh; do
  $G $L/loom/venv/bin/python -I experiments/inkling-loom/session/session.py --model $L/loom/$pkg --preset r3 \
    --jobs $L/experiments/inkling-loom/opt/jobs_base.json --out $L/experiments/inkling-loom/spec/dh.jsonl \
    --label dh-$pkg --port 8098 --serve-arg=--pool_backing=fixed
done
python3 - <<'PY'
import json
for l in open("/home/illusion/Documents/projects/llamacpp-tuner/tmp/experiments/inkling-loom/spec/dh.jsonl"):
    d = json.loads(l)
    if d["name"] == "warm": continue
    md = d.get("metrics_delta", {})
    p = sum(v for k, v in md.items() if "draft_proposed" in k); a = sum(v for k, v in md.items() if "draft_accepted" in k)
    print(f"{d['label']:18s} {d['name']:12s} {d['decode_tps']:6.1f} t/s  accepted/proposed {a:.0f}/{p:.0f} = {a/p if p else 0:.3f}")
PY
