#!/bin/bash
# GPU window, phase B: baseline vs the combined package int-r3, short and long. Results: opt/window-b2.jsonl.
T=/home/illusion/Documents/projects/llamacpp-tuner/tmp; G=$T/loom/notes/gpu-run; O=$T/experiments/inkling-loom/opt
export LCT_HOME=$T; cd /home/illusion/Documents/projects/llamacpp-tuner-inkling
exec > $T/loom/notes/window-b2.log 2>&1
S="experiments/inkling-loom/session/session.py --port 8097 --serve-arg=--pool_backing=fixed --out $O/window-b2.jsonl"
run() { echo "== $3 $(date +%T)"; timeout ${TO:-400} $G $T/loom/venv/bin/python -I $S --model $T/loom/$1 --preset $2 --jobs $4 --label $3 | cut -c1-220; for p in $(pgrep -f "listen=127.0.0.1:[8]097"); do kill $p; done; sleep 3; }
#run kv8w4096 xtx-kv8-196k base-short $O/jobs_base.json
run bis-2 r3 b2-short $O/jobs_base.json
run bis-2 r3-d1 b2d1-short $O/jobs_base.json
TO=1500 run bis-2 r3 b2-long $O/jobs_long.json
TO=1500 run kv8w4096 xtx-kv8-196k base-long $O/jobs_long.json
journalctl -k --since "-40min" --no-pager | grep -i amdgpu | grep -i "fault\|timeout\|reset" | tail -3
echo "== done $(date +%T)"
