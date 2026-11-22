#!/bin/bash
# usage: e2e.sh OUT JOBS "PKG PRESET LABEL" ...   (each run takes the GPU lock separately)
out=$(realpath -m "$1")
cd /home/illusion/Documents/projects/llamacpp-tuner-inkling
export LCT_HOME=/home/illusion/Documents/projects/llamacpp-tuner/tmp
jobs=$2; shift 2
for r in "$@"; do
  set -- $r
  /home/illusion/Documents/projects/llamacpp-tuner/tmp/loom/notes/gpu-run $LCT_HOME/loom/venv/bin/python -I experiments/inkling-loom/session/session.py --model $LCT_HOME/loom/$1 --preset $2 --jobs $jobs --out $out --label $3 --port 8131 $EXTRA > /dev/null 2>&1 || echo "FAILED $r"
  journalctl -k --since "-10min" | grep -i amdgpu | tail -2
done
python3 -c "
import json,sys
for l in open('$out'):
    d=json.loads(l); m=d.get('metrics_delta',{}); p=a=0; dep={'d3':3,'l3':3,'d2':2}.get(d.get('preset'),1)
    for k,v in m.items():
        if 'draft_proposed' in k: p=v
        if 'draft_accepted' in k: a=v
    print(f\"{d['label']:14s} {d['name']:12s} {d['decode_tps']:7.1f} t/s  gen {d['gen_tokens']:5d}  prop {p:6.0f} acc {a:6.0f}  step {1000*d['gen_tokens']/d['decode_tps']/max(p/dep,1):5.2f} ms  tok/step {(a+p/dep)/max(p/dep,1):4.2f}\")"
