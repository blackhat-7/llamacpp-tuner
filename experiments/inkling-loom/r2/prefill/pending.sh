#!/bin/bash
# GPU runs still needed by the prefill workstream. Each waits for the GPU hold via gpu-run.
set -u
export LCT_HOME=/home/illusion/Documents/projects/llamacpp-tuner/tmp
R=$LCT_HOME/loom/notes/gpu-run; PY=$LCT_HOME/loom/venv/bin/python; D=$LCT_HOME/experiments/inkling-loom/prefill
cd /home/illusion/Documents/projects/llamacpp-tuner-inkling
S="experiments/inkling-loom/session/session.py --port 8095 --serve-arg=--pool_backing=fixed --out $D/res.jsonl"
# 1. Split-count sweep at chunk 1024 (8K / 64K / 187K TTFT), distinct prompts per session.
$R $PY -I $S --model $LCT_HOME/loom/prefill-b2 --preset p1024 --jobs $D/jobs_F.json --label pf-b2-F
$R $PY -I $S --model $LCT_HOME/loom/prefill-b8 --preset p1024 --jobs $D/jobs_G.json --label pf-b8-G
# 2. Short greedy outputs + decode with the split kernel (compare to pf-base-short with session/compare.py).
$R $PY -I $S --model $LCT_HOME/loom/prefill-b --preset p1024 --jobs $LCT_HOME/experiments/inkling-loom/opt/jobs_base.json --label pf-b-short
# 3. 195K recall 3/3 (probe_loom) on prefill-b, chunk 1024.
$R bash -c "$LCT_HOME/loom/venv/bin/loom-serve --device=amdgpu --model=$LCT_HOME/loom/prefill-b --preset=p1024 --pool_backing=fixed --listen=127.0.0.1:8095 > $D/serve-recall.log 2>&1 & pid=\$!
  until curl -s 127.0.0.1:8095/v1/models | grep -q '\"id\"'; do sleep 2; done
  $PY experiments/inkling-loom/probe_loom.py run --server http://127.0.0.1:8095 --model Inkling-2-Mini --prompt $LCT_HOME/model-hunt/prompt_inkling_195000.txt --label pf-b-recall --kinds recall | cut -c1-400
  kill \$pid; wait \$pid"
# 4. Repeat baseline and prefill-b once more (noise), with a 64K profile of prefill-b.
$R $PY -I $S --model $LCT_HOME/loom/kv8w4096 --preset xtx-kv8-196k --jobs $D/jobs_H.json --label pf-base-H
$R $PY -I $S --model $LCT_HOME/loom/prefill-b --preset p1024 --jobs $D/jobs_D.json --label pf-b-D
journalctl -k --since "-60min" | grep -i amdgpu | tail -3
# 5. Prefetch kernel (ts2_pf): correctness vs ts2, then kernel timing of all variants (~2 min each).
B=$LCT_HOME/loom/venv/bin
(cd $D/pftest8 && $R $B/loom-test test.loom --device=amdgpu | grep -o '"case":"[a-z0-9]*","sample_ordinal":0,"passed":[a-z]*')
(cd $D/pftest4 && $R $B/loom-test test.loom --device=amdgpu | grep -o '"case":"[a-z0-9]*","sample_ordinal":0,"passed":[a-z]*')
(cd $D/skbench && $R $B/loom-benchmark bench.loom --device=amdgpu --measure=dispatch_complete --batch-size=4 --output-format=jsonl --output=r.jsonl > r.log 2>&1; tail -3 r.log)
# 6. E2E of the best pf variant (prefill-c = QL 8, prefill-c4 = QL 4), as in step 1.
$R $PY -I $S --model $LCT_HOME/loom/prefill-c4 --preset p1024 --jobs $D/jobs_B.json --label pf-c4-B
