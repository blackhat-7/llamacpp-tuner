#!/bin/bash
# GPU window, phase A: every new kernel's correctness test (fast). Log: window-a.log next to this file.
T=/home/illusion/Documents/projects/llamacpp-tuner/tmp; V=$T/loom/venv/bin; G=$T/loom/notes/gpu-run
exec > $T/loom/notes/window-a.log 2>&1
echo "== start $(date +%T)"
echo "== attn fin/neg (neg must FAIL)"; A=$T/experiments/inkling-loom/attn; cd $A
for d in fin neg; do
  (cd $d && $G $V/loom-test test_p.loom --device=amdgpu | grep -o '"failed_sample_count":[0-9]*'
   $G $V/loom-test test_e.loom --device=amdgpu --file-output-root=. | grep -o '"failed_sample_count":[0-9]*')
  uv run -q --no-project --with numpy python check_e.py $d
done
echo "== attn v2fin"; (cd v2fin && $G $V/loom-test test_e.loom --device=amdgpu --file-output-root=. | grep -o '"failed_sample_count":[0-9]*'); uv run -q --no-project --with numpy python check_e.py v2fin
echo "== moe padding"; bash $T/loom/moe/deliver/pad_run.sh
echo "== dense sconv / scneg / mx / mxneg"; D=$T/experiments/inkling-loom/dense; cd $D
(cd sc && $G $V/loom-test test.loom --device=amdgpu > test.json 2> test.err; jq -c '[.samples[]|{case,passed}]' test.json)
(cd scneg && $G $V/loom-test test.loom --device=amdgpu > test.json 2> test.err; jq -c '[.samples[]|{case,passed}]' test.json)
(cd mx && $G $V/loom-test test.loom --device=amdgpu > test.json 2> test.err; jq -c '{failed_sample_count, n: (.samples|length)}' test.json)
(cd mxneg && $G $V/loom-test test.loom --device=amdgpu > test.json 2> test.err; jq -c '[.samples[]|select(.passed|not)|.case]' test.json)
echo "== prefill pftest8 / pftest4"; P=$T/experiments/inkling-loom/prefill
(cd $P/pftest8 && $G $V/loom-test test.loom --device=amdgpu | grep -o '"case":"[a-z0-9]*","sample_ordinal":0,"passed":[a-z]*')
(cd $P/pftest4 && $G $V/loom-test test.loom --device=amdgpu | grep -o '"case":"[a-z0-9]*","sample_ordinal":0,"passed":[a-z]*')
echo "== spec draft-head test (match, numpy pass; neg fails)"; bash $T/loom/spec/gpu_plan.sh 1
journalctl -k --since "-30min" --no-pager | grep -i amdgpu | grep -i "fault\|timeout\|reset" | tail -3
echo "== done $(date +%T)"
