#!/bin/bash
# The dense workstream's remaining GPU runs, in order (each step takes the GPU through notes/gpu-run).
set -x
D=/home/illusion/Documents/projects/llamacpp-tuner/tmp/experiments/inkling-loom/dense
L=/home/illusion/Documents/projects/llamacpp-tuner/tmp
LV=$L/loom/venv/bin; G=$L/loom/notes/gpu-run
cd $D
# 1. sconv kernels: bitwise test, negative control (must fail), microbench
(cd sc && $G $LV/loom-test test.loom --device=amdgpu > test.json 2> test.err; jq -c '[.samples[]|{case,passed}]' test.json)
(cd scneg && $G $LV/loom-test test.loom --device=amdgpu > test.json 2> test.err; jq -c '[.samples[]|{case,passed}]' test.json)
$G $LV/loom-benchmark sc/bench.loom --device=amdgpu --measure=dispatch_complete --batch-size=4 --output-format=jsonl --output=sc/r.jsonl > sc/r.log 2>&1
jq -r 'select(.row=="benchmark") | "\(.benchmark_result.benchmark) \(.benchmark_result.measurement.operation_timing_ns.p50)"' sc/r.jsonl
# 1b. mix+norm kernels: bitwise test, negative control (m3 cases must fail), microbench
(cd mx && $G $LV/loom-test test.loom --device=amdgpu > test.json 2> test.err; jq -c '{failed_sample_count, n: (.samples|length)}' test.json)
(cd mxneg && $G $LV/loom-test test.loom --device=amdgpu > test.json 2> test.err; jq -c '[.samples[]|select(.passed|not)|.case]' test.json)
$G $LV/loom-benchmark mx/bench.loom --device=amdgpu --measure=dispatch_complete --batch-size=4 --output-format=jsonl --output=mx/r.jsonl > mx/r.log 2>&1
jq -r 'select(.row=="benchmark") | "\(.benchmark_result.benchmark) \(.benchmark_result.measurement.operation_timing_ns.p50)"' mx/r.jsonl
# 2. packages register (p5 = default edit incl. mix fusion; p4 = default + --sconv), then e2e, then e2e (short, depth 3 and 1)
./regcheck.sh $L/loom/dense-p5 d3
./regcheck.sh $L/loom/dense-p4 d3
J=$L/experiments/inkling-loom/opt/jobs_base.json
for r in "dense-p5 d3 p5-d3" "dense-p4 d3 p4-d3" "dense-p5 xtx-kv8-196k p5-d1" "dense-p4 xtx-kv8-196k p4-d1"; do set -- $r; timeout 900 ./sess.sh $L/loom/$1 $2 $J $D/e2e.jsonl $3 > /dev/null 2>&1; done
# 3. repeat A/B (noise): base vs p2 (vs p3), depth 3 and 1
for r in "kv8w4096-d d3 base-d3b" "dense-p2 d3 p2-d3b" "kv8w4096-d xtx-kv8-196k base-d1b" "dense-p2 xtx-kv8-196k p2-d1b"; do set -- $r; timeout 900 ./sess.sh $L/loom/$1 $2 $J $D/e2e.jsonl $3 > /dev/null 2>&1; done
# 4. 64K: base vs p2, depth 1 and 3
for r in "kv8w4096-d xtx-kv8-196k base-64k-d1" "dense-p2 xtx-kv8-196k p2-64k-d1" "kv8w4096-d d3 base-64k-d3" "dense-p2 d3 p2-64k-d3"; do set -- $r; timeout 1200 ./sess.sh $L/loom/$1 $2 $D/jobs_64k.json $D/e2e64.jsonl $3 > /dev/null 2>&1; done
# 5. profile of p2 at depth 3 (per-kernel device time after the edits)
timeout 900 ./sess.sh $L/loom/dense-p4 d3 $L/experiments/inkling-loom/opt/jobs_prof.json $D/prof.jsonl dense-prof-p4-d3 prof > /dev/null 2>&1
for p in "base-d3 p5-d3" "base-d3 p4-d3" "base-d1 p5-d1" "base-d1 p4-d1" "base-d3b p2-d3b" "base-d1b p2-d1b"; do /usr/bin/python3 same.py e2e.jsonl $p; done
/usr/bin/python3 same.py e2e64.jsonl base-64k-d1 p2-64k-d1; /usr/bin/python3 same.py e2e64.jsonl base-64k-d3 p2-64k-d3
journalctl -k --since "-2h" | grep -i amdgpu | tail -3
