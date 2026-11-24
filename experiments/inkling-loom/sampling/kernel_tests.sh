#!/bin/bash
# The select kernel's distribution test and its timing, each under the GPU lock.
# Usage: gen_select_test.py OUT NEW_LLM OLD_LLM first, then: kernel_tests.sh OUT
R=$(cd "$(dirname "$0")" && pwd); T=${LCT_HOME:-$(cd "$R/../../.." && pwd)/tmp}; V=$T/loom/venv/bin
cd "${1:?output directory of gen_select_test.py}"
echo "== loom-test $(date +%T)"
flock /tmp/xtx-gpu.lock $V/loom-test test.loom --device=amdgpu --file-output-root=. > test.json 2> test.err; echo "rc=$?"
tail -c 600 test.json; echo; tail -5 test.err
echo "== loom-benchmark $(date +%T)"
flock /tmp/xtx-gpu.lock $V/loom-benchmark bench.loom --device=amdgpu --measure=dispatch_complete --batch-size=8 --profile-final-batch=true --output-format=jsonl --output=r.jsonl > r.log 2>&1; echo "rc=$?"
tail -3 r.log
echo "== done $(date +%T)"
