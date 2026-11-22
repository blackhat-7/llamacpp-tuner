#!/bin/bash
# GPU: padding-skip kernel test (bit-exact live outputs, zero dead slots) + negative control.
D=/home/illusion/Documents/projects/llamacpp-tuner/tmp; P=$D/experiments/inkling-loom/moe/pad
cd $P && $D/loom/notes/gpu-run $D/loom/venv/bin/loom-test test.loom --device=amdgpu --file-output-root=$P > out.json 2> err.txt
python3 -c "import json; d=json.load(open('$P/out.json')); print([(s['case'], s['passed']) for s in d['samples']])"
uv run --no-project --with numpy python $D/loom/moe/deliver/check_pad.py $P && echo PAD-OK
uv run --no-project --with numpy python $D/loom/moe/deliver/check_pad.py $P --neg >/dev/null && echo NEG-UNEXPECTED-PASS || echo NEG-FAILS-AS-EXPECTED
