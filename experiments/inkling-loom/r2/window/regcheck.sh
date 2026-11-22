#!/bin/bash
# regcheck.sh PKG PRESET: load PKG with PRESET on the XTX, print the registration result (ready or the error).
T=/home/illusion/Documents/projects/llamacpp-tuner/tmp
flock /tmp/xtx-gpu.lock bash -c "
$T/loom/venv/bin/loom-serve --device=amdgpu --model=$1 --preset=$2 --pool_backing=fixed --listen=127.0.0.1:8098 > /tmp/regcheck.log 2>&1 & pid=\$!
for i in \$(seq 1 90); do
  r=\$(curl -s --max-time 3 127.0.0.1:8098/v1/status | python3 -c 'import sys,json; m=json.load(sys.stdin)[\"data\"][0][\"loom\"]; print(m[\"registration\"], m.get(\"error\",{}).get(\"message\",\"\"))' 2>/dev/null)
  case \$r in ready*|failed*) break;; esac; sleep 2
done
echo \"$(basename $1) $2: \$r\"; kill \$pid; wait \$pid 2>/dev/null"
