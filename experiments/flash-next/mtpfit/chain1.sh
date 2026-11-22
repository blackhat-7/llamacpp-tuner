#!/usr/bin/env bash
# chain 1: parity run (patched engine) -> dump of the transcripts so far -> transcript chunk 2. Each step takes gpu.lock.
set -u
M=$(dirname "$(readlink -f "$0")"); P=$M/..; D=$M/data
PY=~/Documents/projects/Strata/.venv/bin/python
until grep -q "built\|error\|STOP" $D/build.out 2>/dev/null; do sleep 10; done
cat $D/build.out
E=~/Documents/projects/Strata-wt/r9-sdmtp/build-hip/strata
if grep -q built $D/build.out; then
  rm -f $D/par.bin $D/par_r.bin
  $P/r8_run.sh r9par_prose prose.ids $E STRATA_MTP_PARITY=$D/par.bin STRATA_PREFILL_DUMP_R_ALL=$D/par_r.bin \
    STRATA_PREFILL_CPU_SHARE=0 -- --short-read 0 --prompt-cache 0
  ls -la $D/par.bin $D/par_r.bin
fi
n=$(wc -l < $D/transcripts.jsonl); echo "dump d1: $n transcripts"; echo $n > $D/d1.count
$PY -I $M/gen_data.py dump $D/transcripts.jsonl --out $D/d1
$PY -I $M/gen_data.py check $D/d1
$PY -I $M/gen_data.py transcripts $D/prompts.jsonl --out $D/transcripts.jsonl --max-new 768 --think 30 --minutes 17
echo CHAIN1 DONE
