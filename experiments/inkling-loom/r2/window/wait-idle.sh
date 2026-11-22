#!/bin/bash
# Exit when the user's inkling server is gone, or has had no request for $1 minutes (default 30).
need=${1:-30}; idle=0; last=""
while true; do
  pgrep -x loom-serve >/dev/null || { echo "inkling server stopped $(date +%H:%M)"; exit 0; }
  port=$(grep -o 'Loom at 127.0.0.1:[0-9]*' /home/illusion/Documents/projects/llamacpp-tuner/tmp/servers/inkling.log | tail -1 | cut -d: -f2)
  m=$(curl -s --max-time 5 "http://127.0.0.1:$port/metrics")
  active=$(grep '^loom_serve_server_active_requests\|^loom_serve_model_active_requests' <<<"$m" | awk '{s+=$NF} END {print s+0}')
  total=$(grep '^loom_serve_server_requests_total' <<<"$m" | awk '{print $NF}')
  if [[ $active == 0 && $total == "$last" ]]; then idle=$((idle+1)); else idle=0; fi
  last=$total
  ((idle >= need)) && { echo "inkling idle ${need} min at $(date +%H:%M) (requests_total $total)"; exit 0; }
  sleep 60
done
