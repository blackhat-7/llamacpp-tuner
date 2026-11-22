#!/bin/bash
# usage: prof.sh SERVE_LOG -> kernels by total device ms, per verify step (steps = MoE gate/up calls / 44)
grep " dispatch " $1 | sed 's/.*dispatch //' | awk '{c=$2; sub("count=","",c); t=$3; sub("total=","",t); u=$4; if(u=="us") t/=1000; if(u=="s") t*=1000; tot[$1]+=t; cnt[$1]+=c} END{for(k in tot) print k, cnt[k], tot[k]}' | sort -k3 -g -r > /tmp/prof.$$
steps=$(awk '$1=="g_ink_moe_gu_nvfp4_w1_k2304_r4"{print $2/44}' /tmp/prof.$$)
awk -v s=$steps 'BEGIN{printf "steps %d\n", s} {all+=$3; printf "%-40s %8d calls %8.1f ms  %6.3f ms/step  %6.1f us/call\n", $1, $2, $3, $3/s, 1000*$3/$2} END{printf "ALL %.3f ms/step\n", all/s}' /tmp/prof.$$ | head -${N:-25}
rm /tmp/prof.$$
