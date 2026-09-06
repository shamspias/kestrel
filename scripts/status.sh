#!/bin/bash
# One-shot status of the experiment pipeline. Usage: bash scripts/status.sh
cd "$(dirname "$0")/.."
echo "== GPU"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "== running jobs"; ps -eo pid,etime,args | grep -E "train.py|evaluate.py|analysis.py|train_yolo.py|eval_yolo.py|pretrain_distill.py|trt_latency.py|queue_cuda" | grep -v -E "grep|tail|pt_data_worker" | awk '{printf "  %-8s %-10s ", $1, $2; for(i=3;i<=NF;i++) if($i!~/^--(device|resume|out)$/ && $i!~/^cuda$/) printf "%s ", $i; print ""}' | cut -c1-140
echo "== queue (last 3 lines)"; for q in runs/queue_cuda4.log runs/queue_cuda3.log runs/queue_cuda2.log; do [ -f $q ] && { tail -3 $q; break; }; done
echo "== KESTREL runs: last EVAL line (gated AP; AP_nogate = gate off)"
for f in runs/kestrel_n.log runs/kestrel_n2.log runs/kestrel_s.log runs/ab_*.log runs/abl_*.log; do [ -f $f ] || continue
  st=$(grep -q "training complete" $f && echo done || echo running); last=$(grep EVAL $f | tail -1 | sed -E 's/ images=[^ ]+ ms_per_img_batched=[^ ]+//; s/ +\([0-9.]+ min\)//')
  cur=$(grep -E "^ep [0-9]+ it" $f | tail -1 | sed -E 's/^ep ([0-9]+) it ([0-9]+) lr [^ ]+ gn [^ ]+ ([0-9.]+) img\/s eta ([0-9.]+)h.*/ep \1 it \2 \3 img\/s eta \4h/'); printf "  %-22s %-8s %s | %s\n" "$(basename $f .log)" "$st" "$cur" "$last"; done
echo "== YOLO baselines: last epoch val (mAP50 mAP50-95)"
for f in runs/baselines/*.log; do [ -f $f ] || continue; case $f in *.eval.log|*.trt.log|*.latency.log) continue;; esac
  st=$(grep -q "epochs completed" $f && echo done || echo running); ep=$(tr '\r' '\n' < $f | grep -E "^ +[0-9]+/[0-9]+ " | tail -1 | awk '{print $1}')
  v=$(tr '\r' '\n' < $f | grep -E "^ +all +4952" | tail -1 | awk '{print $(NF-1), $NF}'); printf "  %-22s %-8s epoch %-8s %s\n" "$(basename $f .log)" "$st" "$ep" "$v"; done
echo "== finished evaluations"; ls runs/*/eval.json runs/baselines/*.json 2>/dev/null | sed 's/^/  /'
