#!/bin/bash
# Stage 2 queue: after the main N run, a short recipe A/B (10 epochs @416, one change each) before committing the
# ablation/S budget to a recipe that was only validated for one epoch on the M2; then the two key YOLO baselines.
# Usage: nohup bash scripts/queue_cuda2.sh > runs/queue_cuda2.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EP_AB=${EP_AB:-10}; SZ_AB=${SZ_AB:-416}; EP_BASE=${EP_BASE:-80}; SZ_MAIN=${SZ_MAIN:-512}; BS_BASE=${BS_BASE:-32}
mkdir -p runs/baselines
until /usr/bin/grep -q "training complete" runs/kestrel_n.log 2>/dev/null; do sleep 60; done; log "kestrel_n finished"
k() { name=$1; shift; for try in 1 2 3; do /usr/bin/grep -q "training complete" runs/$name.log 2>/dev/null && break
      log "run $name (try $try): $*"; python train.py --device cuda --out runs/$name --resume "$@" >> runs/$name.log 2>&1; sleep 20; done; }
b() { name=$1; shift; for try in 1 2 3; do /usr/bin/grep -q "epochs completed" runs/baselines/$name.log 2>/dev/null && break
      log "baseline $name (try $try): $*"; python baselines/train_yolo.py "$@" --epochs $EP_BASE --imgsz $SZ_MAIN --batch $BS_BASE --device 0 --amp --name $name --resume >> runs/baselines/$name.log 2>&1; sleep 20; done; }
ev_bg() { run=$1; sz=$2; shift 2; [ -f runs/$run/best.pt ] || return; [ -f runs/$run/eval.json ] && return
      log "eval (bg) $run @ $sz"; (python evaluate.py --ckpt runs/$run/best.pt --size $sz --device cuda --reparam --gate-sweep --out runs/$run/eval.json "$@" > runs/$run/eval.log 2>&1;
      python analysis.py --ckpt runs/$run/best.pt --size $sz --device cuda --out runs/$run/calib.npz > runs/$run/calib.log 2>&1) & }
evb_bg() { name=$1; w=runs/baselines/$name/weights/best.pt; [ -f $w ] || return; [ -f runs/baselines/$name.json ] && return
      log "eval (bg) baseline $name"; (python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --device 0 --out runs/baselines/$name.json > runs/baselines/$name.eval.log 2>&1
      case $name in yolo26*) python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --device 0 --end2end false --out runs/baselines/${name}_nms.json > runs/baselines/${name}_nms.eval.log 2>&1;; esac) & }
ev_bg kestrel_n $SZ_MAIN --static-sweep --anytime-sweep
# recipe A/B (N, bs 16, 10 epochs @416): reference / half lr / fp32 / no distillation init
AB="--model N --bs 16 --size $SZ_AB --epochs $EP_AB --eval-every 5"
k ab_ref     $AB --lr 2e-3 --amp fp16 --init runs/distill_n/backbone.pt
k ab_lr1e3   $AB --lr 1e-3 --amp fp16 --init runs/distill_n/backbone.pt
k ab_fp32    $AB --lr 2e-3 --amp none --init runs/distill_n/backbone.pt
k ab_noinit  $AB --lr 2e-3 --amp fp16
log "A/B done"
b yolo26n_scratch yolo26n.yaml;   evb_bg yolo26n_scratch
b yolo11n_scratch yolo11n.yaml;   evb_bg yolo11n_scratch
wait; log "queue2 done"
