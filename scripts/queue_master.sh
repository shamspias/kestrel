#!/bin/bash
# Master queue: key results first. Usage: nohup bash scripts/queue_master.sh > runs/queue_master.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1
log() { echo "[$(date '+%F %T')] $*"; }
EP_MAIN=${EP_MAIN:-60}; EP_ABL=${EP_ABL:-30}; SZ_ABL=${SZ_ABL:-416}; LR=${LR:-1e-3}; EP_BASE=${EP_BASE:-60}
until /usr/bin/grep -q "epoch 9 saved" runs/distill_n.log 2>/dev/null; do sleep 120; done; log "distilled backbone ready"
k() { name=$1; shift; for try in 1 2 3; do /usr/bin/grep -q "training complete" runs/$name.log 2>/dev/null && break
      log "run $name (try $try): $*"; python train.py --model N --bs 8 --lr $LR --out runs/$name --resume "$@" >> runs/$name.log 2>&1; sleep 30; done; }
b() { name=$1; shift; mkdir -p runs/baselines; for try in 1 2 3; do /usr/bin/grep -q "epochs completed" runs/baselines/$name.log 2>/dev/null && break
      log "baseline $name (try $try)"; python baselines/train_yolo.py "$@" --epochs $EP_BASE --imgsz 512 --name $name --resume >> runs/baselines/$name.log 2>&1; sleep 30; done; }
k kestrel_n      --size 512 --epochs $EP_MAIN --init runs/distill_n/backbone.pt
k abl_full       --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt
k abl_nogolsd    --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-golsd
k abl_deform     --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --local-attn deform
b yolo26n_scratch yolo26n.yaml
b yolo11n_scratch yolo11n.yaml
k abl_nopresence --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-presence
k abl_scratch    --size $SZ_ABL --epochs $EP_ABL
k abl_nodn       --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-dn
b yolo26n_coco   yolo26n.yaml --pretrained
log "queue done"
