#!/bin/bash
# Ultralytics baselines, sequential. Usage: EP=60 nohup bash scripts/queue_baselines.sh > runs/queue_baselines.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1
EP=${EP:-60}; SZ=${SZ:-512}
log() { echo "[$(date '+%F %T')] $*"; }
mkdir -p runs/baselines
b() { name=$1; shift
  if [ ! -f runs/baselines/$name/weights/best.pt ] || ! grep -q "epochs completed" runs/baselines/$name.log 2>/dev/null; then
    log "baseline $name"; python baselines/train_yolo.py "$@" --epochs $EP --imgsz $SZ --name $name --resume >> runs/baselines/$name.log 2>&1 || python baselines/train_yolo.py "$@" --epochs $EP --imgsz $SZ --name $name >> runs/baselines/$name.log 2>&1
  fi; }
b yolo26n_scratch yolo26n.yaml
b yolo11n_scratch yolo11n.yaml
b yolo26n_coco yolo26n.yaml --pretrained
log "baselines done"
