#!/bin/bash
# Extra same-condition baselines, run after the master queue finishes.
# Usage: nohup bash scripts/queue_more_baselines.sh > runs/queue_more_baselines.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1
log() { echo "[$(date '+%F %T')] $*"; }
EP_BASE=${EP_BASE:-60}
until /usr/bin/grep -q "queue done" runs/queue_master.log 2>/dev/null; do sleep 300; done; log "master queue finished"
b() { name=$1; shift; mkdir -p runs/baselines; for try in 1 2 3; do /usr/bin/grep -q "epochs completed" runs/baselines/$name.log 2>/dev/null && break
      log "baseline $name (try $try)"; python baselines/train_yolo.py "$@" --epochs $EP_BASE --imgsz 512 --name $name --resume >> runs/baselines/$name.log 2>&1; sleep 30; done; }
b yolov8n_scratch  yolov8n.yaml
b yolov10n_scratch yolov10n.yaml
b yolo12n_scratch  yolo12n.yaml
b yolov9t_scratch  yolov9t.yaml
log "extra baselines done"
