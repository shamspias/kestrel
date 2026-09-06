#!/bin/bash
# Lane B (ultralytics baselines), concurrent with lane A. Same data, image size, epoch budget and evaluator as KESTREL.
# Usage: nohup bash scripts/queue5_yolo.sh > runs/queue5_yolo.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EP_BASE=${EP_BASE:-80}; SZ_MAIN=${SZ_MAIN:-512}; BS_BASE=${BS_BASE:-32}; WK=${WK:-4}
mkdir -p runs/baselines
b() { name=$1; shift; for try in 1 2 3; do /usr/bin/grep -q "epochs completed" runs/baselines/$name.log 2>/dev/null && break
      log "baseline $name (try $try): $*"; python baselines/train_yolo.py "$@" --epochs $EP_BASE --imgsz $SZ_MAIN --batch $BS_BASE --device 0 --workers $WK --amp --name $name --resume >> runs/baselines/$name.log 2>&1; sleep 20; done
      w=runs/baselines/$name/weights/best.pt; [ -f $w ] || return; [ -f runs/baselines/$name.json ] && return
      log "eval baseline $name"; python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --device 0 --out runs/baselines/$name.json > runs/baselines/$name.eval.log 2>&1
      case $name in yolo26*) python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --device 0 --end2end false --out runs/baselines/${name}_nms.json > runs/baselines/${name}_nms.eval.log 2>&1;; esac; }
b yolo26n_scratch  yolo26n.yaml
b yolo11n_scratch  yolo11n.yaml
b yolo26n_coco     yolo26n.yaml --pretrained
b yolo12n_scratch  yolo12n.yaml
b yolov10n_scratch yolov10n.yaml
b yolov9t_scratch  yolov9t.yaml
b yolov8n_scratch  yolov8n.yaml
log "lane B done"
