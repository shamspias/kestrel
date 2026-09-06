#!/bin/bash
# Full baseline suite for the paper: every model trained from scratch on the SAME data, image size, epoch
# budget and device as KESTREL, and scored by the SAME pycocotools evaluator (baselines/eval_yolo.py).
#
# Ordered by value so that stopping at any point still leaves a coherent, publishable table:
#   tier 1  nano dense detectors, matched to KESTREL-N (5.31 M)   -- six architecture generations
#   tier 2  RT-DETR, the transformer set-prediction family KESTREL belongs to  -- the key non-YOLO baseline
#   tier 3  small dense detectors, matched to KESTREL-S (13.0 M)
#   tier 4  COCO-pretrained reference rows (NOT like-for-like; context only)
#   tier 5  older generations, for the architecture-progress column
#
# Usage: nohup bash scripts/queue_baselines_full.sh > runs/queue_baselines_full.log 2>&1 &
# Env:   EP_BASE (default 80), SZ (512), BS (32), BS_BIG (16 for models over ~25 M params)
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EP_BASE=${EP_BASE:-80}; SZ=${SZ:-512}; BS=${BS:-32}; BS_BIG=${BS_BIG:-16}; WK=${WK:-4}
mkdir -p runs/baselines

# train one baseline then score it with OUR evaluator; both steps are skipped if already complete
b() { name=$1; cfg=$2; bs=${3:-$BS}; extra=${4:-}
  for try in 1 2 3; do
    /usr/bin/grep -q "epochs completed" runs/baselines/$name.log 2>/dev/null && break
    log "train $name ($cfg, batch $bs $extra)"
    python baselines/train_yolo.py "$cfg" $extra --epochs $EP_BASE --imgsz $SZ --batch $bs --device 0 \
        --workers $WK --amp --name "$name" --resume >> runs/baselines/$name.log 2>&1
    sleep 20
  done
  w=runs/baselines/$name/weights/best.pt
  [ -f "$w" ] || { log "  no checkpoint for $name, skipping evaluation"; return; }
  [ -f runs/baselines/$name.json ] || {
    log "eval $name"
    python baselines/eval_yolo.py "$w" --imgsz $SZ --device 0 --out runs/baselines/$name.json \
        > runs/baselines/$name.eval.log 2>&1; }
  # YOLO26 and YOLOv10 are natively end-to-end; also score their NMS path so both are in the table
  case $name in yolo26*|yolov10*)
    [ -f runs/baselines/${name}_nms.json ] || python baselines/eval_yolo.py "$w" --imgsz $SZ --device 0 \
        --end2end false --out runs/baselines/${name}_nms.json > runs/baselines/${name}_nms.eval.log 2>&1;;
  esac; }

log "=== tier 1: nano dense detectors (matched to KESTREL-N)"
b yolo26n_scratch   yolo26n.yaml
b yolo11n_scratch   yolo11n.yaml
b yolov8n_scratch   yolov8n.yaml
b yolov10n_scratch  yolov10n.yaml
b yolov9t_scratch   yolov9t.yaml
b yolo12n_scratch   yolo12n.yaml

log "=== tier 2: RT-DETR — transformer set prediction, the family KESTREL belongs to"
# RT-DETR-l is 33 M against KESTREL-N's 5.3 M, so it is not a size-matched row; it is the architecture-matched
# one, and the table reports parameters so the reader can see that.
b rtdetrl_scratch   rtdetr-l.yaml    $BS_BIG

log "=== tier 3: small dense detectors (matched to KESTREL-S)"
b yolo26s_scratch   yolo26s.yaml
b yolo11s_scratch   yolo11s.yaml
b yolov8s_scratch   yolov8s.yaml
b yolov10s_scratch  yolov10s.yaml
b yolov9s_scratch   yolov9s.yaml

log "=== tier 4: COCO-pretrained reference rows (context only, NOT like-for-like)"
b yolo26n_coco      yolo26n.yaml     $BS  --pretrained
b yolo26s_coco      yolo26s.yaml     $BS  --pretrained

log "=== tier 5: older generations, for the architecture-progress column"
b yolov6n_scratch   yolov6n.yaml
b yolov5n_scratch   yolov5n.yaml

log "baseline suite complete"
