#!/bin/bash
# Lane B (rev 2): the full baseline suite, reordered so the risky row is settled first.
#
# Same data, image size, epoch budget and device as KESTREL, scored by the same pycocotools evaluator.
# Ordering differs from scripts/queue_baselines_full.sh in one way that matters: rtdetr-l runs SECOND, right after the
# nano row already in flight, because it is the only baseline whose training loop is unproven in this ultralytics
# version — construction from YAML succeeds, but RT-DETR's loss and collate path have never been exercised here. If it
# cannot train we need that answer in an hour, not in two days. It is also the memory hazard on this box (a ~33 M model
# whose parent process alone is ~3 GB), so it runs with two workers and never beside another baseline.
#
# Usage: nohup bash scripts/queue7_baselines.sh > runs/queue7_baselines.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EP_BASE=${EP_BASE:-80}; SZ=${SZ:-512}; BS=${BS:-32}; BS_BIG=${BS_BIG:-16}; WK=${WK:-2}   # worker forks, not model size, are what exhausts host RAM on this box; the loader has never been the bottleneck
mkdir -p runs/baselines

b() { name=$1; cfg=$2; bs=${3:-$BS}; wk=${4:-$WK}; extra=${5:-}
  for try in 1 2 3; do
    /usr/bin/grep -q "epochs completed" runs/baselines/$name.log 2>/dev/null && break
    log "train $name ($cfg, batch $bs, $wk workers $extra)"
    python baselines/train_yolo.py "$cfg" $extra --epochs $EP_BASE --imgsz $SZ --batch $bs --device 0 \
      --workers $wk --amp --name "$name" --resume >> runs/baselines/$name.log 2>&1
    sleep 20
  done
  w=runs/baselines/$name/weights/best.pt
  [ -f "$w" ] || { log "  $name produced no checkpoint — skipping its evaluation"; return; }
  [ -f runs/baselines/$name.json ] && return
  log "score $name with the unified evaluator"
  python baselines/eval_yolo.py "$w" --imgsz $SZ --device 0 --out runs/baselines/$name.json > runs/baselines/$name.eval.log 2>&1
  case $name in yolo26*|yolov10*)      # end-to-end families: also score the NMS path
    python baselines/eval_yolo.py "$w" --imgsz $SZ --device 0 --end2end false \
      --out runs/baselines/${name}_nms.json > runs/baselines/${name}_nms.eval.log 2>&1;; esac; }

# b() already skips a completed run, so no external guard is needed (a pgrep guard here also self-matched).
b yolo26n_scratch   yolo26n.yaml
# 1. the unproven row, settled first
b rtdetrl_scratch   rtdetr-l.yaml    $BS_BIG 2
# 2. the rest of the nano tier, matched to KESTREL-N at 5.31 M parameters
b yolo11n_scratch   yolo11n.yaml
b yolov8n_scratch   yolov8n.yaml
b yolov10n_scratch  yolov10n.yaml
b yolov9t_scratch   yolov9t.yaml
b yolo12n_scratch   yolo12n.yaml
# 3. COCO-pretrained context rows (not like-for-like)
b yolo26n_coco      yolo26n.yaml     $BS $WK --pretrained
# 4. the small tier, matched to KESTREL-S at 13.0 M — only useful once KESTREL-S itself exists
b yolo26s_scratch   yolo26s.yaml
b yolo11s_scratch   yolo11s.yaml
b yolov8s_scratch   yolov8s.yaml
b yolov10s_scratch  yolov10s.yaml
b yolov9s_scratch   yolov9s.yaml
b yolo26s_coco      yolo26s.yaml     $BS $WK --pretrained
# 5. older generations, for the architecture-progress column
b yolov6n_scratch   yolov6n.yaml
b yolov5n_scratch   yolov5n.yaml
log "lane B done"
