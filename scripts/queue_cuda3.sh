#!/bin/bash
# Stage 3 queue (after the recipe A/B showed lr 2e-3 @ bs 16 overshoots: lr 1e-3 → 14.2 vs 5.0 AP at 10 epochs).
# Remaining A/B arms, the corrected main N run, the two key baselines, ablations, S, and the other baselines.
# Idempotent: stages with a completion marker are skipped. Usage: nohup bash scripts/queue_cuda3.sh > runs/queue_cuda3.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EP_MAIN=${EP_MAIN:-80}; EP_ABL=${EP_ABL:-30}; SZ_MAIN=${SZ_MAIN:-512}; SZ_ABL=${SZ_ABL:-416}; EP_BASE=${EP_BASE:-$EP_MAIN}; EP_AB=${EP_AB:-10}
BS_N=${BS_N:-16}; LR_N=${LR_N:-1e-3}; BS_S=${BS_S:-8}; BS_BASE=${BS_BASE:-32}; AMP=${AMP:-fp16}; EXIT=${EXIT:-0.6 0.15 0.05}
mkdir -p runs/baselines
# wait for the orphaned A/B arm from queue 2 to finish before launching anything
while pgrep -f "train.py .*runs/ab_fp32" >/dev/null; do sleep 30; done; log "ab_fp32 finished (or not running)"
k() { name=$1; shift; for try in 1 2 3; do /usr/bin/grep -q "training complete" runs/$name.log 2>/dev/null && break
      log "run $name (try $try): $*"; python train.py --device cuda --amp $AMP --out runs/$name --resume "$@" >> runs/$name.log 2>&1; sleep 20; done; }
b() { name=$1; shift; for try in 1 2 3; do /usr/bin/grep -q "epochs completed" runs/baselines/$name.log 2>/dev/null && break
      log "baseline $name (try $try): $*"; python baselines/train_yolo.py "$@" --epochs $EP_BASE --imgsz $SZ_MAIN --batch $BS_BASE --device 0 --amp --name $name --resume >> runs/baselines/$name.log 2>&1; sleep 20; done; }
# accuracy-only evaluation in the background (overlaps the next training job; latency is measured later on an idle GPU)
ev_bg() { run=$1; sz=$2; shift 2; [ -f runs/$run/best.pt ] || return; [ -f runs/$run/eval.json ] && return
      log "eval (bg) $run @ $sz"; (python evaluate.py --ckpt runs/$run/best.pt --size $sz --device cuda --reparam --gate-sweep --out runs/$run/eval.json "$@" > runs/$run/eval.log 2>&1;
      python analysis.py --ckpt runs/$run/best.pt --size $sz --device cuda --out runs/$run/calib.npz > runs/$run/calib.log 2>&1) & }
evb_bg() { name=$1; w=runs/baselines/$name/weights/best.pt; [ -f $w ] || return; [ -f runs/baselines/$name.json ] && return
      log "eval (bg) baseline $name"; (python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --device 0 --out runs/baselines/$name.json > runs/baselines/$name.eval.log 2>&1
      case $name in yolo26*) python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --device 0 --end2end false --out runs/baselines/${name}_nms.json > runs/baselines/${name}_nms.eval.log 2>&1;; esac) & }
MAINSWEEP="--static-sweep --anytime-sweep --sweep-u 0.15 0.30 --anytime --exit $EXIT"
ABLSWEEP="--static-sweep --anytime-sweep --sweep-p 0.5 0.7 --sweep-u 0.15 0.30 --anytime --exit $EXIT"
AB="--model N --bs $BS_N --size $SZ_ABL --epochs $EP_AB --eval-every 5"
k ab_noinit  $AB --lr 2e-3
k ab_lr5e4   $AB --lr 5e-4 --init runs/distill_n/backbone.pt
log "A/B done"
N="--model N --bs $BS_N --lr $LR_N"
k kestrel_n2     $N --size $SZ_MAIN --epochs $EP_MAIN --init runs/distill_n/backbone.pt
ev_bg kestrel_n2 $SZ_MAIN $MAINSWEEP
b yolo26n_scratch yolo26n.yaml;   evb_bg yolo26n_scratch
b yolo11n_scratch yolo11n.yaml;   evb_bg yolo11n_scratch
k abl_full       $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt;                     ev_bg abl_full $SZ_ABL $ABLSWEEP
k abl_nogolsd    $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-golsd;          ev_bg abl_nogolsd $SZ_ABL $ABLSWEEP
k abl_deform     $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --local-attn deform; ev_bg abl_deform $SZ_ABL $ABLSWEEP
/usr/bin/grep -q "epoch 9 saved" runs/distill_s.log 2>/dev/null || { log "distill S"; python pretrain_distill.py --model S --size 448 --bs 16 --lr 5e-4 --epochs 10 --device cuda --workers 6 --out runs/distill_s >> runs/distill_s.log 2>&1; }
LR_S=$(cat runs/LR_S 2>/dev/null || echo 7e-4)          # decided from the lr 5e-4 arm; see runs/LR_S
k kestrel_s      --model S --bs $BS_S --lr $LR_S --size $SZ_MAIN --epochs $EP_MAIN --init runs/distill_s/backbone.pt
ev_bg kestrel_s $SZ_MAIN $MAINSWEEP
b yolo12n_scratch  yolo12n.yaml;  evb_bg yolo12n_scratch
b yolov10n_scratch yolov10n.yaml; evb_bg yolov10n_scratch
k abl_nopresence $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-presence;       ev_bg abl_nopresence $SZ_ABL $ABLSWEEP
k abl_scratch    $N --size $SZ_ABL --epochs $EP_ABL;                                                       ev_bg abl_scratch $SZ_ABL $ABLSWEEP
k abl_nodn       $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-dn;             ev_bg abl_nodn $SZ_ABL $ABLSWEEP
b yolov9t_scratch  yolov9t.yaml;  evb_bg yolov9t_scratch
b yolov8n_scratch  yolov8n.yaml;  evb_bg yolov8n_scratch
b yolo26n_coco     yolo26n.yaml --pretrained; evb_bg yolo26n_coco
wait; log "queue3 done"
