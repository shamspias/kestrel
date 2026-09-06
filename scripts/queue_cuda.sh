#!/bin/bash
# Master queue for a single CUDA GPU (written for an RTX 2080 Ti with ~5.5 GB usable): key results first.
# Each stage is idempotent (skipped when its "training complete" / "epochs completed" marker exists) and retried
# with --resume, so the script can be re-run after an interruption.
# Usage: nohup bash scripts/queue_cuda.sh > runs/queue_cuda.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # never let ultralytics pip-install into the venv
log() { echo "[$(date '+%F %T')] $*"; }
EP_MAIN=${EP_MAIN:-80}; EP_ABL=${EP_ABL:-30}; SZ_MAIN=${SZ_MAIN:-512}; SZ_ABL=${SZ_ABL:-416}; EP_BASE=${EP_BASE:-$EP_MAIN}
BS_N=${BS_N:-16}; LR_N=${LR_N:-2e-3}; BS_S=${BS_S:-8}; LR_S=${LR_S:-1e-3}; BS_BASE=${BS_BASE:-32}; AMP=${AMP:-fp16}
mkdir -p runs/baselines
until /usr/bin/grep -q "epoch 9 saved" runs/distill_n.log 2>/dev/null; do sleep 60; done; log "distilled N backbone ready"
k() { name=$1; shift; for try in 1 2 3; do /usr/bin/grep -q "training complete" runs/$name.log 2>/dev/null && break
      log "run $name (try $try): $*"; python train.py --device cuda --amp $AMP --out runs/$name --resume "$@" >> runs/$name.log 2>&1; sleep 20; done; }
b() { name=$1; shift; for try in 1 2 3; do /usr/bin/grep -q "epochs completed" runs/baselines/$name.log 2>/dev/null && break
      log "baseline $name (try $try): $*"; python baselines/train_yolo.py "$@" --epochs $EP_BASE --imgsz $SZ_MAIN --batch $BS_BASE --device 0 --amp --name $name --resume >> runs/baselines/$name.log 2>&1; sleep 20; done; }
# accuracy-only evaluation, launched in the background so it overlaps the next training job (latency is measured separately on an idle GPU)
ev_bg() { run=$1; sz=$2; shift 2; [ -f runs/$run/best.pt ] || return; [ -f runs/$run/eval.json ] && return
      log "eval (bg) $run @ $sz"; (python evaluate.py --ckpt runs/$run/best.pt --size $sz --device cuda --reparam --out runs/$run/eval.json "$@" > runs/$run/eval.log 2>&1;
      python analysis.py --ckpt runs/$run/best.pt --size $sz --device cuda --out runs/$run/calib.npz > runs/$run/calib.log 2>&1) & }
evb_bg() { name=$1; w=runs/baselines/$name/weights/best.pt; [ -f $w ] || return; [ -f runs/baselines/$name.json ] && return
      log "eval (bg) baseline $name"; (python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --device 0 --out runs/baselines/$name.json > runs/baselines/$name.eval.log 2>&1
      case $name in yolo26*) python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --device 0 --end2end false --out runs/baselines/${name}_nms.json > runs/baselines/${name}_nms.eval.log 2>&1;; esac) & }
N="--model N --bs $BS_N --lr $LR_N"; S="--model S --bs $BS_S --lr $LR_S"
k kestrel_n      $N --size $SZ_MAIN --epochs $EP_MAIN --init runs/distill_n/backbone.pt
ev_bg kestrel_n $SZ_MAIN --static-sweep --anytime-sweep
b yolo26n_scratch yolo26n.yaml;   evb_bg yolo26n_scratch
b yolo11n_scratch yolo11n.yaml;   evb_bg yolo11n_scratch
k abl_full       $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt;                 ev_bg abl_full $SZ_ABL --static-sweep --anytime-sweep --anytime --exit ${EXIT:-0.6 0.15 0.05}
k abl_nogolsd    $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-golsd;      ev_bg abl_nogolsd $SZ_ABL --static-sweep --anytime-sweep --anytime --exit ${EXIT:-0.6 0.15 0.05}
k abl_deform     $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --local-attn deform; ev_bg abl_deform $SZ_ABL --static-sweep --anytime-sweep --anytime --exit ${EXIT:-0.6 0.15 0.05}
# phase-1 distillation for S (small memory footprint) — only if not already done
/usr/bin/grep -q "epoch 9 saved" runs/distill_s.log 2>/dev/null || { log "distill S"; python pretrain_distill.py --model S --size 448 --bs 16 --lr 5e-4 --epochs 10 --device cuda --workers 6 --out runs/distill_s >> runs/distill_s.log 2>&1; }
k kestrel_s      $S --size $SZ_MAIN --epochs $EP_MAIN --init runs/distill_s/backbone.pt
ev_bg kestrel_s $SZ_MAIN --static-sweep --anytime-sweep
b yolo12n_scratch  yolo12n.yaml;  evb_bg yolo12n_scratch
b yolov10n_scratch yolov10n.yaml; evb_bg yolov10n_scratch
k abl_nopresence $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-presence;   ev_bg abl_nopresence $SZ_ABL --static-sweep --anytime-sweep --anytime --exit ${EXIT:-0.6 0.15 0.05}
k abl_scratch    $N --size $SZ_ABL --epochs $EP_ABL;                                                   ev_bg abl_scratch $SZ_ABL --static-sweep --anytime-sweep --anytime --exit ${EXIT:-0.6 0.15 0.05}
k abl_nodn       $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-dn;         ev_bg abl_nodn $SZ_ABL --static-sweep --anytime-sweep --anytime --exit ${EXIT:-0.6 0.15 0.05}
b yolov9t_scratch  yolov9t.yaml;  evb_bg yolov9t_scratch
b yolov8n_scratch  yolov8n.yaml;  evb_bg yolov8n_scratch
b yolo26n_coco     yolo26n.yaml --pretrained; evb_bg yolo26n_coco
wait; log "queue done"
