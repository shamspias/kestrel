#!/bin/bash
# Lane A (KESTREL models). Runs concurrently with lane B (scripts/queue5_yolo.sh) now that the GPU is free of the
# llama-server allocation. Recipe (precision/batch/lr) is read from runs/RECIPE_N, written after the A/B arms finish.
# Usage: nohup bash scripts/queue5_kestrel.sh > runs/queue5_kestrel.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EP_MAIN=${EP_MAIN:-80}; EP_ABL=${EP_ABL:-30}; SZ_MAIN=${SZ_MAIN:-512}; SZ_ABL=${SZ_ABL:-416}; EP_AB=${EP_AB:-10}; WK=${WK:-5}
EXIT=${EXIT:-0.6 0.15 0.05}
k() { name=$1; amp=$2; shift 2; for try in 1 2 3; do /usr/bin/grep -q "training complete" runs/$name.log 2>/dev/null && break
      log "run $name (try $try): --amp $amp $*"; python train.py --device cuda --amp $amp --workers $WK --out runs/$name --resume "$@" >> runs/$name.log 2>&1; sleep 20; done; }
ev_bg() { run=$1; sz=$2; shift 2; [ -f runs/$run/best.pt ] || return; [ -f runs/$run/eval.json ] && return
      log "eval (bg) $run @ $sz"; (python evaluate.py --ckpt runs/$run/best.pt --size $sz --device cuda --reparam --gate-sweep --out runs/$run/eval.json "$@" > runs/$run/eval.log 2>&1;
      python analysis.py --ckpt runs/$run/best.pt --size $sz --device cuda --out runs/$run/calib.npz > runs/$run/calib.log 2>&1) & }
# anytime sweeps: both exit modes, both minimum depths, background-exit thresholds (the foreground/entropy exit is swept too)
MAINSWEEP="--static-sweep --anytime-sweep --sweep-mode remove freeze --sweep-min-layers 1 2 --sweep-p 0.5 1.1 --sweep-u 0.3 --sweep-bg 0.05 0.1 0.2 0.3 --anytime --exit $EXIT"
ABLSWEEP="--static-sweep --anytime-sweep --sweep-mode remove freeze --sweep-min-layers 1 2 --sweep-p 1.1 --sweep-u 0.0 --sweep-bg 0.1 0.3 --anytime --exit $EXIT"
while pgrep -f "train.py .*runs/ab_" >/dev/null; do sleep 30; done; log "A/B arm free"
k ab_fp32_lr1e3 none --model N --bs 16 --size $SZ_ABL --epochs $EP_AB --eval-every 5 --lr 1e-3 --init runs/distill_n/backbone.pt
log "A/B done — waiting for runs/RECIPE_N (AMP_N BS_N LR_N AMP_S BS_S LR_S)"
until [ -f runs/RECIPE_N ]; do sleep 30; done; source runs/RECIPE_N; log "recipe: N=$AMP_N/bs$BS_N/lr$LR_N  S=$AMP_S/bs$BS_S/lr$LR_S"
N="--model N --bs $BS_N --lr $LR_N"
k kestrel_n2     $AMP_N $N --size $SZ_MAIN --epochs $EP_MAIN --init runs/distill_n/backbone.pt
ev_bg kestrel_n2 $SZ_MAIN $MAINSWEEP
k abl_full       $AMP_N $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt;                     ev_bg abl_full $SZ_ABL $ABLSWEEP
k abl_nogolsd    $AMP_N $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-golsd;          ev_bg abl_nogolsd $SZ_ABL $ABLSWEEP
k abl_deform     $AMP_N $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --local-attn deform; ev_bg abl_deform $SZ_ABL $ABLSWEEP
k abl_scratch    $AMP_N $N --size $SZ_ABL --epochs $EP_ABL;                                                       ev_bg abl_scratch $SZ_ABL $ABLSWEEP
k abl_nopresence $AMP_N $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-presence;       ev_bg abl_nopresence $SZ_ABL $ABLSWEEP
k abl_nodn       $AMP_N $N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt --no-dn;             ev_bg abl_nodn $SZ_ABL $ABLSWEEP
/usr/bin/grep -q "epoch 9 saved" runs/distill_s.log 2>/dev/null || { log "distill S"; python pretrain_distill.py --model S --size 448 --bs 16 --lr 5e-4 --epochs 10 --device cuda --workers $WK --out runs/distill_s >> runs/distill_s.log 2>&1; }
k kestrel_s      $AMP_S --model S --bs $BS_S --lr $LR_S --size $SZ_MAIN --epochs $EP_MAIN --init runs/distill_s/backbone.pt
ev_bg kestrel_s $SZ_MAIN $MAINSWEEP
wait; log "lane A done"
