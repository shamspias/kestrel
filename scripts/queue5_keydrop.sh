#!/bin/bash
# Lane F: the constructive fix for the removal failure. Removing exited queries collapses the survivors' class scores
# because the decoder is trained with a fixed peer set of 100 queries and never sees a smaller one; the damage is
# monotone in HOW MANY keys vanish and independent of WHICH ones. Training with self-attention key dropout exposes the
# decoder to the whole range of peer-set sizes, which should make removal work as well as freezing — and removal is
# strictly cheaper than freezing, since it drops the retained keys' projections too.
# Two strengths, sampled per layer and per image from [0, p]: 0.9 covers the full range seen at inference, 0.5 is the
# conservative setting in case the aggressive one costs clean accuracy. Compare against runs/abl_full (same schedule).
# Usage: nohup bash scripts/queue5_keydrop.sh > runs/queue5_keydrop.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EP_ABL=${EP_ABL:-30}; SZ_ABL=${SZ_ABL:-416}; EXIT=${EXIT:-0.6 0.15 0.05}
# wait for the ablations AND for lane C, so at most one extra training job shares the GPU with lane A's KESTREL-S
until /usr/bin/grep -q "training complete" runs/abl_nodn.log 2>/dev/null; do sleep 120; done; log "ablations finished"
until /usr/bin/grep -q "lane C done" runs/queue5_deep.log 2>/dev/null; do sleep 120; done; log "lane C finished"
until [ -f runs/RECIPE_N ]; do sleep 30; done; source runs/RECIPE_N
SWEEP="--static-sweep --anytime-sweep --sweep-mode remove freeze --sweep-min-layers 1 2 --sweep-policy confidence random --sweep-p 1.1 --sweep-u 0.0 --sweep-bg 0.05 0.1 0.2 0.3 --sweep-random-p 0.3 0.6 0.9 --anytime --exit $EXIT"
kd() { name=$1; p=$2
  for try in 1 2 3; do /usr/bin/grep -q "training complete" runs/$name.log 2>/dev/null && break
    log "run $name (try $try): key dropout $p"
    python train.py --device cuda --amp $AMP_N --model N --bs $BS_N --lr $LR_N --size $SZ_ABL --epochs $EP_ABL \
      --key-dropout $p --init runs/distill_n/backbone.pt --out runs/$name --resume >> runs/$name.log 2>&1; sleep 20; done
  [ -f runs/$name/eval.json ] && return
  log "eval $name"
  python evaluate.py --ckpt runs/$name/best.pt --size $SZ_ABL --device cuda --reparam --workers 1 --gate-power 0 $SWEEP --out runs/$name/eval_nogate.json > runs/$name/eval_nogate.log 2>&1
  python evaluate.py --ckpt runs/$name/best.pt --size $SZ_ABL --device cuda --reparam --workers 1 --gate-sweep --static-sweep --out runs/$name/eval.json > runs/$name/eval.log 2>&1
  python scripts/duplicate_analysis.py --ckpt runs/$name/best.pt --size $SZ_ABL --subset 600 --gate-power 0 \
    --configs full freeze:1:0.2 remove:1:0.2 freeze:2:0.05 remove:2:0.05 --out runs/$name/dupes.json > runs/$name/dupes.log 2>&1; }
kd abl_keydrop 0.9
kd abl_keydrop_mild 0.5
log "lane F done"
