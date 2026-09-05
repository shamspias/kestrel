#!/bin/bash
# Sequential experiment queue. Each stage is skipped if its final artifact exists, so the script can be re-run.
# Usage: nohup bash scripts/queue.sh > runs/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONUNBUFFERED=1
log() { echo "[$(date '+%F %T')] $*"; }
EP_MAIN=${EP_MAIN:-60}; EP_ABL=${EP_ABL:-30}; SZ_ABL=${SZ_ABL:-416}; LR=${LR:-1e-3}

# 0. wait for the distilled backbone (10 epochs)
until grep -q "epoch 9 saved" runs/distill_n.log 2>/dev/null; do sleep 120; done
log "distilled backbone ready"

# 1. main model
if [ ! -f runs/kestrel_n/best.pt ] || ! grep -q "training complete" runs/kestrel_n.log 2>/dev/null; then
  log "main run: kestrel_n"
  python train.py --model N --size 512 --bs 8 --lr $LR --epochs $EP_MAIN --init runs/distill_n/backbone.pt --out runs/kestrel_n --resume >> runs/kestrel_n.log 2>&1
fi
# 2. ablations (416, shorter)
run_abl() { name=$1; shift
  if ! grep -q "training complete" runs/$name.log 2>/dev/null; then
    log "ablation: $name $*"
    python train.py --model N --size $SZ_ABL --bs 8 --lr $LR --epochs $EP_ABL --out runs/$name --resume "$@" >> runs/$name.log 2>&1
  fi; }
run_abl abl_full       --init runs/distill_n/backbone.pt
run_abl abl_nogolsd    --init runs/distill_n/backbone.pt --no-golsd
run_abl abl_deform     --init runs/distill_n/backbone.pt --local-attn deform
run_abl abl_nopresence --init runs/distill_n/backbone.pt --no-presence
run_abl abl_scratch
run_abl abl_nodn       --init runs/distill_n/backbone.pt --no-dn
log "queue done"
