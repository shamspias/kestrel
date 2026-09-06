#!/bin/bash
# Lane C: KESTREL-N with a 6-layer decoder instead of 3. The anytime mechanism's headroom is bounded by the decoder's
# share of compute (3.7 % of FLOPs for N with RoI attention), so this variant tests whether the saving scales with
# decoder depth — the trend that decides whether the mechanism matters at the M/L sizes we cannot train here.
# Starts only once lane A has finished the last ablation, so it never competes with the headline runs.
# Usage: nohup bash scripts/queue5_deep.sh > runs/queue5_deep.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EP_ABL=${EP_ABL:-30}; SZ_ABL=${SZ_ABL:-416}; EXIT=${EXIT:-0.6 0.15 0.05}
until /usr/bin/grep -q "training complete" runs/abl_nodn.log 2>/dev/null; do sleep 120; done; log "ablations finished"
until [ -f runs/RECIPE_N ]; do sleep 30; done; source runs/RECIPE_N
SWEEP="--static-sweep --anytime-sweep --sweep-mode remove freeze --sweep-min-layers 1 2 --sweep-policy confidence random --sweep-p 1.1 --sweep-u 0.0 --sweep-bg 0.05 0.2 0.5 --sweep-random-p 0.3 0.6 0.9 --anytime --exit $EXIT"
for try in 1 2 3; do /usr/bin/grep -q "training complete" runs/abl_deep.log 2>/dev/null && break
  log "run abl_deep (try $try): 6 decoder layers"
  python train.py --device cuda --amp $AMP_N --model N --bs $BS_N --lr $LR_N --size $SZ_ABL --epochs $EP_ABL \
    --dec-layers 6 --init runs/distill_n/backbone.pt --out runs/abl_deep --resume >> runs/abl_deep.log 2>&1; sleep 20; done
[ -f runs/abl_deep/eval.json ] || { log "eval abl_deep"; python evaluate.py --ckpt runs/abl_deep/best.pt --size $SZ_ABL --device cuda --reparam --workers 1 --gate-sweep $SWEEP --out runs/abl_deep/eval.json > runs/abl_deep/eval.log 2>&1
  python analysis.py --ckpt runs/abl_deep/best.pt --size $SZ_ABL --device cuda --out runs/abl_deep/calib.npz > runs/abl_deep/calib.log 2>&1; }
log "lane C done"
