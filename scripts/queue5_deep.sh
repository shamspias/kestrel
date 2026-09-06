#!/bin/bash
# Lane C: decoder-depth variants of KESTREL-N (6 layers instead of 3), with the RoI sampler and with deformable
# attention. The anytime mechanism's headroom is bounded by the decoder's share of compute (3.7 % of N's FLOPs with
# RoI attention, ~16 % with deformable), so these two runs measure whether the saving scales with decoder depth and
# with a more expensive sampler — the configuration where an anytime decoder should finally pay for itself end to end.
# Starts only after lane A's last ablation, so it never competes with the headline runs.
# Usage: nohup bash scripts/queue5_deep.sh > runs/queue5_deep.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EP_ABL=${EP_ABL:-30}; SZ_ABL=${SZ_ABL:-416}; EXIT=${EXIT:-0.6 0.15 0.05}
until /usr/bin/grep -q "training complete" runs/abl_nodn.log 2>/dev/null; do sleep 120; done; log "ablations finished"
until [ -f runs/RECIPE_N ]; do sleep 30; done; source runs/RECIPE_N
SWEEP="--static-sweep --anytime-sweep --sweep-mode remove freeze --sweep-min-layers 1 2 4 --sweep-policy confidence random --sweep-p 1.1 --sweep-u 0.0 --sweep-bg 0.05 0.2 0.5 --sweep-random-p 0.3 0.6 0.9 --anytime --exit $EXIT"
deep() { name=$1; shift
  for try in 1 2 3; do /usr/bin/grep -q "training complete" runs/$name.log 2>/dev/null && break
    log "run $name (try $try): 6 decoder layers $*"
    python train.py --device cuda --amp $AMP_N --model N --bs $BS_N --lr $LR_N --size $SZ_ABL --epochs $EP_ABL \
      --dec-layers 6 --init runs/distill_n/backbone.pt --out runs/$name --resume "$@" >> runs/$name.log 2>&1; sleep 20; done
  [ -f runs/$name/eval.json ] && return
  log "eval $name"
  python evaluate.py --ckpt runs/$name/best.pt --size $SZ_ABL --device cuda --reparam --workers 1 --gate-sweep $SWEEP --out runs/$name/eval.json > runs/$name/eval.log 2>&1
  python evaluate.py --ckpt runs/$name/best.pt --size $SZ_ABL --device cuda --reparam --workers 1 --gate-power 0 $SWEEP --out runs/$name/eval_nogate.json > runs/$name/eval_nogate.log 2>&1
  python analysis.py --ckpt runs/$name/best.pt --size $SZ_ABL --device cuda --out runs/$name/calib.npz > runs/$name/calib.log 2>&1
  python scripts/duplicate_analysis.py --ckpt runs/$name/best.pt --size $SZ_ABL --subset 600 --gate-power 0 \
    --configs full freeze:1:0.2 remove:1:0.2 freeze:2:0.05 remove:2:0.05 --out runs/$name/dupes.json > runs/$name/dupes.log 2>&1; }
deep abl_deep
deep abl_deep_deform --local-attn deform
log "lane C done"
