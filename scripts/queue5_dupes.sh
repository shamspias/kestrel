#!/bin/bash
# Lane E: the duplicate / score-collapse analysis on the corrected main runs. It confirms on a properly trained model
# that removing exited queries destroys recall by collapsing the survivors' class scores (a train/test mismatch in the
# self-attention key set), not by breaking the one-to-one duplicate suppression that replaces NMS.
# Cheap, so it waits for the GPU to be free of training jobs before running.
# Usage: nohup bash scripts/queue5_dupes.sh > runs/queue5_dupes.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
d() { run=$1; sz=$2
  until [ -f runs/$run/best.pt ] && /usr/bin/grep -q "training complete" runs/$run.log 2>/dev/null; do sleep 120; done
  [ -f runs/$run/dupes.json ] && return
  while pgrep -f "python train.py" > /dev/null; do sleep 120; done
  log "duplicate / score-collapse analysis $run @ $sz"
  python scripts/duplicate_analysis.py --ckpt runs/$run/best.pt --size $sz --subset 600 --gate-power 0 \
    --configs full freeze:1:0.1 remove:1:0.1 freeze:1:0.2 remove:1:0.2 freeze:2:0.05 remove:2:0.05 \
    --out runs/$run/dupes.json > runs/$run/dupes.log 2>&1; }
d kestrel_n2 ${SZ_MAIN:-512}
d kestrel_s  ${SZ_MAIN:-512}
log "lane E done"
