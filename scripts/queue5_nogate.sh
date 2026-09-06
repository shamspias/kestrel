#!/bin/bash
# Lane D: the anytime sweeps again with the presence gate OFF, for every finished KESTREL run.
# The headline AP in the paper is the ungated one (the gate costs ~3 AP), so the anytime table must be ungated too,
# otherwise it mixes gated anytime rows with ungated static rows. make_tables.py prefers eval_nogate.json when present.
# Usage: nohup bash scripts/queue5_nogate.sh > runs/queue5_nogate.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EXIT=${EXIT:-0.6 0.15 0.05}
SWEEP="--static-sweep --anytime-sweep --sweep-mode remove freeze --sweep-min-layers 1 2 --sweep-policy confidence random --sweep-p 1.1 --sweep-u 0.0 --sweep-bg 0.05 0.1 0.2 0.3 --sweep-random-p 0.3 0.6 0.9 --anytime --exit $EXIT"
ng() { run=$1; sz=$2; until [ -f runs/$run/eval.json ]; do sleep 120; done
  [ -f runs/$run/eval_nogate.json ] && return
  while pgrep -f "evaluate.py --ckpt runs/$run" > /dev/null; do sleep 60; done      # let the gated sweep finish first
  log "ungated sweep $run @ $sz"
  python evaluate.py --ckpt runs/$run/best.pt --size $sz --device cuda --reparam --workers 1 --gate-power 0 \
    $SWEEP --out runs/$run/eval_nogate.json > runs/$run/eval_nogate.log 2>&1; }
ng kestrel_n2 ${SZ_MAIN:-512}
ng kestrel_s  ${SZ_MAIN:-512}
log "lane D done"
