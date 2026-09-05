#!/bin/bash
# Evaluate every finished run: full/static/anytime sweeps + latency for KESTREL, unified scoring for baselines, calibration npz.
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate; export PYTHONUNBUFFERED=1
log() { echo "[$(date '+%F %T')] $*"; }
SZ_MAIN=${SZ_MAIN:-512}; SZ_ABL=${SZ_ABL:-416}
ev() { run=$1; sz=$2; shift 2
  [ -f runs/$run/best.pt ] || { log "skip $run (no best.pt)"; return; }
  [ -f runs/$run/eval.json ] && [ "${FORCE:-0}" = 0 ] && { log "skip $run (eval.json exists)"; return; }
  log "eval $run @ $sz"; python evaluate.py --ckpt runs/$run/best.pt --size $sz --reparam --out runs/$run/eval.json "$@" > runs/$run/eval.log 2>&1
  [ -f runs/$run/calib.npz ] || python analysis.py --ckpt runs/$run/best.pt --size $sz --out runs/$run/calib.npz > runs/$run/calib.log 2>&1; }
ev kestrel_n $SZ_MAIN --static-sweep --anytime-sweep --latency --latency-anytime --exit ${EXIT:-0.6 0.15 0.05}
for r in abl_full abl_nogolsd abl_deform abl_nopresence abl_scratch abl_nodn; do ev $r $SZ_ABL --static-sweep --anytime-sweep --anytime --exit ${EXIT:-0.6 0.15 0.05}; done
for b in yolo26n_scratch yolo11n_scratch yolo26n_coco; do
  w=runs/baselines/$b/weights/best.pt; [ -f $w ] || continue
  [ -f runs/baselines/$b.json ] || python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --latency --out runs/baselines/$b.json > runs/baselines/$b.eval.log 2>&1
  case $b in yolo26*) [ -f runs/baselines/${b}_nms.json ] || python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --end2end false --out runs/baselines/${b}_nms.json > runs/baselines/${b}_nms.eval.log 2>&1;; esac
done
python make_tables.py && log "tables done"
