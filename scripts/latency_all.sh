#!/bin/bash
# Latency pass — run ONLY on an otherwise idle GPU (after the training queues are done).
#
# This is the only place the paper's compute claim can be settled. Mean decoder depth is NOT a compute axis: a frozen
# query still costs its key/value projections in every later layer, every layer pays the exit test, and the anytime path
# runs a Python loop with dynamic indexing. So we time the operating points the paper actually reports (background rule
# only, minimum depth 1) against the static-depth ladder, rather than converting a depth reduction into a saving.
#
# KESTREL: batch-1 PyTorch latency at every static depth; batch-1 anytime wall-clock at each reported operating point in
#          both exit modes; TensorRT FP16 engines per depth with CUDA-graph replay for the main models.
# YOLO:    batch-1 PyTorch raw forward and a TensorRT FP16 engine, timed by the same runner.
# Usage: bash scripts/latency_all.sh > runs/latency_all.log 2>&1
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/tensorrt_libs
log() { echo "[$(date '+%F %T')] $*"; }
SZ_MAIN=${SZ_MAIN:-512}; SZ_ABL=${SZ_ABL:-416}
if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "llama|^$" | grep -q .; then
  log "WARNING: other compute processes are on the GPU — latency numbers would be unreliable"; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; [ "${FORCE:-0}" = 1 ] || exit 1
fi
# Reported operating points as MODE:MIN_LAYERS:TAU_BG. tau_p 1.1 and tau_H 0 disable the foreground/entropy rule, so
# these are the background rule alone — the policy the results section reports.
POINTS="freeze:1:0.1 freeze:1:0.2 freeze:1:0.3 remove:1:0.1 remove:1:0.2 freeze:2:0.05 remove:2:0.05"
# TensorRT engines take 10-15 min each to build on this GPU, so: all depths for the main models, final depth only for the
# ablations whose deployment graph differs (RoI vs deformable, 3 vs 6 decoder layers), none for the rest.
lat() { run=$1; sz=$2; trt=${3:-none}; [ -f runs/$run/best.pt ] || { log "skip $run"; return; }
  log "latency $run @ $sz (torch, static ladder)"
  python evaluate.py --ckpt runs/$run/best.pt --size $sz --device cuda --reparam --workers 1 --update --skip-full \
    --latency --out runs/$run/eval.json > runs/$run/latency.log 2>&1
  for pt in $POINTS; do
    md=${pt%%:*}; rest=${pt#*:}; ml=${rest%%:*}; bg=${rest##*:}
    log "latency $run @ $sz (torch, anytime $md min_l=$ml tau_bg=$bg)"
    python evaluate.py --ckpt runs/$run/best.pt --size $sz --device cuda --reparam --workers 1 --update --skip-full \
      --latency-anytime --exit-mode $md --exit-min-layers $ml --exit 1.1 0.0 $bg \
      --out runs/$run/lat_${md}_${ml}_${bg}.json >> runs/$run/latency.log 2>&1
  done
  [ "$trt" = none ] && return
  log "latency $run @ $sz (TensorRT, $trt)"
  python scripts/trt_latency.py kestrel --ckpt runs/$run/best.pt --size $sz $([ "$trt" = all ] && echo --all-depths) --out runs/$run/trt.json > runs/$run/trt.log 2>&1 || log "  TensorRT failed for $run (see runs/$run/trt.log)"; }
latb() { name=$1; w=runs/baselines/$name/weights/best.pt; [ -f $w ] || { log "skip $name"; return; }
  log "latency baseline $name (torch)"; python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --device 0 --latency --update --out runs/baselines/$name.json > runs/baselines/$name.latency.log 2>&1
  log "latency baseline $name (TensorRT)"; python scripts/trt_latency.py yolo --ckpt $w --size $SZ_MAIN --out runs/baselines/$name.trt.json > runs/baselines/$name.trt.log 2>&1 || log "  TensorRT failed for $name"; }
lat kestrel_n2 $SZ_MAIN all; lat kestrel_s $SZ_MAIN all; lat kestrel_n $SZ_MAIN all
lat abl_full $SZ_ABL final; lat abl_deform $SZ_ABL final; lat abl_deep $SZ_ABL final; lat abl_deep_deform $SZ_ABL final
for r in abl_nogolsd abl_nopresence abl_scratch abl_nodn abl_keydrop abl_keydrop_mild; do lat $r $SZ_ABL; done
for b in yolo26n_scratch yolo11n_scratch yolo26n_coco yolo12n_scratch yolov10n_scratch yolov9t_scratch yolov8n_scratch; do latb $b; done
log "latency pass done"
