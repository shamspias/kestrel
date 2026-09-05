#!/bin/bash
# Latency pass — run ONLY on an otherwise idle GPU (after the training queue is done).
# KESTREL: PyTorch batch-1 (static depths + anytime) merged into eval.json, and TensorRT FP16 engines per depth → trt.json
# YOLO:    PyTorch batch-1 raw forward merged into <name>.json, and TensorRT FP16 engine → <name>.trt.json
# Usage: bash scripts/latency_all.sh > runs/latency_all.log 2>&1
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/tensorrt_libs
log() { echo "[$(date '+%F %T')] $*"; }
SZ_MAIN=${SZ_MAIN:-512}; SZ_ABL=${SZ_ABL:-416}; EXIT=${EXIT:-0.6 0.15 0.05}
if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "llama|^$" | grep -q .; then
  log "WARNING: other compute processes are on the GPU — latency numbers would be unreliable"; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader; [ "${FORCE:-0}" = 1 ] || exit 1
fi
# TensorRT engines take 10-15 min each to build on this GPU, so: all depths for the main models, final depth only for the
# two ablations whose deployment graph differs (RoI vs deformable), none for the rest (same graph as abl_full).
lat() { run=$1; sz=$2; trt=${3:-none}; [ -f runs/$run/best.pt ] || { log "skip $run"; return; }
  log "latency $run @ $sz (torch)"; python evaluate.py --ckpt runs/$run/best.pt --size $sz --device cuda --reparam --update --skip-full --latency --latency-anytime --exit $EXIT --out runs/$run/eval.json > runs/$run/latency.log 2>&1
  [ "$trt" = none ] && return
  log "latency $run @ $sz (TensorRT, $trt)"; python scripts/trt_latency.py kestrel --ckpt runs/$run/best.pt --size $sz $([ "$trt" = all ] && echo --all-depths) --out runs/$run/trt.json > runs/$run/trt.log 2>&1 || log "  TensorRT failed for $run (see runs/$run/trt.log)"; }
latb() { name=$1; w=runs/baselines/$name/weights/best.pt; [ -f $w ] || { log "skip $name"; return; }
  log "latency baseline $name (torch)"; python baselines/eval_yolo.py $w --imgsz $SZ_MAIN --device 0 --latency --update --out runs/baselines/$name.json > runs/baselines/$name.latency.log 2>&1
  log "latency baseline $name (TensorRT)"; python scripts/trt_latency.py yolo --ckpt $w --size $SZ_MAIN --out runs/baselines/$name.trt.json > runs/baselines/$name.trt.log 2>&1 || log "  TensorRT failed for $name"; }
lat kestrel_n $SZ_MAIN all; lat kestrel_s $SZ_MAIN all
lat abl_full $SZ_ABL final; lat abl_deform $SZ_ABL final
for r in abl_nogolsd abl_nopresence abl_scratch abl_nodn; do lat $r $SZ_ABL; done
for b in yolo26n_scratch yolo11n_scratch yolo12n_scratch yolov10n_scratch yolov9t_scratch yolov8n_scratch yolo26n_coco; do latb $b; done
log "latency pass done"
