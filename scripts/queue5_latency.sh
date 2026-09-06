#!/bin/bash
# Lane G: the latency pass, which must run on an otherwise idle GPU and is therefore last. It waits for every other
# lane to finish and for the GPU to be free of training and evaluation jobs, then measures, for every finished run:
# PyTorch batch-1 latency at each static decoder depth, batch-1 anytime wall-clock in both exit modes, and TensorRT
# FP16 engines with CUDA-graph replay for the main models and the two ablations whose deployment graph differs.
# Usage: nohup bash scripts/queue5_latency.sh > runs/queue5_latency.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/tensorrt_libs
log() { echo "[$(date '+%F %T')] $*"; }
# 1. every lane reports "done" in its own log
for lane in kestrel yolo deep keydrop nogate dupes; do
  until /usr/bin/grep -q "lane . done\|queue.* done" runs/queue5_$lane.log 2>/dev/null; do sleep 300; done
  log "lane $lane finished"
done
# 2. and the GPU is genuinely quiet (a stray evaluation would corrupt the timings)
while pgrep -f "python (train|evaluate|analysis|scripts/duplicate)" > /dev/null || pgrep -f "train_yolo|eval_yolo" > /dev/null; do sleep 120; done
log "GPU idle — starting the latency pass"
FORCE=1 bash scripts/latency_all.sh
log "lane G done"
