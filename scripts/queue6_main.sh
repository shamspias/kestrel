#!/bin/bash
# Lane A (rev 2): every KESTREL training run, in strict order of value to the paper, so that stopping at any point
# still leaves a coherent result set. Replaces queue5_kestrel.sh, queue5_deep.sh and queue5_keydrop.sh, which between
# them ordered the constructive fix and the deep-decoder variant AFTER six ablations and would not have reached them.
#
# Why the budget changed: measured throughput under two-lane contention is about 10.6 min per epoch at 416 px and
# 13.6 at 512, so the original plan (six ablations at 30 epochs, plus 12-config sweeps after each) came to roughly
# 90 GPU-hours for lane A alone. Ablations are now 15 epochs, and the per-run sweeps are cut to what the tables
# actually print. The comparison stays valid because every ablation row, including the reference, uses this schedule.
# Usage: nohup bash scripts/queue6_main.sh > runs/queue6_main.log 2>&1 &
set -u; cd "$(dirname "$0")/.."; source .venv/bin/activate
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=False PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] $*"; }
EP_MAIN=${EP_MAIN:-80}; EP_ABL=${EP_ABL:-15}; SZ_MAIN=${SZ_MAIN:-512}; SZ_ABL=${SZ_ABL:-416}; EP_AB=${EP_AB:-10}; WK=${WK:-5}

# Sweeps. The foreground/entropy rule is inert (tau_p 1.1, tau_H 0 disable it), so only the background rule is swept.
# Headline run: both exit modes, both minimum depths, four thresholds — 16 anytime configurations.
MAINSWEEP="--static-sweep --anytime-sweep --sweep-mode remove freeze --sweep-min-layers 1 2 --sweep-p 1.1 --sweep-u 0.0 --sweep-bg 0.05 0.1 0.2 0.3"
# Ablation rows print full-depth AP and one anytime operating point, so six configurations suffice.
ABLSWEEP="--static-sweep --anytime-sweep --sweep-mode remove freeze --sweep-min-layers 1 --sweep-p 1.1 --sweep-u 0.0 --sweep-bg 0.1 0.2 0.3"

k() { name=$1; shift; for try in 1 2 3; do /usr/bin/grep -q "training complete" runs/$name.log 2>/dev/null && break
      log "run $name (try $try): $*"
      python train.py --device cuda --amp $AMP_N --workers $WK --out runs/$name --resume "$@" >> runs/$name.log 2>&1; sleep 20; done; }
# evaluation in the background so it overlaps the next training job; latency is NOT measured here (needs an idle GPU)
ev() { run=$1; sz=$2; shift 2; [ -f runs/$run/best.pt ] || return; [ -f runs/$run/eval.json ] && return
      log "eval (bg) $run @ $sz"
      (python evaluate.py --ckpt runs/$run/best.pt --size $sz --device cuda --reparam --workers 1 --gate-sweep --out runs/$run/eval.json "$@" > runs/$run/eval.log 2>&1
       python evaluate.py --ckpt runs/$run/best.pt --size $sz --device cuda --reparam --workers 1 --gate-power 0 "$@" --out runs/$run/eval_nogate.json > runs/$run/eval_nogate.log 2>&1
       python analysis.py --ckpt runs/$run/best.pt --size $sz --device cuda --out runs/$run/calib.npz > runs/$run/calib.log 2>&1) & }

# 0. the last recipe arm, then the recipe itself
while pgrep -f "train.py .*runs/ab_lr" > /dev/null; do sleep 30; done
AMP_N=none k ab_fp32_lr1e3 --model N --bs 16 --size $SZ_ABL --epochs $EP_AB --eval-every 5 --lr 1e-3 --init runs/distill_n/backbone.pt
log "recipe arms done — waiting for runs/RECIPE_N"
until [ -f runs/RECIPE_N ]; do sleep 30; done; source runs/RECIPE_N
log "recipe: N=$AMP_N/bs$BS_N/lr$LR_N  S=$AMP_S/bs$BS_S/lr$LR_S"
N="--model N --bs $BS_N --lr $LR_N"; A="$N --size $SZ_ABL --epochs $EP_ABL --init runs/distill_n/backbone.pt"

# 1. the headline model
k kestrel_n2 $N --size $SZ_MAIN --epochs $EP_MAIN --init runs/distill_n/backbone.pt
ev kestrel_n2 $SZ_MAIN $MAINSWEEP
# 2. the ablations the paper's claims rest on: the reference row, the GO-LSD calibration claim, RoI versus deformable
k abl_full    $A;                     ev abl_full    $SZ_ABL $ABLSWEEP
k abl_nogolsd $A --no-golsd;          ev abl_nogolsd $SZ_ABL $ABLSWEEP
k abl_deform  $A --local-attn deform; ev abl_deform  $SZ_ABL $ABLSWEEP
# 3. the constructive fix, and whether the mechanism scales with decoder depth
k abl_keydrop $A --key-dropout 0.9;   ev abl_keydrop $SZ_ABL $ABLSWEEP
k abl_deep    $A --dec-layers 6;      ev abl_deep    $SZ_ABL $ABLSWEEP
# 4. KESTREL-S: the size where the decoder is a larger share of compute, and the second row of the main table
/usr/bin/grep -q "epoch 9 saved" runs/distill_s.log 2>/dev/null || { log "distill S"
  python pretrain_distill.py --model S --size 448 --bs 16 --lr 5e-4 --epochs 10 --device cuda --workers $WK --out runs/distill_s >> runs/distill_s.log 2>&1; }
AMP_N=$AMP_S k kestrel_s --model S --bs $BS_S --lr $LR_S --size $SZ_MAIN --epochs $EP_MAIN --init runs/distill_s/backbone.pt
ev kestrel_s $SZ_MAIN $MAINSWEEP
# 5. the remaining ablation rows, in descending order of what they settle
k abl_scratch    $N --size $SZ_ABL --epochs $EP_ABL;  ev abl_scratch    $SZ_ABL $ABLSWEEP
k abl_nopresence $A --no-presence;                    ev abl_nopresence $SZ_ABL $ABLSWEEP
k abl_nodn       $A --no-dn;                          ev abl_nodn       $SZ_ABL $ABLSWEEP
# 6. second-order variants, run only if everything above has finished
k abl_keydrop_mild  $A --key-dropout 0.5;                    ev abl_keydrop_mild  $SZ_ABL $ABLSWEEP
k abl_deep_deform   $A --dec-layers 6 --local-attn deform;   ev abl_deep_deform   $SZ_ABL $ABLSWEEP
wait; log "lane A done"
