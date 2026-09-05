# KESTREL

Real-time, NMS-free object detector with **entropy-gated anytime decoding**: every object query leaves the decoder as soon as its per-edge box distributions are sharp, and later decoder layers run only on the queries that are still uncertain. Design write-up: *From YOLO to Kestrel* (survey + proposal). Paper sources in `paper/`.

## Layout

| File | Purpose |
|---|---|
| `kestrel.py` | Architecture: re-parameterisable conv stages, windowed/global attention with registers + 2-D RoPE, PAN neck, dense seed head, RoI-gathered (or deformable, for ablation) decoder with FDR distributions, per-query anytime exit, presence head, foldable vocabulary, mask/keypoint heads, slot memory. |
| `data.py` | PASCAL VOC parsing, mosaic / affine / HSV / flip augmentation, letterbox, COCO-format GT writer. |
| `assign.py` | Task-aligned assigner (+ small-target rule) and Hungarian matcher. |
| `criterion.py` | Dense one-to-many loss, per-layer set loss, contrastive denoising, GO-LSD self-distillation, presence loss. |
| `train.py` | Detection training (AdamW, EMA, cosine, progressive loss weights). |
| `pretrain_distill.py` | Phase 1: distil a frozen DINOv2-S/14 teacher into the backbone's stride-16/32 ports. |
| `evaluate.py` | pycocotools evaluation, static-depth and anytime-threshold sweeps, batch-1 latency. |
| `analysis.py` | Entropy-vs-IoU calibration statistics for the anytime exit. |
| `baselines/` | Ultralytics YOLO26 / YOLO11 training and unified evaluation. |
| `scripts/` | Experiment queues (`queue_master.sh`, `eval_all.sh`). |
| `make_tables.py` | Results JSON → LaTeX tables, macros and figures for the paper. |

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision numpy pillow scipy pycocotools tqdm pyyaml matplotlib onnx opencv-python-headless ultralytics timm
bash scripts/download_voc.sh   # VOC07 trainval/test + VOC12 trainval → data/voc/VOCdevkit
python baselines/prepare_voc_yolo.py
```

## Reproduce

```bash
python pretrain_distill.py --model N --size 448 --epochs 10 --out runs/distill_n
python train.py --model N --size 512 --bs 8 --lr 1e-3 --epochs 60 --init runs/distill_n/backbone.pt --out runs/kestrel_n
python evaluate.py --ckpt runs/kestrel_n/best.pt --size 512 --reparam --static-sweep --anytime-sweep --latency --latency-anytime --exit 0.6 0.15 0.05 --out runs/kestrel_n/eval.json
python analysis.py --ckpt runs/kestrel_n/best.pt --out runs/kestrel_n/calib.npz
bash scripts/queue_master.sh   # everything (main run, ablations, baselines), sequentially
bash scripts/eval_all.sh && (cd paper && latexmk -pdf main.tex)
```

`python kestrel.py` runs the architecture self-test (shapes, FLOPs, exact re-parameterisation, ONNX export). All experiments in the paper were run on one Apple M2 Pro (16 GB) with the PyTorch MPS backend.
