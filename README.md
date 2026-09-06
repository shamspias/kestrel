# KESTREL

A real-time, NMS-free object detector with a **per-query anytime decoder**: every object query leaves the
decoder as soon as its own predictions stop changing, so different queries in the same image run different
numbers of decoder layers, and decoder cost tracks how hard the image is.

This repository is both the detector and the study of whether that idea works. It does, with two corrections
that turned out to matter more than the idea itself — see [Findings](#findings).

```bash
python predict.py --ckpt runs/kestrel_n/best.pt --source images/ --out out/ \
    --anytime --exit-mode freeze --min-layers 1 --exit-bg 0.1
# 000001.jpg   3 detections   47.6 ms   mean depth 1.18/3
```

---

## Contents

1. [Findings](#findings) · 2. [Benchmark results](#benchmark-results) · 3. [Install](#install) ·
4. [Run the model](#run-the-model) · 5. [Reproduce the study](#reproduce-the-study) ·
6. [Use your own data or model](#use-your-own-data-or-model) · 7. [Repository layout](#repository-layout) ·
8. [Status and caveats](#status-and-caveats)

---

## Findings

**1. Per-query early exit works, but only if exited queries stay visible.** The obvious implementation removes
an exited query from the remaining decoder layers. That is *worse than simply truncating the decoder*. Keeping
it as a frozen self-attention key — never recomputed, never hidden — recovers essentially all of the accuracy.
At a mean depth of 1.07 of 3 layers, freezing scores 40.62 AP and removal 27.73.

**2. Removal fails by collapsing scores, not by producing duplicates.** We expected removal to break the
one-to-one duplicate suppression that replaces NMS. It does the opposite: it produces *fewer* duplicates and
halves recall. The surviving queries' attention peer set shrinks below anything seen in training, and their
class scores collapse. Retaining **ten randomly chosen** exited queries as keys recovers 11.4 AP — it is the
*number* of keys that matters, not which ones.

**3. The exit rule helps when queries are frozen and hurts when they are removed.** Against a depth-matched
random control, the confidence rule is worth +1.4 AP with frozen context and **−4.9 AP** with removal, because
a good rule concentrates the damage on exactly the foreground queries that carry detections. A better policy
cannot rescue removal.

**4. Localisation entropy is not a useful exit signal.** The correlation between a query's per-edge
distribution entropy and the IoU it still gains from later layers is −0.05, and the weak trend runs backwards.
All of the usable saving comes from stopping confidently-*background* queries.

**5. Depth is not compute.** Cutting mean decoder depth by 54 % reduces network FLOPs by 1.2 %, because the
decoder is only 3.7 % of this model. On the measured-arithmetic axis the policy is worth **1.6 AP**, not the
5.9 the depth axis suggests. Anyone reporting an anytime detector in layers or exit rates should re-express it
in measured FLOPs before believing it.

**6. Two components did not earn their place.** The SAM-3-style presence gate costs 2.9 AP under average
precision, and initialising the backbone from a DINOv2 distillation stage is worth ~0 at a short schedule.

---

## Benchmark results

> **Preliminary.** Every number below comes from an 80-epoch run that used a learning rate the recipe
> comparison later showed was too high (see [Status](#status-and-caveats)). The corrected run and the full
> baseline suite are still training. Absolute accuracy will move; the comparisons will not, because each is an
> inference-time transformation of one checkpoint.

**Setup.** PASCAL VOC 2007+2012 `trainval` (16,551 images) → VOC 2007 `test` (4,952 images), 20 classes,
512², trained from scratch, single seed, one RTX 2080 Ti. Every model is scored by the *same* pycocotools
evaluator on the same ground-truth file.

### KESTREL-N

| | Params | GFLOPs @512 | AP | AP50 | AP75 |
|---|---|---|---|---|---|
| KESTREL-N, presence gate off | 5.31 M | 7.11 | **45.71** | **67.07** | **49.33** |
| KESTREL-N, product gate (as designed) | 5.31 M | 7.11 | 42.81 | 62.70 | 46.25 |
| dense one-to-many head + NMS | — | — | 35.92 | — | — |

### Anytime decoding (the paper's central result)

Mean decoder depth against AP, background-exit rule, presence gate on so every row is measured identically.

| Mode | Mean depth | AP | vs. static truncation at that depth |
|---|---|---|---|
| static truncation, ℓ=1 | 1.00 | 34.77 | — |
| static truncation, ℓ=2 | 2.00 | 41.15 | — |
| static truncation, ℓ=3 (full) | 3.00 | 42.81 | — |
| **freeze**, τ_bg 0.3 | 1.07 | **40.62** | **+5.9** |
| **freeze**, τ_bg 0.1 | 1.43 | **42.71** | dominates |
| remove, τ_bg 0.3 | 1.07 | 27.73 | −7.0 |
| remove, τ_bg 0.1 | 1.41 | 35.89 | −1.6 |

On the **measured FLOP** axis rather than the depth axis, freezing at 7.03 GFLOPs scores 42.71 where static
truncation at 7.02 GFLOPs scores 41.15 — a real but much smaller **+1.6 AP at equal arithmetic**.

### How many attention keys does a surviving query need?

Exited queries are removed from all computation; a random subset is left visible as keys (ungated, 2,000 images).

| Keys retained | 0 | 5 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|
| AP | 31.71 | 40.44 | 43.07 | 44.36 | 45.13 | 45.35 |

### What an anytime decoder can save, by architecture

Decoder share of total network FLOPs at 512² — the ceiling on any decoder-only anytime policy.

| Preset | RoI-gathered (ours) | Multi-scale deformable |
|---|---|---|
| N | 3.7 % | 16.1 % |
| S | 11.1 % | 24.8 % |
| M | 14.0 % | 30.5 % |
| L | 9.4 % | 21.8 % |

The export-friendly RoI sampler makes the decoder ~4× cheaper than deformable attention — and removes most of
the headroom an anytime policy could reclaim.

### Baselines

Trained under identical conditions (same data, size, epochs, GPU, evaluator). **Still training**; this table
is filled by `make_tables.py` as runs complete.

The suite spans six YOLO generations at two parameter tiers plus RT-DETR, the transformer set-prediction
family KESTREL itself belongs to. Run it with `bash scripts/queue_baselines_full.sh`.

| Tier | Models | Matched to | Status |
|---|---|---|---|
| nano (2.1–4.5 M) | YOLO26n, YOLO11n, YOLO12n, YOLOv10n, YOLOv9t, YOLOv8n, YOLOv6n, YOLOv5n | KESTREL-N (5.31 M) | YOLO26n at epoch 50/80 (intermediate AP 39.04); rest queued |
| transformer | RT-DETR-L (33 M) | KESTREL by *architecture*, not size — it has no nano/small variant | queued |
| small (7.3–11.2 M) | YOLO26s, YOLO11s, YOLOv10s, YOLOv9s, YOLOv8s | KESTREL-S (13.0 M) | queued |
| reference | YOLO26n, YOLO26s, COCO-pretrained | context only, **not** like-for-like | queued |

The full suite is roughly 173 GPU-hours on one RTX 2080 Ti. It is ordered so that stopping at any point still
leaves a coherent table, and `make_tables.py` fills rows as runs land.

Both the end-to-end and the NMS decoding paths are scored for YOLO26 and YOLOv10.

---

## Install

Requires a CUDA GPU (tested on an 11 GB RTX 2080 Ti, compute capability 7.5) and Python 3.12.

```bash
git clone -b experiments git@github.com:shamspias/kestrel.git && cd kestrel

curl -LsSf https://astral.sh/uv/install.sh | sh          # or use plain python -m venv
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy pillow scipy pycocotools tqdm pyyaml matplotlib onnx \
               opencv-python-headless ultralytics timm onnxruntime-gpu

python kestrel.py        # architecture self-test: shapes, FLOPs, exact re-parameterisation, ONNX export
```

Optional, for TensorRT latency measurement:

```bash
uv pip install tensorrt-cu12
ln -sf libnvinfer_plugin.so.11 .venv/lib/python3.12/site-packages/tensorrt_libs/libnvinfer_vc_plugin.so.11
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/tensorrt_libs
```

> Always run Ultralytics with `YOLO_AUTOINSTALL=False`. Its dependency check will otherwise pip-install
> packages that replace your PyTorch build.

---

## Run the model

`predict.py` is the entry point for using a trained checkpoint.

```bash
# full-depth inference on a folder, annotated copies written to out/
python predict.py --ckpt runs/kestrel_n/best.pt --source images/ --out out/

# a single image, presence gate off (the gate costs ~3 AP)
python predict.py --ckpt runs/kestrel_n/best.pt --source dog.jpg --gate-power 0

# anytime decoding: queries that are confidently background stop early
python predict.py --ckpt runs/kestrel_n/best.pt --source images/ --out out/ \
    --anytime --exit-mode freeze --min-layers 1 --exit-bg 0.1 --json dets.json
```

| Flag | What it does |
|---|---|
| `--anytime` | enable the per-query early exit; prints mean executed depth per image |
| `--exit-mode freeze\|remove` | **use `freeze`.** `remove` reproduces the failure mode documented above |
| `--exit-bg` | a query below this class probability stops. Higher = shallower and faster |
| `--min-layers` | minimum layers before any query may exit |
| `--exit-p` | foreground-exit threshold; `>1` disables it (it does not help) |
| `--gate-power` | presence gate exponent: `1` product, `0` off |
| `--size`, `--conf`, `--max-det`, `--names` | inference size, score threshold, cap, class-name JSON |

Any input resolution works — images are letterboxed to a multiple of 32 and boxes are mapped back to original
pixel coordinates. The exit thresholds are **inference-time knobs on one checkpoint**: no retraining is needed
to move along the accuracy/compute curve.

### Export

```python
from evaluate import load_checkpoint
from kestrel import ExportWrapper
import torch

model, _ = load_checkpoint("runs/kestrel_n/best.pt", torch.device("cpu"))
model.reparameterize()                      # folds every multi-branch conv into one dense conv
torch.onnx.export(ExportWrapper(model), torch.randn(1, 3, 512, 512), "kestrel.onnx", opset_version=17)
```

The exported graph contains convolutions, matmuls, softmax, norms, `RoiAlign` and a `TopK` — no NMS, no
`grid_sample`, no FlashAttention dependency. Folding changes outputs by at most 6e-5.

---

## Reproduce the study

```bash
# 1. data
bash scripts/download_voc.sh          # VOC07 trainval/test + VOC12 trainval → data/voc/VOCdevkit
python baselines/prepare_voc_yolo.py  # YOLO-format mirror for the baselines

# 2. optional phase-1 backbone distillation from DINOv2-S (worth ~0 at short schedules; see Findings)
python pretrain_distill.py --model N --size 448 --bs 16 --lr 5e-4 --epochs 10 \
    --device cuda --out runs/distill_n

# 3. train
python train.py --model N --size 512 --bs 16 --lr 1e-3 --epochs 80 --device cuda --amp fp16 \
    --init runs/distill_n/backbone.pt --out runs/kestrel_n2

# 4. evaluate: accuracy, presence-gate sweep, static-depth ladder, anytime sweep over both exit modes
python evaluate.py --ckpt runs/kestrel_n2/best.pt --size 512 --device cuda --reparam \
    --gate-sweep --static-sweep --anytime-sweep \
    --sweep-mode remove freeze --sweep-min-layers 1 2 \
    --sweep-p 1.1 --sweep-u 0.0 --sweep-bg 0.05 0.1 0.2 0.3 0.5 \
    --out runs/kestrel_n2/eval.json

# 5. the analyses behind the findings
python analysis.py --ckpt runs/kestrel_n2/best.pt --size 512 --device cuda --out runs/kestrel_n2/calib.npz
python scripts/duplicate_analysis.py --ckpt runs/kestrel_n2/best.pt --size 512 --gate-power 0
python scripts/decoder_share.py --out runs/decoder_share.json
python scripts/anytime_flops.py --ckpt runs/kestrel_n2/best.pt --size 512 --device cpu --min-layers 1

# 6. baselines, identical conditions
YOLO_AUTOINSTALL=False python baselines/train_yolo.py yolo26n.yaml --epochs 80 --imgsz 512 \
    --batch 32 --device 0 --amp --name yolo26n_scratch
python baselines/eval_yolo.py runs/baselines/yolo26n_scratch/weights/best.pt --imgsz 512 --device 0 \
    --out runs/baselines/yolo26n_scratch.json

# 7. latency — on an IDLE GPU only
bash scripts/latency_all.sh

# 8. tables and figures
python make_tables.py && python make_figures.py
```

Whole-pipeline drivers live in `scripts/queue*.sh`; `bash scripts/status.sh` prints a one-shot view of every
run, the GPU, and which evaluations have completed.

**Training knobs that matter.** `--lr 1e-3` at batch 16 (2e-3 is ~10 AP worse; do not scale linearly from a
different batch size without measuring), `--amp fp16`, `--key-dropout` to train the decoder against varying
attention peer-set sizes, and the ablation switches `--no-golsd`, `--no-presence`, `--no-dn`,
`--local-attn deform`.

---

## Use your own data or model

### Your own dataset

Training reads a list of plain dicts, so a new dataset needs one function. Each record is:

```python
{"id": "000001", "file": "/abs/path/000001.jpg", "width": 500, "height": 375,
 "boxes": np.ndarray((N, 4), np.float32),   # xyxy, absolute pixels
 "labels": np.ndarray((N,), np.int64),      # 0-based class indices
 "difficult": np.ndarray((N,), np.int64)}   # 1 = ignored by the metric, dropped from training
```

Write a loader that returns `list[record]`, then:

1. point `train.py` / `evaluate.py` at it instead of `data.load_split`;
2. pass your class count to `build_model(name, num_classes=...)`;
3. `data.write_coco_gt(records, path)` produces the COCO-format ground truth the evaluator needs, so your
   dataset is scored by exactly the same code as everything else here.

Nothing else is VOC-specific — there are no absolute position tables, so any input whose sides are multiples
of 32 works at its native aspect ratio.

### Your own model, benchmarked against KESTREL

The point of `baselines/eval_yolo.py` is that a competitor is scored by *our* evaluator rather than its own,
which is what makes the comparison meaningful. To add a model, produce detections in COCO format and call the
shared scorer:

```python
from data import load_split, write_coco_gt
from evaluate import coco_eval

recs = load_split("data/voc", [("2007", "test")], cache="data/voc/cache_test07.json")
id_map = write_coco_gt(recs, "data/voc/coco_gt_test07.json")

results = []
for r in recs:
    for (x1, y1, x2, y2), cls, score in your_model.predict(r["file"]):
        results.append(dict(image_id=id_map[r["id"]], category_id=int(cls) + 1,
                            bbox=[x1, y1, x2 - x1, y2 - y1], score=float(score)))

print(coco_eval(results, "data/voc/coco_gt_test07.json"))
```

Boxes are absolute xyxy pixels in the *original* image, `category_id` is 1-based. Add the run to `BASELINES`
in `make_tables.py` and it appears in the generated table.

### A new KESTREL size

Presets live in `evaluate.PRESETS`. Add an entry and it is constructible everywhere:

```python
PRESETS["XS"] = dict(stem_ch=12, conv_dims=(24, 48), conv_depths=(1, 2), attn_dims=(96, 128),
                     attn_depths=(2, 2), neck_dim=96, d_model=96, embed_dim=96,
                     dec_layers=2, dec_heads=3, num_queries=100)
```

`python scripts/decoder_share.py --presets XS` then tells you what share of it the decoder is — which,
per finding 5, is the number that decides whether an anytime decoder is worth having at all.

---

## Repository layout

| File | Purpose |
|---|---|
| `kestrel.py` | architecture: re-parameterisable conv stages, windowed/global attention with registers and 2-D RoPE, PAN neck, dense seed head, RoI-gathered decoder with per-edge distributions, **per-query anytime exit** (`forward_anytime`, and an equivalent fast `forward_anytime_batched` for scoring), presence head, foldable vocabulary, mask/keypoint heads, slot memory |
| `train.py` | detection training: AdamW, EMA, cosine, progressive dense→one-to-one weighting, denoising, key dropout |
| `evaluate.py` | the shared pycocotools evaluator, gate/static/anytime sweeps, batch-1 latency |
| `predict.py` | run a checkpoint on images, with the anytime decoder exposed |
| `data.py` · `assign.py` · `criterion.py` | VOC pipeline and augmentation · TAL + Hungarian assignment · all losses incl. GO-LSD |
| `pretrain_distill.py` | phase-1 DINOv2-S distillation into the backbone |
| `analysis.py` | entropy-vs-IoU exit calibration |
| `scripts/decoder_share.py` | decoder share of network FLOPs, per preset and sampler |
| `scripts/anytime_flops.py` | FLOPs *actually executed* by the anytime decoder |
| `scripts/duplicate_analysis.py` | duplicate rate, recall and score collapse per exit mode |
| `scripts/trt_latency.py` | TensorRT FP16 engines with CUDA-graph replay, for KESTREL and YOLO |
| `scripts/latency_all.sh` · `scripts/status.sh` · `scripts/queue*.sh` | idle-GPU latency pass · pipeline status · experiment drivers |
| `baselines/` | Ultralytics training and **evaluation through our scorer** |
| `make_tables.py` · `make_figures.py` | result JSONs → LaTeX tables and figures |

---

## Status and caveats

- **Results are preliminary.** The 80-epoch run above used lr 2e-3, which the recipe comparison shows is
  ~10 AP worse than 1e-3 at a short schedule. The corrected run, six ablations, KESTREL-S and six of seven
  baselines are still training.
- **Single seed, no error bars.** Differences below a few tenths of an AP point are not resolvable here.
- **VOC, not COCO.** The design targets COCO; the available GPU cannot train it in a workable time. VOC has
  fewer objects per image, which is the regime where the background-exit rule does most of its work.
- **Latency is one consumer GPU.** Not comparable to published T4 tables. Within this repo it is fair, because
  everything is timed identically on the same device.
- **Untrained parts.** Open-vocabulary prompt folding, masks, keypoints and the video slot memory are
  implemented and shape-checked but never trained; no claim is made about them.
- **Not a novel mechanism.** Per-query routing in a DETR decoder, halt-and-copy, entropy-thresholded exit and
  exit-consistency distillation are all published. The contribution here is measurement.
