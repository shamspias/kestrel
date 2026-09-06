"""Does removing exited queries break NMS-free duplicate suppression?

A one-to-one-matched decoder has no NMS: query self-attention is what stops two queries from claiming the same
object. If exited queries are deleted from later layers, the surviving queries can no longer see them, and the
prediction is that survivors re-converge onto objects the exited queries had already claimed -- i.e. duplicates.
Freezing the exited queries as self-attention keys should prevent this.

This script tests that prediction directly: for a set of exit configurations it counts, among confident
detections, how many are duplicates (same predicted class, IoU above a threshold, with a higher-scoring
detection in the same image), and reports recall over ground-truth objects so a drop in duplicates cannot be
confused with a drop in detections.

Usage:
  python scripts/duplicate_analysis.py --ckpt runs/kestrel_n/best.pt --size 512 --subset 500 \
      --configs "full" "freeze:1:0.2" "remove:1:0.2" "freeze:2:0.05" "remove:2:0.05" --out runs/kestrel_n/dupes.json
Each config is MODE:MIN_LAYERS:TAU_BG, or the literal "full" for the unmodified full-depth forward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torchvision.ops import box_iou

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import normalize                                     # noqa: E402
from evaluate import build_test, load_checkpoint                # noqa: E402


@torch.no_grad()
def measure(model, loader, dev, conf=0.30, iou_thr=0.70, gt_iou=0.5, anytime=False):
    n_det = n_dup = n_img = 0
    gt_total = gt_hit = 0
    depths = []
    for imgs, boxes_gt, labels_gt, _ in loader:
        x = normalize(imgs.to(dev))
        out = model(x, return_masks=False, anytime=anytime)
        if anytime:
            depths.append(out["exit_layer"].float().mean(1).cpu())
        scores, boxes = out["scores"], out["boxes"]            # (B, K, C), (B, K, 4)
        conf_q, cls_q = scores.max(-1)
        for b in range(boxes.shape[0]):
            keep = conf_q[b] > conf
            if keep.sum() == 0:
                n_img += 1
                gt_total += len(boxes_gt[b])
                continue
            bx, sc, cl = boxes[b][keep], conf_q[b][keep], cls_q[b][keep]
            order = sc.argsort(descending=True)
            bx, sc, cl = bx[order], sc[order], cl[order]
            iou = box_iou(bx, bx)
            same = cl[:, None] == cl[None, :]
            # a detection is a duplicate if a HIGHER-scoring detection of the same class overlaps it
            higher = torch.triu(torch.ones_like(iou, dtype=torch.bool), diagonal=1).t()
            dup = ((iou > iou_thr) & same & higher).any(1)
            n_det += bx.shape[0]
            n_dup += int(dup.sum())
            # recall: a ground-truth object is hit if any kept detection of its class overlaps it
            g, gl = boxes_gt[b].to(dev), labels_gt[b].to(dev)
            gt_total += g.shape[0]
            if g.shape[0] and bx.shape[0]:
                m = (box_iou(g, bx) > gt_iou) & (gl[:, None] == cl[None, :])
                gt_hit += int(m.any(1).sum())
            n_img += 1
    d = torch.cat(depths).mean().item() if depths else float(model.cfg.dec_layers)
    return dict(images=n_img, detections=n_det, duplicates=n_dup,
                dup_rate=n_dup / max(1, n_det), dets_per_image=n_det / max(1, n_img),
                dups_per_image=n_dup / max(1, n_img), recall=gt_hit / max(1, gt_total), mean_depth=d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--subset", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conf", type=float, default=0.30, help="score threshold for a 'confident' detection")
    ap.add_argument("--iou", type=float, default=0.70, help="IoU above which two same-class detections are duplicates")
    ap.add_argument("--gate-power", type=float, default=None)
    ap.add_argument("--configs", nargs="+", default=["full", "freeze:1:0.2", "remove:1:0.2", "freeze:2:0.05", "remove:2:0.05"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, ck = load_checkpoint(a.ckpt, dev)
    model.reparameterize()
    model.cfg.anytime_batched = True
    if a.gate_power is not None:
        model.cfg.presence_power = a.gate_power
    loader, _, _, _ = build_test(a.size, subset=a.subset, workers=2, batch=8)
    model.cfg.exit_p, model.cfg.exit_u = 1.1, 0.0                # background rule only
    rows = []
    for cfg in a.configs:
        if cfg == "full":
            r = measure(model, loader, dev, a.conf, a.iou, anytime=False)
            r.update(config="full", mode="-", min_layers="-", exit_bg="-")
        else:
            mode, ml, bg = cfg.split(":")
            model.cfg.exit_mode, model.cfg.exit_min_layers, model.cfg.exit_bg = mode, int(ml), float(bg)
            r = measure(model, loader, dev, a.conf, a.iou, anytime=True)
            r.update(config=cfg, mode=mode, min_layers=int(ml), exit_bg=float(bg))
        rows.append(r)
        print(f"{cfg:16s} depth {r['mean_depth']:.2f} | {r['dets_per_image']:5.2f} det/img | "
              f"{r['dups_per_image']:5.3f} dup/img | dup rate {100 * r['dup_rate']:5.2f}% | recall {100 * r['recall']:5.2f}%",
              flush=True)
    if a.out:
        json.dump(dict(ckpt=a.ckpt, size=a.size, subset=a.subset, conf=a.conf, iou=a.iou, rows=rows), open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
