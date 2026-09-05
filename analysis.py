"""Calibration analysis for the anytime exit: for every decoder layer l and every query, record the normalised
localisation entropy H_q^(l), the top class probability p_q^(l), the IoU with the (final-layer Hungarian-matched)
ground truth after layer l, and the IoU after the final layer. Writes a .npz that make_figures.py bins into
Fig. 'calibration' (IoU and IoU-gain vs entropy). Also records per-image object counts for the depth-vs-difficulty plot."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from assign import HungarianMatcher
from criterion import box_iou_diag
from data import normalize
from evaluate import build_test, load_checkpoint


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, ck = load_checkpoint(a.ckpt, dev)
    loader, gt_path, id_map, recs = build_test(a.size, subset=a.subset)
    matcher = HungarianMatcher()
    L = model.cfg.dec_layers
    H, P, IOU, MATCHED, IMG, NOBJ, FIN_P = [], [], [], [], [], [], []
    for imgs, boxes, labels, metas in loader:
        x = normalize(imgs.to(dev))
        out = model(x, return_masks=False)
        layers = out["aux"] + [dict(logits=out["logits"], boxes=out["boxes"], fdr=out["fdr"])]
        B, K = out["logits"].shape[:2]
        M = max(1, max(len(b) for b in boxes))
        gb = torch.zeros(B, M, 4, device=dev); gl = torch.zeros(B, M, dtype=torch.long, device=dev); gm = torch.zeros(B, M, dtype=torch.bool, device=dev)
        for i, (b, l) in enumerate(zip(boxes, labels)):
            gb[i, :len(b)], gl[i, :len(b)], gm[i, :len(b)] = b.to(dev), l.to(dev), True
        idx = matcher(layers[-1]["logits"], layers[-1]["boxes"], gl, gb, gm, (a.size, a.size))
        iou = torch.zeros(L, B, K, device=dev); matched = torch.zeros(B, K, dtype=torch.bool, device=dev)
        for b, (q, g) in enumerate(idx):
            if q.numel():
                q, g = q.to(dev), g.to(dev); matched[b, q] = True
                for l in range(L):
                    iou[l, b, q] = box_iou_diag(layers[l]["boxes"][b, q], gb[b, g])
        ent = torch.stack([model.decoder.fdr.entropy(layers[l]["fdr"]) for l in range(L)])               # (L, B, K)
        prob = torch.stack([layers[l]["logits"].sigmoid().amax(-1) for l in range(L)])                   # (L, B, K)
        H.append(ent.cpu().numpy()); P.append(prob.cpu().numpy()); IOU.append(iou.cpu().numpy()); MATCHED.append(matched.cpu().numpy())
        IMG.append(np.array([m["idx"] for m in metas]).repeat(K).reshape(B, K)); NOBJ.append(np.array([len(b) for b in boxes]).repeat(K).reshape(B, K))
    cat = lambda xs, ax: np.concatenate(xs, ax)
    np.savez_compressed(a.out, H=cat(H, 1), P=cat(P, 1), IOU=cat(IOU, 1), MATCHED=cat(MATCHED, 0), IMG=cat(IMG, 0), NOBJ=cat(NOBJ, 0), L=L)
    Hm, Im, Mm = cat(H, 1), cat(IOU, 1), cat(MATCHED, 0)
    for l in range(L - 1):
        h, i0, i1 = Hm[l][Mm], Im[l][Mm], Im[-1][Mm]
        for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1.01)):
            s = (h >= lo) & (h < hi)
            if s.sum():
                print(f"layer {l + 1}: H in [{lo:.1f},{hi:.1f}) n={s.sum():6d}  IoU@l={i0[s].mean():.3f}  IoU@L={i1[s].mean():.3f}  gain={(i1[s] - i0[s]).mean():+.3f}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
