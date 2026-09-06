"""Run a trained KESTREL checkpoint on images and write annotated copies.

This is the entry point for using the model rather than reproducing the study. It also exposes the anytime
decoder, so the per-query early exit can be tried on real images and its effect on executed depth observed:

    # plain full-depth inference on a folder
    python predict.py --ckpt runs/kestrel_n/best.pt --source images/ --out out/

    # anytime decoding: queries that are confidently background stop after the first layer.
    # "freeze" keeps them as attention keys, which is the mode the paper recommends.
    python predict.py --ckpt runs/kestrel_n/best.pt --source images/ --out out/ \
        --anytime --exit-mode freeze --min-layers 1 --exit-bg 0.1

    # a single image, no presence gate (the gate costs accuracy; see the paper)
    python predict.py --ckpt runs/kestrel_n/best.pt --source dog.jpg --gate-power 0

Any input size is accepted; images are letterboxed to a multiple of 32 and boxes are mapped back to the
original pixel coordinates.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from data import VOC_CLASSES, letterbox, normalize
from evaluate import load_checkpoint

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def collect(source: str) -> list:
    if os.path.isdir(source):
        return sorted(f for f in glob.glob(os.path.join(source, "**", "*"), recursive=True)
                      if f.lower().endswith(IMG_EXT))
    if any(ch in source for ch in "*?["):
        return sorted(f for f in glob.glob(source) if f.lower().endswith(IMG_EXT))
    return [source]


def colour(i: int) -> tuple:
    """Deterministic, reasonably distinct BGR colour per class index."""
    h = (i * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw(img: np.ndarray, boxes, scores, labels, names) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, labels):
        col = colour(int(c))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, col, 2)
        tag = f"{names[int(c)] if int(c) < len(names) else int(c)} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], max(0, p1[1] - th - 6)), (p1[0] + tw + 4, p1[1]), col, -1)
        cv2.putText(out, tag, (p1[0] + 2, max(th, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Run a trained KESTREL checkpoint on images.")
    ap.add_argument("--ckpt", required=True, help="checkpoint written by train.py (best.pt / last.pt)")
    ap.add_argument("--source", required=True, help="image file, directory, or glob")
    ap.add_argument("--out", default=None, help="directory for annotated images (omit to only print)")
    ap.add_argument("--size", type=int, default=512, help="inference size; rounded up to a multiple of 32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--conf", type=float, default=0.3, help="score threshold for a shown detection")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--names", default=None, help="JSON list of class names (default: the 20 VOC classes)")
    ap.add_argument("--gate-power", type=float, default=None,
                    help="presence-gate exponent: 1 = product (as designed), 0 = gate off. The gate costs AP")
    ap.add_argument("--no-reparam", action="store_true", help="skip folding multi-branch convs (slower, identical output)")
    # --- anytime decoding
    ap.add_argument("--anytime", action="store_true", help="enable the per-query early exit")
    ap.add_argument("--exit-mode", default="freeze", choices=["remove", "freeze"],
                    help="what happens to an exited query: 'freeze' keeps it as an attention key (recommended)")
    ap.add_argument("--min-layers", type=int, default=1, help="minimum decoder layers before a query may exit")
    ap.add_argument("--exit-bg", type=float, default=0.1, help="a query below this class probability stops (the rule that works)")
    ap.add_argument("--exit-p", type=float, default=1.1, help="foreground-exit threshold; >1 disables it (it does not help)")
    ap.add_argument("--exit-u", type=float, default=0.0, help="entropy threshold for the foreground rule")
    ap.add_argument("--json", default=None, help="also write detections to this JSON file")
    a = ap.parse_args()

    files = collect(a.source)
    if not files:
        sys.exit(f"no images found at {a.source!r}")
    names = json.load(open(a.names)) if a.names else VOC_CLASSES
    dev = torch.device(a.device)
    model, ck = load_checkpoint(a.ckpt, dev)
    if not a.no_reparam:
        model.reparameterize()
    if a.gate_power is not None:
        model.cfg.presence_power = a.gate_power
    model.cfg.exit_mode, model.cfg.exit_min_layers = a.exit_mode, a.min_layers
    model.cfg.exit_bg, model.cfg.exit_p, model.cfg.exit_u = a.exit_bg, a.exit_p, a.exit_u
    model.cfg.anytime_batched = False                      # the sequential path is the one that saves compute
    size = ((a.size + 31) // 32) * 32
    print(f"{a.ckpt}: {model.cfg.num_queries} queries, {model.cfg.dec_layers} decoder layers, "
          f"trained {ck.get('epoch', '?')} epochs | {len(files)} image(s) at {size} on {dev}"
          + (f" | anytime: {a.exit_mode}, min depth {a.min_layers}, tau_bg {a.exit_bg}" if a.anytime else ""),
          flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)

    records, depths, times = [], [], []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"  skip (unreadable): {path}"); continue
        h0, w0 = img.shape[:2]
        lb, ratio, (dx, dy) = letterbox(img, size)
        x = normalize(torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1])).permute(2, 0, 1)[None]).to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(x, return_masks=False, anytime=a.anytime)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - t0))

        scores = out["scores"][0]                                       # (K, C)
        k = min(a.max_det, scores.numel())
        top = scores.flatten().topk(k)
        qi, ci = top.indices // scores.shape[1], top.indices % scores.shape[1]
        keep = top.values > a.conf
        qi, ci, sc = qi[keep], ci[keep], top.values[keep]
        bx = out["boxes"][0][qi].clone()
        bx[:, [0, 2]] = ((bx[:, [0, 2]] - dx) / ratio).clamp(0, w0)     # undo the letterbox
        bx[:, [1, 3]] = ((bx[:, [1, 3]] - dy) / ratio).clamp(0, h0)
        bx, sc, ci = bx.cpu().numpy(), sc.cpu().numpy(), ci.cpu().numpy()

        depth = out["exit_layer"].float().mean().item() if a.anytime else float(model.cfg.dec_layers)
        depths.append(depth)
        print(f"  {os.path.basename(path):40s} {len(bx):3d} detections  {times[-1]:6.1f} ms"
              + (f"  mean depth {depth:.2f}/{model.cfg.dec_layers}" if a.anytime else ""), flush=True)
        records.append(dict(image=path, boxes=bx.tolist(), scores=sc.tolist(),
                            labels=[int(c) for c in ci], mean_depth=depth))
        if a.out:
            cv2.imwrite(os.path.join(a.out, os.path.basename(path)), draw(img, bx, sc, ci, names))

    if times:
        print(f"\n{len(times)} image(s): {np.mean(times):.1f} ms/image mean, {np.median(times):.1f} median"
              + (f" | mean decoder depth {np.mean(depths):.2f} of {model.cfg.dec_layers}" if a.anytime else ""))
    if a.json:
        json.dump(records, open(a.json, "w"), indent=1)
        print("wrote", a.json)
    if a.out:
        print("annotated images in", a.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
