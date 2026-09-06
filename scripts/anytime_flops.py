"""FLOPs actually executed by the anytime decoder, per exit mode, on real images.

The anytime policy's saving is usually reported as "mean decoder layers per query", which ignores that the two
exit modes do different amounts of work per executed layer: "remove" drops an exited query entirely, while
"freeze" keeps it as a self-attention key/value (so its key/value projection is still paid, but its own
attention, RoI gather, cross-attention and feed-forward are not). This script measures the difference by
running the real sequential implementation under PyTorch's FLOP counter on test images.

Usage: python scripts/anytime_flops.py --ckpt runs/kestrel_n/best.pt --size 512 --n 32 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import normalize                                    # noqa: E402
from evaluate import build_test, load_checkpoint              # noqa: E402


def flops_of(fn) -> int:
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n", type=int, default=32, help="test images to average over")
    ap.add_argument("--device", default="cpu", help="cpu keeps the count free of GPU contention; the count is device-independent")
    ap.add_argument("--exit", type=float, nargs=3, default=None, metavar=("P", "U", "BG"))
    ap.add_argument("--bg-grid", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--min-layers", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, ck = load_checkpoint(a.ckpt, dev)
    model.reparameterize()
    model.cfg.anytime_batched = False                          # the sequential path is the one that saves compute
    if a.exit:
        model.cfg.exit_p, model.cfg.exit_u, model.cfg.exit_bg = a.exit
    if a.min_layers is not None:
        model.cfg.exit_min_layers = a.min_layers
    L = model.cfg.dec_layers
    loader, _, _, _ = build_test(a.size, subset=a.n, workers=2, batch=1)
    imgs = [normalize(b[0].to(dev)) for b in loader][: a.n]
    print(f"{len(imgs)} images @ {a.size}, decoder layers {L}, queries {model.cfg.num_queries}", flush=True)

    out = {"ckpt": a.ckpt, "size": a.size, "n": len(imgs), "static": {}, "anytime": []}
    for l in range(1, L + 1):
        f = sum(flops_of(lambda: model(x, return_masks=False, max_layers=l)) for x in imgs) / len(imgs)
        out["static"][f"L{l}"] = f
        print(f"static  L={l}: {f / 1e9:7.2f} GFLOPs", flush=True)
    base = out["static"][f"L{L}"]
    for mode in ("remove", "freeze"):
        model.cfg.exit_mode = mode
        for bg in a.bg_grid:
            model.cfg.exit_bg = bg
            tot, depth = 0.0, 0.0
            for x in imgs:
                tot += flops_of(lambda: model(x, return_masks=False, anytime=True))
                with torch.no_grad():
                    depth += model(x, return_masks=False, anytime=True)["exit_layer"].float().mean().item()
            f, d = tot / len(imgs), depth / len(imgs)
            out["anytime"].append(dict(mode=mode, exit_bg=bg, exit_p=model.cfg.exit_p, exit_u=model.cfg.exit_u,
                                       min_layers=model.cfg.exit_min_layers, gflops=f / 1e9, mean_depth=d,
                                       saving_vs_full=1 - f / base))
            print(f"anytime {mode:6s} bg={bg:<5}: {f / 1e9:7.2f} GFLOPs  mean depth {d:.2f}  saving {100 * (1 - f / base):5.1f}% of the full-depth graph", flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
