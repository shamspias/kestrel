"""How much of a KESTREL forward pass is the decoder?

The per-query anytime exit can only save what the decoder costs, so the decoder's share of the whole network
bounds the end-to-end benefit. This measures that share for every size preset, at several resolutions, for both
local-attention variants (RoI-gathered vs multi-scale deformable) and for a range of query counts — which is
what says at which scale an anytime decoder is worth having.

Usage: python scripts/decoder_share.py --out runs/decoder_share.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import PRESETS, build_model                      # noqa: E402
from kestrel import count_params                               # noqa: E402

# The reference implementation ships N/S/M; L is the fourth rung of the design's size ladder, registered here so
# build_model() can construct it like any other preset.
PRESETS.setdefault("L", dict(stem_ch=48, conv_dims=(96, 192), conv_depths=(3, 6), attn_dims=(384, 512),
                             attn_depths=(8, 6), neck_dim=320, d_model=320, embed_dim=320, dec_layers=6,
                             dec_heads=10, num_queries=300))


def gflops(model, x, **kw) -> float:
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x, return_masks=False, **kw)
    return fc.get_total_flops() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 640])
    ap.add_argument("--presets", nargs="+", default=["N", "S", "M", "L"])
    ap.add_argument("--queries", type=int, nargs="+", default=None, help="also sweep query count on the M preset")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for name in a.presets:
        for local in ("roi", "deform"):
            model = build_model(name, 20, local_attn=local).eval().reparameterize()
            L = model.cfg.dec_layers
            for size in a.sizes:
                x = torch.randn(1, 3, size, size)
                full = gflops(model, x)
                one = gflops(model, x, max_layers=1)
                per_layer = (full - one) / max(1, L - 1)
                dec = per_layer * L                       # decoder cost = L layers of the same size
                rows.append(dict(preset=name, local_attn=local, size=size, queries=model.cfg.num_queries,
                                 layers=L, d_model=model.cfg.d_model, params_M=count_params(model) / 1e6,
                                 total_gflops=full, decoder_gflops=dec, decoder_share=dec / full,
                                 per_layer_gflops=per_layer))
                print(f"{name}-{local:6s} @{size}: total {full:6.2f} GF | decoder {dec:5.3f} GF "
                      f"({100 * dec / full:4.1f}%) | per layer {per_layer:5.3f} GF | {model.cfg.num_queries} queries, "
                      f"{L} layers, d={model.cfg.d_model}", flush=True)
            del model
    if a.queries:
        for q in a.queries:
            model = build_model("M", 20, num_queries=q).eval().reparameterize()
            x = torch.randn(1, 3, 640, 640)
            full, one = gflops(model, x), gflops(model, x, max_layers=1)
            per_layer = (full - one) / (model.cfg.dec_layers - 1); dec = per_layer * model.cfg.dec_layers
            rows.append(dict(preset="M", local_attn="roi", size=640, queries=q, layers=model.cfg.dec_layers,
                             d_model=model.cfg.d_model, params_M=count_params(model) / 1e6, total_gflops=full,
                             decoder_gflops=dec, decoder_share=dec / full, per_layer_gflops=per_layer))
            print(f"M-roi @640 queries={q}: total {full:6.2f} GF | decoder {dec:5.3f} GF ({100 * dec / full:4.1f}%)", flush=True)
            del model
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
