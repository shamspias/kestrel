"""Unified COCO-style evaluation on VOC07 test (pycocotools), static-depth sweeps, anytime-exit sweeps and
batch-1 latency. Every model in the paper (KESTREL and the ultralytics baselines) is scored by `coco_eval`
on the same GT file, so the numbers are comparable."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import VOCDetection, collate, load_split, normalize, write_coco_gt
from kestrel import KESTREL, KestrelConfig, count_params

PRESETS = {
    "N": dict(stem_ch=16, conv_dims=(32, 64), conv_depths=(1, 2), attn_dims=(128, 192), attn_depths=(3, 2), neck_dim=128,
              d_model=128, embed_dim=128, dec_layers=3, dec_heads=4, num_queries=100),
    "S": dict(stem_ch=24, conv_dims=(48, 96), conv_depths=(2, 3), attn_dims=(192, 288), attn_depths=(4, 3), neck_dim=192,
              d_model=192, embed_dim=192, dec_layers=4, dec_heads=6, num_queries=300),
    "M": dict(),
}


def sync(device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def build_model(name: str, num_classes: int = 20, **over) -> KESTREL:
    kw = dict(PRESETS[name]); kw.update(over)
    return KESTREL(replace(KestrelConfig(), num_classes=num_classes, **kw))


def load_checkpoint(path: str, device, ema: bool = True) -> Tuple[KESTREL, Dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck["args"]
    model = build_model(args["model"], 20, local_attn=args.get("local_attn", "roi"), use_presence=not args.get("no_presence", False),
                        fdr_scale=args.get("fdr_scale", 0.5), ls_init=args.get("ls_init", 1e-2), presence_power=args.get("presence_power", 1.0))
    sd = ck["ema"] if (ema and "ema" in ck) else ck["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.startswith(("mask_head", "kpt_head", "slots", "temporal"))], missing
    return model.to(device).eval(), ck


def build_test(size: int, root: str = "data/voc", subset: Optional[int] = None, workers: int = 4, batch: int = 16):
    recs = load_split(root, [("2007", "test")], cache=f"{root}/cache_test07.json")
    if subset:
        recs = recs[:subset]
    gt_path = f"{root}/coco_gt_test07{'_sub' + str(subset) if subset else ''}.json"
    id_map = write_coco_gt(recs, gt_path)
    ds = VOCDetection(recs, size=size, train=False)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate, persistent_workers=workers > 0)
    return loader, gt_path, id_map, recs


@torch.no_grad()
def predict(model: KESTREL, loader, device, id_map: Dict, recs: List[Dict], max_layers: Optional[int] = None,
            anytime: bool = False, max_det: int = 300, num_queries: Optional[int] = None, dense_nms: bool = False):
    """dense_nms=True decodes the one-to-many dense head with NMS (IoU 0.6) instead of the decoder — the 'NMS path'."""
    model.eval()
    results, exits, n_img, t_fwd = [], [], 0, 0.0
    for imgs, _, _, metas in loader:
        x = normalize(imgs.to(device))
        sync(device)
        t0 = time.perf_counter()
        out = model(x, return_masks=False, max_layers=max_layers, anytime=anytime, num_queries=num_queries)
        sync(device)
        t_fwd += time.perf_counter() - t0
        if dense_nms:
            from torchvision.ops import batched_nms
            dsc = out["dense_logits"].sigmoid() * out["dense_quality"].sigmoid()[..., None]      # (B, N, C)
            B, N, C = dsc.shape
            top = dsc.flatten(1).topk(min(1000, N * C), dim=1)
            qidx, cidx = top.indices // C, top.indices % C
            bx_all = out["dense_boxes"].gather(1, qidx[..., None].expand(-1, -1, 4))
            boxes_l, sc_l, c_l = [], [], []
            for b in range(B):
                keep = batched_nms(bx_all[b], top.values[b], cidx[b], 0.6)[:max_det]
                pad = max_det - keep.numel()
                boxes_l.append(torch.cat([bx_all[b][keep], bx_all.new_zeros(pad, 4)])); sc_l.append(torch.cat([top.values[b][keep], top.values.new_zeros(pad)])); c_l.append(torch.cat([cidx[b][keep], cidx.new_zeros(pad)]))
            boxes, sc, cidx = torch.stack(boxes_l).cpu(), torch.stack(sc_l).cpu(), torch.stack(c_l).cpu()
        else:
            scores = out["scores"]                                                  # (B, K, C)
            B, K, C = scores.shape
            top = scores.flatten(1).topk(min(max_det, K * C), dim=1)
            qidx, cidx = top.indices // C, top.indices % C
            boxes = out["boxes"].gather(1, qidx[..., None].expand(-1, -1, 4)).cpu()
            sc, cidx = top.values.cpu(), cidx.cpu()
        if anytime:
            exits.append(out["exit_layer"].cpu())
        for b in range(B):
            m = metas[b]; r = recs[m["idx"]]
            dx, dy = m["pad"]; ratio = m["ratio"]
            bx = boxes[b].clone()
            bx[:, [0, 2]] = ((bx[:, [0, 2]] - dx) / ratio).clamp(0, r["width"])
            bx[:, [1, 3]] = ((bx[:, [1, 3]] - dy) / ratio).clamp(0, r["height"])
            iid = id_map[r["id"]]
            for j in range(bx.shape[0]):
                if sc[b, j] <= 0:
                    continue
                x1, y1, x2, y2 = bx[j].tolist()
                results.append(dict(image_id=iid, category_id=int(cidx[b, j]) + 1, bbox=[x1, y1, x2 - x1, y2 - y1], score=float(sc[b, j])))
        n_img += B
    info = dict(images=n_img, ms_per_img_batched=1000 * t_fwd / max(n_img, 1))
    if exits:
        e = torch.cat(exits, 0).float()
        info.update(mean_exit_layer=e.mean().item(), exit_hist=torch.bincount(e.long().flatten(), minlength=model.cfg.dec_layers + 1)[1:].tolist(),
                    mean_exit_per_image=e.mean(1).tolist())
    return results, info


def coco_eval(results: List[Dict], gt_path: str, quiet: bool = True) -> Dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not results:
        return dict(AP=0.0, AP50=0.0, AP75=0.0, APs=0.0, APm=0.0, APl=0.0)
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        gt = COCO(gt_path)
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    return dict(AP=100 * s[0], AP50=100 * s[1], AP75=100 * s[2], APs=100 * s[3], APm=100 * s[4], APl=100 * s[5])


def run_eval(model, loader, gt_path, id_map, recs, device, gate_power: Optional[float] = None, **kw) -> Dict:
    """gate_power overrides cfg.presence_power for this evaluation only (None = leave the model's setting)."""
    saved = model.cfg.presence_power
    if gate_power is not None:
        model.cfg.presence_power = gate_power
    try:
        res, info = predict(model, loader, device, id_map, recs, **kw)
    finally:
        model.cfg.presence_power = saved
    stats = coco_eval(res, gt_path)
    stats.update({k: v for k, v in info.items() if k != "mean_exit_per_image"})
    return stats


@torch.no_grad()
def latency(model: KESTREL, device, size: int, n: int = 50, warm: int = 10, **kw) -> float:
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warm):
        model(x, return_masks=False, **kw)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        model(x, return_masks=False, **kw)
    sync(device)
    return 1000 * (time.perf_counter() - t0) / n


@torch.no_grad()
def latency_anytime(model: KESTREL, loader, device, n_images: int = 200, warm: int = 10):
    """Batch-1 wall-clock of the anytime forward on real test images (the exit pattern depends on content)."""
    model.eval()
    times, depths, k = [], [], 0
    for imgs, _, _, _ in loader:
        for i in range(imgs.shape[0]):
            x = normalize(imgs[i:i + 1].to(device))
            sync(device)
            t0 = time.perf_counter()
            out = model(x, return_masks=False, anytime=True)
            sync(device)
            dt = time.perf_counter() - t0
            k += 1
            if k > warm:
                times.append(1000 * dt); depths.append(out["exit_layer"].float().mean().item())
            if k >= n_images + warm:
                return dict(ms_mean=float(np.mean(times)), ms_median=float(np.median(times)), mean_depth=float(np.mean(depths)), n=len(times))
    return dict(ms_mean=float(np.mean(times)), ms_median=float(np.median(times)), mean_depth=float(np.mean(depths)), n=len(times))


@torch.no_grad()
def latency_fixed_images(model: KESTREL, loader, device, n_images: int = 200, warm: int = 10, max_layers=None):
    model.eval(); times, k = [], 0
    for imgs, _, _, _ in loader:
        for i in range(imgs.shape[0]):
            x = normalize(imgs[i:i + 1].to(device))
            sync(device)
            t0 = time.perf_counter(); model(x, return_masks=False, max_layers=max_layers)
            sync(device)
            k += 1
            if k > warm: times.append(1000 * (time.perf_counter() - t0))
            if k >= n_images + warm: return dict(ms_mean=float(np.mean(times)), ms_median=float(np.median(times)), n=len(times))
    return dict(ms_mean=float(np.mean(times)), ms_median=float(np.median(times)), n=len(times))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--static-sweep", action="store_true", help="AP for max_layers = 1..L")
    ap.add_argument("--anytime-sweep", action="store_true", help="AP vs mean depth over exit thresholds")
    ap.add_argument("--sweep-p", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8], help="foreground-exit thresholds for --anytime-sweep")
    ap.add_argument("--sweep-u", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.30], help="entropy thresholds for --anytime-sweep")
    ap.add_argument("--sweep-bg", type=float, nargs="+", default=[0.02, 0.05, 0.10], help="background-exit thresholds for --anytime-sweep")
    ap.add_argument("--sweep-min-layers", type=int, nargs="+", default=None, help="minimum layers before a query may exit (default: the model's setting)")
    ap.add_argument("--sweep-mode", nargs="+", default=None, choices=["remove", "freeze"], help="exit modes to sweep (default: the model's setting)")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--latency-anytime", action="store_true", help="batch-1 anytime latency on real images at the current cfg thresholds")
    ap.add_argument("--exit", type=float, nargs=3, default=None, metavar=("P", "U", "BG"), help="exit thresholds for --latency-anytime / --anytime")
    ap.add_argument("--anytime", action="store_true", help="single anytime evaluation at --exit thresholds")
    ap.add_argument("--reparam", action="store_true", help="fold multi-branch convs before evaluating/timing")
    ap.add_argument("--exit-mode", default=None, choices=["remove", "freeze"], help="anytime exit: remove exited queries from later layers (default) or keep them frozen as self-attention keys")
    ap.add_argument("--gate-power", type=float, default=None, help="presence gate exponent for all evaluations (1 = product, 0.5 = geometric mean, 0 = off); default: checkpoint setting")
    ap.add_argument("--gate-sweep", action="store_true", help="also report full-depth AP for gate exponents 1, 0.5 and 0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--update", action="store_true", help="merge into an existing --out JSON instead of overwriting it (e.g. add latency later on an idle GPU)")
    ap.add_argument("--skip-full", action="store_true", help="with --update: keep the stored full-depth result instead of recomputing it")
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, ck = load_checkpoint(a.ckpt, dev)
    loader, gt_path, id_map, recs = build_test(a.size, subset=a.subset)
    L = model.cfg.dec_layers
    if a.reparam:
        model.reparameterize()
    if a.exit:
        model.cfg.exit_p, model.cfg.exit_u, model.cfg.exit_bg = a.exit
    if a.gate_power is not None:
        model.cfg.presence_power = a.gate_power
    if a.exit_mode:
        model.cfg.exit_mode = a.exit_mode
    out = json.load(open(a.out)) if (a.update and a.out and os.path.exists(a.out)) else {}
    out.update(ckpt=a.ckpt, params_M=count_params(model) / 1e6, epoch=ck.get("epoch"), exit=a.exit or out.get("exit"), size=a.size, device=str(dev),
               gate_power=model.cfg.presence_power if model.cfg.use_presence else 0.0, exit_mode=model.cfg.exit_mode)
    print(f"model {ck['args']['model']}  params {out['params_M']:.2f}M  epoch {ck.get('epoch')}")
    if not (a.skip_full and "full" in out):
        out["full"] = run_eval(model, loader, gt_path, id_map, recs, dev)
    print("full depth:", {k: round(v, 2) for k, v in out["full"].items()})
    if a.gate_sweep and model.cfg.use_presence:
        out["gate"] = {}
        for g in (1.0, 0.5, 0.0):
            out["gate"][str(g)] = run_eval(model, loader, gt_path, id_map, recs, dev, gate_power=g)
            print(f"presence gate power {g}:", {k: round(v, 2) for k, v in out["gate"][str(g)].items() if k.startswith("AP")})
    if a.static_sweep:
        out["static"] = {}
        for l in range(1, L):
            out["static"][l] = run_eval(model, loader, gt_path, id_map, recs, dev, max_layers=l)
            print(f"static L={l}:", {k: round(v, 2) for k, v in out["static"][l].items()})
    if a.anytime_sweep:
        out["anytime"] = []
        saved_exit = (model.cfg.exit_p, model.cfg.exit_u, model.cfg.exit_bg)
        saved_ml, saved_mode = model.cfg.exit_min_layers, model.cfg.exit_mode
        grid = [(p, u, bg, ml, md) for md in (a.sweep_mode or [saved_mode]) for ml in (a.sweep_min_layers or [saved_ml])
                for p in a.sweep_p for u in a.sweep_u for bg in a.sweep_bg]
        for p, u, bg, ml, md in grid:
            model.cfg.exit_p, model.cfg.exit_u, model.cfg.exit_bg = p, u, bg
            model.cfg.exit_min_layers, model.cfg.exit_mode = ml, md
            r = run_eval(model, loader, gt_path, id_map, recs, dev, anytime=True)
            r.update(exit_p=p, exit_u=u, exit_bg=bg, min_layers=ml, mode=md)
            out["anytime"].append(r)
            print(f"anytime mode={md} min_l={ml} p={p} u={u} bg={bg}: AP {r['AP']:.2f} mean depth {r['mean_exit_layer']:.2f} hist {r['exit_hist']}")
        model.cfg.exit_min_layers, model.cfg.exit_mode = saved_ml, saved_mode
        model.cfg.exit_p, model.cfg.exit_u, model.cfg.exit_bg = saved_exit      # restore --exit thresholds for --anytime / --latency-anytime
    if a.anytime:
        out["anytime_single"] = run_eval(model, loader, gt_path, id_map, recs, dev, anytime=True)
        print("anytime:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out["anytime_single"].items()})
    if a.latency:
        out["latency_ms"] = {f"L{l}": latency(model, dev, a.size, max_layers=l) for l in range(1, L + 1)}
        out["latency_ms_images"] = {f"L{l}": latency_fixed_images(model, loader, dev, max_layers=l) for l in range(1, L + 1)}
        print("latency ms (batch 1, random):", out["latency_ms"]); print("latency ms (batch 1, images):", out["latency_ms_images"])
    if a.latency_anytime:
        out["latency_anytime"] = latency_anytime(model, loader, dev)
        print("anytime latency:", out["latency_anytime"])
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
