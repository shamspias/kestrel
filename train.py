"""KESTREL detection training on PASCAL VOC (07+12 trainval → 07 test). Single-device (mps / cuda / cpu)."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from criterion import KestrelCriterion, build_denoising, denoising_attn_mask
from data import VOCDetection, collate, load_split, normalize
from evaluate import build_model, build_test, run_eval, PRESETS
from kestrel import count_params


class EMA:
    def __init__(self, model, decay=0.9998, tau=2000):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or any(t in n for t in ("embed", "registers", "tgt", ".pos", "grn", ".ls", "logit_", "slot_pos", "temporal_gate")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)]


def pad_targets(boxes, labels, max_gt, device):
    B = len(boxes)
    gb = torch.zeros(B, max_gt, 4); gl = torch.zeros(B, max_gt, dtype=torch.long); gm = torch.zeros(B, max_gt, dtype=torch.bool)
    for i, (b, l) in enumerate(zip(boxes, labels)):
        if len(b) > max_gt:                       # keep the largest boxes (rare, crowded mosaics)
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            keep = area.topk(max_gt).indices
            b, l = b[keep], l[keep]
        n = len(b)
        gb[i, :n], gl[i, :n], gm[i, :n] = b, l, True
    return dict(boxes=gb.to(device), labels=gl.to(device), mask=gm.to(device))


def worker_init(wid):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s + wid); random.seed(s + wid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="N", choices=list(PRESETS))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-iters", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-gt", type=int, default=40)
    ap.add_argument("--dn-groups", type=int, default=2)
    ap.add_argument("--close-mosaic", type=int, default=5)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--out", default="runs/kestrel_n")
    ap.add_argument("--init", default=None, help="distilled backbone checkpoint (pretrain_distill.py)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-attn", default="roi", choices=["roi", "deform"])
    ap.add_argument("--no-dn", action="store_true")
    ap.add_argument("--no-golsd", action="store_true")
    ap.add_argument("--no-presence", action="store_true")
    ap.add_argument("--no-stal", action="store_true")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-train", action="store_true", help="evaluate on the training images (overfit sanity check)")
    ap.add_argument("--w-lsd", type=float, default=1.0)
    ap.add_argument("--fdr-scale", type=float, default=1.0, help="max edge offset as a fraction of the seed box side")
    ap.add_argument("--ls-init", type=float, default=1e-2, help="LayerScale init")
    ap.add_argument("--presence-power", type=float, default=1.0, help="eval-time presence gate exponent (1 product, 0.5 geometric mean, 0 off)")
    ap.add_argument("--dec-layers", type=int, default=None, help="override the preset's decoder depth (the anytime mechanism has more headroom the deeper the decoder is)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # Exclusive lock on the output directory: several experiment queues may be driving this repo at once, and two
    # processes writing the same last.pt/best.pt would corrupt both. A duplicate start exits quietly instead.
    _lock = open(os.path.join(a.out, ".train.lock"), "w")
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"another process is already training {a.out} (lock held); exiting"); return
    _lock.write(f"{os.getpid()}\n"); _lock.flush()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)

    # ---------------- data
    root = "data/voc"
    recs = load_split(root, [("2007", "trainval"), ("2012", "trainval")], cache=f"{root}/cache_trainval0712.json")
    if a.subset:
        recs = recs[:a.subset]
    ds = VOCDetection(recs, size=a.size, train=True, mosaic=a.mosaic, scale=a.scale)
    def make_loader():
        return DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True,
                          persistent_workers=a.workers > 0, worker_init_fn=worker_init, prefetch_factor=4 if a.workers else None)
    loader = make_loader()
    if a.eval_train:
        from data import VOCDetection as _VD, write_coco_gt as _wg
        test_recs = recs; gt_path = f"{a.out}/coco_gt_train_subset.json"; id_map = _wg(test_recs, gt_path)
        test_loader = DataLoader(_VD(test_recs, size=a.size, train=False), batch_size=16, shuffle=False, num_workers=2, collate_fn=collate)
    else:
        test_loader, gt_path, id_map, test_recs = build_test(a.size, root, subset=a.subset, workers=min(a.workers, 4))
    print(f"train images {len(recs)}  test images {len(test_recs)}  iters/epoch {len(loader)}")

    # ---------------- model / loss / optim
    over = dict(dec_layers=a.dec_layers) if a.dec_layers else {}
    model = build_model(a.model, 20, local_attn=a.local_attn, use_presence=not a.no_presence, fdr_scale=a.fdr_scale, ls_init=a.ls_init,
                        presence_power=a.presence_power, **over).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded = [k for k in sd if k not in unexpected]
        print(f"init: loaded {len(loaded)} tensors from {a.init} ({len(unexpected)} unexpected)")
    crit = KestrelCriterion(20, model.decoder.fdr, dn=not a.no_dn, golsd=not a.no_golsd, presence=not a.no_presence,
                            stal_min=0.0 if a.no_stal else 24.0, w_lsd=a.w_lsd)
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.999))
    ema = EMA(model)
    print(f"KESTREL-{a.model}: {count_params(model) / 1e6:.2f}M params  local_attn={a.local_attn} dn={not a.no_dn} golsd={not a.no_golsd} presence={not a.no_presence}")
    total_iters = a.epochs * len(loader)
    def lr_at(it):
        if it < a.warmup_iters:
            return a.lr * (it + 1) / a.warmup_iters
        t = (it - a.warmup_iters) / max(1, total_iters - a.warmup_iters)
        return a.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))
    start_epoch, it, best = 0, 0, -1.0
    if a.resume and os.path.exists(f"{a.out}/last.pt"):
        ck = torch.load(f"{a.out}/last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        start_epoch, it, best, ema.updates = ck["epoch"] + 1, ck["iter"], ck.get("best", -1.0), ck.get("ema_updates", 0)
        print(f"resumed from epoch {ck['epoch']} (iter {it}, best {best:.2f})")
    amp_dtype = dict(none=None, bf16=torch.bfloat16, fp16=torch.float16)[a.amp]
    scaler = torch.amp.GradScaler(dev.type, enabled=amp_dtype == torch.float16)   # fp16 needs loss scaling; bf16/fp32 do not
    K, heads = model.cfg.num_queries, model.cfg.dec_heads
    logf = open(f"{a.out}/log.jsonl", "a")
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)

    # ---------------- loop
    for epoch in range(start_epoch, a.epochs):
        if epoch >= a.epochs - a.close_mosaic and ds.mosaic > 0:
            ds.mosaic = 0.0; loader = make_loader(); print("mosaic closed")
        model.train()
        t_ep, t_last, seen, agg = time.time(), time.time(), 0, {}
        for imgs, boxes, labels, _ in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            x = normalize(imgs.to(dev, non_blocking=True))
            gt = pad_targets(boxes, labels, a.max_gt, dev)
            dn = None
            if not a.no_dn:
                dn = build_denoising(gt, 20, a.dn_groups, img_hw=(a.size, a.size))
                dn["attn_mask"] = denoising_attn_mask(K, dn, heads)
            progress = min(1.0, it / max(1, total_iters))
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x, return_masks=False, dn=dn)
            out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()} if amp_dtype else out
            if amp_dtype:
                out["aux"] = [{k: v.float() for k, v in l.items()} for l in out["aux"]]
                if "dn_layers" in out:
                    out["dn_layers"] = [{k: v.float() for k, v in l.items()} for l in out["dn_layers"]]
            loss, log = crit(out, gt, (a.size, a.size), progress)
            if not torch.isfinite(loss):
                print("non-finite loss, skipping step", log); opt.zero_grad(set_to_none=True); it += 1; continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt); scaler.update()
            ema.update(model)
            it += 1; seen += imgs.shape[0]
            for k, v in log.items():
                agg[k] = agg.get(k, 0.0) + v
            agg["_n"] = agg.get("_n", 0) + 1
            if it % a.log_every == 0:
                n = agg.pop("_n"); avg = {k: v / n for k, v in agg.items()}; agg = {}
                dt = time.time() - t_last; t_last = time.time()
                ips = a.log_every * a.bs / dt
                eta = (total_iters - it) * a.bs / ips / 3600
                keys = ["total", "d_cls", "d_box", "d_qual", "cls", "l1", "giou", "fdr", "dn_cls", "dn_giou", "lsd", "pres", "n_fg"]
                s = " ".join(f"{k}={avg[k]:.3f}" for k in keys if k in avg)
                print(f"ep {epoch} it {it} lr {lr_at(it):.2e} gn {float(gn):.2f} {ips:.1f} img/s eta {eta:.1f}h | {s}", flush=True)
                logf.write(json.dumps(dict(epoch=epoch, iter=it, lr=lr_at(it), ips=ips, **avg)) + "\n"); logf.flush()
        print(f"epoch {epoch} done in {(time.time() - t_ep) / 60:.1f} min", flush=True)
        ck = dict(model=model.state_dict(), ema=ema.ema.state_dict(), opt=opt.state_dict(), epoch=epoch, iter=it, best=best,
                  ema_updates=ema.updates, args=vars(a))
        torch.save(ck, f"{a.out}/last.pt")
        if (epoch + 1) % a.eval_every == 0 or epoch == a.epochs - 1:
            t0 = time.time()
            stats = run_eval(ema.ema, test_loader, gt_path, id_map, test_recs, dev)
            stats["AP_dense"] = run_eval(ema.ema, test_loader, gt_path, id_map, test_recs, dev, dense_nms=True)["AP"]
            if not a.no_presence and a.presence_power > 0:
                stats["AP_nogate"] = run_eval(ema.ema, test_loader, gt_path, id_map, test_recs, dev, gate_power=0.0)["AP"]
            print(f"EVAL epoch {epoch}: " + " ".join(f"{k}={v:.2f}" for k, v in stats.items()) + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            logf.write(json.dumps(dict(epoch=epoch, eval=stats)) + "\n"); logf.flush()
            if stats["AP"] > best:
                best = stats["AP"]; ck["best"] = best
                torch.save(ck, f"{a.out}/best.pt")
                torch.save(ck, f"{a.out}/last.pt")            # keep last.pt's `best` current so a resume cannot overwrite best.pt with a worse model
    print(f"training complete. best AP {best:.2f}")


if __name__ == "__main__":
    main()
