"""Phase 1: distil a frozen DINOv2-S/14 (with registers) teacher into the KESTREL backbone's stride-16 / stride-32
'distillation ports' on unlabelled VOC trainval images: per-token cosine at s16 and s32, a Gram-structure loss at s16
(DINOv3-style anchoring of the patch-similarity matrix) and a pooled-token cosine to the teacher CLS."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import hsv_jitter, load_split, normalize
from evaluate import build_model
from kestrel import count_params


class Unlabeled(Dataset):
    def __init__(self, files, size):
        self.files, self.size = files, size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i]); h, w = img.shape[:2]
        # random resized crop
        for _ in range(10):
            area = h * w * random.uniform(0.35, 1.0); ar = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
            cw, ch = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
            if 0 < cw <= w and 0 < ch <= h:
                x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch); img = img[y0:y0 + ch, x0:x0 + cw]; break
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = hsv_jitter(img)
        if random.random() < 0.5:
            img = img[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="N")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--gram", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--teacher", default="vit_small_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--out", default="runs/distill_n")
    ap.add_argument("--subset", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    dev = torch.device(a.device)
    recs = load_split("data/voc", [("2007", "trainval"), ("2012", "trainval")], cache="data/voc/cache_trainval0712.json")
    files = [r["file"] for r in recs][: a.subset] if a.subset else [r["file"] for r in recs]
    loader = DataLoader(Unlabeled(files, a.size), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)

    teacher = timm.create_model(a.teacher, pretrained=True, num_classes=0, dynamic_img_size=True).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    n_prefix = 1 + getattr(teacher, "num_reg_tokens", 0) if hasattr(teacher, "num_reg_tokens") else 1 + 4
    td = teacher.embed_dim
    model = build_model(a.model, 20).to(dev)
    d16, d32 = model.cfg.attn_dims
    heads = nn.ModuleDict(dict(p16=nn.Conv2d(d16, td, 1), p32=nn.Conv2d(d32, td, 1), pc=nn.Linear(d32, td))).to(dev)
    student_mods = [model.stem, model.s4, model.s8, model.s16, model.s32]
    params = [p for m in student_mods for p in m.parameters()] + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    total = a.epochs * len(loader); it = 0
    print(f"student backbone {sum(p.numel() for m in student_mods for p in m.parameters()) / 1e6:.2f}M, teacher {a.teacher} {count_params(teacher) / 1e6:.1f}M, iters {total}")
    logf = open(f"{a.out}/log.jsonl", "a")
    for epoch in range(a.epochs):
        model.train(); t0 = time.time(); agg = {}
        for imgs in loader:
            lr = a.lr * (it / 300 if it < 300 else 0.5 * (1 + math.cos(math.pi * (it - 300) / max(1, total - 300))))
            for g in opt.param_groups:
                g["lr"] = lr
            x = normalize(imgs.to(dev))
            with torch.no_grad():
                tf = teacher.forward_features(x)                                  # (B, prefix + N, td)
                tcls, tp = tf[:, 0], tf[:, n_prefix:]
                g = int(math.sqrt(tp.shape[1]))
                tp = tp.transpose(1, 2).reshape(-1, td, g, g)
                t16 = F.interpolate(tp, size=(a.size // 16, a.size // 16), mode="bilinear", align_corners=False, antialias=True)
                t32 = F.interpolate(tp, size=(a.size // 32, a.size // 32), mode="bilinear", align_corners=False, antialias=True)
                t16n = F.normalize(t16.flatten(2).transpose(1, 2), dim=-1)      # (B, T, td)
                Gt = t16n @ t16n.transpose(1, 2)
            c3, c4, c5, mem, _ = model.backbone(x)
            s16, s32 = heads["p16"](c4), heads["p32"](c5)
            s16n = F.normalize(s16.flatten(2).transpose(1, 2), dim=-1)
            l16 = (1 - F.cosine_similarity(s16, t16, dim=1)).mean()
            l32 = (1 - F.cosine_similarity(s32, t32, dim=1)).mean()
            lc = (1 - F.cosine_similarity(heads["pc"](mem.mean(1)), tcls, dim=-1)).mean()
            lg = ((s16n @ s16n.transpose(1, 2) - Gt) ** 2).mean()
            loss = l16 + l32 + 0.5 * lc + a.gram * lg
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0); opt.step(); it += 1
            for k, v in dict(loss=loss, l16=l16, l32=l32, lcls=lc, gram=lg).items():
                agg[k] = agg.get(k, 0.0) + float(v)
            agg["n"] = agg.get("n", 0) + 1
            if it % 50 == 0:
                n = agg.pop("n"); s = {k: v / n for k, v in agg.items()}; agg = {}
                ips = 50 * a.bs / (time.time() - t0); t0 = time.time()
                print(f"ep {epoch} it {it}/{total} lr {lr:.2e} {ips:.1f} img/s eta {(total - it) * a.bs / ips / 3600:.2f}h | " + " ".join(f"{k}={v:.4f}" for k, v in s.items()), flush=True)
                logf.write(json.dumps(dict(epoch=epoch, iter=it, **s)) + "\n"); logf.flush()
        sd = {k: v for k, v in model.state_dict().items() if k.startswith(("stem.", "s4.", "s8.", "s16.", "s32."))}
        torch.save(sd, f"{a.out}/backbone.pt")
        print(f"epoch {epoch} saved {a.out}/backbone.pt ({len(sd)} tensors)", flush=True)


if __name__ == "__main__":
    main()
