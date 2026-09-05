"""VOC data pipeline for KESTREL: XML parsing, mosaic / affine / HSV / flip augmentation, letterbox,
and a COCO-format ground-truth writer so every model in the paper is scored by the same pycocotools code."""
from __future__ import annotations

import json
import math
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
CLS2ID = {c: i for i, c in enumerate(VOC_CLASSES)}


# ----------------------------------------------------------------------------------------------
# Annotation parsing / caching
# ----------------------------------------------------------------------------------------------
def parse_voc_xml(path: str) -> Dict:
    root = ET.parse(path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    boxes, labels, difficult = [], [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLS2ID:
            continue
        bb = obj.find("bndbox")
        # VOC is 1-based inclusive → 0-based [x1, y1, x2, y2)
        x1, y1 = float(bb.find("xmin").text) - 1, float(bb.find("ymin").text) - 1
        x2, y2 = float(bb.find("xmax").text), float(bb.find("ymax").text)
        d = obj.find("difficult")
        boxes.append([x1, y1, x2, y2]); labels.append(CLS2ID[name]); difficult.append(int(d.text) if d is not None else 0)
    return dict(width=w, height=h, boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(labels, dtype=np.int64), difficult=np.array(difficult, dtype=np.int64))


def load_split(root: str, splits: List[Tuple[str, str]], cache: Optional[str] = None) -> List[Dict]:
    """splits: [(year, set)] e.g. [("2007","trainval"),("2012","trainval")]. Returns list of records."""
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    recs = []
    for year, s in splits:
        base = Path(root) / "VOCdevkit" / f"VOC{year}"
        ids = [l.strip() for l in open(base / "ImageSets" / "Main" / f"{s}.txt") if l.strip()]
        for i in ids:
            a = parse_voc_xml(str(base / "Annotations" / f"{i}.xml"))
            recs.append(dict(id=f"{year}_{i}", file=str(base / "JPEGImages" / f"{i}.jpg"), width=a["width"], height=a["height"],
                             boxes=a["boxes"].tolist(), labels=a["labels"].tolist(), difficult=a["difficult"].tolist()))
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        json.dump(recs, open(cache, "w"))
    return recs


def write_coco_gt(recs: List[Dict], path: str) -> Dict[str, int]:
    """COCO-format GT. VOC 'difficult' objects become iscrowd=1 (ignored by pycocotools), which is the usual
    VOC→COCO convention. Returns mapping record id → integer image id."""
    images, anns, id_map = [], [], {}
    for k, r in enumerate(recs):
        id_map[r["id"]] = k
        images.append(dict(id=k, file_name=os.path.basename(r["file"]), width=r["width"], height=r["height"]))
        for b, l, d in zip(r["boxes"], r["labels"], r["difficult"]):
            x1, y1, x2, y2 = b
            anns.append(dict(id=len(anns) + 1, image_id=k, category_id=int(l) + 1, bbox=[x1, y1, x2 - x1, y2 - y1],
                             area=(x2 - x1) * (y2 - y1), iscrowd=int(d), ignore=int(d)))
    cats = [dict(id=i + 1, name=c) for i, c in enumerate(VOC_CLASSES)]
    tmp = f"{path}.{os.getpid()}.tmp"                    # atomic replace: concurrent readers never see a truncated file
    with open(tmp, "w") as f:
        json.dump(dict(images=images, annotations=anns, categories=cats), f)
    os.replace(tmp, path)
    return id_map


# ----------------------------------------------------------------------------------------------
# Augmentation (YOLO-style, numpy/cv2, all box-aware)
# ----------------------------------------------------------------------------------------------
def letterbox(img: np.ndarray, size: int, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out, r, (left, top)


def hsv_jitter(img: np.ndarray, hgain=0.015, sgain=0.7, vgain=0.4) -> np.ndarray:
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h, lut_s, lut_v = ((x * r[0]) % 180).astype(np.uint8), np.clip(x * r[1], 0, 255).astype(np.uint8), np.clip(x * r[2], 0, 255).astype(np.uint8)
    im = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(im, cv2.COLOR_HSV2BGR)


def random_affine(img: np.ndarray, boxes: np.ndarray, labels: np.ndarray, size: int, degrees=0.0, translate=0.1,
                  scale=0.5, shear=0.0, border=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random scale/translate (YOLOv5 random_perspective without perspective). `border` < 0 crops a mosaic."""
    h, w = img.shape[:2]
    out_h, out_w = h + border * 2, w + border * 2
    C = np.eye(3); C[0, 2], C[1, 2] = -w / 2, -h / 2
    R = np.eye(3)
    a, s = random.uniform(-degrees, degrees), random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    S = np.eye(3); S[0, 1], S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180), math.tan(random.uniform(-shear, shear) * math.pi / 180)
    T = np.eye(3); T[0, 2], T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out_w, random.uniform(0.5 - translate, 0.5 + translate) * out_h
    M = T @ S @ R @ C
    img = cv2.warpAffine(img, M[:2], dsize=(out_w, out_h), borderValue=(114, 114, 114))
    if len(boxes):
        n = len(boxes)
        xy = np.ones((n * 4, 3)); xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = (xy @ M.T)[:, :2].reshape(n, 8)
        x, y = xy[:, [0, 2, 4, 6]], xy[:, [1, 3, 5, 7]]
        nb = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)], 1)
        nb[:, [0, 2]] = nb[:, [0, 2]].clip(0, out_w); nb[:, [1, 3]] = nb[:, [1, 3]].clip(0, out_h)
        # keep boxes that survive (area ratio, min size, aspect)
        w0, h0 = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        w1, h1 = nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]
        ar = np.maximum(w1 / (h1 + 1e-16), h1 / (w1 + 1e-16))
        keep = (w1 > 2) & (h1 > 2) & (w1 * h1 / (w0 * h0 * s * s + 1e-16) > 0.1) & (ar < 20)
        boxes, labels = nb[keep], labels[keep]
    return img, boxes.astype(np.float32), labels


class VOCDetection(Dataset):
    """Training: mosaic (p) + random affine + HSV + flip → size×size. Eval: letterbox only, returns meta for un-scaling."""

    def __init__(self, recs: List[Dict], size: int = 512, train: bool = True, mosaic: float = 1.0, scale: float = 0.5,
                 translate: float = 0.1, fliplr: float = 0.5, mixup: float = 0.0):
        self.recs, self.size, self.train = recs, size, train
        self.mosaic, self.scale, self.translate, self.fliplr, self.mixup = mosaic, scale, translate, fliplr, mixup

    def __len__(self):
        return len(self.recs)

    def load(self, i: int):
        r = self.recs[i]
        img = cv2.imread(r["file"])
        assert img is not None, r["file"]
        boxes = np.array(r["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.array(r["labels"], dtype=np.int64)
        if self.train:  # drop 'difficult' boxes from training targets (ultralytics VOC convention)
            keep = np.array(r["difficult"], dtype=bool) == False
            boxes, labels = boxes[keep], labels[keep]
        h, w = img.shape[:2]
        rr = self.size / max(h, w)
        if rr != 1:
            img = cv2.resize(img, (int(round(w * rr)), int(round(h * rr))), interpolation=cv2.INTER_LINEAR if (rr > 1 or self.train) else cv2.INTER_AREA)
            boxes = boxes * rr
        return img, boxes, labels, rr

    def load_mosaic(self, i: int):
        s = self.size
        yc, xc = (int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2))
        idx = [i] + random.choices(range(len(self.recs)), k=3)
        canvas = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        all_b, all_l = [], []
        for k, j in enumerate(idx):
            img, b, l, _ = self.load(j)
            h, w = img.shape[:2]
            if k == 0:   x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc; x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif k == 1: x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc; x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2: x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h); x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:        x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h); x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pw, ph = x1a - x1b, y1a - y1b
            if len(b):
                bb = b.copy(); bb[:, [0, 2]] += pw; bb[:, [1, 3]] += ph
                all_b.append(bb); all_l.append(l)
        boxes = np.concatenate(all_b, 0) if all_b else np.zeros((0, 4), np.float32)
        labels = np.concatenate(all_l, 0) if all_l else np.zeros((0,), np.int64)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, 2 * s); boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, 2 * s)
        img, boxes, labels = random_affine(canvas, boxes, labels, s, translate=self.translate, scale=self.scale, border=-s // 2)
        return img, boxes, labels

    def __getitem__(self, i: int):
        if self.train:
            if random.random() < self.mosaic:
                img, boxes, labels = self.load_mosaic(i)
                if random.random() < self.mixup:
                    img2, b2, l2 = self.load_mosaic(random.randrange(len(self.recs)))
                    r = np.random.beta(32.0, 32.0)
                    img = (img * r + img2 * (1 - r)).astype(np.uint8)
                    boxes, labels = np.concatenate([boxes, b2]), np.concatenate([labels, l2])
            else:
                img, boxes, labels, _ = self.load(i)
                img, r, (dx, dy) = letterbox(img, self.size)
                boxes = boxes * 1.0; boxes[:, [0, 2]] += dx; boxes[:, [1, 3]] += dy
                img, boxes, labels = random_affine(img, boxes, labels, self.size, translate=self.translate, scale=self.scale)
            img = hsv_jitter(img)
            if random.random() < self.fliplr:
                img = np.ascontiguousarray(img[:, ::-1])
                w = img.shape[1]
                boxes = boxes.copy(); boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            meta = dict(idx=i, ratio=1.0, pad=(0.0, 0.0))
        else:
            img, boxes, labels, rr = self.load(i)
            img, r, (dx, dy) = letterbox(img, self.size)
            boxes = boxes * r; boxes[:, [0, 2]] += dx; boxes[:, [1, 3]] += dy
            meta = dict(idx=i, ratio=rr * r, pad=(float(dx), float(dy)))
        img = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).permute(2, 0, 1)   # BGR→RGB, CHW, uint8
        return img, torch.from_numpy(boxes).float().reshape(-1, 4), torch.from_numpy(labels).long(), meta


def collate(batch):
    imgs, boxes, labels, metas = zip(*batch)
    return torch.stack(imgs, 0), list(boxes), list(labels), list(metas)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(imgs_uint8: torch.Tensor) -> torch.Tensor:
    x = imgs_uint8.float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)
