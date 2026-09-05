"""KESTREL training criterion.
   L = w_dense(t)·L_dense  +  w_o2o(t)·(L_o2o + L_dn)  +  λ_lsd·L_GO-LSD  +  λ_pres·L_presence
   L_dense : one-to-many (TAL + STAL): varifocal cls, GIoU box, quality-focal IoU quality
   L_o2o   : Hungarian-matched per decoder layer: varifocal (IoU-aware) cls, L1 + GIoU, FDR two-bin CE
   L_dn    : same losses on contrastive-denoising queries (known assignment, no matching)
   L_GO-LSD: KL(final layer sharpened distributions || layer l) for every query (matched weighted higher)
   L_pres  : BCE on image-level multi-label presence."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

from assign import TaskAlignedAssigner, HungarianMatcher, box_cxcywh
from kestrel import FDR, golsd_loss


def varifocal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """VFL (Zhang et al.): soft IoU target on positives, focal down-weighting on negatives. Returns element-wise loss."""
    p = logits.sigmoid()
    w = torch.where(target > 0, target, alpha * p.pow(gamma))
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w


def quality_focal_loss(logits: torch.Tensor, target: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    p = logits.sigmoid()
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (target - p).abs().pow(beta)


class KestrelCriterion(nn.Module):
    def __init__(self, num_classes: int, fdr: FDR, tal_topk: int = 10, stal_min: float = 24.0, dn: bool = True,
                 golsd: bool = True, presence: bool = True, lsd_T: float = 5.0,
                 w_cls: float = 1.0, w_l1: float = 5.0, w_giou: float = 2.0, w_fdr: float = 0.5, w_lsd: float = 1.0,
                 w_pres: float = 1.0, w_dense_cls: float = 1.0, w_dense_box: float = 2.0, w_dense_qual: float = 0.5):
        super().__init__()
        self.C, self.fdr = num_classes, fdr
        self.tal = TaskAlignedAssigner(topk=tal_topk, stal_min=stal_min)
        self.matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
        self.use_dn, self.use_golsd, self.use_presence, self.T = dn, golsd, presence, lsd_T
        self.w = dict(cls=w_cls, l1=w_l1, giou=w_giou, fdr=w_fdr, lsd=w_lsd, pres=w_pres, dcls=w_dense_cls, dbox=w_dense_box, dqual=w_dense_qual)

    # ------------------------------------------------------------------ dense one-to-many
    def dense_loss(self, out: Dict, gt: Dict) -> Dict[str, torch.Tensor]:
        logits, boxes, qual = out["dense_logits"], out["dense_boxes"], out["dense_quality"]
        a = self.tal(logits.detach().sigmoid(), boxes.detach(), out["dense_anchors"], gt["labels"], gt["boxes"], gt["mask"])
        ts, fg, tb = a["target_scores"], a["fg_mask"], a["target_boxes"]
        norm = ts.sum().clamp(min=1.0)
        l_cls = varifocal_loss(logits, ts).sum() / norm
        if fg.any():
            w = ts.sum(-1)[fg]
            l_box = (generalized_box_iou_loss(boxes[fg], tb[fg], reduction="none") * w).sum() / norm
            iou_t = torch.zeros_like(qual)
            iou_t[fg] = box_iou_diag(boxes[fg].detach(), tb[fg])
        else:
            l_box = boxes.sum() * 0
            iou_t = torch.zeros_like(qual)
        l_q = quality_focal_loss(qual, iou_t).sum() / fg.sum().clamp(min=1.0)
        return dict(d_cls=l_cls * self.w["dcls"], d_box=l_box * self.w["dbox"], d_qual=l_q * self.w["dqual"], n_fg=fg.sum().float())

    # ------------------------------------------------------------------ set losses for one decoder layer
    def layer_loss(self, layer: Dict, gt: Dict, indices, box_init: torch.Tensor, img_hw, num_gt: float, prefix: str) -> Dict[str, torch.Tensor]:
        logits, boxes, fdr = layer["logits"], layer["boxes"], layer["fdr"]
        B, K, C = logits.shape
        H, W = img_hw
        scale = boxes.new_tensor([W, H, W, H])
        target = torch.zeros_like(logits)
        bi = torch.cat([torch.full_like(q, b) for b, (q, _) in enumerate(indices)]).to(logits.device)
        qi = torch.cat([q for q, _ in indices]).to(logits.device)
        gi = torch.cat([g for _, g in indices]).to(logits.device)
        if qi.numel():
            # gather matched gt (gt tensors are padded; indices refer to the compacted valid gts → map back)
            valid_idx = [gt["mask"][b].nonzero().squeeze(1) for b in range(B)]
            gsel = torch.cat([valid_idx[b][g.to(logits.device)] for b, (_, g) in enumerate(indices)])
            tb = gt["boxes"][bi, gsel]
            tl = gt["labels"][bi, gsel]
            pb = boxes[bi, qi]
            iou = box_iou_diag(pb.detach(), tb)
            target[bi, qi, tl] = iou
            l_cls = varifocal_loss(logits, target).sum() / num_gt
            l_l1 = F.l1_loss(box_cxcywh(pb) / scale, box_cxcywh(tb) / scale, reduction="sum") / num_gt
            l_giou = generalized_box_iou_loss(pb, tb, reduction="sum") / num_gt
            # FDR: two-bin target on edge offsets relative to the seed box
            b0 = box_init[bi, qi]
            w0 = (b0[:, 2] - b0[:, 0]).clamp(min=1.0); h0 = (b0[:, 3] - b0[:, 1]).clamp(min=1.0)
            off = (tb - b0) / torch.stack([w0, h0, w0, h0], -1)
            tdist = self.fdr.target_distribution(off)
            l_fdr = -(tdist * fdr[bi, qi].log_softmax(-1)).sum(-1).mean(-1)          # (n,)
            l_fdr = (l_fdr * iou.clamp(min=0.1)).sum() / num_gt                          # IoU-weighted like D-FINE's FGL
        else:
            l_cls = varifocal_loss(logits, target).sum() / num_gt
            l_l1 = l_giou = l_fdr = logits.sum() * 0
        return {f"{prefix}cls": l_cls * self.w["cls"], f"{prefix}l1": l_l1 * self.w["l1"], f"{prefix}giou": l_giou * self.w["giou"], f"{prefix}fdr": l_fdr * self.w["fdr"]}

    # ------------------------------------------------------------------ denoising part (known assignment)
    def dn_loss(self, layers: List[Dict], dn: Dict, gt: Dict, img_hw, num_gt: float) -> Dict[str, torch.Tensor]:
        """dn['gt_idx'] (B, Kd) index into padded gt, dn['positive'] (B, Kd) bool, dn['valid'] (B, Kd) bool."""
        H, W = img_hw
        tot: Dict[str, torch.Tensor] = {}
        B, Kd = dn["gt_idx"].shape
        scale = layers[0]["boxes"].new_tensor([W, H, W, H])
        valid, pos = dn["valid"], dn["positive"] & dn["valid"]
        tb = gt["boxes"].gather(1, dn["gt_idx"][..., None].expand(-1, -1, 4))
        tl = gt["labels"].gather(1, dn["gt_idx"])
        b0 = dn["boxes"]
        w0 = (b0[..., 2] - b0[..., 0]).clamp(min=1.0); h0 = (b0[..., 3] - b0[..., 1]).clamp(min=1.0)
        tdist = self.fdr.target_distribution((tb - b0) / torch.stack([w0, h0, w0, h0], -1))
        n_pos = pos.sum().clamp(min=1.0)
        for li, layer in enumerate(layers):
            logits, boxes, fdr = layer["logits"], layer["boxes"], layer["fdr"]
            target = torch.zeros_like(logits)
            iou = box_iou_diag(boxes.detach().reshape(-1, 4), tb.reshape(-1, 4)).view(B, Kd)
            target.scatter_(2, tl.clamp(min=0)[..., None], (iou * pos)[..., None])
            l_cls = (varifocal_loss(logits, target) * valid[..., None]).sum() / n_pos
            l_l1 = (F.l1_loss(box_cxcywh(boxes) / scale, box_cxcywh(tb) / scale, reduction="none").sum(-1) * pos).sum() / n_pos
            l_giou = (generalized_box_iou_loss(boxes.reshape(-1, 4), tb.reshape(-1, 4), reduction="none").view(B, Kd) * pos).sum() / n_pos
            l_fdr = ((-(tdist * fdr.log_softmax(-1)).sum(-1).mean(-1)) * pos * iou.clamp(min=0.1)).sum() / n_pos
            for k, v in dict(cls=l_cls * self.w["cls"], l1=l_l1 * self.w["l1"], giou=l_giou * self.w["giou"], fdr=l_fdr * self.w["fdr"]).items():
                tot[f"dn_{k}"] = tot.get(f"dn_{k}", 0) + v
        return tot

    # ------------------------------------------------------------------ everything
    def forward(self, out: Dict, gt: Dict, img_hw: Tuple[int, int], progress: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: Dict[str, torch.Tensor] = {}
        B = gt["mask"].shape[0]
        num_gt = float(gt["mask"].sum().clamp(min=1))
        losses.update(self.dense_loss(out, gt))
        layers = out["aux"] + [dict(logits=out["logits"], boxes=out["boxes"], fdr=out["fdr"])]
        box_init = out["box_init"]
        final_idx = None
        for li, layer in enumerate(layers):
            idx = self.matcher(layer["logits"].detach(), layer["boxes"].detach(), gt["labels"], gt["boxes"], gt["mask"], img_hw)
            if li == len(layers) - 1:
                final_idx = idx
            pre = "" if li == len(layers) - 1 else f"a{li}_"
            losses.update(self.layer_loss(layer, gt, idx, box_init, img_hw, num_gt, pre))
        if self.use_dn and "dn_layers" in out:
            losses.update(self.dn_loss(out["dn_layers"], out["dn"], gt, img_hw, num_gt))
        if self.use_golsd and len(layers) > 1:
            # D-FINE-style GO-LSD: final-layer distributions (sharpened, T) teach every earlier layer on ALL queries;
            # matched queries weighted by their final IoU, unmatched by the teacher's max class probability.
            final = layers[-1]["fdr"]
            teacher_conf = layers[-1]["logits"].detach().sigmoid().amax(-1)                 # (B, K)
            w = teacher_conf.clone()
            matched = torch.zeros_like(w, dtype=torch.bool)
            for b, (q, g) in enumerate(final_idx):
                if q.numel():
                    q = q.to(final.device); gsel = gt["mask"][b].nonzero().squeeze(1)[g.to(final.device)]
                    w[b, q] = box_iou_diag(layers[-1]["boxes"][b, q].detach(), gt["boxes"][b, gsel])
                    matched[b, q] = True
            p_t = (final.detach() / self.T).softmax(-1)
            n_pos, n_neg = matched.sum().float().sqrt(), (~matched).sum().float().sqrt()
            l = 0
            for layer in layers[:-1]:
                kl = F.kl_div((layer["fdr"] / self.T).log_softmax(-1), p_t, reduction="none").sum(-1).mean(-1) * self.T * self.T
                kl = kl * w
                l_pos = kl[matched].mean() if matched.any() else kl.sum() * 0
                l_neg = kl[~matched].mean() if (~matched).any() else kl.sum() * 0
                l = l + (l_pos * n_pos + l_neg * n_neg) / (n_pos + n_neg)
            losses["lsd"] = l * self.w["lsd"]
        if self.use_presence:
            tp = torch.zeros(B, self.C, device=out["presence"].device)
            lab = torch.where(gt["mask"], gt["labels"], torch.full_like(gt["labels"], -1))
            for b in range(B):
                v = lab[b][lab[b] >= 0]
                tp[b, v] = 1.0
            losses["pres"] = F.binary_cross_entropy_with_logits(out["presence"], tp) * self.w["pres"]
        # progressive weighting: dense 1.0→0.5, o2o 0.5→1.0
        w_dense, w_o2o = 1.0 - 0.5 * progress, 0.5 + 0.5 * progress
        total = 0
        for k, v in losses.items():
            if k == "n_fg":
                continue
            if k.startswith("d_"):
                total = total + w_dense * v
            elif k in ("lsd", "pres"):
                total = total + v
            else:
                total = total + w_o2o * v
        log = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in losses.items()}
        log["total"] = float(total)
        return total, log


def generalized_box_iou_loss(a: torch.Tensor, b: torch.Tensor, reduction: str = "none", eps: float = 1e-7) -> torch.Tensor:
    """1 - GIoU for paired boxes (N, 4) xyxy. Mask-free (torchvision's version uses boolean-mask assignment,
    which is fragile on the MPS backend)."""
    lt = torch.max(a[:, :2], b[:, :2]); rb = torch.min(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[:, 0] * wh[:, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a + area_b - inter + eps
    iou = inter / union
    elt = torch.min(a[:, :2], b[:, :2]); erb = torch.max(a[:, 2:], b[:, 2:])
    ewh = (erb - elt).clamp(min=0); enc = ewh[:, 0] * ewh[:, 1] + eps
    giou = iou - (enc - union) / enc
    loss = 1 - giou
    return loss.sum() if reduction == "sum" else loss.mean() if reduction == "mean" else loss


def box_iou_diag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.max(a[:, :2], b[:, :2]); rb = torch.min(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[:, 0] * wh[:, 1]
    area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / (area + 1e-7)


# ----------------------------------------------------------------------------------------------
# Contrastive denoising query construction (DN-DETR / DINO), fixed shapes for MPS
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def build_denoising(gt: Dict, num_classes: int, groups: int, box_noise: float = 0.4, label_noise: float = 0.5,
                    img_hw: Tuple[int, int] = (512, 512)) -> Dict[str, torch.Tensor]:
    """Returns dict with labels (B, Kd) [C = background/negative content], boxes (B, Kd, 4) noised xyxy,
    gt_idx, positive, valid; Kd = 2 * groups * M. Group g: [M positives | M negatives]."""
    B, M = gt["mask"].shape
    H, W = img_hw
    dev = gt["boxes"].device
    boxes = gt["boxes"].repeat(1, 2 * groups, 1)                                    # (B, Kd, 4)
    labels = gt["labels"].clamp(min=0).repeat(1, 2 * groups)
    valid = gt["mask"].repeat(1, 2 * groups)
    gt_idx = torch.arange(M, device=dev).repeat(2 * groups)[None].expand(B, -1)
    positive = torch.cat([torch.ones(M, dtype=torch.bool, device=dev), torch.zeros(M, dtype=torch.bool, device=dev)]).repeat(groups)[None].expand(B, -1)
    # label noise on positives
    flip = (torch.rand(B, boxes.shape[1], device=dev) < label_noise) & positive
    labels = torch.where(flip, torch.randint(0, num_classes, labels.shape, device=dev), labels)
    labels = torch.where(positive, labels, torch.full_like(labels, num_classes))   # negatives get the 'no-object' embedding
    # box noise: positives jitter within ±box_noise·(w/2,h/2); negatives in the ring [1, 2]×
    c = box_cxcywh(boxes)
    half = c[..., 2:] / 2
    diff = torch.cat([half, half], -1)
    sign = torch.randint(0, 2, boxes.shape, device=dev) * 2 - 1
    r = torch.rand(boxes.shape, device=dev)
    r = torch.where(positive[..., None], r, r + 1.0)
    noised = boxes + sign * r * diff * box_noise
    noised[..., 0::2] = noised[..., 0::2].clamp(0, W); noised[..., 1::2] = noised[..., 1::2].clamp(0, H)
    # ensure well-formed
    x1 = torch.minimum(noised[..., 0], noised[..., 2]); x2 = torch.maximum(noised[..., 0], noised[..., 2])
    y1 = torch.minimum(noised[..., 1], noised[..., 3]); y2 = torch.maximum(noised[..., 1], noised[..., 3])
    noised = torch.stack([x1, y1, x2 + (x2 - x1 < 1).float(), y2 + (y2 - y1 < 1).float()], -1)
    return dict(labels=labels, boxes=noised, gt_idx=gt_idx, positive=positive, valid=valid, groups=groups, M=M)


def denoising_attn_mask(K: int, dn: Dict, heads: int) -> torch.Tensor:
    """Bool mask (B*heads, K+Kd, K+Kd), True = blocked. Matching queries cannot see dn queries; dn group i cannot
    see group j; nobody sees padded dn slots except themselves (a query always sees itself)."""
    B, Kd = dn["valid"].shape
    G, M = dn["groups"], dn["M"]
    T = K + Kd
    mask = torch.zeros(B, T, T, dtype=torch.bool, device=dn["valid"].device)
    mask[:, :K, K:] = True
    for g in range(G):
        s, e = K + 2 * M * g, K + 2 * M * (g + 1)
        mask[:, s:e, K:s] = True
        mask[:, s:e, e:] = True
    mask[:, :, K:] |= ~dn["valid"][:, None, :]
    eye = torch.eye(T, dtype=torch.bool, device=mask.device)[None]
    mask = mask & ~eye
    return mask[:, None].expand(-1, heads, -1, -1).reshape(B * heads, T, T)
