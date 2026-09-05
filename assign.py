"""Label assignment for KESTREL.
  * TaskAlignedAssigner (TOOD / YOLOv8-style, top-k by t = s^alpha * u^beta) with YOLO26's small-target-aware
    candidate selection (STAL: an enlarged surrogate box is used ONLY to decide which anchors are candidates).
  * HungarianMatcher (DETR) with focal-class + L1 + GIoU cost.
Both work on padded targets: gt_boxes (B, M, 4) xyxy in pixels, gt_labels (B, M), gt_mask (B, M) bool."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, generalized_box_iou


def box_cxcywh(b):
    return torch.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2, b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], -1)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a (B, M, 4), b (B, N, 4) xyxy → (B, M, N)."""
    lt = torch.max(a[:, :, None, :2], b[:, None, :, :2])
    rb = torch.min(a[:, :, None, 2:], b[:, None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]))[:, :, None]
    area_b = ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]))[:, None, :]
    return inter / (area_a + area_b - inter + 1e-7)


class TaskAlignedAssigner:
    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0, stal_min: float = 0.0, stal_scale: float = 1.0):
        """stal: anchors are candidates if they fall inside max(gt, enlarged surrogate). surrogate side = max(side*stal_scale, stal_min px)."""
        self.topk, self.alpha, self.beta, self.stal_min, self.stal_scale = topk, alpha, beta, stal_min, stal_scale

    @torch.no_grad()
    def __call__(self, pred_scores: torch.Tensor, pred_boxes: torch.Tensor, anchors: torch.Tensor,
                 gt_labels: torch.Tensor, gt_boxes: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """pred_scores (B, N, C) sigmoid probs; pred_boxes (B, N, 4) xyxy; anchors (N, 2) centres;
        gt_* padded to M. Returns target_labels (B,N) [C = background], target_boxes (B,N,4), target_scores (B,N,C)
        (soft, IoU-normalised as in TOOD/YOLOv8), fg_mask (B,N), and the matched gt index (B,N)."""
        B, N, C = pred_scores.shape
        M = gt_boxes.shape[1]
        if M == 0:
            return dict(target_labels=torch.full((B, N), C, dtype=torch.long, device=pred_scores.device),
                        target_boxes=torch.zeros(B, N, 4, device=pred_scores.device), target_scores=torch.zeros(B, N, C, device=pred_scores.device),
                        fg_mask=torch.zeros(B, N, dtype=torch.bool, device=pred_scores.device), gt_idx=torch.zeros(B, N, dtype=torch.long, device=pred_scores.device))
        # --- candidates: anchor centre inside the (surrogate-enlarged) gt box  (B, M, N)
        cx, cy = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2, (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        w, h = gt_boxes[..., 2] - gt_boxes[..., 0], gt_boxes[..., 3] - gt_boxes[..., 1]
        sw, sh = torch.maximum(w * self.stal_scale, torch.full_like(w, self.stal_min)), torch.maximum(h * self.stal_scale, torch.full_like(h, self.stal_min))
        sw, sh = torch.maximum(sw, w), torch.maximum(sh, h)
        sur = torch.stack([cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2], -1)
        ax, ay = anchors[None, None, :, 0], anchors[None, None, :, 1]
        in_box = (ax > sur[..., 0:1]) & (ax < sur[..., 2:3]) & (ay > sur[..., 1:2]) & (ay < sur[..., 3:4])
        in_box = in_box & gt_mask[..., None]
        # --- alignment metric
        ious = pairwise_iou(gt_boxes, pred_boxes)                                          # (B, M, N)
        cls_p = pred_scores.gather(2, gt_labels.clamp(min=0)[:, None, :].expand(-1, N, -1)).transpose(1, 2)   # (B, M, N)
        metric = cls_p.clamp(min=0).pow(self.alpha) * ious.clamp(min=0).pow(self.beta)
        metric = metric * in_box
        # --- top-k per gt
        topk_val, topk_idx = metric.topk(min(self.topk, N), dim=-1)
        cand = torch.zeros_like(metric, dtype=torch.bool).scatter_(-1, topk_idx, topk_val > 0)
        cand = cand & in_box
        # --- resolve anchors claimed by several gts: keep the highest IoU
        multi = cand.sum(1) > 1                                                           # (B, N)
        if multi.any():
            best = (ious * cand).argmax(1)                                                 # (B, N) highest IoU among CLAIMANT gts only
            one_hot = F.one_hot(best, M).permute(0, 2, 1).bool()
            cand = torch.where(multi[:, None, :], one_hot & cand.any(1, keepdim=True), cand)
        fg = cand.any(1)                                                                   # (B, N)
        gt_idx = cand.float().argmax(1)                                                    # (B, N)
        tl = gt_labels.gather(1, gt_idx)
        tl = torch.where(fg, tl, torch.full_like(tl, C))
        tb = gt_boxes.gather(1, gt_idx[..., None].expand(-1, -1, 4))
        # --- soft scores: normalise the alignment metric per gt so the best anchor gets the gt's max IoU (TOOD)
        met = metric * cand
        max_met = met.amax(-1, keepdim=True)
        max_iou = (ious * cand).amax(-1, keepdim=True)
        norm = (met * max_iou / (max_met + 1e-9)).amax(1)                                # (B, N)
        ts = F.one_hot(tl.clamp(max=C - 1), C).float() * (norm * fg)[..., None]
        return dict(target_labels=tl, target_boxes=tb, target_scores=ts, fg_mask=fg, gt_idx=gt_idx)


class HungarianMatcher:
    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0, alpha: float = 0.25, gamma: float = 2.0):
        self.cc, self.cb, self.cg, self.alpha, self.gamma = cost_class, cost_bbox, cost_giou, alpha, gamma

    @torch.no_grad()
    def __call__(self, logits: torch.Tensor, boxes: torch.Tensor, gt_labels: torch.Tensor, gt_boxes: torch.Tensor,
                 gt_mask: torch.Tensor, img_hw: Tuple[int, int]):
        """logits (B, K, C); boxes (B, K, 4) xyxy pixels. Returns list of (query_idx, gt_idx) LongTensors per image."""
        B, K, C = logits.shape
        H, W = img_hw
        scale = boxes.new_tensor([W, H, W, H])
        prob = logits.sigmoid()
        out = []
        for b in range(B):
            m = gt_mask[b]
            n = int(m.sum())
            if n == 0:
                out.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
                continue
            gl, gb = gt_labels[b][m], gt_boxes[b][m]
            p = prob[b][:, gl]                                                              # (K, n)
            neg = (1 - self.alpha) * p.pow(self.gamma) * (-(1 - p + 1e-8).log())
            pos = self.alpha * (1 - p).pow(self.gamma) * (-(p + 1e-8).log())
            c_cls = pos - neg
            c_l1 = torch.cdist(box_cxcywh(boxes[b]) / scale, box_cxcywh(gb) / scale, p=1)
            c_giou = -generalized_box_iou(boxes[b], gb)
            cost = (self.cc * c_cls + self.cb * c_l1 + self.cg * c_giou).cpu().numpy()
            cost[~torch.isfinite(torch.from_numpy(cost)).numpy()] = 1e5
            qi, gi = linear_sum_assignment(cost)
            out.append((torch.as_tensor(qi, dtype=torch.long), torch.as_tensor(gi, dtype=torch.long)))
        return out
