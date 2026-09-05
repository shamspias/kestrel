"""
KESTREL — a reference PyTorch skeleton for a real-time, NMS-free, open-vocabulary,
anytime (adaptive-compute) vision architecture.

STATUS: design proposal. This file defines the ARCHITECTURE (forward pass, re-parameterisation,
vocabulary folding, anytime early-exit, slot memory) and the distribution-based loss primitives.
It has NOT been trained; the self-test at the bottom only verifies shapes, parameter counts,
FLOPs, re-parameterisation equivalence and ONNX exportability.

Design summary (see the accompanying write-up for the rationale and provenance of each idea)
--------------------------------------------------------------------------------------------
  input (any H, W multiple of 32; 2-D RoPE, no absolute position tables)
   │
   ├─ Stem + ConvStage(s4) + ConvStage(s8)        re-parameterisable dilated large-kernel DW blocks + GRN
   ├─ AttnStage(s16)  windowed attention, every k-th block global, 4 register tokens, 2-D RoPE, QK-norm, SwiGLU
   ├─ AttnStage(s32)  global attention (few tokens)               ← "distillation ports" for DINOv3 / SigLIP-2 teachers
   │       └─ temporal gate (zero-init) to previous-frame s32 tokens (video mode)
   ├─ PyramidFuser    PAN-style CNN cross-scale fusion (RepConv/CSP)            → P3, P4, P5
   ├─ DenseHead       one-to-many trained (TAL + STAL); 4 plain scalars per location (no DFL), quality, region embedding
   │       └─ QuerySelector: top-K by quality·class  → initial boxes + content queries (mixed query selection)
   ├─ AnytimeDecoder  L ≤ 6 layers: self-attn → RoI-gathered cross-attn (roi_align, export-friendly; replaces deformable attn)
   │                  → global cross-attn to s32 tokens+registers → SwiGLU;  FDR box distributions refined residually,
   │                  GO-LSD self-distillation makes early layers calibrated ⇒ per-query EARLY EXIT on distribution entropy
   ├─ PresenceHead    one token per concept attends to s32 tokens → "is the concept in the image?"  score = presence × query
   ├─ Vocabulary      region-embedding ↔ class/text embeddings; fold_vocabulary() bakes text prompts into static weights
   ├─ MaskHead        pixel-embedding dot product (stride 4)
   ├─ KeypointHead    SimCC-style coordinate classification, box-relative bins
   └─ SlotMemory      K object slots carried across frames as track queries (bootstrapped from detections), GRU update

Everything continuous (box edges, keypoint coordinates) is predicted as a categorical distribution over bins and
decoded by expectation, so *uncertainty = entropy* has one meaning across tasks and the same self-distillation applies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, box_iou


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
@dataclass
class KestrelConfig:
    # backbone
    in_ch: int = 3
    stem_ch: int = 32
    conv_dims: Tuple[int, int] = (64, 128)       # stages at stride 4 and 8
    conv_depths: Tuple[int, int] = (2, 4)
    conv_mlp_ratio: int = 3
    attn_dims: Tuple[int, int] = (256, 384)      # stages at stride 16 and 32
    attn_depths: Tuple[int, int] = (6, 4)
    head_dim: int = 32
    window: int = 8                              # window size (tokens) at stride 16
    global_every: int = 3                        # every k-th s16 block is global
    n_registers: int = 4
    # neck / heads
    neck_dim: int = 256
    d_model: int = 256
    embed_dim: int = 256                         # region/text embedding size
    num_classes: int = 80
    num_queries: int = 300
    dec_layers: int = 6
    dec_heads: int = 8
    fdr_bins: int = 16
    fdr_scale: float = 0.5                       # max edge offset as a fraction of the box side
    roi_grid: int = 3
    roi_context: float = 1.5                     # RoI enlargement factor for context
    mask_dim: int = 32
    num_keypoints: int = 17
    kpt_bins: int = 64
    kpt_roi: int = 7
    # video
    slots: int = 64
    # anytime exit thresholds (eval only)
    exit_p: float = 0.70                         # confident foreground
    exit_bg: float = 0.05                        # confident background
    exit_u: float = 0.15                         # normalised localisation entropy
    exit_min_layers: int = 2
    # ablation / training switches
    local_attn: str = "roi"                      # "roi" (RoI-gathered, export-native) | "deform" (multi-scale deformable)
    deform_points: int = 4
    use_presence: bool = True                    # gate query scores with the presence head
    presence_power: float = 1.0                  # score = sigmoid(z) * sigmoid(pi)^power; 1 = SAM-3 product, 0.5 = geometric mean, 0 = no gate
    detach_seeds: bool = True                    # detach seed boxes/content before the decoder (train stability)
    ls_init: float = 1e-2                        # LayerScale init for conv/attention residual branches

    @property
    def strides(self) -> Tuple[int, int, int]:
        return (8, 16, 32)


# ----------------------------------------------------------------------------------------------
# Re-parameterisable convolution primitives
# ----------------------------------------------------------------------------------------------
def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold a BatchNorm (eval statistics) into the preceding convolution. Works for grouped/depthwise convs."""
    std = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / std
    w = conv.weight * scale.reshape(-1, 1, 1, 1)
    b = bn.bias - bn.running_mean * scale
    if conv.bias is not None:
        b = b + conv.bias * scale
    return w, b


def pad_kernel(w: torch.Tensor, k: int) -> torch.Tensor:
    """Zero-pad a (O, I, kh, kw) kernel to (O, I, k, k), centred."""
    kh, kw = w.shape[-2:]
    ph, pw = (k - kh) // 2, (k - kw) // 2
    return F.pad(w, (pw, k - kw - pw, ph, k - kh - ph))


def dilated_to_dense(w: torch.Tensor, d: int) -> torch.Tensor:
    """Convert a small kernel with dilation d into the equivalent dense (sparse-valued) kernel."""
    if d == 1:
        return w
    k = w.shape[-1]
    dense = w.new_zeros(*w.shape[:-2], (k - 1) * d + 1, (k - 1) * d + 1)
    dense[..., ::d, ::d] = w
    return dense


class ConvBN(nn.Sequential):
    def __init__(self, cin, cout, k, s=1, p=None, d=1, groups=1):
        p = (k // 2) * d if p is None else p
        super().__init__(nn.Conv2d(cin, cout, k, s, p, dilation=d, groups=groups, bias=False), nn.BatchNorm2d(cout))

    @property
    def conv(self):
        return self[0]

    @property
    def bn(self):
        return self[1]


class RepConv(nn.Module):
    """RepVGG-style block: train with 3x3 + 1x1 (+ identity) branches, infer with one dense conv."""

    def __init__(self, cin, cout, k=3, s=1, act=True):
        super().__init__()
        self.cin, self.cout, self.k, self.s = cin, cout, k, s
        self.dense = ConvBN(cin, cout, k, s)
        self.pw = ConvBN(cin, cout, 1, s, 0) if k > 1 else None
        self.idn = nn.BatchNorm2d(cin) if (cin == cout and s == 1) else None
        self.act = nn.SiLU() if act else nn.Identity()
        self.fused: Optional[nn.Conv2d] = None

    def forward(self, x):
        if self.fused is not None:
            return self.act(self.fused(x))
        y = self.dense(x)
        if self.pw is not None:
            y = y + self.pw(x)
        if self.idn is not None:
            y = y + self.idn(x)
        return self.act(y)

    @torch.no_grad()
    def reparameterize(self):
        if self.fused is not None:
            return
        w, b = fuse_conv_bn(self.dense.conv, self.dense.bn)
        if self.pw is not None:
            w1, b1 = fuse_conv_bn(self.pw.conv, self.pw.bn)
            w, b = w + pad_kernel(w1, self.k), b + b1
        if self.idn is not None:
            eye = torch.zeros(self.cout, self.cin, 1, 1, device=w.device, dtype=w.dtype)
            eye[torch.arange(self.cin), torch.arange(self.cin)] = 1.0
            std = (self.idn.running_var + self.idn.eps).sqrt()
            w = w + pad_kernel(eye * (self.idn.weight / std).reshape(-1, 1, 1, 1), self.k)
            b = b + self.idn.bias - self.idn.running_mean * self.idn.weight / std
        self.fused = nn.Conv2d(self.cin, self.cout, self.k, self.s, self.k // 2, bias=True).to(w.device)
        self.fused.weight.copy_(w)
        self.fused.bias.copy_(b)
        del self.dense, self.pw, self.idn
        self.dense = self.pw = self.idn = None


class DilatedRepDW(nn.Module):
    """Large-kernel depthwise conv with parallel dilated small-kernel branches (UniRepLKNet-style
    'Dilated Reparam Block'). At inference all branches fold into ONE k×k depthwise conv."""

    def __init__(self, c, k=7, dilations=(1, 2, 3)):
        super().__init__()
        assert all((3 - 1) * d + 1 <= k for d in dilations)
        self.c, self.k = c, k
        self.main = ConvBN(c, c, k, groups=c)
        self.branches = nn.ModuleList([ConvBN(c, c, 3, d=d, groups=c) for d in dilations])
        self.dilations = dilations
        self.fused: Optional[nn.Conv2d] = None

    def forward(self, x):
        if self.fused is not None:
            return self.fused(x)
        y = self.main(x)
        for br in self.branches:
            y = y + br(x)
        return y

    @torch.no_grad()
    def reparameterize(self):
        if self.fused is not None:
            return
        w, b = fuse_conv_bn(self.main.conv, self.main.bn)
        for br, d in zip(self.branches, self.dilations):
            wb, bb = fuse_conv_bn(br.conv, br.bn)
            w, b = w + pad_kernel(dilated_to_dense(wb, d), self.k), b + bb
        self.fused = nn.Conv2d(self.c, self.c, self.k, 1, self.k // 2, groups=self.c, bias=True).to(w.device)
        self.fused.weight.copy_(w)
        self.fused.bias.copy_(b)
        del self.main, self.branches
        self.main = self.branches = None


class GRN(nn.Module):
    """Global Response Normalisation (ConvNeXt V2) — prevents channel collapse under MIM/distillation."""

    def __init__(self, c):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvBlock(nn.Module):
    """DW7x7(reparam,+BN) → 1x1 expand → GELU → GRN → 1x1 project, LayerScale residual."""

    def __init__(self, c, mlp_ratio=3, ls_init=1e-2):
        super().__init__()
        self.dw = DilatedRepDW(c)
        self.pw1 = nn.Conv2d(c, c * mlp_ratio, 1)
        self.act = nn.GELU()
        self.grn = GRN(c * mlp_ratio)
        self.pw2 = nn.Conv2d(c * mlp_ratio, c, 1)
        self.ls = nn.Parameter(ls_init * torch.ones(c, 1, 1))

    def forward(self, x):
        return x + self.ls * self.pw2(self.grn(self.act(self.pw1(self.dw(x)))))


class ConvStage(nn.Module):
    def __init__(self, cin, c, depth, mlp_ratio, down: bool, ls_init=1e-2):
        super().__init__()
        self.down = RepConv(cin, c, 3, 2) if down else nn.Identity()
        self.blocks = nn.Sequential(*[ConvBlock(c, mlp_ratio, ls_init) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.down(x))


# ----------------------------------------------------------------------------------------------
# Transformer primitives: RMSNorm, 2-D RoPE, SwiGLU, attention block with windows and registers
# ----------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope2d(h: int, w: int, dim_head: int, device, theta: float = 100.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Axial 2-D rotary embedding tables for an h×w token grid: returns cos, sin of shape (h*w, dim_head/2)."""
    n = dim_head // 4
    inv = 1.0 / (theta ** (torch.arange(n, device=device, dtype=torch.float32) / n))
    ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] * inv[None]      # (h, n)
    xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] * inv[None]      # (w, n)
    ang = torch.cat([ys[:, None, :].expand(h, w, n), xs[None, :, :].expand(h, w, n)], -1).reshape(h * w, dim_head // 2)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., T, dh); cos/sin broadcastable to (..., T, dh/2). Rotates interleaved pairs."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d, ratio=8 / 3):
        super().__init__()
        hidden = int(round(d * ratio / 32) * 32)
        self.w12 = nn.Linear(d, 2 * hidden)
        self.w3 = nn.Linear(hidden, d)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class AttnBlock(nn.Module):
    """Pre-RMSNorm attention block with QK-norm, 2-D RoPE, SwiGLU and LayerScale."""

    def __init__(self, d, head_dim=32, ls_init=1e-2):
        super().__init__()
        self.h, self.dh = d // head_dim, head_dim
        self.n1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.qn, self.kn = RMSNorm(head_dim), RMSNorm(head_dim)
        self.proj = nn.Linear(d, d)
        self.ls1 = nn.Parameter(ls_init * torch.ones(d))
        self.n2 = RMSNorm(d)
        self.ffn = SwiGLU(d)
        self.ls2 = nn.Parameter(ls_init * torch.ones(d))

    def attend(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.qn(q), cos, sin), apply_rope(self.kn(k), cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(o.transpose(1, 2).reshape(B, T, -1))

    def forward_tokens(self, x, cos, sin, mask=None):
        x = x + self.ls1 * self.attend(self.n1(x), cos, sin, mask)
        return x + self.ls2 * self.ffn(self.n2(x))

    def forward_global(self, tokens, regs, cos, sin):
        """Registers + all spatial tokens attend jointly; registers get identity rotation."""
        R = regs.shape[1]
        cos = torch.cat([cos.new_ones(R, cos.shape[-1]), cos], 0)
        sin = torch.cat([sin.new_zeros(R, sin.shape[-1]), sin], 0)
        x = self.forward_tokens(torch.cat([regs, tokens], 1), cos, sin)
        return x[:, R:], x[:, :R]

    def forward_window(self, tokens, H, W, win):
        """Non-overlapping window attention; RoPE uses global grid coordinates (relative ⇒ window-invariant)."""
        B, T, d = tokens.shape
        x = tokens.view(B, H, W, d)
        ph, pw = (-H) % win, (-W) % win
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
        Hp, Wp = H + ph, W + pw
        nh, nw = Hp // win, Wp // win
        cos, sin = rope2d(Hp, Wp, self.dh, tokens.device)
        cos = cos.view(nh, win, nw, win, -1).permute(0, 2, 1, 3, 4).reshape(nh * nw, win * win, -1)
        sin = sin.view(nh, win, nw, win, -1).permute(0, 2, 1, 3, 4).reshape(nh * nw, win * win, -1)
        x = x.view(B, nh, win, nw, win, d).permute(0, 1, 3, 2, 4, 5).reshape(B * nh * nw, win * win, d)
        mask = None
        if ph or pw:  # mask padded keys
            valid = torch.zeros(Hp, Wp, dtype=torch.bool, device=tokens.device)
            valid[:H, :W] = True
            valid = valid.view(nh, win, nw, win).permute(0, 2, 1, 3).reshape(nh * nw, win * win)
            mask = valid.repeat(B, 1)[:, None, None, :]           # (B*nW, 1, 1, T)
        cos = cos.repeat(B, 1, 1)[:, None]                          # (B*nW, 1, T, dh/2)
        sin = sin.repeat(B, 1, 1)[:, None]
        x = self.forward_tokens(x, cos, sin, mask)
        x = x.view(B, nh, nw, win, win, d).permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, d)[:, :H, :W]
        return x.reshape(B, T, d)


class AttnStage(nn.Module):
    """Patch-merge downsample → depth × AttnBlock. window=None ⇒ all-global."""

    def __init__(self, cin, d, depth, head_dim, window: Optional[int], global_every: int, n_registers: int, reg_in: Optional[int], ls_init=1e-2):
        super().__init__()
        self.down = nn.Conv2d(cin, d, 2, 2)
        self.blocks = nn.ModuleList([AttnBlock(d, head_dim, ls_init) for _ in range(depth)])
        self.window = window
        self.is_global = [window is None or (i % global_every == global_every - 1) or i == depth - 1 for i in range(depth)]
        self.registers = nn.Parameter(torch.randn(1, n_registers, d) * 0.02) if reg_in is None else None
        self.reg_in = nn.Linear(reg_in, d) if reg_in is not None else None
        self.dh = head_dim

    def forward(self, x, regs=None):
        x = self.down(x)
        B, d, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        regs = self.registers.expand(B, -1, -1) if regs is None else self.reg_in(regs)
        cos, sin = rope2d(H, W, self.dh, x.device)
        for blk, g in zip(self.blocks, self.is_global):
            if g:
                tokens, regs = blk.forward_global(tokens, regs, cos, sin)
            else:
                tokens = blk.forward_window(tokens, H, W, self.window)
        return tokens.transpose(1, 2).reshape(B, d, H, W), regs


# ----------------------------------------------------------------------------------------------
# Neck: PAN-style cross-scale fusion with re-parameterisable CSP blocks
# ----------------------------------------------------------------------------------------------
class CSPRep(nn.Module):
    def __init__(self, cin, cout, n=1):
        super().__init__()
        h = cout // 2
        self.c1, self.c2 = RepConv(cin, h, 1), RepConv(cin, h, 1)
        self.m = nn.Sequential(*[RepConv(h, h, 3) for _ in range(n)])
        self.c3 = RepConv(2 * h, cout, 1)

    def forward(self, x):
        return self.c3(torch.cat([self.m(self.c1(x)), self.c2(x)], 1))


class PyramidFuser(nn.Module):
    def __init__(self, cins: Sequence[int], nd: int):
        super().__init__()
        self.lat = nn.ModuleList([RepConv(c, nd, 1) for c in cins])
        self.td4, self.td3 = CSPRep(2 * nd, nd), CSPRep(2 * nd, nd)
        self.dn3, self.dn4 = RepConv(nd, nd, 3, 2), RepConv(nd, nd, 3, 2)
        self.bu4, self.bu5 = CSPRep(2 * nd, nd), CSPRep(2 * nd, nd)

    def forward(self, c3, c4, c5):
        l3, l4, l5 = (lat(c) for lat, c in zip(self.lat, (c3, c4, c5)))
        p5 = l5
        p4 = self.td4(torch.cat([F.interpolate(p5, size=l4.shape[-2:], mode="nearest"), l4], 1))
        p3 = self.td3(torch.cat([F.interpolate(p4, size=l3.shape[-2:], mode="nearest"), l3], 1))
        n4 = self.bu4(torch.cat([self.dn3(p3), p4], 1))
        n5 = self.bu5(torch.cat([self.dn4(n4), p5], 1))
        return p3, n4, n5


# ----------------------------------------------------------------------------------------------
# Vocabulary (closed-set class embeddings or folded text prompts) and presence head
# ----------------------------------------------------------------------------------------------
class Vocabulary(nn.Module):
    """logits = s · <normalise(region), normalise(class_embed)> + b.
    Closed-set: class_embed is a learned matrix. Open-vocabulary: text embeddings pass through a small
    adapter (RepRTA-like) and are then FOLDED into a static buffer, so deployment cost equals closed-set."""

    def __init__(self, num_classes, embed_dim, text_dim=None):
        super().__init__()
        self.class_embed = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.adapter = nn.Sequential(nn.Linear(text_dim or embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.logit_bias = nn.Parameter(torch.tensor(-4.0))       # prior: positives are rare
        self.register_buffer("folded", torch.zeros(0), persistent=False)

    @property
    def embeddings(self):
        return self.folded if self.folded.numel() else self.class_embed

    @property
    def num_classes(self):
        return self.embeddings.shape[0]

    @torch.no_grad()
    def fold_vocabulary(self, text_embeds: torch.Tensor):
        """Bake a prompt vocabulary (N, text_dim) into the head. The text encoder is no longer needed."""
        self.folded = F.normalize(self.adapter(text_embeds), dim=-1)

    def clear_vocabulary(self):
        self.folded = torch.zeros(0, device=self.class_embed.device)

    def forward(self, region: torch.Tensor):
        e = F.normalize(self.embeddings, dim=-1)
        return self.logit_scale.exp() * F.normalize(region, dim=-1) @ e.t() + self.logit_bias


class PresenceHead(nn.Module):
    """One token per concept attends to the global (s32 + register) tokens and answers
    'is this concept anywhere in the image?'. Final score = presence × per-query score (SAM 3 idea)."""

    def __init__(self, d, embed_dim, heads=8):
        super().__init__()
        self.c_proj = nn.Linear(embed_dim, d)
        self.xa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, class_embed: torch.Tensor, mem: torch.Tensor):
        B = mem.shape[0]
        c = self.c_proj(class_embed)[None].expand(B, -1, -1)
        c = c + self.xa(c, mem, mem, need_weights=False)[0]
        return self.mlp(self.norm(c)).squeeze(-1)                 # (B, C)


# ----------------------------------------------------------------------------------------------
# Dense one-to-many head and query selection
# ----------------------------------------------------------------------------------------------
class DenseHead(nn.Module):
    """Per-location: region embedding (→ class logits via Vocabulary), 4 edge distances in stride units
    (plain scalars — cheap, unbounded, quantisation-friendly; the decoder does the fine distributional work),
    and an IoU-aware quality score. Trained one-to-many (TAL top-k + STAL); at inference it seeds the queries."""

    def __init__(self, nd, embed_dim, strides, width=128):
        super().__init__()
        self.strides = strides
        # light per-level tower: 1x1 squeeze → one dense 3x3 (both re-parameterisable)
        self.tower = nn.ModuleList([nn.Sequential(RepConv(nd, width, 1), RepConv(width, width, 3)) for _ in strides])
        self.region = nn.Conv2d(width, embed_dim, 1)
        self.reg = nn.Conv2d(width, 4, 1)
        self.qual = nn.Conv2d(width, 1, 1)
        nn.init.constant_(self.qual.bias, -4.0)
        self.width = width

    def forward(self, feats: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat, region, ltrb, qual, anchors, strides = [], [], [], [], [], []
        for f, tower, s in zip(feats, self.tower, self.strides):
            t = tower(f)
            B, C, H, W = t.shape
            feat.append(t.flatten(2).transpose(1, 2))
            region.append(self.region(t).flatten(2).transpose(1, 2))
            ltrb.append(F.softplus(self.reg(t)).flatten(2).transpose(1, 2))
            qual.append(self.qual(t).flatten(2).squeeze(1))
            ys, xs = torch.meshgrid(torch.arange(H, device=f.device), torch.arange(W, device=f.device), indexing="ij")
            anchors.append(torch.stack([(xs + 0.5) * s, (ys + 0.5) * s], -1).reshape(-1, 2).float())
            strides.append(torch.full((H * W,), float(s), device=f.device))
        out = dict(feat=torch.cat(feat, 1), region=torch.cat(region, 1), ltrb=torch.cat(ltrb, 1),
                   quality=torch.cat(qual, 1), anchors=torch.cat(anchors, 0), strides=torch.cat(strides, 0))
        a, s, d = out["anchors"], out["strides"][:, None], out["ltrb"]
        out["boxes"] = torch.stack([a[:, 0] - d[..., 0] * s[:, 0], a[:, 1] - d[..., 1] * s[:, 0],
                                    a[:, 0] + d[..., 2] * s[:, 0], a[:, 1] + d[..., 3] * s[:, 0]], -1)
        return out


class QuerySelector(nn.Module):
    """Top-K locations by quality·class → initial boxes (positional part) and content queries
    (mixed: projected local feature + learned embedding)."""

    def __init__(self, feat_dim, d, num_queries):
        super().__init__()
        self.k = num_queries
        self.proj = nn.Linear(feat_dim, d)
        self.tgt = nn.Parameter(torch.randn(num_queries, d) * 0.02)

    def forward(self, dense: Dict[str, torch.Tensor], logits: torch.Tensor, k: Optional[int] = None, detach: bool = False):
        k = k or self.k
        score = dense["quality"].sigmoid() * logits.sigmoid().amax(-1)
        idx = score.topk(k, dim=1).indices                                  # (B, K)
        gather = lambda t: t.gather(1, idx[..., None].expand(-1, -1, t.shape[-1]))
        feat, boxes = gather(dense["feat"]), gather(dense["boxes"])
        if detach:                                                          # cut the graph UPSTREAM of the projection so proj/tgt still train
            feat, boxes = feat.detach(), boxes.detach()
        q = self.proj(feat) + self.tgt[:k][None]
        return q, boxes, gather(logits), idx


# ----------------------------------------------------------------------------------------------
# Distributional regression (FDR / DFL-style) and its losses
# ----------------------------------------------------------------------------------------------
class FDR(nn.Module):
    """Fine-grained Distribution Refinement: each box edge is a categorical distribution over `bins`
    offsets (fraction of the initial box side, finer near zero), decoded by expectation.
    Entropy of the distribution is the localisation uncertainty used for early exit."""

    def __init__(self, bins=16, scale=0.5):
        super().__init__()
        u = torch.linspace(-1, 1, bins)
        self.register_buffer("offsets", scale * u * u.abs())        # quadratic spacing: fine near 0

    def decode(self, logits: torch.Tensor, box_init: torch.Tensor) -> torch.Tensor:
        off = (logits.softmax(-1) * self.offsets).sum(-1)              # (B, K, 4)
        w = (box_init[..., 2] - box_init[..., 0]).clamp(min=1.0)
        h = (box_init[..., 3] - box_init[..., 1]).clamp(min=1.0)
        return box_init + off * torch.stack([w, h, w, h], -1)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        p = logits.softmax(-1)
        return -(p * (p + 1e-9).log()).sum(-1).mean(-1) / math.log(logits.shape[-1])   # (B, K) in [0, 1]

    def target_distribution(self, target_off: torch.Tensor) -> torch.Tensor:
        """Two-bin linear interpolation of a continuous target (as in DFL) → soft target over bins."""
        offs = self.offsets
        t = target_off.clamp(offs[0], offs[-1])
        idx = torch.searchsorted(offs, t.contiguous()).clamp(1, len(offs) - 1)
        lo, hi = offs[idx - 1], offs[idx]
        w_hi = (t - lo) / (hi - lo).clamp(min=1e-6)
        dist = torch.zeros(*t.shape, len(offs), device=t.device)
        dist.scatter_(-1, (idx - 1)[..., None], (1 - w_hi)[..., None])
        dist.scatter_add_(-1, idx[..., None], w_hi[..., None])
        return dist


def fdr_loss(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between predicted edge distributions and the soft two-bin target."""
    return -(target_dist * logits.log_softmax(-1)).sum(-1).mean()


def golsd_loss(early_logits: torch.Tensor, final_logits: torch.Tensor, T: float = 5.0) -> torch.Tensor:
    """Global Optimal Localisation Self-Distillation: the last layer's (sharpened) distributions teach
    earlier layers. This is also what makes EARLY EXIT safe — every layer is trained to agree with the last."""
    p_t = (final_logits.detach() / T).softmax(-1)
    return F.kl_div((early_logits / T).log_softmax(-1), p_t, reduction="batchmean") * T * T


def sine_embed(x: torch.Tensor, num_feats=64, temperature=10000.0) -> torch.Tensor:
    """(…, n) normalised coordinates → (…, n*num_feats) sine/cosine features."""
    dim_t = temperature ** (2 * torch.div(torch.arange(num_feats, device=x.device), 2, rounding_mode="floor") / num_feats)
    pos = x[..., None] * 2 * math.pi / dim_t
    return torch.stack([pos[..., 0::2].sin(), pos[..., 1::2].cos()], -1).flatten(-3)


def boxes_to_norm_cxcywh(boxes: torch.Tensor, img_hw: Tuple[int, int]) -> torch.Tensor:
    H, W = img_hw
    cx, cy = (boxes[..., 0] + boxes[..., 2]) / 2 / W, (boxes[..., 1] + boxes[..., 3]) / 2 / H
    w, h = (boxes[..., 2] - boxes[..., 0]) / W, (boxes[..., 3] - boxes[..., 1]) / H
    return torch.stack([cx, cy, w, h], -1)


# ----------------------------------------------------------------------------------------------
# Anytime decoder with RoI-gathered cross-attention
# ----------------------------------------------------------------------------------------------
class RoICrossAttention(nn.Module):
    """Box-conditioned cross-attention. Keys/values are gathered with roi_align from K/V maps that were projected
    ONCE per image (shared by all decoder layers), so the per-query cost is just the attention itself.
    roi_align is a standard ONNX/TensorRT/CoreML op — this is the export-friendly replacement for deformable attention."""

    def __init__(self, d, heads, grid, n_levels, context, strides):
        super().__init__()
        self.h, self.dh, self.g, self.ctx, self.strides = heads, d // heads, grid, context, strides
        self.q_proj, self.out = nn.Linear(d, d), nn.Linear(d, d)
        self.pos = nn.Parameter(torch.randn(1, n_levels * grid * grid, d) * 0.02)   # level + grid-cell embedding

    def gather(self, boxes: torch.Tensor, kv_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        B, K, _ = boxes.shape
        cx, cy = (boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2
        w, h = (boxes[..., 2] - boxes[..., 0]) * self.ctx / 2, (boxes[..., 3] - boxes[..., 1]) * self.ctx / 2
        rois = torch.stack([cx - w, cy - h, cx + w, cy + h], -1).reshape(B * K, 4)
        bidx = torch.arange(B, device=boxes.device, dtype=boxes.dtype)[:, None].expand(B, K).reshape(-1, 1)   # tracer-safe (no repeat_interleave on a traced int)
        rois = torch.cat([bidx, rois], 1)
        toks = [roi_align(f, rois, (self.g, self.g), spatial_scale=1.0 / s, sampling_ratio=2, aligned=True).flatten(2)
                for f, s in zip(kv_maps, self.strides)]
        return torch.cat(toks, -1).transpose(1, 2)                    # (B*K, L*g*g, 2d)

    def forward(self, q, boxes, kv_maps):
        B, K, d = q.shape
        kv = self.gather(boxes, kv_maps)
        k, v = kv.chunk(2, -1)
        k = k + self.pos
        T = k.shape[1]
        qh = self.q_proj(q).reshape(B * K, 1, self.h, self.dh).transpose(1, 2)          # (BK, h, 1, dh)
        kh = k.reshape(B * K, T, self.h, self.dh).transpose(1, 2)
        vh = v.reshape(B * K, T, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(qh, kh, vh)                                   # (BK, h, 1, dh)
        return self.out(o.transpose(1, 2).reshape(B, K, d))


class DeformCrossAttention(nn.Module):
    """Multi-scale deformable attention (Deformable DETR / DINO, box-modulated offsets) in pure PyTorch with
    grid_sample — the ablation baseline for RoICrossAttention. Values are projected per layer over the whole maps."""

    def __init__(self, d, heads, n_levels, n_points, strides):
        super().__init__()
        self.h, self.dh, self.L, self.P, self.strides = heads, d // heads, n_levels, n_points, strides
        self.offset = nn.Linear(d, heads * n_levels * n_points * 2)
        self.attn = nn.Linear(d, heads * n_levels * n_points)
        self.value = nn.ModuleList([nn.Conv2d(2 * d, d, 1) for _ in range(n_levels)])   # takes the shared K/V maps (2d) → values
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.offset.weight)
        th = torch.arange(heads, dtype=torch.float32) * (2.0 * math.pi / heads)
        grid = torch.stack([th.cos(), th.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(heads, 1, 1, 2).repeat(1, n_levels, n_points, 1)
        for i in range(n_points):
            grid[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.offset.bias.copy_(grid.view(-1))
        nn.init.zeros_(self.attn.weight); nn.init.zeros_(self.attn.bias)

    def forward(self, q, boxes, kv_maps):
        B, K, d = q.shape
        H_img = None
        off = self.offset(q).view(B, K, self.h, self.L, self.P, 2)
        w = self.attn(q).view(B, K, self.h, self.L * self.P).softmax(-1).view(B, K, self.h, self.L, self.P)
        c = torch.stack([(boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2], -1)      # (B, K, 2) px
        wh = torch.stack([boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]], -1).clamp(min=1.0)
        loc = c[:, :, None, None, None, :] + off * wh[:, :, None, None, None, :] / (2 * self.P)                  # (B,K,h,L,P,2) px
        outs = []
        for l, (f, s) in enumerate(zip(kv_maps, self.strides)):
            v = self.value[l](f)                                                                                  # (B, d, Hl, Wl)
            Hl, Wl = v.shape[-2:]
            v = v.view(B * self.h, self.dh, Hl, Wl)
            g = loc[:, :, :, l]                                                                                    # (B, K, h, P, 2)
            g = torch.stack([g[..., 0] / (Wl * s) * 2 - 1, g[..., 1] / (Hl * s) * 2 - 1], -1)                    # normalised
            g = g.permute(0, 2, 1, 3, 4).reshape(B * self.h, K, self.P, 2)
            samp = F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)               # (B*h, dh, K, P)
            outs.append(samp.view(B, self.h, self.dh, K, self.P))
        samp = torch.cat(outs, -1)                                                                                # (B, h, dh, K, L*P)
        w = w.permute(0, 2, 1, 3, 4).reshape(B, self.h, 1, K, self.L * self.P)
        o = (samp * w).sum(-1).permute(0, 3, 1, 2).reshape(B, K, d)
        return self.out(o)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: KestrelConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.dec_heads
        self.n1, self.sa = RMSNorm(d), nn.MultiheadAttention(d, h, batch_first=True)
        self.n2 = RMSNorm(d)
        if cfg.local_attn == "deform":
            self.roi_xa = DeformCrossAttention(d, h, len(cfg.strides), cfg.deform_points, cfg.strides)
        else:
            self.roi_xa = RoICrossAttention(d, h, cfg.roi_grid, len(cfg.strides), cfg.roi_context, cfg.strides)
        self.n3, self.glob_xa = RMSNorm(d), nn.MultiheadAttention(d, h, batch_first=True)
        self.n4, self.ffn = RMSNorm(d), SwiGLU(d)
        self.delta_fdr = nn.Linear(d, 4 * cfg.fdr_bins)
        nn.init.zeros_(self.delta_fdr.weight); nn.init.zeros_(self.delta_fdr.bias)
        self.region = nn.Linear(d, cfg.embed_dim)

    def forward(self, q, qpos, boxes, kv_maps, mem, mem_pos, sa_mask=None):
        x = self.n1(q) + qpos
        q = q + self.sa(x, x, self.n1(q), attn_mask=sa_mask, need_weights=False)[0]
        q = q + self.roi_xa(self.n2(q), boxes.detach(), kv_maps)
        q = q + self.glob_xa(self.n3(q) + qpos, mem + mem_pos, mem, need_weights=False)[0]
        q = q + self.ffn(self.n4(q))
        return q


class AnytimeDecoder(nn.Module):
    def __init__(self, cfg: KestrelConfig, vocab: Vocabulary):
        super().__init__()
        self.cfg = cfg
        object.__setattr__(self, "vocab", vocab)          # shared with KESTREL.vocab; not registered twice (state_dict / EMA)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.dec_layers)])
        self.fdr = FDR(cfg.fdr_bins, cfg.fdr_scale)
        self.pos_mlp = nn.Sequential(nn.Linear(4 * 64, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
        self.mem_proj = nn.Linear(cfg.attn_dims[1], cfg.d_model)
        # K/V maps for RoI cross-attention: projected once per image, shared by every decoder layer
        self.kv_proj = nn.ModuleList([nn.Conv2d(cfg.neck_dim, 2 * cfg.d_model, 1) for _ in cfg.strides])

    def qpos(self, boxes, img_hw):
        return self.pos_mlp(sine_embed(boxes_to_norm_cxcywh(boxes.detach(), img_hw)))

    def kv_maps(self, feats):
        return [p(f) for p, f in zip(self.kv_proj, feats)]

    def forward(self, q, box_init, feats, mem, mem_pos, img_hw, sa_mask=None, max_layers: Optional[int] = None) -> List[Dict[str, torch.Tensor]]:
        """Standard (training / batched) path: run `max_layers` layers, return every layer's outputs (aux losses)."""
        B, K, _ = q.shape
        mem, kv = self.mem_proj(mem), self.kv_maps(feats)
        fdr_logits = q.new_zeros(B, K, 4, self.cfg.fdr_bins)
        boxes, outs = box_init, []
        for layer in self.layers[: max_layers or len(self.layers)]:
            q = layer(q, self.qpos(boxes, img_hw), boxes, kv, mem, mem_pos, sa_mask)
            fdr_logits = fdr_logits + layer.delta_fdr(q).view(B, K, 4, -1)
            boxes = self.fdr.decode(fdr_logits, box_init)
            region = layer.region(q)
            outs.append(dict(q=q, boxes=boxes, fdr=fdr_logits, region=region, logits=self.vocab(region),
                             uncertainty=self.fdr.entropy(fdr_logits)))
        return outs

    @torch.no_grad()
    def forward_anytime(self, q, box_init, feats, mem, mem_pos, img_hw) -> Dict[str, torch.Tensor]:
        """Per-query early exit (batch element by batch element). A query leaves the refinement loop when it is
        confidently foreground with sharp edge distributions, or confidently background. Converged queries are
        physically removed from later layers' compute (only the remaining ones attend to each other)."""
        cfg, B, K, d = self.cfg, *q.shape
        mem, kv = self.mem_proj(mem), self.kv_maps(feats)
        results = []
        for b in range(B):
            qb, boxb0 = q[b:b + 1], box_init[b:b + 1]
            fdr_b = qb.new_zeros(1, K, 4, cfg.fdr_bins)
            boxes_b, region_b = boxb0.clone(), qb.new_zeros(1, K, cfg.embed_dim)
            exit_layer = torch.full((K,), len(self.layers), dtype=torch.long, device=q.device)
            active = torch.ones(K, dtype=torch.bool, device=q.device)
            fb, mb, mpb = [f[b:b + 1] for f in kv], mem[b:b + 1], mem_pos
            for li, layer in enumerate(self.layers):
                ia = active.nonzero().squeeze(1)
                if ia.numel() == 0:
                    break
                qa = layer(qb[:, ia], self.qpos(boxes_b[:, ia], img_hw), boxes_b[:, ia], fb, mb, mpb)
                fdr_a = fdr_b[:, ia] + layer.delta_fdr(qa).view(1, -1, 4, cfg.fdr_bins)
                qb[:, ia], fdr_b[:, ia] = qa, fdr_a
                boxes_b[:, ia] = self.fdr.decode(fdr_a, boxb0[:, ia])
                region_b[:, ia] = layer.region(qa)
                p = self.vocab(region_b[:, ia]).sigmoid().amax(-1)[0]
                u = self.fdr.entropy(fdr_a)[0]
                if li + 1 >= cfg.exit_min_layers:
                    done = ((p > cfg.exit_p) & (u < cfg.exit_u)) | (p < cfg.exit_bg)
                    exit_layer[ia[done]] = li + 1
                    active[ia[done]] = False
            results.append(dict(q=qb, boxes=boxes_b, fdr=fdr_b, region=region_b, logits=self.vocab(region_b),
                                uncertainty=self.fdr.entropy(fdr_b), exit_layer=exit_layer[None]))
        return {k: torch.cat([r[k] for r in results], 0) for k in results[0]}


# ----------------------------------------------------------------------------------------------
# Task heads: masks (pixel-embedding dot product) and keypoints (coordinate classification)
# ----------------------------------------------------------------------------------------------
class MaskHead(nn.Module):
    def __init__(self, nd, d, mask_dim):
        super().__init__()
        self.pixel = nn.Sequential(RepConv(nd, 128, 1), nn.Upsample(scale_factor=2, mode="nearest"),
                                   RepConv(128, 64, 3), nn.Conv2d(64, mask_dim, 1))
        self.query = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, mask_dim))

    def forward(self, p3, q):
        return torch.einsum("bkc,bchw->bkhw", self.query(q), self.pixel(p3))       # (B, K, H/4, W/4) logits


class KeypointHead(nn.Module):
    """SimCC/RTMO-style: per keypoint, two 1-D distributions (x, y) over box-relative bins + visibility."""

    def __init__(self, nd, d, num_kpts, bins, roi):
        super().__init__()
        self.J, self.bins, self.roi = num_kpts, bins, roi
        self.conv = nn.Sequential(nn.Conv2d(nd, nd, 3, padding=1), nn.GELU(), nn.Conv2d(nd, 32, 1))
        self.fc = nn.Linear(32 * roi * roi + d, d)
        self.out = nn.Linear(d, num_kpts * (2 * bins + 1))
        self.register_buffer("grid", torch.linspace(-0.25, 1.25, bins))            # margin around the box

    def forward(self, p3, q, boxes, stride=8):
        B, K, d = q.shape
        bidx = torch.arange(B, device=q.device, dtype=boxes.dtype).repeat_interleave(K)[:, None]
        rois = torch.cat([bidx, boxes.reshape(B * K, 4)], 1)
        f = self.conv(roi_align(p3, rois, (self.roi, self.roi), spatial_scale=1.0 / stride, sampling_ratio=2, aligned=True))
        z = F.gelu(self.fc(torch.cat([f.flatten(1), q.reshape(B * K, d)], 1)))
        o = self.out(z).view(B, K, self.J, 2 * self.bins + 1)
        xy_logits, vis = o[..., : 2 * self.bins].view(B, K, self.J, 2, self.bins), o[..., -1]
        rel = (xy_logits.softmax(-1) * self.grid).sum(-1)                                   # (B, K, J, 2) in box coords
        wh = torch.stack([boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]], -1)[:, :, None]
        xy = boxes[..., None, :2] + rel * wh
        return dict(kpt_xy=xy, kpt_vis=vis, kpt_logits=xy_logits)


# ----------------------------------------------------------------------------------------------
# Slot memory (video): object slots carried across frames as track queries
# ----------------------------------------------------------------------------------------------
class SlotMemory(nn.Module):
    """A fixed bank of K object slots (constant latency regardless of object count). Alive slots are appended to
    the decoder as track queries; after decoding they are updated with a GRU; unmatched confident detections
    spawn new slots; slots that stay low-confidence for `patience` frames are freed. (Training would use a
    tracklet-aware assignment; only the inference mechanics are implemented here.)"""

    def __init__(self, d, slots):
        super().__init__()
        self.K, self.d = slots, d
        self.gru = nn.GRUCell(d, d)
        self.slot_pos = nn.Parameter(torch.randn(1, slots, d) * 0.02)

    def init_state(self, B, device):
        return dict(emb=torch.zeros(B, self.K, self.d, device=device), box=torch.zeros(B, self.K, 4, device=device),
                    score=torch.zeros(B, self.K, device=device), age=torch.zeros(B, self.K, dtype=torch.long, device=device),
                    alive=torch.zeros(B, self.K, dtype=torch.bool, device=device), ids=torch.full((B, self.K), -1, dtype=torch.long, device=device),
                    next_id=0)

    @torch.no_grad()
    def update(self, state, track_q, track_boxes, track_scores, det_q, det_boxes, det_scores,
               spawn_thr=0.5, kill_thr=0.3, patience=5, iou_thr=0.5):
        B = track_q.shape[0]
        emb = self.gru(track_q.reshape(-1, self.d), state["emb"].reshape(-1, self.d)).view(B, self.K, self.d)
        alive = state["alive"]
        state["emb"] = torch.where(alive[..., None], emb, state["emb"])
        state["box"] = torch.where(alive[..., None], track_boxes, state["box"])
        state["score"] = torch.where(alive, track_scores, state["score"])
        state["age"] = torch.where(alive & (track_scores < kill_thr), state["age"] + 1, torch.zeros_like(state["age"]))
        state["alive"] = alive & (state["age"] < patience)
        for b in range(B):                                        # spawn new slots from confident, unmatched detections
            cand = (det_scores[b] > spawn_thr).nonzero().squeeze(1)
            if cand.numel() == 0:
                continue
            if state["alive"][b].any():
                ious = box_iou(det_boxes[b, cand], state["box"][b][state["alive"][b]]).amax(1)
                cand = cand[ious < iou_thr]
            free = (~state["alive"][b]).nonzero().squeeze(1)
            n = min(cand.numel(), free.numel())
            if n:
                c, f = cand[:n], free[:n]
                state["emb"][b, f], state["box"][b, f], state["score"][b, f] = det_q[b, c], det_boxes[b, c], det_scores[b, c]
                state["age"][b, f], state["alive"][b, f] = 0, True
                state["ids"][b, f] = torch.arange(state["next_id"], state["next_id"] + n, device=det_q.device)
                state["next_id"] += n
        return state


# ----------------------------------------------------------------------------------------------
# The full model
# ----------------------------------------------------------------------------------------------
class KESTREL(nn.Module):
    def __init__(self, cfg: KestrelConfig = KestrelConfig(), text_dim: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        c4, c8 = cfg.conv_dims
        d16, d32 = cfg.attn_dims
        self.stem = nn.Sequential(RepConv(cfg.in_ch, cfg.stem_ch, 3, 2), RepConv(cfg.stem_ch, c4, 3, 2))
        self.s4 = ConvStage(c4, c4, cfg.conv_depths[0], cfg.conv_mlp_ratio, down=False, ls_init=cfg.ls_init)
        self.s8 = ConvStage(c4, c8, cfg.conv_depths[1], cfg.conv_mlp_ratio, down=True, ls_init=cfg.ls_init)
        self.s16 = AttnStage(c8, d16, cfg.attn_depths[0], cfg.head_dim, cfg.window, cfg.global_every, cfg.n_registers, reg_in=None, ls_init=cfg.ls_init)
        self.s32 = AttnStage(d16, d32, cfg.attn_depths[1], cfg.head_dim, None, cfg.global_every, cfg.n_registers, reg_in=d16, ls_init=cfg.ls_init)
        # temporal context: zero-initialised gate ⇒ the single-image model is unchanged when no memory is given
        self.temporal_xa = nn.MultiheadAttention(d32, d32 // cfg.head_dim, batch_first=True)
        self.temporal_gate = nn.Parameter(torch.zeros(1))
        self.neck = PyramidFuser((c8, d16, d32), cfg.neck_dim)
        self.vocab = Vocabulary(cfg.num_classes, cfg.embed_dim, text_dim)
        self.dense = DenseHead(cfg.neck_dim, cfg.embed_dim, cfg.strides)
        self.select = QuerySelector(self.dense.width, cfg.d_model, cfg.num_queries)
        self.decoder = AnytimeDecoder(cfg, self.vocab)
        self.presence = PresenceHead(cfg.d_model, cfg.embed_dim, cfg.dec_heads)
        self.mem_pos_proj = nn.Linear(2 * 64, cfg.d_model)
        self.mask_head = MaskHead(cfg.neck_dim, cfg.d_model, cfg.mask_dim)
        self.kpt_head = KeypointHead(cfg.neck_dim, cfg.d_model, cfg.num_keypoints, cfg.kpt_bins, cfg.kpt_roi)
        self.slots = SlotMemory(cfg.d_model, cfg.slots)
        self.dn_embed = nn.Embedding(cfg.num_classes + 1, cfg.d_model)             # denoising content queries (training only)

    # ----- backbone --------------------------------------------------------------------------
    def backbone(self, images, prev_mem: Optional[torch.Tensor] = None):
        x = self.s4(self.stem(images))
        c3 = self.s8(x)
        c4, regs = self.s16(c3)
        c5, regs = self.s32(c4, regs)
        B, d, H, W = c5.shape
        mem = torch.cat([regs, c5.flatten(2).transpose(1, 2)], 1)                  # (B, R + H*W, d32)
        if prev_mem is not None:                                                    # video: gated temporal attention
            mem = mem + torch.tanh(self.temporal_gate) * self.temporal_xa(mem, prev_mem, prev_mem, need_weights=False)[0]
            c5 = mem[:, regs.shape[1]:].transpose(1, 2).reshape(B, d, H, W)
        ys, xs = torch.meshgrid(torch.arange(H, device=c5.device), torch.arange(W, device=c5.device), indexing="ij")
        grid = torch.stack([(xs + 0.5) / W, (ys + 0.5) / H], -1).reshape(-1, 2).float()
        mem_pos = self.mem_pos_proj(sine_embed(grid))
        mem_pos = torch.cat([mem_pos.new_zeros(regs.shape[1], mem_pos.shape[-1]), mem_pos], 0)[None]   # (1, R+HW, d)
        return c3, c4, c5, mem, mem_pos

    # ----- full forward ----------------------------------------------------------------------
    def forward(self, images: torch.Tensor, anytime: bool = False, state: Optional[dict] = None,
                return_masks: bool = True, return_kpts: bool = False, max_layers: Optional[int] = None,
                num_queries: Optional[int] = None, dn: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        B, _, H, W = images.shape
        assert H % 32 == 0 and W % 32 == 0, "input sides must be multiples of 32"
        prev_mem = state.get("scene_mem") if state else None
        c3, c4, c5, mem, mem_pos = self.backbone(images, prev_mem)
        p3, p4, p5 = self.neck(c3, c4, c5)
        feats = (p3, p4, p5)

        dense = self.dense(feats)
        dense_logits = self.vocab(dense["region"])
        q, box_init, init_logits, idx = self.select(dense, dense_logits, num_queries, detach=self.cfg.detach_seeds)
        K = q.shape[1]
        # training: contrastive denoising queries appended after the detection queries
        sa_mask = None
        if dn is not None:
            q = torch.cat([q, self.dn_embed(dn["labels"])], 1)
            box_init = torch.cat([box_init, dn["boxes"]], 1)
            sa_mask = dn["attn_mask"]

        # video: append alive slots as track queries (dead slots masked out of self-attention)
        if state is not None:
            q = torch.cat([q, state["emb"] + self.slots.slot_pos], 1)
            box_init = torch.cat([box_init, torch.where(state["alive"][..., None], state["box"], box_init[:, :1].expand(-1, self.cfg.slots, -1))], 1)
            dead = torch.cat([torch.zeros(B, K, dtype=torch.bool, device=q.device), ~state["alive"]], 1)   # (B, K+S)
            sa_mask = dead[:, None, None, :].expand(-1, self.cfg.dec_heads, q.shape[1], -1).reshape(B * self.cfg.dec_heads, q.shape[1], -1)

        out: Dict[str, torch.Tensor] = dict(dense_logits=dense_logits, dense_boxes=dense["boxes"], dense_quality=dense["quality"],
                                            dense_anchors=dense["anchors"], dense_strides=dense["strides"], query_idx=idx,
                                            init_logits=init_logits, box_init=box_init[:, :K])
        if anytime and not self.training and state is None:
            final = self.decoder.forward_anytime(q, box_init, feats, mem, mem_pos, (H, W))
            out["exit_layer"] = final["exit_layer"]
        else:
            layers = self.decoder(q, box_init, feats, mem, mem_pos, (H, W), sa_mask, max_layers)
            if dn is not None:                                                             # split off the denoising part
                out["dn_layers"] = [{k: v[:, K:] for k, v in l.items()} for l in layers]
                out["dn"] = dn
                layers = [{k: v[:, :K] for k, v in l.items()} for l in layers]
            out["aux"] = layers[:-1]
            final = layers[-1]

        presence = self.presence(self.vocab.embeddings, self.decoder.mem_proj(mem))       # (B, C)
        scores = final["logits"].sigmoid()
        if self.cfg.use_presence and self.cfg.presence_power > 0:
            gate = presence.sigmoid()[:, None, :]
            scores = scores * (gate if self.cfg.presence_power == 1.0 else gate.pow(self.cfg.presence_power))
        out.update(presence=presence, boxes=final["boxes"], logits=final["logits"], scores=scores, fdr=final["fdr"],
                   uncertainty=final["uncertainty"], query=final["q"])
        if return_masks:
            out["masks"] = self.mask_head(p3, final["q"])
        if return_kpts:
            out.update(self.kpt_head(p3, final["q"], final["boxes"]))

        if state is not None:                                                              # split detections / tracks, update slots
            det_s, trk_s = slice(0, K), slice(K, K + self.cfg.slots)
            sc = scores.amax(-1)
            state = self.slots.update(state, final["q"][:, trk_s], final["boxes"][:, trk_s], sc[:, trk_s],
                                      final["q"][:, det_s], final["boxes"][:, det_s], sc[:, det_s])
            state["scene_mem"] = mem.detach()
            out["state"] = state
            out["track_boxes"], out["track_ids"], out["track_alive"] = state["box"], state["ids"], state["alive"]
            for k in ("boxes", "logits", "scores", "uncertainty", "query"):
                out[k] = out[k][:, det_s]
            if "masks" in out:
                out["masks"] = out["masks"][:, det_s]
        return out

    # ----- deployment utilities ---------------------------------------------------------------
    @torch.no_grad()
    def reparameterize(self):
        """Fold every multi-branch conv into a single dense conv (call in eval mode, before export)."""
        for m in self.modules():
            if isinstance(m, (RepConv, DilatedRepDW)):
                m.reparameterize()
        return self

    def fold_vocabulary(self, text_embeds: torch.Tensor):
        self.vocab.fold_vocabulary(text_embeds)
        return self

    def init_video_state(self, B: int, device):
        return self.slots.init_state(B, device)


class ExportWrapper(nn.Module):
    """Fixed-signature, fixed-shape forward for ONNX / TensorRT export (closed or folded vocabulary, no anytime)."""

    def __init__(self, model: KESTREL, max_layers: Optional[int] = None):
        super().__init__()
        self.model, self.max_layers = model, max_layers

    def forward(self, images):
        o = self.model(images, return_masks=False, max_layers=self.max_layers)
        return o["boxes"], o["scores"]


# ----------------------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------------------
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _randomise_bn(model: nn.Module):
    """Give BatchNorms non-trivial statistics so the re-parameterisation test is meaningful."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5); m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5); m.bias.data.uniform_(-0.5, 0.5)


if __name__ == "__main__":
    import time
    from torch.utils.flop_counter import FlopCounterMode

    torch.manual_seed(0)
    cfg = KestrelConfig()
    model = KESTREL(cfg).eval()
    _randomise_bn(model)

    print("=" * 78)
    print("KESTREL-M reference skeleton — self-test")
    print("=" * 78)
    groups = dict(backbone=[model.stem, model.s4, model.s8, model.s16, model.s32, model.temporal_xa], neck=[model.neck],
                  dense_head=[model.dense, model.select], decoder=[model.decoder], presence=[model.presence, model.vocab],
                  mask_head=[model.mask_head], kpt_head=[model.kpt_head], slot_memory=[model.slots])
    total = count_params(model)
    for k, mods in groups.items():
        n = sum(count_params(m) for m in mods)
        print(f"  params  {k:<12} {n / 1e6:7.2f} M  ({100 * n / total:4.1f}%)")
    print(f"  params  {'TOTAL':<12} {total / 1e6:7.2f} M")

    x = torch.randn(2, 3, 640, 640)

    def gflops(fn):
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops() / 1e9

    with torch.no_grad():
        out = model(x, return_kpts=True)
    g_det = gflops(lambda: model(x[:1], return_masks=False))
    g_seg = gflops(lambda: model(x[:1], return_masks=True))
    g_all = gflops(lambda: model(x[:1], return_masks=True, return_kpts=True))
    print(f"\n  forward @ 640x640 (training graph, before folding): detect {g_det:.1f} | +masks {g_seg:.1f} | "
          f"+masks+kpts {g_all:.1f} GFLOPs   (MACs x2; roi_align/softmax not counted)")
    for k in ("boxes", "logits", "scores", "presence", "uncertainty", "masks", "kpt_xy", "dense_boxes"):
        print(f"  out[{k!r}]".ljust(28), tuple(out[k].shape))
    print(f"  aux decoder outputs      : {len(out['aux'])} layers (deep supervision / GO-LSD)")

    # loss primitives
    tgt = model.decoder.fdr.target_distribution(torch.rand(2, 300, 4) * 0.6 - 0.3)
    l_fdr = fdr_loss(out["aux"][0]["fdr"], tgt)
    l_lsd = golsd_loss(out["aux"][0]["fdr"] + 0.5 * torch.randn_like(out["aux"][0]["fdr"]), out["fdr"] if "fdr" in out else out["aux"][-1]["fdr"])
    print(f"  loss primitives          : fdr_loss={l_fdr.item():.3f}  golsd_loss={l_lsd.item():.4f}  "
          f"(target dists sum to {tgt.sum(-1).mean():.3f})")

    # non-square resolution
    with torch.no_grad():
        o2 = model(torch.randn(1, 3, 512, 768), return_masks=False)
    print(f"\n  arbitrary resolution 512x768 → boxes {tuple(o2['boxes'].shape)}  (no position tables to interpolate)")

    # elastic knobs: fewer decoder layers / queries without retraining
    with torch.no_grad():
        o3 = model(x[:1], return_masks=False, max_layers=2, num_queries=100)
    g_el = gflops(lambda: model(x[:1], return_masks=False, max_layers=2, num_queries=100))
    print(f"  elastic: max_layers=2, num_queries=100 → boxes {tuple(o3['boxes'].shape)}, {g_el:.1f} GFLOPs")

    # anytime early exit — mechanics demo. With untrained weights nothing is confident, so we loosen the
    # background threshold to show queries leaving early and later layers running on fewer queries.
    with torch.no_grad():
        oa = model(x[:1], anytime=True, return_masks=False)
        el = oa["exit_layer"][0]
        print(f"  anytime (real thresholds, untrained): exit-layer histogram "
              f"{torch.bincount(el, minlength=cfg.dec_layers + 1)[1:].tolist()} (layers 1..{cfg.dec_layers})")
        model.cfg.exit_bg = 0.20
        oa = model(x[:1], anytime=True, return_masks=False)
        el = oa["exit_layer"][0]
        model.cfg.exit_bg = 0.05
    print(f"  anytime (demo exit_bg=0.20)          : exit-layer histogram "
          f"{torch.bincount(el, minlength=cfg.dec_layers + 1)[1:].tolist()}, mean depth {el.float().mean():.2f}")

    # video / slot memory over 3 frames (demo spawn threshold — scores are ~uniform without training)
    state = model.init_video_state(1, x.device)
    with torch.no_grad():
        for t in range(3):
            ov = model(torch.randn(1, 3, 640, 640), state=state, return_masks=False)
            state = ov["state"]
            if t == 0:   # spawn a few slots for the demo (top-2% detections by score)
                sc = ov["scores"].amax(-1)
                state = model.slots.update(state, state["emb"], state["box"], state["score"], ov["query"], ov["boxes"], sc,
                                           spawn_thr=sc.quantile(0.98).item())
    print(f"  video: 3 frames with slot memory → alive slots {int(state['alive'].sum())}, ids issued {state['next_id']}, "
          f"scene_mem {tuple(state['scene_mem'].shape)}, det boxes {tuple(ov['boxes'].shape)}")

    # open-vocabulary folding
    text = torch.randn(12, cfg.embed_dim)              # stand-in for 12 prompt embeddings from a text encoder
    model.fold_vocabulary(text)
    with torch.no_grad():
        ov = model(x[:1], return_masks=False)
    print(f"  open-vocab: folded 12 prompts → scores {tuple(ov['scores'].shape)}, presence {tuple(ov['presence'].shape)}")
    model.vocab.clear_vocabulary()

    # re-parameterisation equivalence (continuous outputs; top-K selection is discrete and may reorder near-ties)
    with torch.no_grad():
        ref = model(x[:1], return_masks=True)
        model.reparameterize()
        rep = model(x[:1], return_masks=True)
    diff = max((ref[k] - rep[k]).abs().max().item() for k in ("dense_boxes", "dense_logits", "dense_quality", "presence"))
    same_set = len(set(ref["query_idx"][0].tolist()) & set(rep["query_idx"][0].tolist()))
    n_conv = sum(1 for m in model.modules() if isinstance(m, (RepConv, DilatedRepDW)))
    g_fold = gflops(lambda: model(x[:1], return_masks=False))
    print(f"\n  reparameterize(): {n_conv} multi-branch convs folded → max |Δ| on dense outputs = {diff:.2e}; "
          f"same top-{cfg.num_queries} anchors selected: {same_set}/{cfg.num_queries}")
    print(f"  deployment graph: {count_params(model) / 1e6:.2f} M params, {g_fold:.1f} GFLOPs @640 (detect)")

    # CPU timing (illustrative only — this sandbox has no GPU)
    with torch.no_grad():
        model(x[:1], return_masks=False)
        t0 = time.perf_counter()
        for _ in range(3):
            model(x[:1], return_masks=False)
        dt = (time.perf_counter() - t0) / 3
    print(f"  CPU latency (this sandbox, fp32, {torch.get_num_threads()} threads): {dt * 1000:.0f} ms / image @640")

    # ONNX export (best effort)
    try:
        import onnx  # noqa: F401
        wrapper = ExportWrapper(model).eval()
        torch.onnx.export(wrapper, x[:1], "/tmp/kestrel.onnx", opset_version=17, input_names=["images"],
                          output_names=["boxes", "scores"], dynamo=False)
        m = onnx.load("/tmp/kestrel.onnx")
        ops = sorted({n.op_type for n in m.graph.node})
        print(f"\n  ONNX export OK (opset 17): {len(m.graph.node)} nodes, op types: {', '.join(ops)}")
    except Exception as e:  # pragma: no cover
        print(f"\n  ONNX export not verified: {type(e).__name__}: {str(e)[:200]}")
    print("\nself-test complete.")
