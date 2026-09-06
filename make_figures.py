"""Figures for the paper, from the measured result JSONs. Run after make_tables.py.

  runs/<run>/eval.json         anytime + static sweeps  -> figures/anytime_curve.pdf
  runs/<run>/calib.npz         entropy vs IoU           -> figures/calibration.pdf
  runs/decoder_share*.json     decoder share of FLOPs   -> figures/decoder_share.pdf
  runs/<run>/calib.npz         depth vs object count    -> figures/depth_difficulty.pdf

Missing inputs are skipped with a message rather than faked. Colours are the validated categorical slots
(blue = ours / RoI / freeze, orange = the alternative, grey = the non-adaptive reference); every figure
carries direct value labels or a companion table, which is the relief the palette's contrast check requires.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = Path("paper")
(P / "figures").mkdir(exist_ok=True, parents=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def pareto(points, x="mean_exit_layer", y="AP"):
    """Points that are not dominated: sorted by depth, keep each new best AP."""
    best, out = -1e9, []
    for r in sorted(points, key=lambda r: r[x]):
        if r[y] > best:
            out.append(r); best = r[y]
    return out


# ------------------------------------------------------------------ 1. anytime curve
def fig_anytime(run="runs/kestrel_n2", fallback="runs/kestrel_n", out="anytime_curve.pdf"):
    j = load(f"{run}/eval.json") or load(f"{fallback}/eval.json")
    if not j or not j.get("anytime"):
        print("skip anytime_curve (no anytime sweep yet)"); return
    tag = run if load(f"{run}/eval.json") else fallback
    L = len(j.get("static", {})) + 1
    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    xs = list(range(1, L + 1))
    ys = [j["static"][str(l)]["AP"] if l < L else j["full"]["AP"] for l in xs]
    ax.plot(xs, ys, "s--", color=INK2, lw=1.2, ms=4, mfc="white", mew=1.0,
            label="static truncation (all queries)", zorder=2)

    pts = j["anytime"]
    modes = [m for m in ("freeze", "remove") if any(r.get("mode") == m for r in pts)] or [None]
    for mode, colour in zip(modes, (BLUE, ORANGE)):
        sel = [r for r in pts if r.get("mode") == mode] if mode else pts
        ax.scatter([r["mean_exit_layer"] for r in sel], [r["AP"] for r in sel], s=7, color=colour,
                   alpha=0.30, linewidths=0, zorder=3)
        pa = pareto(sel)
        lbl = {"freeze": "anytime, exited queries frozen", "remove": "anytime, exited queries removed",
               None: "anytime"}[mode]
        ax.plot([r["mean_exit_layer"] for r in pa], [r["AP"] for r in pa], "o-", color=colour, lw=1.6,
                ms=4.5, mfc="white", mew=1.2, label=lbl, zorder=4)

    ax.set_xlabel("mean decoder layers executed per query")
    ax.set_ylabel("AP (VOC07 test)")
    ax.grid(axis="y", alpha=0.8, zorder=0)
    ax.legend(loc="lower right", handlelength=1.8)
    fig.savefig(P / "figures" / out); plt.close(fig)
    print(f"wrote figures/{out} (from {tag})")


# ------------------------------------------------------------------ 2. decoder share
def fig_decoder_share(size=512, out="decoder_share.pdf"):
    rows = []
    for f in ("runs/decoder_share.json", "runs/decoder_share_L.json"):
        rows += load(f) or []
    rows = [r for r in rows if r["size"] == size and r["queries"] == {"N": 100}.get(r["preset"], 300)]
    if not rows:
        print("skip decoder_share (no measurements)"); return
    presets = [p for p in ("N", "S", "M", "L") if any(r["preset"] == p for r in rows)]
    get = lambda p, la: next((100 * r["decoder_share"] for r in rows if r["preset"] == p and r["local_attn"] == la), None)
    roi = [get(p, "roi") for p in presets]
    dfm = [get(p, "deform") for p in presets]

    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    x = np.arange(len(presets)); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, roi, w, color=BLUE, label="RoI-gathered (ours)", zorder=3)
    b2 = ax.bar(x + w / 2 + 0.01, dfm, w, color=ORANGE, label="multi-scale deformable", zorder=3)
    for bars, vals in ((b1, roi), (b2, dfm)):
        for b, v in zip(bars, vals):
            if v is not None:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(presets)
    ax.set_xlabel(f"model size (at ${size}^2$)")
    ax.set_ylabel("decoder share of network FLOPs (%)")
    ax.set_ylim(0, max([v for v in roi + dfm if v is not None]) * 1.22)
    ax.grid(axis="y", alpha=0.8, zorder=0)
    ax.legend(loc="upper left", handlelength=1.2)
    fig.savefig(P / "figures" / out); plt.close(fig)
    print(f"wrote figures/{out}")


# ------------------------------------------------------------------ 3. depth vs difficulty
def fig_depth_difficulty(run="runs/kestrel_n2", fallback="runs/kestrel_n", bg=0.2, out="depth_difficulty.pdf"):
    path = f"{run}/calib.npz" if os.path.exists(f"{run}/calib.npz") else f"{fallback}/calib.npz"
    if not os.path.exists(path):
        print("skip depth_difficulty (no calib.npz)"); return
    z = np.load(path)
    Pr, NOBJ, L = z["P"], z["NOBJ"][:, 0], int(z["L"])
    # policy: minimum two layers, background rule only -> depth is L minus the queries that exit at layer 2
    depth = L - (Pr[1] < bg).mean(1)
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4–5"), (6, 100, "6+")]
    xs, ys, ns = [], [], []
    for lo, hi, lbl in bins:
        s = (NOBJ >= lo) & (NOBJ <= hi)
        if s.sum() > 10:
            xs.append(lbl); ys.append(depth[s].mean()); ns.append(int(s.sum()))
    if not xs:
        print("skip depth_difficulty (no bins)"); return
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    lo, hi = min(ys), max(ys)
    y0, y1 = lo - 0.35 * (hi - lo) - 0.01, hi + 0.22 * (hi - lo) + 0.01
    ax.set_ylim(y0, y1)                                   # set limits BEFORE placing labels against them
    bars = ax.bar(np.arange(len(xs)), ys, 0.6, color=BLUE, zorder=3)
    for b, v, n in zip(bars, ys, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02 * (y1 - y0), f"{v:.2f}", ha="center", va="bottom",
                fontsize=6.5, color=INK)
        ax.text(b.get_x() + b.get_width() / 2, y0 + 0.03 * (y1 - y0), f"n={n}", ha="center", va="bottom",
                fontsize=6, color="white", zorder=4)
    ax.set_xticks(np.arange(len(xs))); ax.set_xticklabels(xs)
    ax.set_xlabel("annotated objects in the image")
    ax.set_ylabel("mean decoder depth")
    ax.grid(axis="y", alpha=0.8, zorder=0)
    fig.savefig(P / "figures" / out); plt.close(fig)
    print(f"wrote figures/{out}")


# ------------------------------------------------------------------ 4. calibration
def fig_calibration(runs=(("runs/kestrel_n2", "with GO-LSD", BLUE), ("runs/abl_nogolsd", "without GO-LSD", ORANGE)),
                    fallback=("runs/kestrel_n", "with GO-LSD", BLUE), out="calibration.pdf"):
    have = [(p, n, c) for p, n, c in runs if os.path.exists(f"{p}/calib.npz")]
    if not have and os.path.exists(f"{fallback[0]}/calib.npz"):
        have = [fallback]
    if not have:
        print("skip calibration (no calib.npz)"); return
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3))
    edges = np.array([0, .05, .1, .15, .2, .3, .4, .6, 1.001])
    for path, name, colour in have:
        z = np.load(f"{path}/calib.npz")
        H, I, M, L = z["H"], z["IOU"], z["MATCHED"], int(z["L"])
        for l, ls, alpha in ((0, "-", 1.0), (1, "--", 0.75)):
            if l >= L - 1:
                break
            h, i0, i1 = H[l][M], I[l][M], I[-1][M]
            xs, y0, y1 = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = (h >= lo) & (h < hi)
                if s.sum() >= 50:
                    xs.append(h[s].mean()); y0.append(i0[s].mean()); y1.append((i1[s] - i0[s]).mean())
            axs[0].plot(xs, y0, ls, marker="o", ms=3.2, lw=1.4, color=colour, alpha=alpha,
                        label=f"{name}, after layer {l + 1}")
            axs[1].plot(xs, y1, ls, marker="o", ms=3.2, lw=1.4, color=colour, alpha=alpha)
    axs[0].set_ylabel("IoU with matched ground truth")
    axs[1].set_ylabel("IoU still to be gained")
    for a in axs:
        a.set_xlabel("normalised localisation entropy $H_q$")
        a.grid(alpha=0.8, zorder=0)
    axs[1].axhline(0, color=INK2, lw=0.5, ls=":")
    axs[0].legend(loc="best", handlelength=1.8)
    fig.tight_layout()
    fig.savefig(P / "figures" / out); plt.close(fig)
    print(f"wrote figures/{out}")


if __name__ == "__main__":
    fig_anytime()
    fig_decoder_share()
    fig_depth_difficulty()
    fig_calibration()
