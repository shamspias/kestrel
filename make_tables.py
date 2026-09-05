"""Collect results JSONs into LaTeX tables, macros and figures for the paper.
Inputs (all optional; missing entries are rendered as '--'):
  runs/<name>/eval.json          from evaluate.py --out (keys: full, static, anytime, latency_ms(_images), latency_anytime)
  runs/baselines/<name>.json     from baselines/eval_yolo.py --out
  runs/<name>/calib.npz          from analysis.py
Edit RUNS / BASELINES below to map run directories to table rows."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

P = Path("paper"); (P / "tables").mkdir(exist_ok=True, parents=True); (P / "figures").mkdir(exist_ok=True, parents=True)

MAIN_RUNS = [  # (label, eval json, params col override, notes)
    ("\\kestrel{}-N (ours)", "runs/kestrel_n/eval.json", None, ""),
    ("\\kestrel{}-N, no distillation", "runs/kestrel_n_scratch/eval.json", None, ""),
]
BASELINES = [
    ("YOLO26n (e2e, scratch)", "runs/baselines/yolo26n_scratch.json"),
    ("YOLO26n (NMS, scratch)", "runs/baselines/yolo26n_scratch_nms.json"),
    ("YOLO11n (NMS, scratch)", "runs/baselines/yolo11n_scratch.json"),
    ("YOLO12n (NMS, scratch)", "runs/baselines/yolo12n_scratch.json"),
    ("YOLOv10n (e2e, scratch)", "runs/baselines/yolov10n_scratch.json"),
    ("YOLOv9t (NMS, scratch)", "runs/baselines/yolov9t_scratch.json"),
    ("YOLOv8n (NMS, scratch)", "runs/baselines/yolov8n_scratch.json"),
    ("YOLO26n (COCO-pretrained, ref.)", "runs/baselines/yolo26n_coco.json"),
]
# Published COCO val2017 numbers (author-reported; T4 TensorRT FP16 batch 1) for context only — not the same data,
# hardware or schedule as our VOC experiments. Sources: the models' papers / READMEs as compiled in the design survey.
PUBLISHED_COCO = [
    ("YOLOv10-N", 38.5, 1.84, "2.3"), ("YOLO11n", 39.5, 1.5, "2.6"), ("YOLOv12n (turbo)", 40.4, 1.60, "2.5"), ("YOLOv13-N", 41.6, 1.97, "2.5"),
    ("YOLO26n (NMS / e2e 40.1)", 40.9, 1.7, "2.4"), ("D-FINE-N", 42.8, 2.12, "4"), ("DEIMv2-N (HGNetv2)", 43.0, 2.32, "3.6"),
    ("LW-DETR-tiny", 42.6, 2.0, "12"), ("RF-DETR-N (DINOv2-S, 384 px)", 48.4, 2.3, "30.5"),
    ("YOLOv10-S", 46.3, 2.49, "7.2"), ("YOLO11s", 47.0, 2.5, "9.4"), ("YOLO26s", 48.6, 2.5, "9.5"), ("D-FINE-S", 48.5, 3.49, "10"),
    ("RT-DETRv4-S", 49.7, 3.66, "10"), ("DEIMv2-S (DINOv3-distilled)", 50.9, 5.78, "9.7"), ("RF-DETR-S (512 px)", 53.0, 3.5, "32.1"),
]
ABLATIONS = [  # label, run dir
    ("Full model", "runs/abl_full"),
    ("$-$ GO-LSD", "runs/abl_nogolsd"),
    ("$-$ presence gate", "runs/abl_nopresence"),
    ("$-$ denoising queries", "runs/abl_nodn"),
    ("deformable attention instead of RoI", "runs/abl_deform"),
    ("$-$ DINOv2 distillation init", "runs/abl_scratch"),
]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, d=1):
    return "--" if v is None else f"{v:.{d}f}"


macros = {}
# ------------------------------------------------------------------ main table
rows = []
for label, path, params, note in MAIN_RUNS:
    j = load(path)
    if j:
        r = j["full"]; lat = (j.get("latency_ms_images") or {}).get(f"L{len(j.get('static', {})) + 1}", {}).get("ms_median") if j.get("latency_ms_images") else None
        rows.append((label, j["params_M"], r["AP"], r["AP50"], r["AP75"], r["APs"], r["APm"], r["APl"], lat))
    else:
        rows.append((label, None, None, None, None, None, None, None, None))
for label, path in BASELINES:
    j = load(path)
    rows.append((label, j and j.get("params_M"), *(j and (j["AP"], j["AP50"], j["AP75"], j["APs"], j["APm"], j["APl"]) or (None,) * 6), j and j.get("latency_ms_raw")))
with open(P / "tables/main.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{VOC07 test}, all models trained from scratch on VOC07+12 trainval at $512^2$ for the same number of epochs on one Apple M2~Pro, scored by the same pycocotools script. Latency: batch 1, fp32, MPS, median of 50 passes. The COCO-pretrained row is a reference, not a like-for-like comparison.}\\label{tab:main}\n")
    fh.write("\\begin{tabular}{lrrrrrrrr}\\toprule\nModel & Params (M) & AP & AP$_{50}$ & AP$_{75}$ & AP$_S$ & AP$_M$ & AP$_L$ & ms \\\\\\midrule\n")
    for r in rows:
        fh.write(f"{r[0]} & {f(r[1], 2)} & {f(r[2])} & {f(r[3])} & {f(r[4])} & {f(r[5])} & {f(r[6])} & {f(r[7])} & {f(r[8])} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

with open(P / "tables/published.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{Published COCO val2017 results} of nano/small real-time detectors, as reported by their authors (T4, TensorRT FP16, batch 1). Listed for context only: different data, schedule and hardware from our VOC study, so not comparable to Table~\\ref{tab:main}.}\\label{tab:published}\n")
    fh.write("\\begin{tabular}{lrrr}\\toprule\nModel & COCO AP & T4 ms & Params (M) \\\\\\midrule\n")
    for name, apv, ms, pm in PUBLISHED_COCO:
        fh.write(f"{name} & {apv:.1f} & {ms:.2f} & {pm} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

# ------------------------------------------------------------------ anytime table + figure
j = load("runs/kestrel_n/eval.json"); jn = load("runs/abl_nogolsd/eval.json"); jf = load("runs/abl_full/eval.json")
with open(P / "tables/anytime.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{Anytime decoding} of \\kestrel{}-N on VOC07 test. Static: every query runs $\\ell$ layers. Anytime: per-query exit at the listed thresholds; depth is the mean number of layers executed per query; latency is batch-1 wall-clock on real images (median, ms).}\\label{tab:anytime}\n")
    fh.write("\\begin{tabular}{llrrrr}\\toprule\nMode & Setting & Depth & AP & AP$_{50}$ & ms \\\\\\midrule\n")
    if j:
        L = model_L = len(j.get("static", {})) + 1
        for l in range(1, L + 1):
            r = j["full"] if l == L else j["static"][str(l)]
            ms = (j.get("latency_ms_images") or {}).get(f"L{l}", {}).get("ms_median")
            fh.write(f"static & $\\ell={l}$ & {l:.2f} & {f(r['AP'])} & {f(r['AP50'])} & {f(ms)} \\\\\n")
        fh.write("\\midrule\n")
        pts = sorted(j.get("anytime", []), key=lambda r: r["mean_exit_layer"])
        # pick Pareto points
        best, pareto = -1, []
        for r in sorted(j.get("anytime", []), key=lambda r: -r["mean_exit_layer"]):
            pass
        for r in pts:
            if r["AP"] > best:
                pareto.append(r); best = r["AP"]
        for r in pareto:
            fh.write(f"anytime & $\\tau_p{{=}}{r['exit_p']},\\tau_H{{=}}{r['exit_u']},\\tau_{{bg}}{{=}}{r['exit_bg']}$ & {r['mean_exit_layer']:.2f} & {f(r['AP'])} & {f(r['AP50'])} & {f((j.get('latency_anytime') or {}).get('ms_median')) if r is pareto[-1] else '--'} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
    for jj, name, c in ((j, "with GO-LSD (main)", "#2f4f9e"), (jf, "with GO-LSD (ablation schedule)", "#5b7fd0"), (jn, "without GO-LSD", "#b9521a")):
        if not jj: continue
        L = len(jj.get("static", {})) + 1
        xs = list(range(1, L + 1)); ys = [jj["static"][str(l)]["AP"] if l < L else jj["full"]["AP"] for l in xs]
        ax.plot(xs, ys, "s--", color=c, alpha=0.6, label=f"static, {name}")
        pts = sorted(jj.get("anytime", []), key=lambda r: r["mean_exit_layer"])
        best, px, py = -1, [], []
        for r in pts:
            if r["AP"] > best: best = r["AP"]; px.append(r["mean_exit_layer"]); py.append(r["AP"])
        ax.scatter([r["mean_exit_layer"] for r in pts], [r["AP"] for r in pts], s=6, color=c, alpha=0.25)
        if px: ax.plot(px, py, "o-", color=c, label=f"anytime, {name}")
    ax.set_xlabel("mean decoder layers per query"); ax.set_ylabel("AP (VOC07 test)"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(P / "figures/anytime_curve.pdf"); plt.close(fig)
    # calibration figure
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.6), dpi=200)
    for path, name, c in (("runs/kestrel_n/calib.npz", "with GO-LSD", "#2f4f9e"), ("runs/abl_nogolsd/calib.npz", "without GO-LSD", "#b9521a")):
        if not os.path.exists(path): continue
        z = np.load(path); H, I, M, L = z["H"], z["IOU"], z["MATCHED"], int(z["L"])
        edges = np.array([0, .05, .1, .15, .2, .3, .4, .6, 1.001])
        for l, ls in ((0, "-"), (1, "--")):
            if l >= L - 1: break
            h, i0, i1 = H[l][M], I[l][M], I[-1][M]
            xs, y0, y1 = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = (h >= lo) & (h < hi)
                if s.sum() >= 20: xs.append(h[s].mean()); y0.append(i0[s].mean()); y1.append((i1[s] - i0[s]).mean())
            axs[0].plot(xs, y0, ls, marker="o", ms=3, color=c, label=f"{name}, layer {l + 1}")
            axs[1].plot(xs, y1, ls, marker="o", ms=3, color=c)
    axs[0].set_xlabel("normalised entropy $H_q$"); axs[0].set_ylabel("IoU after layer $\\ell$"); axs[1].set_xlabel("normalised entropy $H_q$"); axs[1].set_ylabel("IoU gain from remaining layers")
    for a in axs: a.grid(alpha=0.3)
    axs[0].legend(fontsize=5); fig.tight_layout(); fig.savefig(P / "figures/calibration.pdf"); plt.close(fig)
except Exception as e:
    print("figure error:", e)

# ------------------------------------------------------------------ ablation table
with open(P / "tables/ablation.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{Ablations} (\\kestrel{}-N, \\epochsAbl{} epochs at $416^2$, VOC07 test). Anytime AP and depth are at the thresholds that keep AP within 0.3 of full depth for the full model, applied unchanged to every row.}\\label{tab:ablation}\n")
    fh.write("\\begin{tabular}{lrrrrr}\\toprule\nVariant & AP & AP$_{50}$ & AP$_{75}$ & AP anytime & depth \\\\\\midrule\n")
    for label, d in ABLATIONS:
        jj = load(f"{d}/eval.json")
        if jj:
            r = jj["full"]; an = jj.get("anytime_single") or {}
            fh.write(f"{label} & {f(r['AP'])} & {f(r['AP50'])} & {f(r['AP75'])} & {f(an.get('AP'))} & {f(an.get('mean_exit_layer'), 2)} \\\\\n")
        else:
            fh.write(f"{label} & -- & -- & -- & -- & -- \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

# ------------------------------------------------------------------ latency table
with open(P / "tables/latency.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{Batch-1 latency} on the M2~Pro (MPS, fp32, $512^2$, median ms over real test images). Anytime uses the thresholds of the last row of Table~\\ref{tab:anytime}.}\\label{tab:latency}\n")
    fh.write("\\begin{tabular}{lrrr}\\toprule\nModel & Mode & Depth & ms \\\\\\midrule\n")
    if j:
        L = len(j.get("static", {})) + 1
        for l in range(1, L + 1):
            fh.write(f"\\kestrel{{}}-N & static $\\ell={l}$ & {l} & {f((j.get('latency_ms_images') or {}).get(f'L{l}', {}).get('ms_median'))} \\\\\n")
        la = j.get("latency_anytime") or {}
        fh.write(f"\\kestrel{{}}-N & anytime & {f(la.get('mean_depth'), 2)} & {f(la.get('ms_median'))} \\\\\n")
    for label, path in BASELINES:
        jj = load(path)
        if jj and jj.get("latency_ms_raw") is not None:
            fh.write(f"{label} & -- & -- & {f(jj['latency_ms_raw'])} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

# ------------------------------------------------------------------ macros
if j:
    macros["paramsN"] = f"{j['params_M']:.1f}"; macros["apMain"] = f(j["full"]["AP"]); macros["apFiftyMain"] = f(j["full"]["AP50"])
args = load("runs/kestrel_n/args.json")
macros["epochsMain"] = str(args["epochs"]) if args else "--"
aargs = load("runs/abl_full/args.json")
macros["epochsAbl"] = str(aargs["epochs"]) if aargs else "--"
macros.setdefault("paramsN", "5.4"); macros.setdefault("gflopsN", "7.1")
for k in ("mainResultText", "anytimeResultText", "calibResultText", "ablationResultText", "latencyResultText", "abstractResultText", "conclusionResultText"):
    macros.setdefault(k, "")
with open(P / "results_macros.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("tables/figures/macros written")
