"""Collect results JSONs into LaTeX tables, macros and figures for the paper.
Inputs (all optional; missing entries are rendered as '--'):
  runs/<name>/eval.json          from evaluate.py --out (keys: full, gate{1.0,0.5,0.0}, static, anytime, latency_ms(_images), latency_anytime)
  runs/<name>/eval_nogate.json   the same sweeps with --gate-power 0 (ungated scores)
  runs/<name>/trt.json           from scripts/trt_latency.py kestrel (TensorRT FP16 per depth, + CUDA-graph replay)
  runs/<name>/calib.npz          from analysis.py
  runs/baselines/<name>.json     from baselines/eval_yolo.py --out (+ latency_ms_raw)
  runs/baselines/<name>.trt.json from scripts/trt_latency.py yolo
Edit MAIN_RUNS / BASELINES / ABLATIONS below to map run directories to table rows."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

P = Path("paper"); (P / "tables").mkdir(exist_ok=True, parents=True); (P / "figures").mkdir(exist_ok=True, parents=True)

HW = "RTX~2080~Ti"
MAIN_RUNS = [  # (label, run dir, gate exponent to report: 1.0 = product gate as designed, 0.0 = gate off)
    ("\\kestrel{}-N (product gate, as designed)", "runs/kestrel_n", 1.0),
    ("\\kestrel{}-N (gate off)", "runs/kestrel_n", 0.0),
    ("\\kestrel{}-S (product gate, as designed)", "runs/kestrel_s", 1.0),
    ("\\kestrel{}-S (gate off)", "runs/kestrel_s", 0.0),
]
BASELINES = [
    ("YOLO26n (end-to-end, scratch)", "yolo26n_scratch"),
    ("YOLO26n (NMS path, scratch)", "yolo26n_scratch_nms"),
    ("YOLO11n (NMS, scratch)", "yolo11n_scratch"),
    ("YOLO12n (NMS, scratch)", "yolo12n_scratch"),
    ("YOLOv10n (end-to-end, scratch)", "yolov10n_scratch"),
    ("YOLOv9t (NMS, scratch)", "yolov9t_scratch"),
    ("YOLOv8n (NMS, scratch)", "yolov8n_scratch"),
    ("YOLO26n (COCO-pretrained, ref.)", "yolo26n_coco"),
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
    ("$-$ presence head", "runs/abl_nopresence"),
    ("$-$ denoising queries", "runs/abl_nodn"),
    ("deformable attention instead of RoI", "runs/abl_deform"),
    ("$-$ DINOv2 distillation init", "runs/abl_scratch"),
]
RECIPE_AB = [  # label, run dir (10-epoch recipe check at 416, batch 16, one change per row)
    ("reference: lr 2e-3, fp16, distilled init", "runs/ab_ref"),
    ("lr 1e-3", "runs/ab_lr1e3"),
    ("lr 5e-4", "runs/ab_lr5e4"),
    ("fp32 instead of fp16", "runs/ab_fp32"),
    ("fp32, lr 1e-3", "runs/ab_fp32_lr1e3"),
    ("no distillation init", "runs/ab_noinit"),
]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, d=1):
    return "--" if v is None else f"{v:.{d}f}"


def get(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def full_stats(j, gate):
    """Full-depth AP dict of a KESTREL eval.json at the requested gate exponent."""
    if not j:
        return None
    if gate is None or abs(gate - (j.get("gate_power", 1.0) or 0.0)) < 1e-6:
        return j["full"]
    return get(j, "gate", str(float(gate)))


def last_eval(run):
    """Last EVAL line of a training log (log.jsonl) — used for the recipe A/B table."""
    p = f"{run}/log.jsonl"
    if not os.path.exists(p):
        return None
    ev = [json.loads(l) for l in open(p) if '"eval"' in l]
    return ev[-1] if ev else None


def anytime_points(run, full_only=True, bg_only=True):
    """Every anytime measurement for a run, merged across eval.json and the extra sweep files, tagged with exit mode
    and minimum depth (older files predate those fields: they used removal with a two-layer minimum).

    Files produced with --subset score a different image set, so their AP is NOT comparable with the full-set
    rows and mixing the two in one table is a reporting error. They are skipped unless full_only=False, and
    the number skipped is printed so the omission is never silent. Points that fix a key set as a control
    (keys_kept >= 0) or use the random-exit control policy are also excluded from the main table: they belong
    to their own analyses, not to the accuracy-versus-depth comparison."""
    import glob
    rows = []
    for path in sorted(glob.glob(f"{run}/eval*.json")):
        jj = load(path)
        for r in (jj or {}).get("anytime", []) or []:
            if r.get("keys_kept", -1) not in (-1, None) or r.get("policy", "confidence") != "confidence":
                continue                                       # control experiments belong to their own analyses
            if bg_only and (r.get("exit_p") is not None and r["exit_p"] <= 1.0):
                continue                                       # the reported curve disables the foreground rule
                                                               # (tau_p > 1); mixing in entropy-rule points would
                                                               # put two different policies on one axis
            r = dict(r); r.setdefault("mode", (jj or {}).get("exit_mode", "remove")); r.setdefault("min_layers", 2)
            rows.append(r)
    # Every row records the number of images it was scored on. Rows from a --subset run are not comparable with
    # full-set rows, and mixing them in one table is a reporting error, so keep only the largest image count.
    counts = {r.get("images") for r in rows if r.get("images")}
    keep = max(counts) if (full_only and counts) else None
    pts, seen, skipped = [], set(), 0
    for r in rows:
        if keep and r.get("images") and r["images"] != keep:
            skipped += 1
            continue
        key = (r["mode"], r["min_layers"], r["exit_p"], r["exit_u"], r["exit_bg"])
        if key not in seen:
            seen.add(key); pts.append(r)
    if skipped:
        print(f"  note: {run}: dropped {skipped} anytime points scored on fewer than {keep} images "
              f"(not comparable with the full-set rows)")
    return pts


def torch_ms(j, L=None):
    if not j:
        return None
    L = L or len(j.get("static", {})) + 1
    return get(j, "latency_ms_images", f"L{L}", "ms_median")


def trt_ms(t, L=None, graph=True):
    if not t:
        return None
    if "trt" in t and isinstance(t["trt"], dict) and any(k.startswith("L") for k in t["trt"]):        # kestrel: per depth
        L = L or max(int(k[1:]) for k in t["trt"])
        r = t["trt"].get(f"L{L}")
    else:                                                                                              # yolo
        r = t.get("trt")
    if not r:
        return None
    return get(r, "cuda_graph", "ms_median") if graph else r.get("ms_median")


macros = {}
# ------------------------------------------------------------------ main table
rows = []
for label, d, gate in MAIN_RUNS:
    j, jn, t = load(f"{d}/eval.json"), load(f"{d}/eval_nogate.json"), load(f"{d}/trt.json")
    r = full_stats(jn, gate) if (gate == 0.0 and jn) else full_stats(j, gate)
    if r:
        rows.append((label, j["params_M"], r["AP"], r["AP50"], r["AP75"], r["APs"], r["APm"], r["APl"], torch_ms(j), trt_ms(t)))
    else:
        rows.append((label, None, None, None, None, None, None, None, None, None))
for label, name in BASELINES:
    j, t = load(f"runs/baselines/{name}.json"), load(f"runs/baselines/{name.replace('_nms', '')}.trt.json")
    rows.append((label, j and j.get("params_M"), *(j and (j["AP"], j["AP50"], j["AP75"], j["APs"], j["APm"], j["APl"]) or (None,) * 6),
                 j and j.get("latency_ms_raw"), trt_ms(t)))
with open(P / "tables/main.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3.5pt}\n\\caption{\\textbf{VOC07 test.} All models trained from scratch on VOC07+12 trainval at $512^2$ for \\epochsMain{} epochs on one " + HW + ", scored by the same pycocotools script. Latency is batch~1 at $512^2$ on the same GPU: PyTorch eager fp32 (median over real test images) and TensorRT FP16 with CUDA-graph replay (median of 200 passes; NMS excluded for the NMS-based YOLOs, nothing excluded for the end-to-end models). The COCO-pretrained row is a reference, not a like-for-like comparison.}\\label{tab:main}\n")
    fh.write("\\begin{tabular}{lrrrrrrrrr}\\toprule\nModel & Params (M) & AP & AP$_{50}$ & AP$_{75}$ & AP$_S$ & AP$_M$ & AP$_L$ & torch ms & TRT ms \\\\\\midrule\n")
    for i, r in enumerate(rows):
        if i == len(MAIN_RUNS):
            fh.write("\\midrule\n")
        fh.write(f"{r[0]} & {f(r[1], 2)} & {f(r[2])} & {f(r[3])} & {f(r[4])} & {f(r[5])} & {f(r[6])} & {f(r[7])} & {f(r[8], 2)} & {f(r[9], 2)} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

with open(P / "tables/published.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{Published COCO val2017 results} of nano/small real-time detectors, as reported by their authors (T4, TensorRT FP16, batch 1). Listed for context only: different data, schedule and hardware from our VOC study, so not comparable to Table~\\ref{tab:main}.}\\label{tab:published}\n")
    fh.write("\\begin{tabular}{lrrr}\\toprule\nModel & COCO AP & T4 ms & Params (M) \\\\\\midrule\n")
    for name, apv, ms, pm in PUBLISHED_COCO:
        fh.write(f"{name} & {apv:.1f} & {ms:.2f} & {pm} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

# ------------------------------------------------------------------ presence-gate table
with open(P / "tables/gate.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{Presence gating} on VOC07 test. Final score $=\\sigma(z_{qc})\\,\\sigma(\\pi_c)^{\\gamma}$: $\\gamma{=}1$ is the SAM~3 product used in the design, $\\gamma{=}0$ removes the gate at inference. The gate is inference-only, so all rows use the same weights.}\\label{tab:gate}\n")
    fh.write("\\begin{tabular}{llrrrrrr}\\toprule\nModel & $\\gamma$ & AP & AP$_{50}$ & AP$_{75}$ & AP$_S$ & AP$_M$ & AP$_L$ \\\\\\midrule\n")
    for name, d in (("\\kestrel{}-N", "runs/kestrel_n"), ("\\kestrel{}-S", "runs/kestrel_s")):
        j = load(f"{d}/eval.json")
        if not j or "gate" not in j:
            continue
        for g in ("1.0", "0.5", "0.0"):
            r = j["gate"].get(g)
            if r:
                fh.write(f"{name} & {g} & {f(r['AP'])} & {f(r['AP50'])} & {f(r['AP75'])} & {f(r['APs'])} & {f(r['APm'])} & {f(r['APl'])} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")


def pareto(pts):
    best, out = -1, []
    for r in sorted(pts, key=lambda r: r["mean_exit_layer"]):
        if r["AP"] > best:
            out.append(r); best = r["AP"]
    return out


# ------------------------------------------------------------------ anytime table + figure
MAIN = "runs/kestrel_n2" if os.path.exists("runs/kestrel_n2/eval.json") else "runs/kestrel_n"    # corrected-recipe run once it exists
j_g, j_n = load(f"{MAIN}/eval.json"), load(f"{MAIN}/eval_nogate.json")
j = j_n or j_g                                                     # headline anytime numbers: ungated if available
t_n = load(f"{MAIN}/trt.json")
with open(P / "tables/anytime.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{Anytime decoding} of \\kestrel{}-N on VOC07 test" + (" (gate off)" if j_n else "") + ". Static: every query runs $\\ell$ layers. All anytime rows use the confident-background rule alone ($\\tau_p>1$ disables the foreground rule), which \\cref{sec:results:calib} shows is the only branch that does any work. Anytime: per-query exit at the listed thresholds; depth is the mean number of decoder layers executed per query. Latency: batch~1 PyTorch eager on real images (median ms) and TensorRT FP16 engines of the corresponding static depth. Every row is scored on the same \\detcount{} test images.}\\label{tab:anytime}\n")
    fh.write("\\begin{tabular}{llrrrrr}\\toprule\nMode & Setting & Depth & AP & AP$_{50}$ & torch ms & TRT ms \\\\\\midrule\n")
    if j:
        L = len(j.get("static", {})) + 1
        for l in range(1, L + 1):
            r = j["full"] if l == L else j["static"][str(l)]
            fh.write(f"static & $\\ell={l}$ & {l:.2f} & {f(r['AP'])} & {f(r['AP50'])} & {f(torch_ms(j_g, l), 2)} & {f(trt_ms(t_n, l), 2)} \\\\\n")
        allpts = anytime_points(MAIN)
        n_img = {r.get("images") for r in allpts if r.get("images")}
        if len(n_img) > 1:                                     # must never happen: one table, one image set
            raise SystemExit(f"anytime table would mix image counts {sorted(n_img)} -- refusing to write it")
        for mode, tag in (("remove", "anytime, removal"), ("freeze", "anytime, frozen context")):
            sel = [r for r in allpts if r.get("mode") == mode]
            if not sel:
                continue
            fh.write("\\midrule\n")
            for r in pareto(sel):
                la = (j_g or {}).get("latency_anytime") or {}
                same = la and mode == (j_g or {}).get("exit_mode", "remove") and abs(la.get("mean_depth", -9) - r["mean_exit_layer"]) < 0.15
                fh.write(f"{tag} & $\\ell_{{\\min}}{{=}}{r.get('min_layers', 2)},\\tau_{{bg}}{{=}}{r['exit_bg']}$ & {r['mean_exit_layer']:.2f} & {f(r['AP'])} & {f(r['AP50'])} & {f(la.get('ms_median'), 2) if same else '--'} & -- \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
    if j:
        L = len(j.get("static", {})) + 1
        xs = list(range(1, L + 1)); ys = [j["static"][str(l)]["AP"] if l < L else j["full"]["AP"] for l in xs]
        ax.plot(xs, ys, "s--", color="#77818f", label="static truncation")
        allpts = anytime_points(MAIN)
        for mode, name, c in (("remove", "anytime, queries removed (as designed)", "#b9521a"), ("freeze", "anytime, frozen context (ours)", "#2f4f9e")):
            sel = sorted([r for r in allpts if r.get("mode") == mode], key=lambda r: r["mean_exit_layer"])
            if not sel: continue
            ax.scatter([r["mean_exit_layer"] for r in sel], [r["AP"] for r in sel], s=7, color=c, alpha=0.3)
            pa = pareto(sel)
            ax.plot([r["mean_exit_layer"] for r in pa], [r["AP"] for r in pa], "o-", color=c, label=name)
    ax.set_xlabel("mean decoder layers per query"); ax.set_ylabel("AP (VOC07 test)"); ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="lower right")
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
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{Ablations} (\\kestrel{}-N, \\epochsAbl{} epochs at $416^2$, VOC07 test, one change per row). AP is reported with the product gate ($\\gamma{=}1$) and with the gate off ($\\gamma{=}0$). Anytime AP and depth use the exit thresholds $\\tau_p{=}0.6,\\tau_H{=}0.15,\\tau_{bg}{=}0.05$ for every row.}\\label{tab:ablation}\n")
    fh.write("\\begin{tabular}{lrrrrrr}\\toprule\nVariant & AP ($\\gamma{=}1$) & AP ($\\gamma{=}0$) & AP$_{50}$ ($\\gamma{=}0$) & AP$_{75}$ ($\\gamma{=}0$) & AP anytime & depth \\\\\\midrule\n")
    for label, d in ABLATIONS:
        jj = load(f"{d}/eval.json")
        if jj:
            r1, r0 = full_stats(jj, 1.0), full_stats(jj, 0.0) or jj["full"]; an = jj.get("anytime_single") or {}
            fh.write(f"{label} & {f(r1 and r1['AP'])} & {f(r0['AP'])} & {f(r0['AP50'])} & {f(r0['AP75'])} & {f(an.get('AP'))} & {f(an.get('mean_exit_layer'), 2)} \\\\\n")
        else:
            fh.write(f"{label} & -- & -- & -- & -- & -- & -- \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

# ------------------------------------------------------------------ recipe A/B table
with open(P / "tables/recipe.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{Recipe check} (\\kestrel{}-N, 10 epochs at $416^2$, batch 16, one change per row; VOC07 test AP of the EMA weights after the last epoch, gate off).}\\label{tab:recipe}\n")
    fh.write("\\begin{tabular}{lrr}\\toprule\nRecipe & AP & AP$_{50}$ \\\\\\midrule\n")
    for label, d in RECIPE_AB:
        e = last_eval(d)
        if e:
            ev = e["eval"]; ap = ev.get("AP_nogate", ev["AP"])
            fh.write(f"{label} & {f(ap)} & {f(ev['AP50']) if 'AP_nogate' not in ev else '--'} \\\\\n")
        else:
            fh.write(f"{label} & -- & -- \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

# ------------------------------------------------------------------ latency table
with open(P / "tables/latency.tex", "w") as fh:
    fh.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{\\textbf{Batch-1 latency} on the " + HW + " at $512^2$ (median ms). PyTorch eager fp32 over real test images; TensorRT FP16 engine with plain enqueue and with CUDA-graph replay (200 passes). Anytime uses the thresholds of the last row of Table~\\ref{tab:anytime}.}\\label{tab:latency}\n")
    fh.write("\\begin{tabular}{llrrrr}\\toprule\nModel & Mode & Depth & torch ms & TRT ms & TRT+graph ms \\\\\\midrule\n")
    for name, d in (("\\kestrel{}-N", "runs/kestrel_n"), ("\\kestrel{}-S", "runs/kestrel_s")):
        jj, tt = load(f"{d}/eval.json"), load(f"{d}/trt.json")
        if not jj:
            continue
        L = len(jj.get("static", {})) + 1
        for l in range(1, L + 1):
            fh.write(f"{name} & static $\\ell={l}$ & {l} & {f(torch_ms(jj, l), 2)} & {f(trt_ms(tt, l, graph=False), 2)} & {f(trt_ms(tt, l), 2)} \\\\\n")
        la = jj.get("latency_anytime") or {}
        fh.write(f"{name} & anytime & {f(la.get('mean_depth'), 2)} & {f(la.get('ms_median'), 2)} & -- & -- \\\\\n")
    for label, bname in BASELINES:
        if bname.endswith("_nms"):
            continue
        jj, tt = load(f"runs/baselines/{bname}.json"), load(f"runs/baselines/{bname}.trt.json")
        if jj and (jj.get("latency_ms_raw") is not None or tt):
            fh.write(f"{label} & -- & -- & {f(jj.get('latency_ms_raw'), 2)} & {f(trt_ms(tt, graph=False), 2)} & {f(trt_ms(tt), 2)} \\\\\n")
    fh.write("\\bottomrule\\end{tabular}\\end{table}\n")

# ------------------------------------------------------------------ macros
for tag, d in (("N", "runs/kestrel_n"), ("S", "runs/kestrel_s")):
    jj, jn = load(f"{d}/eval.json"), load(f"{d}/eval_nogate.json")
    if jj:
        r1, r0 = full_stats(jj, 1.0), full_stats(jj, 0.0)
        macros[f"params{tag}"] = f"{jj['params_M']:.1f}"
        if r1: macros[f"apGate{tag}"] = f(r1["AP"]); macros[f"apFiftyGate{tag}"] = f(r1["AP50"])
        if r0: macros[f"apNoGate{tag}"] = f(r0["AP"]); macros[f"apFiftyNoGate{tag}"] = f(r0["AP50"]); macros[f"apSeventyFiveNoGate{tag}"] = f(r0["AP75"])
        macros[f"apMain{tag}"] = macros.get(f"apNoGate{tag}", f(jj["full"]["AP"])); macros[f"apFiftyMain{tag}"] = macros.get(f"apFiftyNoGate{tag}", f(jj["full"]["AP50"]))
macros.setdefault("apMain", macros.get("apMainN", "--")); macros.setdefault("apFiftyMain", macros.get("apFiftyMainN", "--"))
b26 = load("runs/baselines/yolo26n_scratch.json")
if b26: macros["apYoloTwentySix"] = f(b26["AP"]); macros["apFiftyYoloTwentySix"] = f(b26["AP50"])
args = load("runs/kestrel_n/args.json"); macros["epochsMain"] = str(args["epochs"]) if args else "--"
aargs = load("runs/abl_full/args.json"); macros["epochsAbl"] = str(aargs["epochs"]) if aargs else "--"
macros.setdefault("paramsN", "5.4"); macros.setdefault("gflopsN", "7.1"); macros.setdefault("paramsS", "13.0"); macros.setdefault("gflopsS", "17.3")
macros["hardware"] = HW
macros["detcount"] = str(sorted({r.get("images") for r in anytime_points(MAIN) if r.get("images")})[0]) if anytime_points(MAIN) else "4952"
for k in ("mainResultText", "anytimeResultText", "calibResultText", "ablationResultText", "latencyResultText",
          "abstractResultText", "conclusionResultText", "gateResultText", "recipeResultText", "costResultText",
          "staticLossText", "decShareText", "mechanismResultText", "reproText", "KeydropResultText"):
    macros.setdefault(k, "")
prev = {}
if (P / "results_macros.tex").exists():                     # keep hand-written result texts across regenerations
    for line in open(P / "results_macros.tex"):
        if line.startswith("\\newcommand{\\") and "ResultText" in line:
            k = line.split("{\\")[1].split("}")[0]; prev[k] = line[line.index("}{") + 2:].rstrip("\n")[:-1]
for k, v in prev.items():
    if v:
        macros[k] = v
with open(P / "results_macros.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("tables/figures/macros written:", ", ".join(f"{k}={v}" for k, v in macros.items() if "Text" not in k))
