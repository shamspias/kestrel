"""Score an ultralytics checkpoint with the SAME pycocotools evaluator / GT file used for KESTREL (evaluate.coco_eval).
Also times batch-1 raw model forward (+NMS for YOLO) on the device."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ultralytics import YOLO, RTDETR
from data import load_split, write_coco_gt
from evaluate import coco_eval

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--imgsz", type=int, default=512)
ap.add_argument("--device", default="mps")
ap.add_argument("--conf", type=float, default=0.001)
ap.add_argument("--max-det", type=int, default=300)
ap.add_argument("--out", default=None)
ap.add_argument("--latency", action="store_true")
ap.add_argument("--end2end", default=None, choices=["true", "false"], help="YOLO26: force NMS-free (true) or NMS (false) inference")
a = ap.parse_args()
recs = load_split("data/voc", [("2007", "test")], cache="data/voc/cache_test07.json")
gt_path = "data/voc/coco_gt_test07.json"
id_map = write_coco_gt(recs, gt_path)
cls = RTDETR if "rtdetr" in a.ckpt else YOLO
model = cls(a.ckpt)
results, t0, n = [], time.time(), 0
files = [r["file"] for r in recs]
for i in range(0, len(files), 32):
    batch = files[i:i + 32]
    kw = {} if a.end2end is None else dict(end2end=(a.end2end == "true"))
    preds = model.predict(batch, imgsz=a.imgsz, conf=a.conf, iou=0.7, max_det=a.max_det, device=a.device, verbose=False, half=False, **kw)
    for r, p in zip(recs[i:i + 32], preds):
        iid = id_map[r["id"]]
        b = p.boxes
        for xyxy, c, s in zip(b.xyxy.cpu().tolist(), b.cls.cpu().tolist(), b.conf.cpu().tolist()):
            x1, y1, x2, y2 = xyxy
            results.append(dict(image_id=iid, category_id=int(c) + 1, bbox=[x1, y1, x2 - x1, y2 - y1], score=float(s)))
    n += len(batch)
stats = coco_eval(results, gt_path)
stats["params_M"] = sum(p.numel() for p in model.model.parameters()) / 1e6
print(a.ckpt, {k: round(v, 2) for k, v in stats.items()})
if a.latency:
    dev = torch.device(a.device)
    net = model.model.to(dev).eval().float()
    x = torch.randn(1, 3, a.imgsz, a.imgsz, device=dev)
    with torch.no_grad():
        for _ in range(10): net(x)
        if dev.type == "mps": torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(50): net(x)
        if dev.type == "mps": torch.mps.synchronize()
    stats["latency_ms_raw"] = 1000 * (time.perf_counter() - t) / 50
    print("raw forward latency ms:", round(stats["latency_ms_raw"], 2))
if a.out:
    json.dump(stats, open(a.out, "w"), indent=1)
