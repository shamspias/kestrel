"""Train ultralytics baselines (YOLO26 / YOLO11 / YOLOv8 / RT-DETR) on VOC07+12 under the same data, image size,
epoch budget and device as KESTREL. Usage: python baselines/train_yolo.py yolo26n.yaml --epochs 60 --imgsz 512"""
import argparse, os, sys
from ultralytics import YOLO, RTDETR

ap = argparse.ArgumentParser()
ap.add_argument("cfg")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=512)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="mps")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--pretrained", action="store_true", help="start from the COCO checkpoint (.pt) instead of scratch (.yaml)")
ap.add_argument("--name", default=None)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA); off by default because MPS does not support it")
a = ap.parse_args()
name = a.name or a.cfg.replace(".yaml", "").replace(".pt", "") + ("_coco" if a.pretrained else "_scratch") + f"_{a.imgsz}_e{a.epochs}"
cls = RTDETR if "rtdetr" in a.cfg else YOLO
if a.resume and os.path.exists(f"runs/baselines/{name}/weights/last.pt"):
    model = cls(f"runs/baselines/{name}/weights/last.pt"); model.train(resume=True); sys.exit(0)
model = cls(a.cfg if not a.pretrained else a.cfg.replace(".yaml", ".pt"))
model.train(data="data/voc_yolo/voc_yolo.yaml", epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
            pretrained=a.pretrained, project=os.path.abspath("runs/baselines"), name=name, exist_ok=True, seed=0, deterministic=False,
            close_mosaic=5, plots=False, val=True, amp=a.amp)
