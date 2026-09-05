"""Write VOC07+12 trainval / VOC07 test in YOLO txt format (symlinked images) for the ultralytics baselines."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_split, VOC_CLASSES

root = "data/voc"; out = "data/voc_yolo"
for split, parts in dict(train=[("2007", "trainval"), ("2012", "trainval")], test=[("2007", "test")]).items():
    recs = load_split(root, parts, cache=f"{root}/cache_{'trainval0712' if split == 'train' else 'test07'}.json")
    os.makedirs(f"{out}/images/{split}", exist_ok=True); os.makedirs(f"{out}/labels/{split}", exist_ok=True)
    for r in recs:
        name = r["id"]
        dst = f"{out}/images/{split}/{name}.jpg"
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(r["file"]), dst)
        W, H = r["width"], r["height"]
        with open(f"{out}/labels/{split}/{name}.txt", "w") as f:
            for b, l, d in zip(r["boxes"], r["labels"], r["difficult"]):
                if split == "train" and d:
                    continue
                x1, y1, x2, y2 = b
                f.write(f"{l} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
    print(split, len(recs))
with open(f"{out}/voc_yolo.yaml", "w") as f:
    f.write(f"path: {os.path.abspath(out)}\ntrain: images/train\nval: images/test\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(VOC_CLASSES)))
print("wrote", f"{out}/voc_yolo.yaml")
