"""
Evaluate a trained model on REAL labelled images (evaluation only - never train on these).

    python eval_real.py --model best.pt --data <roboflow yolo export>\\data.yaml --splits valid test --out <dir>

- Classes are matched BY NAME between the model and the dataset; dataset classes the model
  doesn't know (e.g. flex, hole) are ignored.
- Real labels are mostly boxes (some polygons), so this reports BOX detection metrics
  (the segmentation model's boxes). Polygon labels are converted to their bounding box.
- Roboflow exports are often stretched to a square. --unstretch W H resizes each image back
  to the original aspect (e.g. 1280 720) before inference; normalized labels stay valid.
- Writes metrics.json, a per-class table and prediction overlays (boxes + masks) to --out.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

IOUV = torch.linspace(0.5, 0.95, 10)


def read_labels(path, map_ids):
    """YOLO txt (box or polygon lines) -> (classes, xyxy normalized) keeping only mapped classes."""
    cls, boxes = [], []
    if not path.exists():
        return np.zeros(0, int), np.zeros((0, 4))
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        c = int(v[0])
        if c not in map_ids:
            continue
        nums = np.array(v[1:], float)
        if len(nums) == 4:
            x, y, w, h = nums
            box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        else:
            pts = nums.reshape(-1, 2)
            box = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        cls.append(map_ids[c])
        boxes.append(box)
    return np.array(cls, int), np.array(boxes, float).reshape(-1, 4)


def match(pred_cls, gt_cls, iou):
    """Greedy one-to-one matching per IoU threshold (same scheme as Ultralytics' validator)."""
    correct = np.zeros((len(pred_cls), len(IOUV)), bool)
    iou = iou.cpu().numpy() * (gt_cls[:, None] == pred_cls[None, :])   # (gt, pred), class must match
    for i, t in enumerate(IOUV.tolist()):
        m = np.nonzero(iou >= t)
        if not len(m[0]):
            continue
        pairs = np.stack(m, 1)
        if len(pairs) > 1:
            pairs = pairs[iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]]
            pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
            pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
        correct[pairs[:, 1], i] = True
    return correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="data.yaml of the real dataset (Roboflow YOLO export)")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--unstretch", type=int, nargs=2, metavar=("W", "H"), default=None)
    ap.add_argument("--overlays", type=int, default=16)
    ap.add_argument("--conf_show", type=float, default=0.35)
    args = ap.parse_args()

    root = Path(args.data).parent
    names_ds = yaml.safe_load(Path(args.data).read_text())["names"]
    if isinstance(names_ds, dict):
        names_ds = [names_ds[k] for k in sorted(names_ds)]
    model = YOLO(args.model)
    names_m = model.names
    by_name = {n: i for i, n in names_m.items()}
    map_ids = {i: by_name[n] for i, n in enumerate(names_ds) if n in by_name}
    ignored = [n for n in names_ds if n not in by_name]

    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    stats, n_img, shown = [], 0, 0
    for split in args.splits:
        for img_path in sorted((root / split / "images").glob("*")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if args.unstretch:
                img = cv2.resize(img, tuple(args.unstretch), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            gt_cls, gt_box = read_labels(root / split / "labels" / (img_path.stem + ".txt"), map_ids)
            gt_box = gt_box * [w, h, w, h]

            r = model.predict(img, imgsz=args.imgsz, conf=0.001, iou=0.6, max_det=100, verbose=False)[0]
            pb = r.boxes.xyxy.cpu()
            pc = r.boxes.cls.cpu().numpy().astype(int)
            pconf = r.boxes.conf.cpu().numpy()
            if len(gt_cls) and len(pc):
                correct = match(pc, gt_cls, box_iou(torch.tensor(gt_box, dtype=torch.float32), pb))
            else:
                correct = np.zeros((len(pc), len(IOUV)), bool)
            stats.append((correct, pconf, pc, gt_cls))
            n_img += 1

            if shown < args.overlays and len(gt_cls):
                keep = r.boxes.conf >= args.conf_show
                vis = r[keep.cpu()].plot(line_width=2, font_size=12)
                for b in gt_box.astype(int):   # ground truth in white, dashed look via thin line
                    cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (255, 255, 255), 1)
                cv2.imwrite(str(out / "overlays" / f"{split}_{img_path.stem[:40]}.jpg"), vis)
                shown += 1

    tp, conf, pcls, tcls = (np.concatenate(x, 0) for x in zip(*stats))
    res = ap_per_class(tp, conf, pcls, tcls, names=names_m)
    _, _, p, r, f1, apc, ucls = res[:7]
    report = {
        "model": str(args.model), "data": str(args.data), "splits": args.splits, "images": n_img,
        "unstretch": args.unstretch, "ignored_dataset_classes": ignored,
        "all": {"P": float(p.mean()), "R": float(r.mean()),
                "mAP50": float(apc[:, 0].mean()), "mAP50-95": float(apc.mean())},
        "per_class": {names_m[int(c)]: {"instances": int((tcls == c).sum()), "P": float(p[i]), "R": float(r[i]),
                                         "AP50": float(apc[i, 0]), "AP50-95": float(apc[i].mean())}
                      for i, c in enumerate(ucls)},
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))

    lines = [f"{'class':<12}{'inst':>6}{'P':>8}{'R':>8}{'AP50':>8}{'AP50-95':>9}"]
    for n, m in report["per_class"].items():
        lines.append(f"{n:<12}{m['instances']:>6}{m['P']:>8.3f}{m['R']:>8.3f}{m['AP50']:>8.3f}{m['AP50-95']:>9.3f}")
    a = report["all"]
    lines.append(f"{'all':<12}{len(tcls):>6}{a['P']:>8.3f}{a['R']:>8.3f}{a['mAP50']:>8.3f}{a['mAP50-95']:>9.3f}")
    table = "\n".join(lines)
    (out / "metrics.txt").write_text(table + "\n")
    print(f"{n_img} images, splits {args.splits}, unstretch {args.unstretch}, ignored classes {ignored}")
    print(table)


if __name__ == "__main__":
    main()
