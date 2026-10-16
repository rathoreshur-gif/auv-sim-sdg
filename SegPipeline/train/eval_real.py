"""
Evaluate a trained model on REAL labelled images (evaluation only - never train on these).

    python eval_real.py --model best.pt --data <roboflow export>\\data.yaml [<another>.zip ...] --out <dir>

- --data: one or more real datasets, each a data.yaml or a Roboflow YOLO .zip (unpacked once, next
  to the zip). Results for each go to <out>/<dataset name>/, plus <out>/summary.json for all of them.
- Classes are matched BY NAME between the model and the dataset; dataset classes the model
  doesn't know (e.g. flex, hole) are ignored.
- Real labels are mostly boxes (some polygons), so this reports BOX detection metrics
  (the segmentation model's boxes). Polygon labels are converted to their bounding box.
- Roboflow exports are often stretched to a square. --unstretch W H resizes each SQUARE image back
  to the original aspect (e.g. 1280 720) before inference; normalized labels stay valid. Images
  that aren't square are left alone.
- Writes metrics.json, metrics.txt and prediction overlays (boxes + masks, ground truth in white).
"""
import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

IOUV = torch.linspace(0.5, 0.95, 10)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


def resolve_data(path):
    """data.yaml, a dataset folder holding one, or a Roboflow .zip (unpacked next to it, once) -> data.yaml."""
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "data.yaml"
    if p.suffix.lower() == ".zip":
        dest = p.with_suffix("")
        if not (dest / "data.yaml").is_file():
            print(f"unpacking {p} -> {dest}")
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
        found = sorted(dest.rglob("data.yaml"), key=lambda q: len(q.parts))
        if not found:
            raise FileNotFoundError(f"{p}: no data.yaml inside")
        p = found[0]
    if not p.is_file():
        raise FileNotFoundError(f"real dataset not found: {path}")
    return p


def set_name(data_yaml):
    """Short name for a real dataset: the Roboflow project (torpedo_rs, tropedo, ...) or its folder."""
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8")) or {}
    rf = d.get("roboflow") or {}
    name = rf.get("project") or Path(data_yaml).parent.name
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


def evaluate(model, data, out, splits=("valid", "test"), imgsz=640, unstretch=None, overlays=16, conf_show=0.35):
    """Score `model` (YOLO or a .pt path) on one real dataset. Writes metrics.json / metrics.txt / overlays
    to `out` and returns the report."""
    model = YOLO(str(model)) if not isinstance(model, YOLO) else model
    data = resolve_data(data)
    root = data.parent
    names_ds = yaml.safe_load(data.read_text(encoding="utf-8"))["names"]
    if isinstance(names_ds, dict):
        names_ds = [names_ds[k] for k in sorted(names_ds)]
    names_m = model.names
    by_name = {n: i for i, n in names_m.items()}
    map_ids = {i: by_name[n] for i, n in enumerate(names_ds) if n in by_name}
    ignored = [n for n in names_ds if n not in by_name]

    out = Path(out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    stats, n_img, shown, used = [], 0, 0, []
    for split in splits:
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            continue
        used.append(split)
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if unstretch and img.shape[0] == img.shape[1]:
                img = cv2.resize(img, tuple(unstretch), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            gt_cls, gt_box = read_labels(root / split / "labels" / (img_path.stem + ".txt"), map_ids)
            gt_box = gt_box * [w, h, w, h]

            r = model.predict(img, imgsz=imgsz, conf=0.001, iou=0.6, max_det=100, verbose=False)[0]
            pb = r.boxes.xyxy.cpu()
            pc = r.boxes.cls.cpu().numpy().astype(int)
            pconf = r.boxes.conf.cpu().numpy()
            if len(gt_cls) and len(pc):
                correct = match(pc, gt_cls, box_iou(torch.tensor(gt_box, dtype=torch.float32), pb))
            else:
                correct = np.zeros((len(pc), len(IOUV)), bool)
            stats.append((correct, pconf, pc, gt_cls))
            n_img += 1

            if shown < overlays and len(gt_cls):
                keep = r.boxes.conf >= conf_show
                vis = r[keep.cpu()].plot(line_width=2, font_size=12)
                for b in gt_box.astype(int):   # ground truth: thin white boxes
                    cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (255, 255, 255), 1)
                cv2.imwrite(str(out / "overlays" / f"{split}_{img_path.stem[:40]}.jpg"), vis)
                shown += 1
    if not stats:
        raise RuntimeError(f"{data}: no images in splits {list(splits)}")

    tp, conf, pcls, tcls = (np.concatenate(x, 0) for x in zip(*stats))
    res = ap_per_class(tp, conf, pcls, tcls, names=names_m)
    _, _, p, r, f1, apc, ucls = res[:7]
    confident = conf >= 0.5
    report = {
        "name": set_name(data), "model": str(getattr(model, "ckpt_path", "") or ""), "data": str(data),
        "splits": used, "images": n_img, "instances": int(len(tcls)), "unstretch": list(unstretch) if unstretch else None,
        "ignored_dataset_classes": ignored,
        "false_detections_conf50": int((confident & ~tp[:, 0]).sum()),
        "all": {"P": float(p.mean()) if len(p) else 0.0, "R": float(r.mean()) if len(r) else 0.0,
                "mAP50": float(apc[:, 0].mean()) if len(apc) else 0.0, "mAP50-95": float(apc.mean()) if len(apc) else 0.0},
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
    (out / "metrics.txt").write_text("\n".join(lines) + "\n")
    return report


def evaluate_sets(model, datas, out, **kw):
    """evaluate() on each real dataset -> <out>/<name>/..., <out>/summary.json; prints one line per set."""
    model = YOLO(str(model)) if not isinstance(model, YOLO) else model
    out = Path(out)
    reports = []
    for d in datas:
        try:
            name = set_name(resolve_data(d))
            rep = evaluate(model, d, out / name, **kw)
        except Exception as e:                                  # one broken set shouldn't lose the others
            print(f"[eval] {d}: FAILED - {e}")
            continue
        reports.append(rep)
    out.mkdir(parents=True, exist_ok=True)
    summary = [{k: r[k] for k in ("name", "data", "images", "instances", "splits", "unstretch", "all",
                                    "false_detections_conf50")} for r in reports]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[eval] real images, box metrics ({out}):")
    print(f"  {'dataset':<16}{'images':>7}{'inst':>6}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}{'false@0.5':>11}")
    for r in reports:
        a = r["all"]
        print(f"  {r['name']:<16}{r['images']:>7}{r['instances']:>6}{a['P']:>7.3f}{a['R']:>7.3f}{a['mAP50']:>8.3f}"
              f"{a['mAP50-95']:>10.3f}{r['false_detections_conf50']:>11}")
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, nargs="+", help="real datasets: data.yaml (or its folder) or Roboflow .zip")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--unstretch", nargs="+", default=None, metavar="W H",
                    help="e.g. 1280 720: resize square (Roboflow-stretched) images back to this before inference; 'none' = off")
    ap.add_argument("--overlays", type=int, default=16)
    ap.add_argument("--conf_show", type=float, default=0.35)
    args = ap.parse_args()
    un = parse_unstretch(args.unstretch)
    evaluate_sets(args.model, args.data, args.out, splits=args.splits, imgsz=args.imgsz, unstretch=un,
                  overlays=args.overlays, conf_show=args.conf_show)


def parse_unstretch(v):
    """["1280", "720"] / "1280 720" / "1280x720" -> (1280, 720); None / "none" / "" -> None."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = " ".join(str(x) for x in v)
    v = str(v).lower().replace("x", " ").replace(",", " ").split()
    if not v or v[0] in ("none", "off", "0"):
        return None
    if len(v) != 2:
        raise SystemExit(f"--unstretch: expected W H, got {' '.join(v)}")
    return int(v[0]), int(v[1])


if __name__ == "__main__":
    main()
