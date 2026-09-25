"""
Run a trained model on real images and save annotated results (masks + boxes + labels).

    python predict_real.py --model <best.pt> --source <folder of images or Roboflow export root> --out <dir>

Options:
  --conf 0.4            confidence threshold for what gets drawn
  --unstretch 1280 720  undo Roboflow's square stretch before inference (use the camera's aspect)
  --gt                  also draw ground-truth boxes (red, marked "GT") when a matching labels/*.txt exists
  --grid 12             also write contact sheets with this many tiles each (0 = off)
Accepts either a plain image folder or a Roboflow export (searches */images/ inside it).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(src: Path):
    if any(p.suffix.lower() in EXTS for p in src.glob("*")):
        return sorted(p for p in src.glob("*") if p.suffix.lower() in EXTS)
    return sorted(p for p in src.glob("*/images/*") if p.suffix.lower() in EXTS)


def draw_gt(img, label_path, names_ds, keep):
    h, w = img.shape[:2]
    for line in label_path.read_text().splitlines():
        v = line.split()
        if len(v) < 5 or (keep and names_ds[int(v[0])] not in keep):
            continue
        nums = np.array(v[1:], float)
        if len(nums) == 4:
            x, y, bw, bh = nums
            x0, y0, x1, y1 = x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2
        else:
            pts = nums.reshape(-1, 2)
            x0, y0, x1, y1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
        p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))
        cv2.rectangle(img, p0, p1, (0, 0, 255), 2, cv2.LINE_AA)            # ground truth = red
        cv2.putText(img, "GT", (p0[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


def contact_sheets(files, out: Path, per_sheet: int, cols: int = 4, tile_w: int = 480):
    for s in range(0, len(files), per_sheet):
        tiles = []
        for f in files[s:s + per_sheet]:
            im = cv2.imread(str(f))
            th = int(im.shape[0] * tile_w / im.shape[1])
            tiles.append(cv2.resize(im, (tile_w, th), interpolation=cv2.INTER_AREA))
        th = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(str(out / f"sheet_{s // per_sheet + 1:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--unstretch", type=int, nargs=2, metavar=("W", "H"))
    ap.add_argument("--gt", action="store_true")
    ap.add_argument("--grid", type=int, default=12)
    args = ap.parse_args()

    src, out = Path(args.source), Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    keep = set(model.names.values())

    names_ds = None
    yml = src / "data.yaml"
    if args.gt and yml.exists():
        import yaml
        names_ds = yaml.safe_load(yml.read_text())["names"]
        if isinstance(names_ds, dict):
            names_ds = [names_ds[k] for k in sorted(names_ds)]

    images = find_images(src)
    written, counts = [], {n: 0 for n in model.names.values()}
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if args.unstretch:
            img = cv2.resize(img, tuple(args.unstretch), interpolation=cv2.INTER_AREA)
        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        for c in r.boxes.cls.tolist():
            counts[model.names[int(c)]] += 1
        vis = r.plot(line_width=2, font_size=14)
        lbl = p.parent.parent / "labels" / (p.stem + ".txt")
        if args.gt and names_ds and lbl.exists():
            draw_gt(vis, lbl, names_ds, keep)
        dst = out / "images" / f"{p.parent.parent.name}_{p.stem[:40]}.jpg"
        cv2.imwrite(str(dst), vis)
        written.append(dst)

    if args.grid:
        contact_sheets(written, out, args.grid)
    print(f"{len(written)} images -> {out / 'images'}")
    print("detections (conf >= %.2f): " % args.conf + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if args.grid:
        print(f"contact sheets -> {out}\\sheet_*.jpg")


if __name__ == "__main__":
    main()
