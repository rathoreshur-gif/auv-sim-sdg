"""
Train YOLO segmentation or pose on an SDG dataset, with the augmentation recipe from train_seg.py.

    ..\\.venv\\Scripts\\python.exe train_yolo.py --task seg  --data <dataset>\\seg.yaml
    ..\\.venv\\Scripts\\python.exe train_yolo.py --task pose --data <dataset>\\pose.yaml

The dataset builder (sdg/post.py) calls this by itself when post.train is set in the config.
Pose: horizontal flips are only used when pose.yaml has flip_idx (the board/bin corner presets set it),
otherwise a flipped image would swap left and right corners without their labels.
"""
import argparse
import os

from ultralytics import YOLO

from train_seg import pixel_augmentations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["seg", "pose"], required=True)
    ap.add_argument("--data", required=True, help="seg.yaml or pose.yaml written by sdg/post.py")
    ap.add_argument("--model", default=None, help="default yolo11s-seg.pt / yolo11s-pose.pt, or a previous best.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--freeze", type=int, default=0, help="freeze first N layers (use ~10 when fine-tuning on few real images)")
    ap.add_argument("--time", type=float, default=None, help="train for this many hours instead of --epochs")
    ap.add_argument("--cache", default="disk", help="disk | ram | none")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        has_flip = "flip_idx:" in f.read()
    aug = dict(
        hsv_h=0.08, hsv_s=0.9, hsv_v=0.6,     # water colour varies a lot
        degrees=15.0, translate=0.2, scale=0.7, shear=4.0, perspective=0.0008,
        fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.15,
    )
    if args.task == "seg":
        aug["copy_paste"] = 0.3
    elif not has_flip:
        aug["fliplr"] = 0.0

    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    model = YOLO(args.model or f"yolo11s-{args.task}.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name or f"sdg_{args.task}",
        exist_ok=False,
        cache=False if args.cache == "none" else args.cache,
        time=args.time,
        patience=40,
        cos_lr=True,
        close_mosaic=15,
        freeze=args.freeze or None,
        plots=True,
        augmentations=pixel_augmentations(),
        **aug,
    )


if __name__ == "__main__":  # required on Windows for dataloader workers
    main()
