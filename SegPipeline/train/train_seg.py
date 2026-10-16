"""
Train a YOLO segmentation model on SegPipeline data with heavy, generalization-oriented augmentation.

    ..\\.venv\\Scripts\\python.exe train_seg.py --data <dataset>\\data.yaml

The goal is a model that relies on SHAPE, not on the look of one pool or one renderer:
  - colour is unreliable underwater  -> strong hue/saturation/RGB shifts, grayscale, channel shuffle
  - cameras differ                   -> blur/defocus/motion blur, sensor noise, JPEG artefacts, low resolution
  - water differs                    -> gamma, contrast, CLAHE (haze comes from the renders' water_fog;
                                        RandomFog is off by default: ~1 s per image, --fog brings it back)
  - viewpoints differ                -> scale, rotation, perspective, translation, mosaic, copy-paste
Spatial transforms use Ultralytics' built-ins (they keep polygons correct);
Albumentations is used for pixel-level transforms only.

Validation here is on synthetic frames, so its mAP will look optimistic. Judge the model on
REAL images (see README) - that is the number that tells you whether it generalizes.
"""
import argparse
import os

from ultralytics import YOLO

import recipe


def pixel_augmentations(fog=False):
    """Image-only transforms (never move pixels, so labels stay valid). Defined in recipe.py."""
    return recipe.pixel_augmentations(fog)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolo11s-seg.pt", help="pretrained weights (COCO) or a previous best.pt")
    ap.add_argument("--epochs", type=int, default=recipe.FULL["epochs"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="torpedo_seg")
    ap.add_argument("--project", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"))
    ap.add_argument("--workers", type=int, default=recipe.WORKERS)
    ap.add_argument("--freeze", type=int, default=0, help="freeze first N layers (use ~10 when fine-tuning on few real images)")
    ap.add_argument("--time", type=float, default=None,
                    help="train for this many hours instead of --epochs (LR schedule is fitted to the time)")
    ap.add_argument("--cache", default="disk",
                    help="disk | ram | none. 'ram' duplicates the cache in every worker on Windows - use disk")
    ap.add_argument("--fog", action="store_true", help="add RandomFog back (slow: ~1 s per image it runs on)")
    args = ap.parse_args()

    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=False,
        cache=False if args.cache == "none" else args.cache,
        time=args.time,
        patience=recipe.FULL["patience"],
        cos_lr=True,
        close_mosaic=recipe.FULL["close_mosaic"],
        freeze=args.freeze or None,
        plots=True,
        augmentations=pixel_augmentations(args.fog),
        **recipe.HYP,
    )


if __name__ == "__main__":  # required on Windows for dataloader workers
    main()
