"""
Train a YOLO segmentation model on SegPipeline data with heavy, generalization-oriented augmentation.

    ..\\.venv\\Scripts\\python.exe train_seg.py --data <dataset>\\data.yaml

The goal is a model that relies on SHAPE, not on the look of one pool or one renderer:
  - colour is unreliable underwater  -> strong hue/saturation/RGB shifts, grayscale, channel shuffle
  - cameras differ                   -> blur/defocus/motion blur, sensor noise, JPEG artefacts, low resolution
  - water differs                    -> fog/haze, gamma, contrast, CLAHE
  - viewpoints differ                -> scale, rotation, perspective, translation, mosaic, copy-paste
Spatial transforms use Ultralytics' built-ins (they keep polygons correct);
Albumentations is used for pixel-level transforms only.

Validation here is on synthetic frames, so its mAP will look optimistic. Judge the model on
REAL images (see README) - that is the number that tells you whether it generalizes.
"""
import argparse
import os

import albumentations as A
from ultralytics import YOLO


def pixel_augmentations():
    """Image-only transforms (never move pixels, so labels stay valid)."""
    return [
        # camera optics / focus / motion
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9)),
            A.MotionBlur(blur_limit=(3, 11)),
            A.MedianBlur(blur_limit=5),
            A.Defocus(radius=(2, 5)),
        ], p=0.4),
        # sensor noise and compression
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.10)),
            A.ISONoise(color_shift=(0.01, 0.06), intensity=(0.1, 0.6)),
        ], p=0.4),
        A.ImageCompression(quality_range=(30, 95), p=0.35),
        A.Downscale(scale_range=(0.35, 0.8), p=0.2),
        # water colour / turbidity / exposure
        A.RGBShift(r_shift_limit=35, g_shift_limit=35, b_shift_limit=35, p=0.45),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.45, p=0.6),
        A.RandomGamma(gamma_limit=(60, 160), p=0.3),
        A.RandomFog(fog_coef_range=(0.1, 0.45), alpha_coef=0.1, p=0.15),
        A.CLAHE(clip_limit=(1, 4), p=0.15),
        # colour independence
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Sharpen(alpha=(0.1, 0.4), p=0.1),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolo11s-seg.pt", help="pretrained weights (COCO) or a previous best.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="torpedo_seg")
    ap.add_argument("--project", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--freeze", type=int, default=0, help="freeze first N layers (use ~10 when fine-tuning on few real images)")
    ap.add_argument("--time", type=float, default=None,
                    help="train for this many hours instead of --epochs (LR schedule is fitted to the time)")
    ap.add_argument("--cache", default="disk",
                    help="disk | ram | none. 'ram' duplicates the cache in every worker on Windows - use disk")
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
        patience=40,
        cos_lr=True,
        close_mosaic=15,
        freeze=args.freeze or None,
        plots=True,
        # --- built-in (label-aware) augmentation, turned up ---
        hsv_h=0.08,          # hue: blue <-> green <-> teal water
        hsv_s=0.9,
        hsv_v=0.6,
        degrees=15.0,        # camera roll
        translate=0.2,
        scale=0.7,           # 0.3x - 1.7x zoom: far and very close targets
        shear=4.0,
        perspective=0.0008,  # oblique views
        fliplr=0.5,          # mirrored drawings are still the same class
        flipud=0.0,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,      # paste objects onto other frames/backgrounds
        # --- pixel-level augmentation (Albumentations) ---
        augmentations=pixel_augmentations(),
    )


if __name__ == "__main__":  # required on Windows for dataloader workers
    main()
