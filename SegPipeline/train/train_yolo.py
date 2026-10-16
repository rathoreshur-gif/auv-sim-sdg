"""
Train YOLO segmentation or pose on an SDG dataset with the recipe in recipe.py, and optionally score
the result on real labelled images right after.

    ..\\.venv\\Scripts\\python.exe train_yolo.py --task seg  --data <dataset>\\seg.yaml
    ..\\.venv\\Scripts\\python.exe train_yolo.py --task seg  --data <dataset>\\seg.yaml --quick --eval <real>\\data.yaml <other>.zip
    ..\\.venv\\Scripts\\python.exe train_yolo.py --task pose --data <dataset>\\pose.yaml

--quick   30 epochs, patience 10 (instead of 150 / 40): for trying things out. Real-image scores barely
          move after that; keep full runs for the model you ship.
--eval    real datasets (data.yaml or Roboflow .zip) to score best.pt on when training ends
          -> <run>/real_eval/<dataset>/ (metrics, overlays) and <run>/real_eval/summary.json
--fog     add RandomFog back to the pixel augmentations (slow: the GPU waits for the CPU)
--no-copy-paste / --no-mixup   switch those off (they cost CPU time and paste drawings off the board)

The dataset builder (sdg/post.py) calls this by itself when post.train is set in the config, and the
SDG app's Train page ("Quick train") runs it with --quick --eval.
Pose: horizontal flips are only used when pose.yaml has flip_idx (the board/bin corner presets set it),
otherwise a flipped image would swap left and right corners without their labels.
"""
import argparse
import os
import time

import recipe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["seg", "pose"], required=True)
    ap.add_argument("--data", required=True, help="seg.yaml or pose.yaml written by sdg/post.py")
    ap.add_argument("--model", default=None, help="default yolo11s-seg.pt / yolo11s-pose.pt, or a previous best.pt")
    ap.add_argument("--quick", action="store_true", help=f"{recipe.QUICK['epochs']} epochs, patience {recipe.QUICK['patience']}")
    ap.add_argument("--epochs", type=int, default=None, help=f"default {recipe.FULL['epochs']} ({recipe.QUICK['epochs']} with --quick)")
    ap.add_argument("--patience", type=int, default=None, help=f"default {recipe.FULL['patience']} ({recipe.QUICK['patience']} with --quick)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--workers", type=int, default=recipe.WORKERS)
    ap.add_argument("--freeze", type=int, default=0, help="freeze first N layers (use ~10 when fine-tuning on few real images)")
    ap.add_argument("--time", type=float, default=None, help="train for this many hours instead of --epochs")
    ap.add_argument("--cache", default="disk", help="disk | ram | none")
    ap.add_argument("--fog", action="store_true", help="add RandomFog back (slow)")
    ap.add_argument("--no-copy-paste", action="store_true")
    ap.add_argument("--no-mixup", action="store_true")
    ap.add_argument("--eval", nargs="*", default=[], metavar="DATA", help="real datasets to score best.pt on at the end")
    ap.add_argument("--unstretch", nargs="+", default=[str(v) for v in recipe.UNSTRETCH], metavar="W H",
                    help="undo Roboflow's square stretch before scoring ('none' = off)")
    ap.add_argument("--splits", nargs="+", default=["valid", "test"], help="splits of the real datasets to score")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        has_flip = "flip_idx:" in f.read()
    sched = dict(recipe.QUICK if args.quick else recipe.FULL)
    if args.epochs is not None:
        sched["epochs"] = args.epochs
    if args.patience is not None:
        sched["patience"] = args.patience
    aug = dict(recipe.HYP)
    if args.task == "pose":
        aug.pop("copy_paste")
        if not has_flip:
            aug["fliplr"] = 0.0
    if args.no_copy_paste:
        aug["copy_paste"] = 0.0
    if args.no_mixup:
        aug["mixup"] = 0.0

    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    from ultralytics import YOLO

    model = YOLO(args.model or f"yolo11s-{args.task}.pt")
    print(f"[train] {'QUICK' if args.quick else 'full'} run: {sched['epochs']} epochs, patience {sched['patience']}, "
          f"workers {args.workers}, RandomFog {'on' if args.fog else 'off'}, copy_paste {aug.get('copy_paste', 0)}, "
          f"mixup {aug['mixup']}" + (f", then scored on {len(args.eval)} real dataset(s)" if args.eval else ""))
    t0 = time.time()
    model.train(
        data=args.data,
        epochs=sched["epochs"],
        patience=sched["patience"],
        close_mosaic=sched["close_mosaic"],
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name or f"sdg_{args.task}",
        exist_ok=False,
        cache=False if args.cache == "none" else args.cache,
        time=args.time,
        cos_lr=True,
        freeze=args.freeze or None,
        plots=True,
        augmentations=recipe.pixel_augmentations(args.fog),
        **aug,
    )
    run_dir = str(model.trainer.save_dir)
    best = str(model.trainer.best) if os.path.isfile(str(model.trainer.best)) else str(model.trainer.last)
    print(f"\n[train] done in {(time.time() - t0) / 3600:.1f} h: {best}")

    if args.eval:
        import torch
        from eval_real import evaluate_sets, parse_unstretch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        evaluate_sets(best, args.eval, os.path.join(run_dir, "real_eval"), splits=args.splits,
                      imgsz=args.imgsz, unstretch=parse_unstretch(args.unstretch))
    print(f"\n[train] run folder: {run_dir}")


if __name__ == "__main__":  # required on Windows for dataloader workers
    main()
