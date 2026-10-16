"""
Where does training time go? Measures, on this PC, with the settings of an earlier run:

  1. each extra pixel augmentation (train/recipe.py, RandomFog included) on one 640x640 image
  2. one training image through the whole Ultralytics pipeline (mosaic, copy-paste, mixup, ...),
     on one CPU core: with / without RandomFog, without extra pixel augs, and also without copy-paste / mixup
  3. the real training loader (worker processes) -> seconds per batch
  4. the GPU alone: forward + backward + optimizer step on one batch, AMP on, deterministic on / off

If (3) is much slower than (4), the GPU is waiting for the CPU.

From the SegPipeline folder (close Unreal and other GPU work first):
  .venv\\Scripts\\python.exe tools\\profile_train.py --run runs\\torpedo_v1_seg

Takes a few minutes; the results are also written to <run>\\profile.txt.
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def cfg_from_run(run_dir, **over):
    from ultralytics.cfg import get_cfg
    with open(os.path.join(run_dir, "args.yaml"), encoding="utf-8") as f:
        a = yaml.safe_load(f)
    keep = {k: v for k, v in a.items() if k not in ("augmentations", "save_dir")}
    keep.update(over)
    return get_cfg(overrides=keep), a


def dataset(cfg, data, aug_list):
    from ultralytics.data import build_yolo_dataset
    from ultralytics.utils import LOGGER
    cfg.augmentations = aug_list
    level = LOGGER.level
    LOGGER.setLevel(logging.WARNING)                         # no dataset scan / augmentation list spam
    try:
        return build_yolo_dataset(cfg, data["train"], cfg.batch, data, mode="train", stride=32)
    finally:
        LOGGER.setLevel(level)


def nvidia_smi():
    """GPU load / clocks / power right now, and what else is on the GPU (e.g. Unreal)."""
    import subprocess
    q = "name,utilization.gpu,memory.used,memory.total,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu,pstate"
    try:
        a = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        b = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
        procs = b[b.find("Processes"):].strip() if "Processes" in b else ""
        return f"   {q}\n   {a}\n" + "\n".join("   " + l for l in procs.splitlines())
    except Exception as e:
        return f"   nvidia-smi not available: {e}"


def per_image_ms(ds, n):
    idx = np.random.default_rng(0).integers(0, len(ds), n)
    ds[int(idx[0])]                                          # warm-up (imports, first read)
    t = time.perf_counter()
    for i in idx:
        ds[int(i)]
    return (time.perf_counter() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run folder with args.yaml (e.g. runs\\torpedo_v1_seg)")
    ap.add_argument("--data", default=None, help="override the dataset yaml from args.yaml")
    ap.add_argument("--images", type=int, default=40, help="images per single-core measurement")
    ap.add_argument("--batches", type=int, default=30, help="batches timed through the real loader")
    ap.add_argument("--workers", type=int, nargs="+", default=None, help="loader worker counts to try (default: the run's)")
    ap.add_argument("--steps", type=int, default=30, help="GPU steps timed")
    ap.add_argument("--batch", type=int, default=None, help="override the run's batch size")
    ap.add_argument("--fog", action="store_true", help="time the loader with RandomFog (the old recipe)")
    ap.add_argument("--skip", nargs="*", default=[], help="parts to skip: aug single loader gpu")
    a = ap.parse_args()
    report = open(os.path.join(a.run, "profile.txt"), "w", encoding="utf-8")

    def say(*parts):
        text = " ".join(str(x) for x in parts)
        print(text, flush=True)
        report.write(text + "\n")
        report.flush()

    from ultralytics import YOLO
    from ultralytics.data import build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.torch_utils import init_seeds, unset_deterministic
    from recipe import pixel_augmentations

    over = {"data": a.data} if a.data else {}
    if a.batch:
        over["batch"] = a.batch
    cfg, args = cfg_from_run(a.run, **over)
    data = check_det_dataset(cfg.data)
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda:0" if cuda else "cpu")
    say(f"torch {torch.__version__}  cuda {torch.version.cuda}  cudnn {torch.backends.cudnn.version()}  "
          f"device {torch.cuda.get_device_name(0) if cuda else 'CPU'}  logical CPUs {os.cpu_count()}")
    say(f"run {a.run}: model {args['model']}  imgsz {cfg.imgsz}  batch {cfg.batch}  workers {cfg.workers}  "
          f"cache {cfg.cache}  amp {cfg.amp}  deterministic {cfg.deterministic}")
    say(f"dataset {cfg.data}: {data['train']}")
    if cuda:
        say("\n0) GPU before the test (anything already using it?)\n" + nvidia_smi())
    results = {}

    # ---- 1. extra pixel augmentations, one at a time, always applied
    if "aug" not in a.skip:
        import albumentations as A
        import cv2
        ds0 = dataset(cfg, data, [])
        img = ds0.load_image(0)[0]
        img = cv2.resize(img, (cfg.imgsz, cfg.imgsz))
        say("\n1) extra pixel augmentations on a 640x640 image (ms per call, when applied)")
        rows = []
        for t in pixel_augmentations(fog=True):
            subs = t.transforms if isinstance(t, A.OneOf) else [t]
            share = t.p / len(subs)                           # OneOf: each member equally likely
            for s in subs:
                s.p = 1.0
                s(image=img)
                t0 = time.perf_counter()
                for _ in range(5):
                    s(image=img)
                ms = (time.perf_counter() - t0) / 5 * 1000
                rows.append((type(s).__name__, ms, share, ms * share))
        for name, ms, p, exp in sorted(rows, key=lambda r: -r[3]):
            say(f"   {name:<24}{ms:8.1f} ms   used {p:5.0%} of images  -> {exp:6.1f} ms per image on average")
        results["pixel_aug_ms"] = sum(r[3] for r in rows)
        say(f"   total on average: {results['pixel_aug_ms']:.1f} ms per training image")

    # ---- 2. whole pipeline, one core
    if "single" not in a.skip:
        say(f"\n2) one training image through the whole pipeline, one CPU core ({a.images} images each)")
        variants = [
            ("recipe with RandomFog (old)", pixel_augmentations(fog=True), {}),
            ("recipe without RandomFog (default)", pixel_augmentations(), {}),
            ("without any extra pixel augs", [], {}),
            ("  ... and without copy_paste / mixup", [], {"copy_paste": 0.0, "mixup": 0.0}),
        ]
        for name, augs, ov in variants:
            c, _ = cfg_from_run(a.run, **over, **ov)
            ms = per_image_ms(dataset(c, data, augs), a.images)
            results[name] = ms
            say(f"   {name:<40}{ms:8.1f} ms per image  -> {ms * cfg.batch / 1000:5.2f} core-seconds per batch of {cfg.batch}")

    # ---- 3. the real loader
    if "loader" not in a.skip:
        say(f"\n3) training loader, recipe {'with' if a.fog else 'without'} RandomFog ({a.batches} batches after warm-up)")
        ds = dataset(cfg, data, pixel_augmentations(fog=a.fog))
        for w in a.workers or [cfg.workers]:
            dl = build_dataloader(ds, cfg.batch, w, shuffle=True, rank=-1, device=dev)
            it = iter(dl)
            for _ in range(min(5, a.batches)):
                next(it)
            t0 = time.perf_counter()
            for _ in range(a.batches):
                next(it)
            s = (time.perf_counter() - t0) / a.batches
            results[f"loader_w{w}"] = s
            say(f"   workers {w:>2} (used {dl.num_workers:>2}): {s:5.2f} s per batch")
            try:
                dl.iterator._shutdown_workers()             # don't let idle workers compete with the next test
            except Exception:
                pass
            del it, dl

    # ---- 4. GPU alone
    if "gpu" not in a.skip:
        say(f"\n4) GPU step alone (one batch reused, {a.steps} steps, AMP {'on' if cuda else 'off (CPU)'})")
        ds = dataset(cfg, data, [])
        batch = ds.collate_fn([ds[i] for i in range(cfg.batch)])
        weights = os.path.join(a.run, "weights", "best.pt")
        weights = weights if os.path.isfile(weights) else args["model"]
        model = YOLO(weights).model.to(dev).train()
        model.args = cfg
        for p in model.parameters():
            p.requires_grad_(True)
        opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(dev)
        batch["img"] = batch["img"].float() / 255
        for det in (True, False):
            init_seeds(0, deterministic=det) if det else unset_deterministic()
            torch.backends.cudnn.benchmark = not det
            for i in range(a.steps + 5):
                if i == 5:
                    torch.cuda.synchronize() if cuda else None
                    t0 = time.perf_counter()
                with torch.autocast(dev.type, enabled=cuda):
                    loss, _ = model(batch)
                scaler.scale(loss.sum()).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize() if cuda else None
            s = (time.perf_counter() - t0) / a.steps
            if cuda and det:
                say("   GPU while stepping:\n" + nvidia_smi().split("\n")[1])
            results[f"gpu_det{det}"] = s
            say(f"   deterministic {str(det):<5}: {s:5.3f} s per batch")
        if cuda:
            say(f"   GPU memory peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
                  f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
        try:
            from ultralytics.utils.checks import check_amp
            if cuda:
                say(f"   AMP check passes: {check_amp(YOLO(args['model']).model.to(dev))}")
        except Exception as e:
            say(f"   AMP check could not run: {e}")

    # ---- summary
    lw = [v for k, v in results.items() if k.startswith("loader_")]
    g = results.get("gpu_detTrue")
    if lw and g:
        say(f"\nloader {min(lw):.2f} s/batch vs GPU {g:.2f} s/batch -> "
              + ("the GPU waits for the CPU: speed up the data pipeline" if min(lw) > 1.3 * g
                 else "the GPU is the limit: smaller model / image size, or fewer images and epochs"))


if __name__ == "__main__":  # required on Windows for loader worker processes
    main()
