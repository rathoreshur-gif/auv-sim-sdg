"""
EXR frames from Movie Render Queue -> YOLO datasets (segmentation + pose), with no mask step in
between. Runs in the SegPipeline venv (numpy + opencv). The editor starts it automatically in
watch mode while MRQ renders; you can also run it by hand:

    python post.py <raw_dir>/<run name>            # convert whatever is there, then finish
    python post.py <raw_dir>/<run name> --watch    # keep converting until the render is done
    python post.py <raw_dir>/<run name> --check    # convert 3 frames and print what was found
    python post.py <raw_dir>/<run name> --zip      # only (re)pack the finished dataset into a shareable zip

Each EXR holds the rendered image (FinalImage) and the SDGLabels pass:
    R = stencil id of the visible tagged surface (0 = nothing)    G = scene depth (cm)
plus the camera pose and FOV in its header. The stage manifests written by the editor map stencil
ids to classes and hold every object's keypoints, so labels come straight out of each frame.

Output (<dataset_dir>/<run name>/):
    images/{train,val}/*.jpg     labels/{train,val}/*.txt     seg.yaml       (YOLO-seg)
    pose/images/...  (hard links) pose/labels/...              pose.yaml      (YOLO-pose)
    previews/*.jpg   meta/<stage>.jsonl (sampled parameters per frame)   stats.json
With post.train off, a full run also writes <dataset_dir>/<run name>_yolo_seg.zip: the YOLO-seg
dataset alone (images, labels, data.yaml without absolute paths) to share or copy to another machine.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exr                       # noqa: E402
import sdg_common as sc          # noqa: E402

FRAME_RE = re.compile(r"\.(\d+)\.exr$", re.IGNORECASE)
DATASET_MARKER = ".sdg_dataset"
PALETTE = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 128, 255), (255, 64, 64),
           (128, 255, 128), (255, 160, 0), (160, 100, 255), (255, 255, 255)]


# ============================================================================ pixels
def _srgb_lut():
    """float16 bit pattern -> sRGB byte (what MRQ's own 8-bit writers do, minus the dither)."""
    v = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    v = np.clip(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


SRGB = _srgb_lut()


def split_layers(img):
    """(rgb_layer, label_layer) names. MRQ writes its first pass as the unnamed layer."""
    names = img.layer_names()
    label = next((n for n in names if n.endswith(sc.LABEL_PASS)), None)
    rgb = next((n for n in names if n.endswith("FinalImage")), None)
    if label is None and rgb not in (None, "") and "" in names:
        label = ""
    if rgb is None and label != "" and "" in names:
        rgb = ""
    if label is None or rgb is None:
        raise exr.ExrError(f"expected FinalImage and {sc.LABEL_PASS} layers, found {names}")
    return rgb, label


def to_bgr8(img, layer):
    planes = []
    for c in "BGR":
        p = img.channels[f"{layer}.{c}" if layer else c]
        if p.dtype != np.float16:
            p = p.astype(np.float16)
        planes.append(SRGB[p.view(np.uint16)])
    return np.dstack(planes)


def camera_from_attrs(attrs):
    """Camera location, rotation (pitch, yaw, roll) and horizontal FOV exactly as MRQ rendered it."""
    try:
        loc = tuple(float(attrs[f"unreal/camera/curPos/{a}"]) for a in "xyz")
        rot = tuple(float(attrs[f"unreal/camera/curRot/{a}"]) for a in ("pitch", "yaw", "roll"))
    except (KeyError, ValueError):
        return None
    fovs = sorted((k, v) for k, v in attrs.items() if k.startswith("unreal/") and k.endswith("/fov"))
    fov = next((float(v) for k, v in fovs if k.endswith("FinalImage/fov")), None)
    if fov is None and fovs:
        fov = float(fovs[0][1])
    return (loc, rot, fov) if fov else None


# ============================================================================ polygons
def mask_polygon(mask, epsilon, min_part):
    """Binary mask -> ONE polygon (N, 2 float, pixel coords) for YOLO-seg.

    Several pieces (an object split by an occluder) and holes are joined into a single outline
    with zero-width bridges, so filling it (even-odd or non-zero) gives back the mask."""
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    hier = hier[0]
    loops = []
    for i, c in enumerate(contours):
        is_hole = hier[i][3] >= 0
        area = cv2.contourArea(c)
        if len(contours) > 1 and area < (min_part / 2.0 if is_hole else min_part):
            continue                                     # specks and pin-holes
        a = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2) if epsilon > 0 else c.reshape(-1, 2)
        if len(a) < 3:
            continue
        loops.append((is_hole, area, a.astype(np.float64)))
    outers = [l for l in loops if not l[0]]
    if not outers:
        return None
    loops.sort(key=lambda l: (l[0], -l[1]))             # biggest outer piece first, holes last
    poly = loops[0][2]
    for _, _, pts in loops[1:]:
        poly = _bridge(poly, pts)
    return poly


def _bridge(main, other):
    """Splice `other` into `main` through its closest vertex pair (out along a slit and back)."""
    d = ((main[:, None, :] - other[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    loop = np.concatenate([other[j:], other[:j + 1]])    # starts and ends at other[j]
    return np.concatenate([main[:i + 1], loop, main[i:]])


def fmt(v):
    return f"{v:.6f}".rstrip("0").rstrip(".") if v else "0"


# ============================================================================ worker
_W = {}


def _worker_init(run_json):
    cv2.setNumThreads(1)
    with open(run_json, encoding="utf-8") as f:
        run = json.load(f)
    stages = []
    for st in run["stages"]:
        with open(st["manifest"], encoding="utf-8") as f:
            man = json.load(f)
        keep = {fr["seq"]: i for i, fr in enumerate(man["frames"])}
        stages.append((man, keep))
    _W.update(run=run, stages=stages)


def process(stage_idx, path, write_preview):
    """One EXR -> jpg + labels. Returns a stats dict (raises exr.ExrIncomplete if still being written)."""
    run, (man, keep) = _W["run"], _W["stages"][stage_idx]
    out = run["output"]
    m = FRAME_RE.search(path)
    seq = int(m.group(1))
    if seq not in keep:                                   # teleport warm-up frame
        if not out["keep_exr"]:
            os.remove(path)
        return {"skipped": 1}
    fr = man["frames"][keep[seq]]
    img = exr.read(path)
    rgb_layer, lab_layer = split_layers(img)
    H, W = img.height, img.width
    ids = np.rint(np.nan_to_num(img.channels[f"{lab_layer}.R" if lab_layer else "R"].astype(np.float32)))
    ids = np.clip(ids, 0, 255).astype(np.uint8)
    depth = img.channels[f"{lab_layer}.G" if lab_layer else "G"].astype(np.float32)

    cam = camera_from_attrs(img.attrs)
    cam_src = "exr"
    if cam is None:
        c = fr["cam"]
        cam, cam_src = ((c[0], c[1], c[2]), (c[3], c[4], c[5]), c[6]), "manifest"

    min_area, eps, max_depth = out["min_area"], out["epsilon"], out["max_label_depth"]
    present = np.bincount(ids.ravel(), minlength=256)
    instances = man["instances"]
    fog = run.get("water_fog", {})
    fog_d = fr["params"].get(sc.FOG_DENSITY) if fog.get("enabled") else None

    def hidden(depths):
        """Too far (max_label_depth) or lost in the water haze (water_fog.min_visibility)."""
        d = float(np.median(depths))
        return bool(max_depth and d > max_depth) or (fog_d is not None and np.exp(-fog_d * d) < fog["min_visibility"])

    # ------------------------------------------------ segmentation: one polygon per stencil id
    seg_lines, seg_polys = [], []
    for sid, info in instances.items():
        s = int(sid)
        if info.get("seg_class") is None or present[s] < min_area:
            continue
        mask = (ids == s).astype(np.uint8)
        if hidden(depth[mask > 0]):
            continue
        poly = mask_polygon(mask, eps, max(8.0, min_area / 4.0))
        if poly is None:
            continue
        cls = info["seg_class"]
        coords = " ".join(f"{fmt((x + 0.5) / W)} {fmt((y + 0.5) / H)}" for x, y in poly)
        seg_lines.append(f"{cls} {coords}")
        seg_polys.append((cls, poly))

    # ------------------------------------------------ pose: bbox from the object's pixels + projected keypoints
    pose_lines, pose_draw = [], []
    K = man["num_keypoints"]
    if K:
        f_px = sc.focal_px(cam[2], W)
        fwd, right, up = (np.array(a) for a in sc.rotator_axes(*cam[1]))
        cam_loc = np.array(cam[0])
        for key, obj in man["objects"].items():
            if obj.get("pose_class") is None:
                continue
            sids = obj["stencils"]
            area = int(sum(present[s] for s in sids))
            if area < min_area:
                continue
            ys, xs = np.nonzero(np.isin(ids, sids))
            if hidden(depth[ys, xs]):
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            t = fr.get("objects", {}).get(key, obj["transform"])
            kp_w = np.array([sc.transform_point(t[0:3], t[3:6], t[6:9], p) for p in obj["keypoints"]])
            d = kp_w - cam_loc
            zc, xc, yc = d @ fwd, d @ right, d @ up
            kps = []
            for j in range(K):
                if j >= len(kp_w) or zc[j] <= 1.0:
                    kps.append((0.0, 0.0, 0))
                    continue
                u = W / 2.0 + f_px * xc[j] / zc[j]
                v = H / 2.0 - f_px * yc[j] / zc[j]
                if not (0.0 <= u < W and 0.0 <= v < H):
                    kps.append((0.0, 0.0, 0))
                    continue
                iu, iv = int(u), int(v)
                scene = depth[max(iv - 1, 0):iv + 2, max(iu - 1, 0):iu + 2].max()
                vis = 2 if zc[j] <= scene * 1.01 + 3.0 else 1
                kps.append((u, v, vis))
            cls = obj["pose_class"]
            box = f"{fmt((x0 + x1) / 2 / W)} {fmt((y0 + y1) / 2 / H)} {fmt((x1 - x0) / W)} {fmt((y1 - y0) / H)}"
            kp_txt = " ".join(f"{fmt(u / W)} {fmt(v / H)} {vis}" for u, v, vis in kps)
            pose_lines.append(f"{cls} {box} {kp_txt}")
            pose_draw.append((cls, (x0, y0, x1, y1), kps))

    # ------------------------------------------------ write
    stem = f"{man['stage']}_{seq:06d}"
    stats = {"frames": 1, "seg": {}, "pose": {}, "kp": [0, 0, 0], "cam": cam_src}
    if not seg_lines and not pose_lines and sc.stable_fraction(stem + "#empty") >= out["empty_keep"]:
        stats = {"empty_dropped": 1}
    else:
        split = "val" if sc.stable_fraction(stem) < out["val_fraction"] else "train"
        ds = run["dataset"]
        bgr = to_bgr8(img, rgb_layer)
        img_path = os.path.join(ds, "images", split, stem + ".jpg")
        cv2.imwrite(img_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, int(out["jpg_quality"])])
        _write(os.path.join(ds, "labels", split, stem + ".txt"), seg_lines)
        if run["pose_classes"]:
            pimg = os.path.join(ds, "pose", "images", split, stem + ".jpg")
            if os.path.exists(pimg):
                os.remove(pimg)
            try:
                os.link(img_path, pimg)
            except OSError:
                shutil.copyfile(img_path, pimg)
            _write(os.path.join(ds, "pose", "labels", split, stem + ".txt"), pose_lines)
        for cls, _ in seg_polys:
            stats["seg"][cls] = stats["seg"].get(cls, 0) + 1
        for cls, _, kps in pose_draw:
            stats["pose"][cls] = stats["pose"].get(cls, 0) + 1
            for _, _, vis in kps:
                stats["kp"][vis] += 1
        if not seg_lines and not pose_lines:
            stats["empty_kept"] = 1
        stats["meta"] = {"image": f"{split}/{stem}.jpg", "stage": man["stage"], "seq": seq,
                         "params": fr.get("params", {}), "n_seg": len(seg_lines), "n_pose": len(pose_lines)}
        if write_preview:
            _preview(os.path.join(ds, "previews", stem + ".jpg"), bgr, seg_polys, pose_draw, run)
    if not out["keep_exr"]:
        os.remove(path)
    return stats


def _write(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def _preview(path, bgr, seg_polys, pose_draw, run):
    im = bgr.copy()
    over = im.copy()
    for cls, poly in seg_polys:
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(over, [pts], PALETTE[cls % len(PALETTE)])
    im = cv2.addWeighted(over, 0.35, im, 0.65, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cls, poly in seg_polys:
        pts = np.round(poly).astype(np.int32)
        col = PALETTE[cls % len(PALETTE)]
        cv2.polylines(im, [pts], True, col, 1, cv2.LINE_AA)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        cv2.putText(im, run["seg_classes"][cls], (int(x), max(int(y) - 4, 12)), font, 0.45, col, 1, cv2.LINE_AA)
    for cls, (x0, y0, x1, y1), kps in pose_draw:
        cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 1)
        cv2.putText(im, run["pose_classes"][cls], (int(x0), int(y1) + 14), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        for j, (u, v, vis) in enumerate(kps):
            if vis:
                col = (0, 255, 0) if vis == 2 else (0, 140, 255)
                cv2.circle(im, (int(u), int(v)), 4, col, -1, cv2.LINE_AA)
                cv2.putText(im, str(j), (int(u) + 5, int(v) - 5), font, 0.4, col, 1, cv2.LINE_AA)
    cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ============================================================================ driver
def prepare_dataset(run, fresh):
    ds = run["dataset"]
    marker = os.path.join(ds, DATASET_MARKER)
    if fresh and os.path.isdir(ds) and os.listdir(ds):
        if not os.path.exists(marker):
            sys.exit(f"{ds} exists and was not made by this pipeline - pick another output.dataset_dir or name")
        for sub in ("images", "labels", "pose", "previews", "meta"):
            shutil.rmtree(os.path.join(ds, sub), ignore_errors=True)
    subs = ["images/train", "images/val", "labels/train", "labels/val", "previews", "meta"]
    if run["pose_classes"]:
        subs += ["pose/images/train", "pose/images/val", "pose/labels/train", "pose/labels/val"]
    for sub in subs:
        os.makedirs(os.path.join(ds, sub), exist_ok=True)
    open(marker, "w").close()


def write_yamls(run, man0):
    ds = run["dataset"].replace("\\", "/")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(run["seg_classes"]))
    body = f"train: images/train\nval: images/val\nnames:\n{names}\n"
    with open(os.path.join(ds, "seg.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {ds}\n{body}")
    with open(os.path.join(ds, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(body)                     # portable copy (no path): for tools that take the dataset folder
    if run["pose_classes"]:
        k = man0["num_keypoints"]
        flip = man0.get("flip_idx")
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(run["pose_classes"]))
        flip_line = f"flip_idx: {flip}\n" if flip else ""
        body = f"train: images/train\nval: images/val\nkpt_shape: [{k}, 3]\n{flip_line}names:\n{names}\n"
        with open(os.path.join(ds, "pose.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {ds}/pose\n{body}")
        with open(os.path.join(ds, "pose", "data.yaml"), "w", encoding="utf-8") as f:
            f.write(body)


def zip_dataset(run):
    """<dataset_dir>/<name>_yolo_seg.zip with one top folder <name>/ holding data.yaml, images/{train,val},
    labels/{train,val} and stats.json. data.yaml has no `path`, so Ultralytics uses the folder it sits in
    (`yolo train data=<unzipped>/<name>/data.yaml`, or data=<the zip> directly)."""
    ds, name = run["dataset"], run["name"]
    out = os.path.join(os.path.dirname(os.path.normpath(ds)), f"{name}_yolo_seg.zip")
    tmp = out + ".part"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(run["seg_classes"]))
    stats = os.path.join(ds, "stats.json")
    counts = {}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}/data.yaml", f"# YOLO segmentation dataset '{name}' (synthetic, Unreal Engine)\n"
                                         f"train: images/train\nval: images/val\nnames:\n{names}\n")
        for split in ("train", "val"):
            imgs = sorted(f for f in os.listdir(os.path.join(ds, "images", split)) if f.lower().endswith(".jpg"))
            for f in imgs:
                # jpgs are already compressed: store them, deflate only the text labels
                z.write(os.path.join(ds, "images", split, f), f"{name}/images/{split}/{f}", zipfile.ZIP_STORED)
                lab = os.path.join(ds, "labels", split, os.path.splitext(f)[0] + ".txt")
                if os.path.exists(lab):
                    z.write(lab, f"{name}/labels/{split}/{os.path.basename(lab)}")
            counts[split] = len(imgs)
        if os.path.exists(stats):
            z.write(stats, f"{name}/stats.json")
    os.replace(tmp, out)
    mb = os.path.getsize(out) / 1e6
    print(f"[post] shareable zip: {out}  ({counts.get('train', 0)} train + {counts.get('val', 0)} val images, {mb:.0f} MB)")
    return out


class Totals:
    def __init__(self, run):
        self.run = run
        self.n = {"frames": 0, "skipped": 0, "empty_kept": 0, "empty_dropped": 0, "errors": 0}
        self.seg = [0] * len(run["seg_classes"])
        self.pose = [0] * len(run["pose_classes"])
        self.kp = [0, 0, 0]
        self.cam_fallback = 0
        self.meta = {}

    def add(self, st):
        for k in self.n:
            self.n[k] += st.get(k, 0)
        for c, v in st.get("seg", {}).items():
            self.seg[c] += v
        for c, v in st.get("pose", {}).items():
            self.pose[c] += v
        for i, v in enumerate(st.get("kp", [0, 0, 0])):
            self.kp[i] += v
        if st.get("cam") == "manifest":
            self.cam_fallback += 1
        if "meta" in st:
            self.meta.setdefault(st["meta"]["stage"], []).append(st["meta"])

    def report(self, secs):
        n = self.n
        print(f"\nframes written: {n['frames']}  (background frames kept {n['empty_kept']}, dropped {n['empty_dropped']}, "
              f"warm-up frames skipped {n['skipped']}, errors {n['errors']})  in {secs:.0f} s")
        for name, c in zip(self.run["seg_classes"], self.seg):
            print(f"  seg   {name:<16} {c} polygons")
        for name, c in zip(self.run["pose_classes"], self.pose):
            print(f"  pose  {name:<16} {c} objects")
        if any(self.kp):
            tot = sum(self.kp)
            print(f"  keypoints: visible {self.kp[2]}  occluded {self.kp[1]}  outside {self.kp[0]}  (of {tot})")
        if self.cam_fallback:
            print(f"  WARNING: {self.cam_fallback} frames had no camera metadata in the EXR; used the baked camera")
        return {"counts": n, "seg": dict(zip(self.run["seg_classes"], self.seg)),
                "pose": dict(zip(self.run["pose_classes"], self.pose)),
                "keypoints": {"visible": self.kp[2], "occluded": self.kp[1], "outside": self.kp[0]}}


def scan(run, done_paths):
    """(stage index, path) of EXRs not handled yet, oldest first."""
    found = []
    for si, st in enumerate(run["stages"]):
        d = st["exr_dir"]
        if not os.path.isdir(d):
            continue
        for e in os.scandir(d):
            if e.is_file() and FRAME_RE.search(e.name) and e.path not in done_paths:
                found.append((si, e.path))
    return sorted(found, key=lambda t: FRAME_RE.search(t[1]).group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="<raw_dir>/<run name> (holds run.json)")
    ap.add_argument("--watch", action="store_true", help="keep going until the editor writes _DONE")
    ap.add_argument("--check", action="store_true", help="convert 3 frames, print diagnostics, keep the EXRs")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="keep what is already in the dataset folder")
    ap.add_argument("--zip", action="store_true", help="only pack the finished dataset into <name>_yolo_seg.zip")
    args = ap.parse_args()

    run_json = os.path.join(args.run_dir, "run.json")
    with open(run_json, encoding="utf-8") as f:
        run = json.load(f)
    if args.zip:
        return zip_dataset(run)
    if args.check:
        return check(run, run_json)
    with open(run["stages"][0]["manifest"], encoding="utf-8") as f:
        man0 = json.load(f)
    prepare_dataset(run, fresh=not args.resume)
    write_yamls(run, man0)
    workers = args.workers or run["post"].get("workers", 6)
    done_flag = os.path.join(args.run_dir, "_DONE")
    fail_flag = os.path.join(args.run_dir, "_FAILED")

    per_stage_prev = {}
    for si, st in enumerate(run["stages"]):
        n = max(1, st["frames"] // max(1, run["output"]["previews"])) if run["output"]["previews"] else 0
        per_stage_prev[si] = n

    tot = Totals(run)
    t0, handled, inflight, retries = time.time(), set(), {}, {}
    last_print = last_progress = 0.0
    stage_handled = [0] * len(run["stages"])
    progress = Progress(args.run_dir, run, tot, stage_handled, t0)
    print(f"[post] {run['name']}: watching {len(run['stages'])} stage folder(s) -> {run['dataset']}")
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(run_json,)) as pool:
        while True:
            finished = os.path.exists(done_flag) or os.path.exists(fail_flag) or not args.watch
            now = time.time()
            for si, path in scan(run, handled):
                if path in inflight:
                    continue
                try:
                    if not finished and now - os.path.getmtime(path) < 1.0:
                        continue                              # MRQ may still be writing it
                except OSError:
                    continue
                seq = int(FRAME_RE.search(path).group(1))
                every = per_stage_prev[si]
                prev = bool(every) and (seq // max(1, run["stages"][si]["hold"])) % every == 0
                inflight[path] = (si, pool.submit(process, si, path, prev))
            for path, (si, fut) in list(inflight.items()):
                if not fut.done():
                    continue
                del inflight[path]
                try:
                    tot.add(fut.result())
                    handled.add(path)
                    stage_handled[si] += 1
                except exr.ExrIncomplete:
                    retries[path] = retries.get(path, 0) + 1
                    if finished and retries[path] > 20:
                        print(f"  giving up on incomplete file {path}")
                        handled.add(path)
                        stage_handled[si] += 1
                        tot.n["errors"] += 1
                except Exception as e:                       # keep going; report at the end
                    print(f"  ERROR {os.path.basename(path)}: {type(e).__name__}: {e}")
                    handled.add(path)
                    stage_handled[si] += 1
                    tot.n["errors"] += 1
            if time.time() - last_progress > 1:
                last_progress = time.time()
                progress.write("converting", in_flight=len(inflight))
            if time.time() - last_print > 5:
                last_print = time.time()
                el = time.time() - t0
                print(f"  {tot.n['frames']} frames  ({tot.n['frames'] / max(el, 1e-6):.1f}/s, {len(inflight)} in flight)")
            if finished and not inflight and not scan(run, handled):
                break
            time.sleep(0.2)

    for stage, rows in tot.meta.items():
        with open(os.path.join(run["dataset"], "meta", f"{stage}.jsonl"), "w", encoding="utf-8") as f:
            for r in sorted(rows, key=lambda r: r["seq"]):
                f.write(json.dumps(r) + "\n")
    summary = tot.report(time.time() - t0)
    with open(os.path.join(run["dataset"], "stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if os.path.exists(fail_flag):
        print("[post] the render reported a failure - the dataset holds what was rendered before it stopped")
    if not tot.n["frames"] and not tot.n["skipped"] and not tot.n["empty_dropped"]:
        print("[post] WARNING: no frames arrived - check the editor's Output Log (LogMovieRenderPipeline)")
    ds = run["dataset"].replace("\\", "/")
    print(f"[post] dataset: {ds}\n       seg:  {ds}/seg.yaml" + (f"\n       pose: {ds}/pose.yaml" if run["pose_classes"] else ""))
    print(f"       check {ds}/previews before training")
    train = run["post"].get("train", "")
    zip_path = None
    if run.get("quick"):
        print(f"[post] --quick run: no " + (f"training (post.train = {train!r})" if train else "zip") + " - that happens after a full render")
    elif not train:
        progress.write("zipping", stats=summary)
        zip_path = zip_dataset(run)
    progress.write("done", stats=summary, zip=zip_path, failed=os.path.exists(fail_flag))
    if run.get("quick"):
        return
    if run["post"].get("open_trainer"):
        open_trainer(zip_path or run["dataset"])
    if train:
        start_training(run, train)


class Progress:
    """<raw>/<name>/progress.json for the SDG app: frames converted per stage, counts, state."""

    def __init__(self, run_dir, run, tot, stage_handled, t0):
        self.path = os.path.join(run_dir, "progress.json")
        self.run, self.tot, self.stage_handled, self.t0 = run, tot, stage_handled, t0

    def write(self, state, **extra):
        data = {"time": time.time(), "state": state, "elapsed": time.time() - self.t0, "quick": self.run.get("quick", 0),
                "counts": dict(self.tot.n), "dataset": self.run["dataset"],
                "stages": [{"name": st["name"], "rendered": n, "expected": st["seq_frames"], "images": st["frames"]}
                           for st, n in zip(self.run["stages"], self.stage_handled)]}
        data.update(extra)
        tmp = self.path + ".part"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except OSError:
            pass                                             # the app is reading it right now: next time


def open_trainer(dataset):
    """Open the SDG app's Train page with this dataset selected (starts the app if it isn't running)."""
    app = os.path.join(sc.ROOT, "app", "server.py")
    if not os.path.isfile(app):
        return
    print(f"[post] opening the SDG app to train on {dataset}")
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    subprocess.Popen([sys.executable, app, "--open", "train", "--dataset", dataset], creationflags=flags)


def start_training(run, which):
    script = os.path.join(sc.ROOT, "train", "train_yolo.py")
    ds = run["dataset"]
    tasks = ["seg", "pose"] if which == "both" else [which]
    for task in tasks:
        if task == "pose" and not run["pose_classes"]:
            continue
        cmd = [sys.executable, script, "--task", task, "--data", os.path.join(ds, f"{task}.yaml"),
               "--name", f"{run['name']}_{task}"] + run["post"].get("train_args", "").split()
        print("[post] training:", " ".join(cmd))
        subprocess.run(cmd, check=False)


def check(run, run_json):
    """Convert up to 3 frames and print what the EXRs contain - use after the first short render."""
    _worker_init(run_json)
    prepare_dataset(run, fresh=False)
    todo = scan(run, set())[:3]
    if not todo:
        print("no EXR files found in", [s["exr_dir"] for s in run["stages"]])
        return
    for si, path in todo:
        img = exr.read(path)
        print(f"\n{os.path.basename(path)}  {img.width}x{img.height}  layers {img.layer_names()}")
        rgb, lab = split_layers(img)
        ids = np.rint(img.channels[f"{lab}.R" if lab else "R"].astype(np.float32)).astype(np.int32)
        dep = img.channels[f"{lab}.G" if lab else "G"].astype(np.float32)
        vals, counts = np.unique(ids, return_counts=True)
        known = _W["stages"][si][0]["instances"]
        print("  stencil ids (pixels):", ", ".join(f"{v}:{c}{'' if v == 0 or str(v) in known else '?'}" for v, c in zip(vals, counts)))
        print(f"  depth range {dep.min():.1f} .. {dep.max():.1f} cm")
        print("  camera:", camera_from_attrs(img.attrs))
        rgb8 = to_bgr8(img, rgb)
        print(f"  image mean BGR {rgb8.reshape(-1, 3).mean(0).round(1)}")
        run["output"]["keep_exr"] = True
        _W["run"]["output"]["keep_exr"] = True
        print("  ->", process(si, path, True))
    print(f"\npreviews: {run['dataset']}/previews   ('?' = stencil id not in the manifest)")


if __name__ == "__main__":
    main()
