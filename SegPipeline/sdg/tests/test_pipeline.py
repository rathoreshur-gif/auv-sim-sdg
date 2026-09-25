"""
Offline tests for the dataset side (no Unreal needed). Run with the SegPipeline venv:

    ..\\.venv\\Scripts\\python.exe tests\\test_pipeline.py
"""
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import zlib

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import exr  # noqa: E402
import post  # noqa: E402
import sdg_common as sc  # noqa: E402


# ---------------------------------------------------------------------------- test EXR writer
def write_exr(path, channels, attrs=None, compression=3):
    """Single-part scanline EXR with any channel names (half/float), NONE/ZIPS/ZIP."""
    names = sorted(channels)
    h, w = channels[names[0]].shape

    def attr(name, typ, data):
        return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data

    ptype = {np.dtype(np.float16): 1, np.dtype(np.float32): 2}
    chl = b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", ptype[channels[n].dtype], 0, 1, 1) for n in names) + b"\0"
    hdr = attr("channels", "chlist", chl) + attr("compression", "compression", bytes([compression]))
    hdr += attr("dataWindow", "box2i", struct.pack("<4i", 0, 0, w - 1, h - 1))
    hdr += attr("displayWindow", "box2i", struct.pack("<4i", 0, 0, w - 1, h - 1))
    hdr += attr("lineOrder", "lineOrder", b"\0") + attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    hdr += attr("screenWindowCenter", "v2f", struct.pack("<2f", 0, 0)) + attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    for k, v in (attrs or {}).items():
        hdr += attr(k, "string", str(v).encode())
    hdr += b"\0"
    lines = {0: 1, 2: 1, 3: 16}[compression]
    blocks = []
    for y0 in range(0, h, lines):
        n = min(lines, h - y0)
        raw = b"".join(channels[name][y].tobytes() for y in range(y0, y0 + n) for name in names)
        data = raw
        if compression in (2, 3):
            b = np.frombuffer(raw, np.uint8)
            t = np.concatenate([b[0::2], b[1::2]]).astype(np.int32)
            p = t.copy()
            p[1:] = (t[1:] - t[:-1] + 128 + 256) % 256
            comp = zlib.compress(p.astype(np.uint8).tobytes())
            data = comp if len(comp) < len(raw) else raw
        blocks.append(struct.pack("<ii", y0, len(data)) + data)
    head = struct.pack("<ii", 20000630, 2) + hdr
    pos = len(head) + 8 * len(blocks)
    offs = []
    for b in blocks:
        offs.append(pos)
        pos += len(b)
    with open(path, "wb") as f:
        f.write(head + struct.pack(f"<{len(offs)}Q", *offs) + b"".join(blocks))


# ---------------------------------------------------------------------------- tests
def test_exr_roundtrip_own_writer(tmp):
    rng = np.random.default_rng(0)
    for comp in (0, 2, 3):
        chans = {"R": rng.random((37, 53)).astype(np.float16), "G": rng.random((37, 53)).astype(np.float16),
                 "Lab.R": (rng.integers(0, 255, (37, 53))).astype(np.float16),
                 "Lab.G": (rng.random((37, 53)) * 5000).astype(np.float32)}
        p = os.path.join(tmp, f"own{comp}.exr")
        write_exr(p, chans, {"unreal/camera/curPos/x": "12.5"}, comp)
        img = exr.read(p)
        for k, v in chans.items():
            assert img.channels[k].dtype == v.dtype and np.array_equal(img.channels[k], v), (comp, k)
        assert img.attrs["unreal/camera/curPos/x"] == "12.5"
        assert img.layer_names() == ["", "Lab"]


def test_exr_vs_opencv(tmp):
    """Independent check: files written by OpenCV (real OpenEXR library) read back identically, and
    OpenCV reads our test files - so writer and reader aren't just agreeing with each other."""
    rng = np.random.default_rng(1)
    bgr = (rng.random((61, 90, 3)) * 4).astype(np.float32)
    if not cv2.haveImageWriter(".exr"):
        print("    skipped: this OpenCV build has no EXR codec (the Unreal test covers real files)")
        return
    for name, flag in (("zip", cv2.IMWRITE_EXR_COMPRESSION_ZIP), ("zips", cv2.IMWRITE_EXR_COMPRESSION_ZIPS),
                       ("none", cv2.IMWRITE_EXR_COMPRESSION_NO), ("rle", cv2.IMWRITE_EXR_COMPRESSION_RLE)):
        for half in (True, False):
            p = os.path.join(tmp, f"cv_{name}_{half}.exr")
            params = [cv2.IMWRITE_EXR_COMPRESSION, flag, cv2.IMWRITE_EXR_TYPE,
                      cv2.IMWRITE_EXR_TYPE_HALF if half else cv2.IMWRITE_EXR_TYPE_FLOAT]
            assert cv2.imwrite(p, bgr, params)
            ref = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            img = exr.read(p)
            mine = np.dstack([img.channels[c].astype(np.float32) for c in "BGR"])
            assert np.array_equal(mine, ref), (name, half, np.abs(mine - ref).max())
    chans = {c: (rng.random((33, 47)) * 2).astype(np.float16) for c in "BGR"}
    p = os.path.join(tmp, "own_for_cv.exr")
    write_exr(p, chans, compression=3)
    ref = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    assert ref is not None and np.array_equal(ref, np.dstack([chans[c].astype(np.float32) for c in "BGR"]))


def test_incomplete_exr(tmp):
    p = os.path.join(tmp, "cut.exr")
    write_exr(p, {"R": np.ones((64, 64), np.float16)}, compression=3)
    data = open(p, "rb").read()
    for cut in (4, 60, len(data) - 10):
        q = os.path.join(tmp, f"cut{cut}.exr")
        open(q, "wb").write(data[:cut])
        try:
            exr.read(q)
            raise AssertionError(f"no error for file cut at {cut}")
        except exr.ExrIncomplete:
            pass
    # offset table still zero = MRQ is still writing
    buf = bytearray(data)
    _, pos = exr._read_header(bytes(buf))
    buf[pos:pos + 8] = b"\0" * 8
    open(os.path.join(tmp, "zero.exr"), "wb").write(buf)
    try:
        exr.read(os.path.join(tmp, "zero.exr"))
        raise AssertionError("zero offset table not detected")
    except exr.ExrIncomplete:
        pass


def iou_after_fill(mask, poly):
    back = np.zeros_like(mask)
    cv2.fillPoly(back, [np.round(poly).astype(np.int32)], 1)
    inter = np.logical_and(back, mask).sum()
    return inter / max(np.logical_or(back, mask).sum(), 1)


def test_polygons():
    h, w = 200, 260
    cases = {}
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (130, 100), 70, 1, -1)
    cv2.circle(m, (130, 100), 30, 0, -1)                     # ring (hole)
    cases["ring"] = m
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (20, 30), (110, 170), 1, -1)
    cv2.rectangle(m, (150, 40), (240, 160), 1, -1)           # two pieces of one occluded object
    cases["split"] = m
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (20, 20), (240, 180), 1, -1)
    cv2.rectangle(m, (60, 60), (100, 100), 0, -1)
    cv2.rectangle(m, (150, 60), (200, 140), 0, -1)
    cv2.circle(m, (175, 100), 12, 1, -1)                      # island inside a hole
    cases["holes+island"] = m
    for name, mask in cases.items():
        poly = post.mask_polygon(mask, 1.0, 15.0)
        iou = iou_after_fill(mask, poly)
        assert iou > 0.95, (name, iou)
        print(f"    {name:<14} {len(poly):>4} points  IoU {iou:.3f}")


def test_math():
    f, r, u = sc.rotator_axes(0, 90, 0)
    assert np.allclose(f, (0, 1, 0)) and np.allclose(r, (-1, 0, 0)) and np.allclose(u, (0, 0, 1))
    f, r, u = sc.rotator_axes(90, 0, 0)
    assert np.allclose(f, (0, 0, 1))
    p, y, _ = sc.look_at((1, 1, math.sqrt(2)))
    assert abs(p - 45) < 1e-9 and abs(y - 45) < 1e-9
    loc, rot, scl = (10, 20, 30), (12, -40, 7), (1.5, 2, 0.5)
    q = sc.transform_point(loc, rot, scl, (3, -4, 5))
    assert np.allclose(sc.inverse_transform_point(loc, rot, scl, q), (3, -4, 5))
    uvd = sc.project((100, 0, 0), (0, 0, 0), (0, 0, 0), 90, 640, 480)
    assert np.allclose(uvd, (320, 240, 100))
    uvd = sc.project((100, 100, 0), (0, 0, 0), (0, 0, 0), 90, 640, 480)
    assert np.allclose(uvd[:2], (640, 240))                   # hfov 90: 45 deg right lands on the right edge
    assert sc.project((-5, 0, 0), (0, 0, 0), (0, 0, 0), 90, 640, 480) is None
    assert abs(sc.focal_mm_to_hfov(sc.hfov_to_focal_mm(73.0, 36.0), 36.0) - 73.0) < 1e-9
    assert sc.unwrap_degrees([170, -175, -160, 175]) == [170, 185, 200, 175]
    pts, flip = sc.preset_keypoints("face4", (-1, -50, -40), (1, 50, 40))    # thin in X -> +X face
    assert flip == [1, 0, 3, 2]
    assert np.allclose(pts, [(1, 50, 40), (1, -50, 40), (1, -50, -40), (1, 50, -40)])
    pts, _ = sc.preset_keypoints("top4", (-10, -20, 0), (10, 20, 30))
    assert all(p[2] == 30 for p in pts)
    lo, hi = (-50, -2.5, 0), (50, 2.5, 80)                    # board thin along local Y, yawed 30 deg
    t = [100, 200, -150, 0, 30, 0, 1, 1, 1]
    assert sc.facing_axis("auto", lo, hi) == (0.0, 1.0, 0.0) and sc.facing_axis("-x", lo, hi) == (-1.0, 0.0, 0.0)
    assert abs(sc.facing_yaw(t, (0, 0, 40), sc.facing_axis("auto", lo, hi)) - 120) < 1e-9
    assert abs(sc.facing_yaw(t, (0, 0, 40), sc.facing_axis("-auto", lo, hi)) + 60) < 1e-9
    assert sc.facing_yaw(t, (0, 0, 40), sc.facing_axis("+z", lo, hi)) is None
    board = ([0, -1, 0], [100, 1, 80])                        # actors duplicated in place -> one object
    assert sc.group_stacked([board, board, ([0.5, -1, 0], [100.5, 1, 80]), board]) == [[0, 1, 2, 3]]
    assert sc.group_stacked([board, ([300, -1, 0], [400, 1, 80]), ([50, -1, 0], [150, 1, 80])]) == [[0], [1], [2]]
    s1, s2 = sc.Sampler("a:b"), sc.Sampler("a:b")
    assert [s1.value([0, 1]) for _ in range(5)] == [s2.value([0, 1]) for _ in range(5)]
    vals = sc.Sampler(3).series([0, 10], 9, every=4, smooth=True)
    assert vals[0] != vals[4] and abs(vals[2] - (vals[0] + vals[4]) / 2) < 1e-9


def test_config():
    cfg = sc.load_config("pool")
    assert cfg["name"] == "pool_v1" and len(cfg["stages"]) == 3
    names = [r["name"] for r in cfg["randomize"]]
    assert names[:2] == [sc.FOG_DENSITY, sc.FOG_COLOR]
    assert cfg["output"]["raw_dir"].endswith("MyProject/Saved/SDG/raw"), cfg["output"]["raw_dir"]
    rz = sc.stage_randomizers(cfg, cfg["stages"][2])
    assert next(r for r in rz if r["name"] == "sun_intensity")["value"] == [0.05, 0.3]
    assert sc.stage_camera(cfg, cfg["stages"][2])["distance"] == [300, 1100]
    assert sc.stage_camera(cfg, cfg["stages"][1])["hold"] == 2
    assert cfg["seg_classes"][0] == "ambulance" and cfg["pose_classes"] == ["torpedo_board", "bin"]
    tc = sc.load_config("torpedo")
    assert tc["seg_classes"] == ["ambulance", "blood", "fire", "firetruck"] and tc["pose_classes"] == []
    assert all(sc.stage_camera(tc, st)["azimuth"] == [-50, 50] for st in tc["stages"])
    bad = os.path.join(HERE, "_bad.toml")
    with open(bad, "w") as f:
        f.write('name = "x"\nmap = "/Game/m"\n[post]\nworkers = 2\nseed = 3\n[[stages]]\nname = "a"\nframes = 1\n')
    try:
        sc.load_config(bad)
        raise AssertionError("misplaced key not caught")
    except sc.ConfigError as e:
        assert "before the first [table]" in str(e), e
    finally:
        os.remove(bad)


def test_zip(tmp):
    import zipfile
    ds = os.path.join(tmp, "zipds", "demo_v1")
    for split in ("train", "val"):
        os.makedirs(os.path.join(ds, "images", split))
        os.makedirs(os.path.join(ds, "labels", split))
        cv2.imwrite(os.path.join(ds, "images", split, "a_000000.jpg"), np.zeros((8, 8, 3), np.uint8))
        np.save(os.path.join(ds, "images", split, "a_000000.npy"), np.zeros(1))      # training cache: left out
        with open(os.path.join(ds, "labels", split, "a_000000.txt"), "w") as f:
            f.write("0 0.1 0.1 0.5 0.1 0.5 0.5\n")
    out = post.zip_dataset({"dataset": ds, "name": "demo_v1", "seg_classes": ["blood", "fire"]})
    z = zipfile.ZipFile(out)
    assert sorted(z.namelist()) == ["demo_v1/data.yaml", "demo_v1/images/train/a_000000.jpg", "demo_v1/images/val/a_000000.jpg",
                                    "demo_v1/labels/train/a_000000.txt", "demo_v1/labels/val/a_000000.txt"], z.namelist()
    y = z.read("demo_v1/data.yaml").decode()
    assert "path:" not in y and "  1: fire" in y


# ---------------------------------------------------------------------------- synthetic frame
def render_boxes(w, h, cam_loc, cam_rot, hfov, boxes):
    """Ray-cast axis-aligned boxes: returns (stencil ids of the visible surface, z-depth)."""
    f, r, u = (np.array(a) for a in sc.rotator_axes(*cam_rot))
    fp = sc.focal_px(hfov, w)
    xs, ys = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dirs = f[None, None] + ((xs - w / 2) / fp)[..., None] * r[None, None] - ((ys - h / 2) / fp)[..., None] * u[None, None]
    best = np.full((h, w), np.inf)
    ids = np.zeros((h, w), np.float32)
    o = np.array(cam_loc, float)
    for sid, lo, hi in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (np.array(lo) - o) / dirs
            t2 = (np.array(hi) - o) / dirs
        tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < best)
        best[hit] = tmin[hit]
        ids[hit] = sid
    depth = np.where(np.isfinite(best), best, 60000.0)       # t along dirs (forward component 1) = z-depth
    return ids, depth.astype(np.float32)


def test_post_end_to_end(tmp):
    w, h, hfov = 320, 240, 80.0
    cam_loc, target = (-420.0, -130.0, 90.0), (0.0, 0.0, 0.0)
    p, y, _ = sc.look_at(sc.sub(target, cam_loc))
    cam_rot = (p, y, 4.0)
    box = (1, (-50, -40, -30), (50, 40, 30))
    wall = (0, (-180, -30, -200), (-170, 200, 200))          # unlabelled occluder in front of part of the box
    ids, depth = render_boxes(w, h, cam_loc, cam_rot, hfov, [box, wall])
    rgb = np.dstack([np.linspace(0, 1, w, dtype=np.float32)[None].repeat(h, 0)] * 3).astype(np.float16)

    raw = os.path.join(tmp, "raw", "t")
    os.makedirs(os.path.join(raw, "s1"))
    attrs = {"unreal/camera/curPos/x": cam_loc[0], "unreal/camera/curPos/y": cam_loc[1], "unreal/camera/curPos/z": cam_loc[2],
             "unreal/camera/curRot/pitch": cam_rot[0], "unreal/camera/curRot/yaw": cam_rot[1],
             "unreal/camera/curRot/roll": cam_rot[2], "unreal/SDG_Camera/FinalImage/fov": hfov}
    chans = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2], "A": np.ones((h, w), np.float16),
             "FinalImageSDGLabels.R": ids.astype(np.float16), "FinalImageSDGLabels.G": depth.astype(np.float16),
             "FinalImageSDGLabels.B": np.zeros((h, w), np.float16)}
    write_exr(os.path.join(raw, "s1", "SDG_t_s1.000000.exr"), chans, attrs)

    kps, flip = sc.preset_keypoints("bounds8", box[1], box[2])
    man = {"stage": "s1", "frames": [{"seq": 0, "cam": [0] * 7, "params": {}}],
           "instances": {"1": {"object": "box#0", "part": "cube", "seg_class": 0}},
           "objects": {"box#0": {"pose_class": 0, "stencils": [1], "keypoints": kps, "flip_idx": flip,
                                 "transform": [0, 0, 0, 0, 0, 0, 1, 1, 1]}},
           "num_keypoints": 8, "flip_idx": flip}
    with open(os.path.join(raw, "s1.manifest.json"), "w") as f:
        json.dump(man, f)
    ds = os.path.join(tmp, "ds")
    out = dict(sc.DEFAULTS["output"], keep_exr=True, empty_keep=1.0, val_fraction=0.0)
    run = {"name": "t", "dataset": ds, "raw": raw, "output": out, "post": {}, "water_fog": {"enabled": False},
           "seg_classes": ["cube"], "pose_classes": ["cube"],
           "stages": [{"name": "s1", "manifest": os.path.join(raw, "s1.manifest.json"),
                       "exr_dir": os.path.join(raw, "s1"), "frames": 1, "hold": 1}]}
    with open(os.path.join(raw, "run.json"), "w") as f:
        json.dump(run, f)
    post.prepare_dataset(run, fresh=True)
    post._worker_init(os.path.join(raw, "run.json"))
    st = post.process(0, os.path.join(raw, "s1", "SDG_t_s1.000000.exr"), True)
    assert st["seg"] == {0: 1} and st["pose"] == {0: 1}, st

    seg = open(os.path.join(ds, "labels", "train", "s1_000000.txt")).read().split()
    poly = np.array(seg[1:], float).reshape(-1, 2) * [w, h] - 0.5
    iou = iou_after_fill((ids == 1).astype(np.uint8), poly)
    assert iou > 0.95, iou
    pose = open(os.path.join(ds, "pose", "labels", "train", "s1_000000.txt")).read().split()
    vals = np.array(pose[5:], float).reshape(-1, 3)
    for j, kp in enumerate(kps):
        uvd = sc.project(kp, cam_loc, cam_rot, hfov, w, h)
        in_img = 0 <= uvd[0] < w and 0 <= uvd[1] < h
        if not in_img:
            assert vals[j, 2] == 0
            continue
        assert np.allclose(vals[j, :2] * [w, h], uvd[:2], atol=1e-3), (j, vals[j], uvd)
        # ground truth visibility from the ray-cast depth: is anything nearer along that pixel?
        scene = depth[int(uvd[1]), int(uvd[0])]
        expect = 2 if uvd[2] <= scene * 1.01 + 3 else 1
        assert vals[j, 2] == expect, (j, vals[j, 2], expect)
    vis = [int(v) for v in vals[:, 2]]
    x0, y0 = np.nonzero((ids == 1).any(0))[0][0], np.nonzero((ids == 1).any(1))[0][0]
    box_l = np.array(pose[1:5], float) * [w, h, w, h]
    assert abs((box_l[0] - box_l[2] / 2) - x0) < 1.01 and abs((box_l[1] - box_l[3] / 2) - y0) < 1.01
    assert os.path.exists(os.path.join(ds, "previews", "s1_000000.jpg"))

    # the whole builder loop, as the editor starts it: progress.json for the SDG app, zip, open-trainer hook
    run["stages"][0]["seq_frames"] = 1
    run["post"] = {"workers": 1, "train": "", "open_trainer": True}
    with open(os.path.join(raw, "run.json"), "w") as f:
        json.dump(run, f)
    opened, real_open = [], post.open_trainer
    post.open_trainer = opened.append
    argv = sys.argv
    sys.argv = ["post.py", raw]
    try:
        post.main()
    finally:
        sys.argv, post.open_trainer = argv, real_open
    with open(os.path.join(raw, "progress.json")) as f:
        prog = json.load(f)
    assert prog["state"] == "done" and prog["stages"][0]["rendered"] == 1 and prog["counts"]["frames"] == 1, prog
    assert prog["zip"] and os.path.isfile(prog["zip"]) and opened == [prog["zip"]], (prog, opened)
    print(f"    seg IoU {iou:.3f}, keypoint visibility {vis}")


def main():
    tmp = tempfile.mkdtemp(prefix="sdg_test_")
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            if fn.__code__.co_argcount:
                fn(tmp)
            else:
                fn()
            print(f"ok    {name}")
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {name}: {e}")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
