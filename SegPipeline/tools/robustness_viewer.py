"""
Robustness viewer: load a YOLO model (.pt) and an image, distort the image with sliders (light, colour,
haze, blur, noise, compression, occlusion, crop, rotation, size ...) and watch the model's predictions
update live, next to its predictions on the untouched image.

    <python with ultralytics> tools/robustness_viewer.py [--model best.pt] [--image frame.jpg]
                                                          [--port 8766] [--no-browser]

Or double-click "SegPipeline/Robustness Viewer.bat". Serves http://127.0.0.1:8766 to this PC only.
Drop a .pt and an image onto the page (or use its file buttons). Every slider change re-runs the model on
the distorted image; "Animate" re-rolls the random effects (noise, particles, random blocks) continuously.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import os
import re
import socket
import tempfile
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "robustness_viewer.html")
UPLOADS = os.path.join(tempfile.gettempdir(), "robustness_viewer")
PORT = 8766


# ============================================================================ controls
# One list drives both the page (sliders are built from it) and the defaults here.
# "rand" = range the page's Randomize button draws from (moderate, realistic distortions).
def S(key, label, lo, hi, step, default, rand=None, doc=""):
    return {"key": key, "type": "range", "label": label, "min": lo, "max": hi, "step": step,
            "default": default, "rand": rand, "doc": doc}


def B(key, label, default=False, doc=""):
    return {"key": key, "type": "bool", "label": label, "default": default, "doc": doc}


def C(key, label, default, doc=""):
    return {"key": key, "type": "color", "label": label, "default": default, "doc": doc}


def E(key, label, options, default, doc=""):
    return {"key": key, "type": "select", "label": label, "options": options, "default": default, "doc": doc}


CONTROLS = [
    {"group": "Model", "model": True, "items": [
        S("conf", "Confidence threshold", 0.01, 0.95, 0.01, 0.25),
        S("iou", "NMS IoU", 0.1, 0.95, 0.05, 0.7),
        S("imgsz", "Inference size (imgsz)", 320, 1280, 32, 640),
        B("draw", "Draw predictions", True),
        B("show_masks", "Masks", True),
        B("show_boxes", "Boxes", True),
        B("show_labels", "Labels + confidence", True),
        B("retina_masks", "Full-resolution masks", False, "sharper mask edges, slower"),
    ]},
    {"group": "Light", "items": [
        S("brightness", "Brightness", -150, 150, 1, 0, [-60, 60]),
        S("contrast", "Contrast", 0.1, 3, 0.05, 1, [0.6, 1.5]),
        S("gamma", "Gamma (>1 darker)", 0.2, 4, 0.05, 1, [0.6, 1.8]),
        S("vignette", "Vignette", 0, 1, 0.05, 0, [0, 0.6]),
    ]},
    {"group": "Colour", "items": [
        S("hue", "Hue shift (deg)", -180, 180, 1, 0, [-25, 25]),
        S("saturation", "Saturation", 0, 3, 0.05, 1, [0.4, 1.6]),
        S("r_gain", "Red gain", 0, 2, 0.02, 1, [0.6, 1.3]),
        S("g_gain", "Green gain", 0, 2, 0.02, 1, [0.8, 1.3]),
        S("b_gain", "Blue gain", 0, 2, 0.02, 1, [0.8, 1.3]),
        S("red_loss", "Underwater red loss", 0, 1, 0.02, 0, [0, 0.7], "red fades first with depth, then green"),
        B("grayscale", "Grayscale"),
        B("swap_rb", "Swap red/blue (BGR bug)"),
    ]},
    {"group": "Water", "items": [
        S("haze", "Haze", 0, 1, 0.01, 0, [0, 0.55]),
        C("haze_color", "Haze colour", "#1e5c5a"),
        S("particles", "Particles (marine snow)", 0, 800, 5, 0, [0, 250]),
        S("particle_size", "Particle size (px)", 0.5, 10, 0.5, 2),
        S("caustics", "Caustic light", 0, 1, 0.05, 0, [0, 0.5], "bright moving ripple pattern"),
        S("caustics_scale", "Caustic size (px)", 10, 300, 5, 80),
    ]},
    {"group": "Blur & sharpness", "items": [
        S("blur", "Gaussian blur (sigma px)", 0, 20, 0.25, 0, [0, 2.5]),
        S("motion", "Motion blur (px)", 0, 80, 1, 0, [0, 15]),
        S("motion_angle", "Motion angle (deg)", 0, 180, 1, 0, [0, 180]),
        S("defocus", "Defocus radius (px)", 0, 25, 0.5, 0, [0, 4]),
        S("sharpen", "Sharpen", 0, 3, 0.05, 0, [0, 0.8]),
    ]},
    {"group": "Noise & compression", "items": [
        S("noise", "Sensor noise (std)", 0, 80, 0.5, 0, [0, 18]),
        S("salt_pepper", "Salt & pepper", 0, 0.2, 0.002, 0, [0, 0.01]),
        S("pixelate", "Low resolution (factor)", 1, 16, 0.25, 1, [1, 3], "shrink, then blow back up"),
        S("jpeg", "JPEG quality", 1, 100, 1, 100, [20, 95], "100 = no compression"),
    ]},
    {"group": "Occlusion", "items": [
        S("occ_count", "Random blocks", 0, 12, 1, 0, [0, 3]),
        S("occ_size", "Block size (% of short side)", 2, 70, 1, 15, [5, 25]),
        E("occ_style", "Block fill", ["black", "gray", "noise", "colour", "blur"], "black"),
        B("occ_on_objects", "Random blocks on detections", False,
          "drop the random blocks on the clean image's detections instead of anywhere"),
    ]},
    {"group": "Crop & geometry", "items": [
        S("crop_left", "Crop left (%)", 0, 45, 1, 0, [0, 15]),
        S("crop_right", "Crop right (%)", 0, 45, 1, 0, [0, 15]),
        S("crop_top", "Crop top (%)", 0, 45, 1, 0, [0, 15]),
        S("crop_bottom", "Crop bottom (%)", 0, 45, 1, 0, [0, 15]),
        S("zoom", "Zoom", 1, 5, 0.05, 1, [1, 1.6]),
        S("pan_x", "Zoom pan x", -1, 1, 0.02, 0, [-1, 1]),
        S("pan_y", "Zoom pan y", -1, 1, 0.02, 0, [-1, 1]),
        S("rotate", "Rotate (deg)", -180, 180, 1, 0, [-15, 15]),
        S("tilt_x", "Perspective tilt (top/bottom)", -0.8, 0.8, 0.02, 0, [-0.25, 0.25]),
        S("tilt_y", "Perspective tilt (left/right)", -0.8, 0.8, 0.02, 0, [-0.25, 0.25]),
        B("flip_h", "Mirror left-right"),
        B("flip_v", "Flip upside down"),
    ]},
    {"group": "Size", "items": [
        S("scale", "Image scale", 0.05, 2, 0.01, 1, [0.4, 1.2], "resolution the model gets"),
        S("aspect", "Stretch width", 0.3, 3, 0.01, 1, [0.8, 1.25]),
        B("square", "Stretch to square", False, "like a Roboflow 640x640 'stretch' export"),
    ]},
]
DEFAULTS = {c["key"]: c["default"] for g in CONTROLS for c in g["items"]}
DEFAULTS.update(seed=0, occ_points=[])


# ============================================================================ distortions
def _rng(p, k):
    """Independent random stream per effect: re-rolling noise doesn't move the blocks."""
    return np.random.default_rng([int(p.get("seed", 0)) & 0xFFFFFFFF, k])


def _resize(img, w, h):
    w, h = max(8, int(round(w))), max(8, int(round(h)))
    if (w, h) == (img.shape[1], img.shape[0]):
        return img
    shrink = w * h < img.shape[0] * img.shape[1]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


def geometry_matrix(p, w, h):
    """Crop / zoom / flip / rotate / tilt / resize as one 3x3 map from source to output pixels,
    plus the output size. Used for the image and to carry clean detections into the distorted frame."""
    M = np.eye(3)

    def then(T):
        nonlocal M
        M = T @ M

    l, r = w * p["crop_left"] / 100, w * (1 - p["crop_right"] / 100)
    t, b = h * p["crop_top"] / 100, h * (1 - p["crop_bottom"] / 100)
    if r - l >= 8 and b - t >= 8:
        then(np.array([[1, 0, -l], [0, 1, -t], [0, 0, 1.0]]))
        w, h = r - l, b - t
    z = float(p["zoom"])
    if z > 1:
        cw, ch = w / z, h / z
        cx, cy = (w - cw) / 2 * (1 + p["pan_x"]), (h - ch) / 2 * (1 + p["pan_y"])
        then(np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]))
        w, h = cw, ch
    if p["flip_h"]:
        then(np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1.0]]))
    if p["flip_v"]:
        then(np.array([[1, 0, 0], [0, -1, h], [0, 0, 1.0]]))
    if p["rotate"]:
        then(np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), float(p["rotate"]), 1.0), [0, 0, 1]]))
    tx, ty = float(p["tilt_x"]), float(p["tilt_y"])
    if tx or ty:
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        dx, dy = w * abs(tx) / 2, h * abs(ty) / 2
        if tx > 0:                                   # top edge narrower: looking up at the scene
            dst[0, 0] += dx
            dst[1, 0] -= dx
        elif tx < 0:
            dst[3, 0] += dx
            dst[2, 0] -= dx
        if ty > 0:                                   # left edge shorter
            dst[0, 1] += dy
            dst[3, 1] -= dy
        elif ty < 0:
            dst[1, 1] += dy
            dst[2, 1] -= dy
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        then(cv2.getPerspectiveTransform(src, dst).astype(np.float64))
    nw, nh = w * p["scale"] * p["aspect"], h * p["scale"]
    if p["square"]:
        nw = nh = math.sqrt(nw * nh)
    nw, nh = max(8, round(nw)), max(8, round(nh))
    then(np.array([[nw / w, 0, 0], [0, nh / h, 0], [0, 0, 1.0]]))
    return M, (int(nw), int(nh))


def geometry(img, p):
    h, w = img.shape[:2]
    M, (nw, nh) = geometry_matrix(p, w, h)
    if np.allclose(M, np.eye(3)):
        return img
    affine = np.allclose(M[2], [0, 0, 1])
    # downscaling with warp* aliases: pre-shrink with INTER_AREA when the result is much smaller
    s = math.sqrt(abs(np.linalg.det(M[:2, :2])))
    if s < 0.7:
        small = _resize(img, w * s, h * s)
        pre = np.diag([small.shape[1] / w, small.shape[0] / h, 1.0])
        img, M = small, M @ np.linalg.inv(pre)
    if affine:
        return cv2.warpAffine(img, M[:2], (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return cv2.warpPerspective(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def _fill(img, x0, y0, x1, y1, style, rng):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return
    patch = img[y0:y1, x0:x1]
    if style == "gray":
        patch[:] = 128
    elif style == "noise":
        patch[:] = rng.integers(0, 256, patch.shape, dtype=np.uint8)
    elif style == "colour":
        patch[:] = rng.integers(0, 256, 3, dtype=np.uint8)
    elif style == "blur":
        k = max(3, min(patch.shape[:2]) / 4)
        patch[:] = cv2.GaussianBlur(patch.copy(), (0, 0), k)
    else:
        patch[:] = 0


def occlude(img, p, targets):
    """targets: clean detection boxes in this frame (for 'blocks on detections')."""
    n, pts = int(p["occ_count"]), p.get("occ_points") or []
    if not n and not pts:
        return img
    img = img.copy()
    h, w = img.shape[:2]
    rng = _rng(p, 1)
    side = p["occ_size"] / 100 * min(h, w)
    for i in range(n):
        a = rng.uniform(0.5, 2.0)
        bw, bh = side * math.sqrt(a), side / math.sqrt(a)
        if p["occ_on_objects"] and len(targets):
            x0, y0, x1, y1 = targets[i % len(targets)]
            cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        else:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        _fill(img, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, p["occ_style"], rng)
    for x, y in pts:                                # placed by clicking the distorted image
        _fill(img, x * w - side / 2, y * h - side / 2, x * w + side / 2, y * h + side / 2, p["occ_style"], rng)
    return img


def _hex_bgr(s):
    s = str(s).lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s):
        return (90, 92, 30)
    return tuple(int(s[i:i + 2], 16) for i in (4, 2, 0))


def colour(img, p):
    x = img.astype(np.float32)
    if p["swap_rb"]:
        x = x[..., ::-1]
    if p["contrast"] != 1 or p["brightness"]:
        x = (x - 128) * p["contrast"] + 128 + p["brightness"]
    if p["gamma"] != 1:
        x = 255 * (np.clip(x, 0, 255) / 255) ** p["gamma"]
    if p["r_gain"] != 1 or p["g_gain"] != 1 or p["b_gain"] != 1:
        x = x * np.array([p["b_gain"], p["g_gain"], p["r_gain"]], np.float32)
    if p["red_loss"]:
        x = x * np.array([1, 1 - 0.35 * p["red_loss"], 1 - p["red_loss"]], np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if p["hue"] or p["saturation"] != 1:
        hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + p["hue"] / 2) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * p["saturation"], 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if p["grayscale"]:
        x = cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if p["vignette"]:
        h, w = x.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        x = (x * np.clip(1 - p["vignette"] * r2 / 2, 0, 1)[..., None].astype(np.float32)).astype(np.uint8)
    return x


def water(img, p):
    if not (p["haze"] or p["particles"] or p["caustics"]):
        return img
    x = img.astype(np.float32)
    h, w = x.shape[:2]
    if p["caustics"]:
        rng = _rng(p, 2)
        s = max(4.0, float(p["caustics_scale"]))
        gw, gh = max(2, int(w / s) + 2), max(2, int(h / s) + 2)
        field = cv2.resize(rng.random((gh, gw), dtype=np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        ridges = np.clip(1 - np.abs(field - 0.5) * 8, 0, 1) ** 2      # thin bright lines where the field crosses 0.5
        x = x * (1 + p["caustics"] * 1.5 * ridges[..., None])
    if p["haze"]:
        x = x * (1 - p["haze"]) + np.array(_hex_bgr(p["haze_color"]), np.float32) * p["haze"]
    n = int(p["particles"])
    if n:
        rng = _rng(p, 3)
        layer = np.zeros((h, w), np.float32)
        size = float(p["particle_size"])
        for cx, cy, r, v in zip(rng.uniform(0, w, n), rng.uniform(0, h, n), rng.uniform(0.4, 1.0, n) * size,
                                rng.uniform(90, 255, n)):
            cv2.circle(layer, (int(cx), int(cy)), max(1, int(round(r))), float(v), -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), max(0.5, size / 3))[..., None]
        x = 255 - (255 - np.clip(x, 0, 255)) * (255 - layer) / 255        # screen blend: specks glow
    return np.clip(x, 0, 255).astype(np.uint8)


def blur(img, p):
    if p["blur"] > 0:
        img = cv2.GaussianBlur(img, (0, 0), float(p["blur"]))
    L = int(p["motion"])
    if L >= 2:
        k = np.zeros((L, L), np.float32)
        a = math.radians(p["motion_angle"])
        c, dx, dy = (L - 1) / 2, math.cos(a) * (L - 1) / 2, math.sin(a) * (L - 1) / 2
        cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
        img = cv2.filter2D(img, -1, k / max(k.sum(), 1))
    r = float(p["defocus"])
    if r >= 0.5:
        n = int(math.ceil(r)) * 2 + 1
        k = np.zeros((n, n), np.float32)
        cv2.circle(k, (n // 2, n // 2), max(1, int(round(r))), 1.0, -1, cv2.LINE_AA)
        img = cv2.filter2D(img, -1, k / k.sum())
    if p["sharpen"] > 0:
        soft = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 1 + p["sharpen"], soft, -p["sharpen"], 0)
    return img


def noise(img, p):
    if p["noise"] > 0:
        rng = _rng(p, 4)
        n = np.empty(img.shape, np.float32)
        cv2.setRNGSeed(int(rng.integers(1, 2 ** 31)))                 # cv2.randn: ~10x faster than numpy's
        cv2.randn(n, 0.0, float(p["noise"]))
        img = cv2.add(img.astype(np.float32), n)
        img = np.clip(img, 0, 255).astype(np.uint8)
    if p["salt_pepper"] > 0:
        rng = _rng(p, 5)
        img = img.copy()
        m = rng.random(img.shape[:2])
        img[m < p["salt_pepper"] / 2] = 0
        img[m > 1 - p["salt_pepper"] / 2] = 255
    return img


def resolution(img, p):
    f = float(p["pixelate"])
    if f > 1:
        h, w = img.shape[:2]
        img = cv2.resize(_resize(img, w / f, h / f), (w, h), interpolation=cv2.INTER_LINEAR)
    q = int(p["jpeg"])
    if q < 100:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def distort(img, p, clean_boxes=()):
    """The full chain, in the order a camera would add them. clean_boxes: xyxy in the source image."""
    h, w = img.shape[:2]
    out = geometry(img, p)
    targets = []
    if p["occ_on_objects"] and len(clean_boxes):
        M, _ = geometry_matrix(p, w, h)
        for x0, y0, x1, y1 in clean_boxes:
            pts = cv2.perspectiveTransform(np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float64), M)[0]
            targets.append((*pts.min(0), *pts.max(0)))
    out = occlude(out, p, targets)
    out = colour(out, p)
    out = water(out, p)
    out = blur(out, p)
    out = noise(out, p)
    return resolution(out, p)


# ============================================================================ model
class Viewer:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.model_path = ""
        self.device = None
        self.image = None
        self.image_name = ""
        self.version = 0                  # bumps on every new model / image: invalidates the clean result
        self.clean = None                 # (key, response part, boxes)
        self.last = None                  # last distorted image (unannotated), for download

    def load_model(self, path):
        import torch
        from ultralytics import YOLO
        model = YOLO(path)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        model.predict(np.zeros((64, 64, 3), np.uint8), device=self.device, verbose=False)   # warm-up
        self.model, self.model_path = model, path
        self.version += 1

    def load_image(self, data, name):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"{name}: not an image OpenCV can read")
        self.image, self.image_name = img, name
        self.version += 1

    def state(self):
        names = getattr(self.model, "names", None) or {}
        return {"model": os.path.basename(self.model_path) if self.model else "", "model_path": self.model_path,
                "task": getattr(self.model, "task", "") if self.model else "",
                "names": [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names),
                "device": ("cuda:0" if self.device == 0 else str(self.device)) if self.device is not None else "",
                "image": self.image_name,
                "size": [int(self.image.shape[1]), int(self.image.shape[0])] if self.image is not None else None,
                "controls": CONTROLS, "defaults": DEFAULTS}

    def _predict(self, img, p):
        t = time.perf_counter()
        r = self.model.predict(img, conf=float(p["conf"]), iou=float(p["iou"]), imgsz=int(p["imgsz"]),
                               retina_masks=bool(p["retina_masks"]), device=self.device, verbose=False)[0]
        ms = (time.perf_counter() - t) * 1000
        dets, boxes = [], []
        if r.boxes is not None:
            for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                dets.append({"cls": int(c), "name": r.names.get(int(c), str(int(c))), "conf": round(s, 4)})
                boxes.append(b)
        if p["draw"]:
            shown = r.plot(boxes=bool(p["show_boxes"]), masks=bool(p["show_masks"]),
                           labels=bool(p["show_labels"]), conf=bool(p["show_labels"]))
        else:
            shown = img
        return {"image": _jpeg_b64(shown), "dets": dets, "ms": round(ms, 1),
                "size": [int(img.shape[1]), int(img.shape[0])]}, boxes

    def infer(self, params, have_clean):
        p = dict(DEFAULTS)
        p.update({k: v for k, v in (params or {}).items() if k in DEFAULTS})
        with self.lock:
            if self.model is None or self.image is None:
                raise ValueError("load a model (.pt) and an image first")
            key = json.dumps([self.version] + [p[k] for k in ("conf", "iou", "imgsz", "retina_masks", "draw",
                                                               "show_masks", "show_boxes", "show_labels")])
            if self.clean is None or self.clean[0] != key:
                part, boxes = self._predict(self.image, p)
                self.clean = (key, part, boxes)
            t = time.perf_counter()
            img = distort(self.image, p, self.clean[2])
            distort_ms = (time.perf_counter() - t) * 1000
            self.last = img
            out, _ = self._predict(img, p)
            out["distort_ms"] = round(distort_ms, 1)
            res = {"distorted": out, "clean_key": key}
            if have_clean != key:
                res["clean"] = self.clean[1]
            return res


def _jpeg_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name or "upload")) or "upload"


# ============================================================================ http
V = Viewer()
# Every model call runs on this one thread. ThreadingHTTPServer starts a new thread per request, and
# PyTorch/CUDA per-thread setup made each prediction ~70 ms there instead of ~10 ms.
WORKER = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")


def on_worker(fn, *args):
    return WORKER.submit(fn, *args).result()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        # headers and body go out as separate writes: without this, Nagle + delayed ACK add ~100-200 ms
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, fmt, *args):            # quiet console: only errors are printed
        pass

    def _send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._send(200, V.state())
            if path == "/api/distorted.png":
                if V.last is None:
                    return self._send(404, {"error": "nothing distorted yet"})
                ok, buf = cv2.imencode(".png", V.last)
                stem = os.path.splitext(V.image_name or "image")[0]
                return self._send(200, buf.tobytes(), "image/png",
                                  {"Content-Disposition": f'attachment; filename="{stem}_distorted.png"'})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/model":
                name = _safe_name(self.headers.get("X-Filename"))
                if not name.lower().endswith(".pt"):
                    raise ValueError("choose a .pt model file")
                os.makedirs(UPLOADS, exist_ok=True)
                dst = os.path.join(UPLOADS, name)
                with open(dst, "wb") as f:
                    f.write(self._body())
                on_worker(V.load_model, dst)          # the worker serializes it with inference: no lock here
                return self._send(200, V.state())
            if path == "/api/image":
                data = self._body()
                with V.lock:
                    V.load_image(data, os.path.basename(self.headers.get("X-Filename") or "pasted image"))
                return self._send(200, V.state())
            if path == "/api/infer":
                req = json.loads(self._body() or b"{}")
                return self._send(200, on_worker(V.infer, req.get("params"), req.get("clean_key")))
            return self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="YOLO .pt to load at start")
    ap.add_argument("--image", help="image to load at start")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if a.model:
        print(f"loading {a.model} ...")
        on_worker(V.load_model, os.path.abspath(a.model))
    if a.image:
        with open(a.image, "rb") as f:
            V.load_image(f.read(), os.path.basename(a.image))
    for port in range(a.port, a.port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {a.port}-{a.port + 19}")
    url = f"http://127.0.0.1:{port}/"
    print(f"Robustness viewer: {url}   (close this window to stop it)")
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
