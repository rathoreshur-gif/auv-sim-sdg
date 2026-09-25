"""
SDG App - a local app for the synthetic data pipeline.

  Recipe  create and edit dataset configs (sdg/configs/*.toml) with forms instead of a text editor
  Render  run setup, a quick test and the full render in the open Unreal Editor; watch progress,
          previews and the log
  Train   pick a dataset, model and augmentation, start training (yolo-training repo) and follow runs

Start it with "SDG App.bat" (or: <venv python> app/server.py). It serves http://127.0.0.1:8765
to this PC only and opens it in an app window. Unreal commands go through sdg/ue_listener.py, which
runs inside the editor and reads a file queue in sdg/.bridge/.

    server.py [--port 8765] [--no-browser] [--open train --dataset <path>]
"""
from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
import traceback
import urllib.request
import uuid
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)                            # SegPipeline/
SDG_DIR = os.path.join(ROOT, "sdg")
PROJECT_DIR = os.path.dirname(ROOT)
sys.path[:0] = [SDG_DIR, APP_DIR]
import sdg_common as sc                                    # noqa: E402
import tomlw                                               # noqa: E402

PORT = 8765
CONFIG_DIR = os.environ.get("SDG_APP_CONFIGS", os.path.join(SDG_DIR, "configs"))
BRIDGE = os.environ.get("SDG_BRIDGE_DIR", os.path.join(SDG_DIR, ".bridge"))
STATE_DIR = os.environ.get("SDG_APP_STATE", os.path.join(APP_DIR, "state"))
STATIC = os.path.join(APP_DIR, "static")
LISTENER = os.path.join(SDG_DIR, "ue_listener.py").replace("\\", "/")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


# ============================================================================ settings
def _editor_log():
    logs = os.path.join(PROJECT_DIR, "Saved", "Logs")
    for f in os.listdir(PROJECT_DIR) if os.path.isdir(PROJECT_DIR) else []:
        if f.endswith(".uproject"):
            return os.path.join(logs, f[:-len(".uproject")] + ".log").replace("\\", "/")
    return ""


def default_settings():
    home = os.path.expanduser("~")
    repo = os.path.join(home, "Downloads", "yolo-training")
    py = os.path.join(repo, ".venv", "Scripts", "python.exe")
    real = os.path.join(home, "Downloads", "Torpedo_RS.v1i.yolov11", "data.yaml")
    return {
        "trainer_repo": repo.replace("\\", "/") if os.path.isdir(repo) else "",
        "trainer_python": (py if os.path.isfile(py) else sys.executable).replace("\\", "/"),
        "real_data": real.replace("\\", "/") if os.path.isfile(real) else "",
        "editor_log": _editor_log(),
        "recent_datasets": [],
    }


class Settings:
    def __init__(self):
        self.path = os.path.join(STATE_DIR, "settings.json")
        self.lock = threading.Lock()
        self.data = default_settings()
        try:
            with open(self.path, encoding="utf-8") as f:
                self.data.update(json.load(f))
        except (OSError, ValueError):
            pass

    def get(self, k, default=None):
        return self.data.get(k, default)

    def update(self, **kw):
        with self.lock:
            self.data.update({k: v for k, v in kw.items() if k in default_settings()})
            os.makedirs(STATE_DIR, exist_ok=True)
            write_json(self.path, self.data)

    def remember_dataset(self, path):
        recent = [p for p in self.data.get("recent_datasets", []) if p != path]
        self.update(recent_datasets=[path] + recent[:9])


def write_json(path, data):
    tmp = f"{path}.{uuid.uuid4().hex[:6]}.part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                           # a reader has it open (Windows): retry
            time.sleep(0.05)
    os.remove(tmp)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def norm(p):
    return os.path.normpath(p).replace("\\", "/") if p else p


# ============================================================================ config schema (for the forms)
H = {   # path -> hint. doc: tooltip | enum | enum_from | example (value used by "Set") | advanced | show_if
    "name": {"doc": "Dataset name: names the output folder and the zip. Letters, digits, _ and -."},
    "map": {"doc": "The level to render (must be open in Unreal), e.g. /Game/volume/water_volume."},
    "seed": {"doc": "Same recipe + same seed = the same images. Change it for a new, different dataset."},

    "output.raw_dir": {"doc": "Rendered EXR frames (temporary: deleted as soon as they are converted).", "advanced": True},
    "output.dataset_dir": {"doc": "The dataset lands in <this folder>/<name>/, the shareable zip next to it."},
    "output.val_fraction": {"doc": "Share of images put in the validation split (0.1 = 10%)."},
    "output.jpg_quality": {"doc": "JPEG quality of the dataset images.", "advanced": True},
    "output.keep_exr": {"doc": "Keep the raw EXR frames after conversion (debugging only - they are big).", "advanced": True},
    "output.previews": {"doc": "Images per stage drawn with their labels (in previews/) - look at them before training."},
    "output.min_area": {"doc": "Pixels: visible parts smaller than this get no label."},
    "output.epsilon": {"doc": "Polygon simplification in pixels (higher = fewer points).", "advanced": True},
    "output.empty_keep": {"doc": "Share of frames with nothing labelled kept as background images."},
    "output.max_label_depth": {"doc": "cm: objects farther than this get no label (0 = off).", "advanced": True},

    "render.resolution": {"doc": "Image size [width, height] in pixels. 960x540 renders ~1.8x faster."},
    "render.executor": {"enum": ["pie", "process"], "doc": "pie = render inside this editor (default); process = separate game process."},
    "render.fps": {"doc": "World time per frame (water and caustic animation speed between images).", "advanced": True},
    "render.anti_aliasing": {"enum": ["tsr", "taa", "fxaa", "none"], "advanced": True, "doc": "Anti-aliasing method."},
    "render.warmup_frames": {"doc": "Frames rendered but not saved at the start of each stage (lighting settles).", "advanced": True},
    "render.texture_streaming": {"enum": ["fully_load", "disabled", "none"], "advanced": True, "doc": "fully_load avoids blurry textures."},
    "render.exr_compression": {"enum": ["zip", "zips", "none"], "advanced": True, "doc": "Compression of the temporary EXR frames."},
    "render.cinematic": {"doc": "MRQ cinematic quality (slower, prettier).", "advanced": True},
    "render.cvars": {"doc": "Any Unreal console variables, e.g. { \"r.Lumen.Reflections.Allow\" = 0 }.", "advanced": True, "example": {}},

    "post.python": {"doc": "Python with numpy + opencv that builds the dataset. Empty = $SDG_PYTHON, else SegPipeline/.venv, else ~/seg-train/.venv.", "advanced": True},
    "post.workers": {"doc": "Parallel workers converting frames to images + labels."},
    "post.train": {"enum": ["", "seg", "pose", "both"], "doc": "Empty = pack a shareable zip after a full render. seg/pose/both = start the pipeline's own training instead."},
    "post.train_args": {"doc": "Extra arguments for the pipeline's own training (post.train).", "advanced": True},
    "post.open_trainer": {"doc": "When a full render is done, open this app's Train page with the new dataset."},

    "classes.seg": {"doc": "Segmentation classes. YOLO ids follow this order - keep it identical to your real dataset (e.g. ambulance, blood, fire, firetruck)."},
    "classes.pose": {"doc": "Keypoint (pose) classes. Leave empty for segmentation only."},

    "camera.mode": {"enum": list(sc.CAMERA_MODES), "doc": "random_path: smooth flight, every frame is an image (fastest). teleport: an independent random view per image. sequence: your own Level Sequence camera path."},
    "camera.fov": {"doc": "Horizontal field of view range in degrees. Use your real camera's UNDERWATER field of view (OAK-D behind a flat port: about 52-60)."},
    "camera.volume": {"widget": "volume", "example": {"min": [0, 0, 0], "max": [0, 0, 0]}, "doc": "Box the camera must stay inside (world cm): keeps it in the pool and under the surface. Fly the Unreal viewport to two opposite corners and press 'Use viewport'."},
    "camera.targets": {"doc": "\"props\" (look at the labelled props) or a list of world points [[x, y, z], ...].", "advanced": True},
    "camera.distance": {"doc": "Distance from the prop being looked at, cm."},
    "camera.elevation": {"doc": "Degrees above (+) or below (-) the prop."},
    "camera.azimuth": {"example": [-50, 50], "doc": "Degrees left/right of straight in front of the prop. Not set = all around."},
    "camera.facing": {"enum": list(sc.FACINGS), "doc": "Which side is the front: auto = the flat face (thinnest side). Use -auto if the previews show the back."},
    "camera.aim_jitter": {"doc": "Degrees the prop may sit off-centre."},
    "camera.roll": {"doc": "Camera roll range, degrees."},
    "camera.speed": {"doc": "random_path: flight speed, cm per frame.", "show_if": {"mode": ["random_path"]}},
    "camera.turn": {"doc": "random_path: max change of view angle between waypoints, degrees.", "show_if": {"mode": ["random_path"]}, "advanced": True},
    "camera.switch_target": {"doc": "random_path: chance to fly on to another prop at each waypoint.", "show_if": {"mode": ["random_path"]}, "advanced": True},
    "camera.hold": {"doc": "teleport: render each view N times and keep the last, so lighting settles (2-3).", "show_if": {"mode": ["teleport"]}},
    "camera.sequence": {"doc": "sequence: the Level Sequence with your planned camera (/Game/...).", "show_if": {"mode": ["sequence"]}},
    "camera.sample_fps": {"doc": "sequence: resample the path at this rate (0 = its own rate).", "show_if": {"mode": ["sequence"]}},
    "camera.repeat": {"doc": "sequence: play the path this many times, re-randomized each time.", "show_if": {"mode": ["sequence"]}},
    "camera.jitter_pos": {"doc": "sequence: smooth random position offset, cm.", "show_if": {"mode": ["sequence"]}},
    "camera.jitter_rot": {"doc": "sequence: smooth random rotation offset, degrees.", "show_if": {"mode": ["sequence"]}},
    "camera.fov_every": {"doc": "sequence: frames between FOV changes.", "show_if": {"mode": ["sequence"]}, "advanced": True},

    "water_fog.enabled": {"doc": "Add a randomized underwater haze over the image."},
    "water_fog.density": {"doc": "Haze density range, 1/cm. 0.0005 = clear, 0.0012 = hazy, 0.003 = murky."},
    "water_fog.color": {"doc": "Haze colour, e.g. { hsv = { h = [0.38, 0.6], s = [0.3, 0.8], v = [0.02, 0.15] } }."},
    "water_fog.every": {"doc": "New haze every N images.", "advanced": True},
    "water_fog.smooth": {"doc": "Blend between haze values (flying stages).", "advanced": True},
    "water_fog.min_visibility": {"doc": "Objects dimmed below this share of their light by the haze get no label."},

    # list items
    "props[].name": {"doc": "Name of this prop entry (used in the setup log)."},
    "props[].type": {"enum": ["actor", "texture_regions", "material_slots"], "doc": "actor: the whole object is one class. texture_regions: drawings printed on a texture (one mask per class). material_slots: parts with their own material."},
    "props[].actor_label": {"doc": "Outliner label of the actor(s), wildcards allowed (e.g. SM_Bin_*)."},
    "props[].actor_tag": {"doc": "Actor tag (Details > Actor > Tags) instead of / as well as the label."},
    "props[].static_mesh": {"doc": "Static mesh asset path pattern, e.g. /Game/Props/SM_Bin_O*."},
    "props[].texture": {"doc": "texture_regions: the texture asset with the drawings (/Game/...)."},
    "props[].regions": {"doc": "texture_regions: class = mask PNG (paths relative to SegPipeline/).", "example": {}},
    "props[].slots": {"doc": "material_slots: \"*SlotOrMaterialPattern*\" = class.", "example": {"*Yellow*": ""}},
    "props[].class": {"enum_from": "classes.seg", "doc": "actor: the class of the whole object."},
    "props[].body_class": {"enum_from": "classes.seg", "doc": "Optional class for the rest of the object."},
    "props[].pose_class": {"enum_from": "classes.pose", "doc": "Keypoint class of this prop (needs keypoints)."},
    "props[].keypoints": {"example": "face4", "doc": "face4 (4 corners of the flat face), face4:-x, top4, bottom4, bounds8, center, or { sockets = [\"KP_TL\", ...] }."},

    "randomize[].name": {"doc": "Name (used by stage overrides and in meta/*.jsonl)."},
    "randomize[].enabled": {"doc": "Switch this randomizer off without deleting it."},
    "randomize[].type": {"enum": list(sc.RANDOMIZER_TYPES), "doc": "What kind of value. Leave out for a number property (float)."},
    "randomize[].actor": {"doc": "Outliner label of the actor(s), wildcards allowed; * = every actor that has the component."},
    "randomize[].tag": {"doc": "Actor tag instead of the label."},
    "randomize[].component": {"doc": "Component name or class, e.g. DirectionalLightComponent, SkyLightComponent."},
    "randomize[].property": {"doc": "C++ property name, e.g. Intensity, LightColor, LightFunctionScale."},
    "randomize[].postprocess": {"doc": "Post Process Volume setting (C++ name): AutoExposureBias, WhiteTemp, VignetteIntensity, BloomIntensity, FilmGrainIntensity, SceneFringeIntensity, SceneColorTint ..."},
    "randomize[].mpc": {"example": sc.SDG_MPC, "doc": "Material Parameter Collection asset (wire its parameter into your material)."},
    "randomize[].param": {"doc": "Parameter name (MPC or material)."},
    "randomize[].material_slot": {"example": 0, "doc": "Material slot index on the mesh."},
    "randomize[].value": {"example": [0.5, 1.5], "doc": "[min, max] = uniform. Also { normal = [mean, std], clip = [lo, hi] }, { choice = [...] }, colours { hsv = {...} } / { kelvin = [k0, k1] }."},
    "randomize[].mode": {"enum": ["set", "add", "multiply"], "doc": "set = use the value; multiply / add = relative to what is set in the level."},
    "randomize[].every": {"example": 1, "doc": "New value every N images (lighting: 10-20 keeps Lumen from ghosting)."},
    "randomize[].smooth": {"example": True, "doc": "Blend between values (flying stages)."},
    "randomize[].per_target": {"example": True, "doc": "Each matched actor gets its own value."},
    "randomize[].location": {"example": {"x": [-50, 50], "y": [-50, 50]}, "doc": "transform: location offsets in cm, e.g. { x = [-50, 50] }."},
    "randomize[].rotation": {"example": {"yaw": [-180, 180]}, "doc": "transform: rotation offsets in degrees (pitch, yaw, roll)."},
    "randomize[].scale": {"example": {"x": [0.9, 1.1]}, "doc": "transform: scale factors."},
    "randomize[].offset": {"example": False, "doc": "transform: false = absolute values instead of offsets."},
    "randomize[].probability": {"example": 0.5, "doc": "visible: chance the actor is shown (0 = always hidden)."},
    "randomize[].default": {"example": 0.0, "doc": "mpc: the value the editor shows outside a render."},

    "stages[].name": {"doc": "Stage name (image file prefix)."},
    "stages[].frames": {"example": 1000, "doc": "Number of images in this stage."},
    "stages[].camera": {"example": {"mode": "random_path"}, "doc": "Camera settings for this stage (anything from the Camera section)."},
    "stages[].override": {"example": {}, "doc": "Different ranges for this stage: randomizer name = [min, max] (or { value = [..], every = 15 })."},
    "stages[].disable": {"enum_from": "randomize.names", "example": [], "doc": "Randomizers switched off in this stage."},
}

OBJECT_KEYS = {  # suggested keys for free-form objects
    "randomize[].location": ["x", "y", "z"],
    "randomize[].rotation": ["pitch", "yaw", "roll"],
    "randomize[].scale": ["x", "y", "z"],
    "camera.volume": ["min", "max"],
}

ITEM_FIELDS = {
    "props": ["name", "type", "actor_label", "actor_tag", "static_mesh", "texture", "regions", "slots", "class",
              "body_class", "pose_class", "keypoints"],
    "randomize": ["name", "enabled", "type", "actor", "tag", "component", "property", "postprocess", "mpc", "param",
                  "material_slot", "value", "mode", "every", "smooth", "per_target", "location", "rotation", "scale",
                  "offset", "probability", "default"],
    "stages": ["name", "frames", "camera", "override", "disable"],
}

TEMPLATES = {
    "props": [
        {"label": "Drawings on a texture", "doc": "e.g. the torpedo board: one mask PNG per drawing",
         "item": {"name": "new_board", "type": "texture_regions", "texture": "/Game/...", "regions": {}}},
        {"label": "Part with its own material", "doc": "e.g. the yellow X / O on a bin",
         "item": {"name": "new_prop", "type": "material_slots", "static_mesh": "/Game/Props/SM_*", "slots": {"*Yellow*": ""}}},
        {"label": "Whole object is one class", "doc": "e.g. a buoy or a gate",
         "item": {"name": "new_object", "type": "actor", "actor_label": "MyActor*", "class": ""}},
    ],
    "randomize": [
        {"label": "Light brightness", "doc": "sun / directional light intensity, relative to the level",
         "item": {"name": "sun_intensity", "actor": "*", "component": "DirectionalLightComponent", "property": "Intensity",
                  "mode": "multiply", "value": [0.3, 1.8], "every": 15, "smooth": True}},
        {"label": "Light colour", "doc": "random tint of a light",
         "item": {"name": "sun_color", "actor": "*", "component": "DirectionalLightComponent", "property": "LightColor",
                  "type": "color", "value": {"hsv": {"h": [0.0, 1.0], "s": [0.0, 0.3], "v": [1.0, 1.0]}}, "every": 15, "smooth": True}},
        {"label": "Light direction", "doc": "rotates the sun: moves shadows and caustics",
         "item": {"name": "sun_direction", "actor": "*", "component": "DirectionalLightComponent", "type": "transform",
                  "rotation": {"pitch": [-20, 20], "yaw": [-180, 180]}, "every": 15, "smooth": True}},
        {"label": "Post-process setting", "doc": "exposure, white balance, vignette, bloom, grain ...",
         "item": {"name": "vignette", "postprocess": "VignetteIntensity", "value": [0.0, 0.7]}},
        {"label": "Material parameter (MPC)", "doc": "any value you wire into a material, e.g. caustic strength",
         "item": {"name": "caustics_strength", "mpc": sc.SDG_MPC, "param": "CausticsStrength", "value": [0.4, 1.8]}},
        {"label": "Mesh material parameter", "doc": "a parameter of one mesh's material, e.g. tile colour",
         "item": {"name": "floor_tint", "actor": "Floor", "material_slot": 0, "param": "Tint",
                  "value": {"hsv": {"h": [0.0, 1.0], "s": [0.0, 0.3], "v": [0.6, 1.0]}}}},
        {"label": "Move / rotate actors", "doc": "random layout of props",
         "item": {"name": "layout", "actor": "SM_*", "type": "transform", "per_target": True,
                  "location": {"x": [-50, 50], "y": [-50, 50]}, "rotation": {"yaw": [-180, 180]}, "every": 25}},
        {"label": "Show / hide actors", "doc": "distractors that appear in some images",
         "item": {"name": "distractors", "actor": "SM_*", "type": "visible", "probability": 0.5}},
        {"label": "Swap material", "doc": "pick one of several materials",
         "item": {"name": "floor_material", "actor": "Floor", "type": "material", "material_slot": 0,
                  "value": {"choice": ["/Game/..."]}}},
    ],
    "stages": [
        {"label": "Smooth flight", "doc": "random_path: every frame is an image (fastest)",
         "item": {"name": "flight", "frames": 1000, "camera": {"mode": "random_path"}}},
        {"label": "Random views", "doc": "teleport: independent view per image, rendered twice",
         "item": {"name": "snap", "frames": 1000, "camera": {"mode": "teleport", "hold": 2}}},
        {"label": "Far and dark", "doc": "far distances, dimmer light",
         "item": {"name": "far_dim", "frames": 1000, "camera": {"mode": "random_path", "distance": [300, 800]}, "override": {}}},
        {"label": "Planned camera path", "doc": "your own Level Sequence camera",
         "item": {"name": "path", "camera": {"mode": "sequence", "sequence": "/Game/...", "repeat": 2}}},
    ],
}

NEW_CONFIG = {
    "name": "new_dataset", "map": "/Game/volume/water_volume", "seed": 1,
    "output": {"raw_dir": "../Saved/SDG/raw", "dataset_dir": "../Saved/SDG/datasets"},
    "post": {"python": "", "open_trainer": True},
    "classes": {"seg": [], "pose": []},
    "camera": {"fov": [52, 60], "distance": [80, 600], "elevation": [-8, 15]},
    "stages": [{"name": "flight", "frames": 500, "camera": {"mode": "random_path"}}],
}


def schema():
    defaults = copy.deepcopy(sc.DEFAULTS)
    return {"defaults": defaults, "hints": H, "object_keys": OBJECT_KEYS, "item_fields": ITEM_FIELDS,
            "templates": TEMPLATES, "new_config": NEW_CONFIG, "listener": LISTENER}


# ============================================================================ configs
def list_configs():
    if not os.path.isdir(CONFIG_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(CONFIG_DIR) if f.endswith(".toml"))


def config_path(stem):
    if not NAME_RE.match(stem or ""):
        raise ApiError(f"invalid config name {stem!r}")
    return os.path.join(CONFIG_DIR, stem + ".toml")


def read_config(stem):
    path = config_path(stem)
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    return {"name": stem, "path": norm(path), "raw": tomllib.loads(text), "text": text,
            "generated": text.startswith(tomlw.HEADER)}


def parse_toml_value(text):
    try:
        return tomllib.loads("v = " + text.strip())["v"], None
    except tomllib.TOMLDecodeError as e:
        return None, f"not a valid value: {e}"


def resolve_placeholders(obj, path, errors):
    """{"$toml": text} (a field typed as TOML in the app) -> its value."""
    if isinstance(obj, dict):
        if set(obj) == {"$toml"}:
            v, err = parse_toml_value(obj["$toml"])
            if err:
                errors.append({"path": path, "message": err})
            return v
        return {k: resolve_placeholders(v, f"{path}.{k}" if path else k, errors) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_placeholders(v, f"{path}[{i}]", errors) for i, v in enumerate(obj)]
    return obj


def strip_nulls(obj):
    if isinstance(obj, dict):
        return {k: strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_nulls(v) for v in obj]
    return obj


def check_config(raw):
    """-> (toml text or None, errors, summary). Validates exactly like the pipeline does."""
    errors = []
    raw = strip_nulls(resolve_placeholders(raw, "", errors))
    if errors:
        return None, errors, {}
    try:
        text = tomlw.dumps(raw)
    except ValueError as e:
        return None, [{"path": "", "message": str(e)}], {}
    tmp_dir = os.path.join(STATE_DIR, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f"check_{uuid.uuid4().hex[:8]}.toml")
    summary = {}
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        cfg = sc.load_config(tmp)
        summary = {
            "images": sum(int(st.get("frames", 0) or 0) for st in cfg["stages"]),
            "stages": [{"name": st["name"], "frames": st.get("frames"), "mode": sc.stage_camera(cfg, st)["mode"]}
                       for st in cfg["stages"]],
            "dataset": norm(os.path.join(cfg["output"]["dataset_dir"], cfg["name"])),
            "seg_classes": cfg["seg_classes"], "pose_classes": cfg["pose_classes"],
        }
    except sc.ConfigError as e:
        errors.append({"path": "", "message": str(e)})
    except tomllib.TOMLDecodeError as e:
        errors.append({"path": "", "message": f"TOML: {e}"})
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    return text, errors, summary


def save_config(stem, raw):
    path = config_path(stem)
    text, errors, summary = check_config(raw)
    if errors:
        return {"ok": False, "errors": errors}
    backup = None
    if os.path.isfile(path):
        with open(path, encoding="utf-8-sig") as f:
            old = f.read()
        if not old.startswith(tomlw.HEADER) and not os.path.exists(path + ".orig"):
            backup = path + ".orig"                      # the hand-written original, with its comments
            shutil.copyfile(path, backup)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return {"ok": True, "path": norm(path), "text": text, "summary": summary, "backup": norm(backup)}


def load_cfg(stem):
    """The merged, validated config (paths resolved) - None if it doesn't load."""
    try:
        return sc.load_config(config_path(stem))
    except Exception:
        return None


# ============================================================================ unreal bridge
def unreal_status():
    hb = read_json(os.path.join(BRIDGE, "heartbeat.json"))
    if not hb:
        return {"connected": False, "never": True}
    age = time.time() - hb.get("time", 0)
    return {"connected": age < 8, "age": round(age, 1), "level": hb.get("level", ""),
            "rendering": bool(hb.get("rendering")) and age < 8}


def send_unreal(command, args):
    if command not in ("setup", "render", "bake", "preview", "where", "cancel", "save_level", "ping"):
        raise ApiError(f"unknown command {command!r}")
    args = [str(a) for a in args]
    if command in ("setup", "render", "bake", "preview"):
        if not args or not NAME_RE.match(args[0]) or not os.path.isfile(config_path(args[0])):
            raise ApiError("choose a saved config first")
    inbox = os.path.join(BRIDGE, "inbox")
    os.makedirs(inbox, exist_ok=True)
    cid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    write_json(os.path.join(inbox, cid + ".json"), {"command": command, "args": args, "sent": time.time()})
    return cid


def unreal_result(cid):
    if not re.match(r"^[\w\-]+$", cid or ""):
        raise ApiError("bad id")
    res = read_json(os.path.join(BRIDGE, "outbox", cid + ".json"))
    if res is not None:
        res["status"] = "done"
        return res
    queued = os.path.exists(os.path.join(BRIDGE, "inbox", cid + ".json"))
    return {"status": "pending", "queued": queued}


# ============================================================================ render progress
def _count(d, ext=".exr"):
    try:
        return sum(1 for e in os.scandir(d) if e.name.lower().endswith(ext))
    except OSError:
        return 0


def _read_text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def run_state(stem):
    cfg = load_cfg(stem)
    if cfg is None:
        return {"ok": False, "error": "the config doesn't load - fix it on the Recipe page"}
    raw_run = norm(os.path.join(cfg["output"]["raw_dir"], cfg["name"]))
    dataset = norm(os.path.join(cfg["output"]["dataset_dir"], cfg["name"]))
    run = read_json(os.path.join(raw_run, "run.json"))
    prog = read_json(os.path.join(raw_run, "progress.json"))
    out = {
        "ok": True, "name": cfg["name"], "raw": raw_run, "dataset": dataset, "map": cfg["map"],
        "planned": {"images": sum(int(st.get("frames", 0) or 0) for st in cfg["stages"]),
                    "stages": [{"name": st["name"], "frames": st.get("frames"), "mode": cam["mode"],
                                "hold": int(cam.get("hold", 1)) if cam["mode"] == "teleport" else 1}
                               for st, cam in ((st, sc.stage_camera(cfg, st)) for st in cfg["stages"])]},
        "run": None, "progress": None, "done_flag": None, "failed_flag": None,
    }
    run_mtime = os.path.getmtime(os.path.join(raw_run, "run.json")) if run else 0
    if run:
        stages = []
        for st in run["stages"]:
            stages.append({"name": st["name"], "images": st["frames"], "expected": st["seq_frames"],
                           "pending": _count(st["exr_dir"])})
        out["run"] = {"created": run.get("created"), "created_t": run_mtime, "quick": run.get("quick", 0), "stages": stages}
    if prog and prog.get("time", 0) >= run_mtime - 1:
        out["progress"] = prog
    out["done_flag"] = _read_text(os.path.join(raw_run, "_DONE"))
    out["failed_flag"] = _read_text(os.path.join(raw_run, "_FAILED"))
    prev_dir = os.path.join(dataset, "previews")
    try:
        prevs = sorted((e for e in os.scandir(prev_dir) if e.name.lower().endswith(".jpg")),
                       key=lambda e: e.stat().st_mtime, reverse=True)
        out["previews"] = [e.name for e in prevs[:48]]
        out["previews_total"] = len(prevs)
    except OSError:
        out["previews"], out["previews_total"] = [], 0
    stats_path = os.path.join(dataset, "stats.json")
    if os.path.isfile(stats_path) and os.path.getmtime(stats_path) >= run_mtime:
        out["stats"] = read_json(stats_path)
    zp = os.path.join(cfg["output"]["dataset_dir"], f"{cfg['name']}_yolo_seg.zip")
    if os.path.isfile(zp):
        out["zip"] = {"path": norm(zp), "size": os.path.getsize(zp), "mtime": os.path.getmtime(zp)}
    m = re.search(r"(\d+) frames in ([\d.]+) s", out["done_flag"] or "")
    if m and float(m.group(2)) > 0:
        out["last_fps"] = int(m.group(1)) / float(m.group(2))
    return out


LOG_KEEP = re.compile(r"\[SDG\]|LogMovieRenderPipeline: (Error|Warning)|LogPython: Error|MoviePipeline.*(Error|error)")
LOG_TS = re.compile(r"^\[(\d{4})\.(\d\d)\.(\d\d)-(\d\d)\.(\d\d)\.(\d\d):\d+\]\[\s*\d+\]")


def log_tail(since, settings):
    path = settings.get("editor_log")
    if not path or not os.path.isfile(path):
        return {"offset": 0, "lines": [], "missing": True}
    size = os.path.getsize(path)
    if since < 0 or since > size:
        since = max(0, size - 256 * 1024)               # first call / new editor session: recent part only
    with open(path, "rb") as f:
        f.seek(since)
        data = f.read(min(size - since, 4 * 1024 * 1024))
    end = data.rfind(b"\n")
    if end < 0:
        return {"offset": since, "lines": []}
    chunk, offset = data[:end + 1], since + end + 1
    lines, keep = [], False
    for ln in chunk.decode("utf-8", errors="replace").splitlines():
        m = LOG_TS.match(ln)
        if m:
            keep = bool(LOG_KEEP.search(ln))
            if keep:
                body = ln[m.end():].replace("LogPython: ", "").replace("[SDG] ", "")
                lines.append(f"{m.group(4)}:{m.group(5)}:{m.group(6)}  {body}")
        elif keep and ln.strip():                        # continuation of a kept entry (tracebacks)
            lines.append("          " + ln)
    return {"offset": offset, "lines": lines[-400:]}


# ============================================================================ datasets
_ds_cache = {}


def dataset_roots(settings):
    roots = []
    for stem in list_configs():
        cfg = load_cfg(stem)
        if cfg:
            roots.append(cfg["output"]["dataset_dir"])
    repo = settings.get("trainer_repo")
    if repo:
        roots += [os.path.join(repo, "datasets"), os.path.join(repo, "dataset")]
    seen, out = set(), []
    for r in roots:
        r = norm(r)
        if r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def _is_dataset_dir(d):
    return any(os.path.isdir(os.path.join(d, *p)) for p in (("images",), ("train", "images"), ("valid", "images")))


def discover_datasets(settings):
    found = []
    for root in dataset_roots(settings):
        for e in sorted(os.scandir(root), key=lambda e: e.name.lower()):
            if e.is_dir() and _is_dataset_dir(e.path):
                found.append(e.path)
                if os.path.isdir(os.path.join(e.path, "pose", "images")):
                    found.append(os.path.join(e.path, "pose"))
            elif e.is_file() and e.name.lower().endswith(".zip"):
                found.append(e.path)
    for p in settings.get("recent_datasets", []):
        if os.path.exists(p):
            found.append(p)
    out, seen = [], set()
    for p in found:
        p = norm(p)
        if p in seen:
            continue
        seen.add(p)
        try:
            out.append(dataset_info(p))
        except Exception as e:
            out.append({"path": p, "name": os.path.basename(p), "error": str(e)})
    return out


def _yaml_load(text):
    import yaml
    return yaml.safe_load(text) or {}


def _names_list(names):
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda k: int(k))]
    return list(names or [])


def _label_task(lines, kpt_shape):
    if kpt_shape:
        return "pose"
    lens = [len(ln.split()) for ln in lines if ln.strip()]
    if not lens:
        return "unknown"
    return "segment" if sum(n > 5 for n in lens) > len(lens) / 2 else "detect"


def dataset_info(path, detail=False):
    path = norm(path)
    st = os.stat(path)
    key = (path, st.st_mtime, detail)
    if key in _ds_cache:
        return _ds_cache[key]
    info = {"path": path, "name": os.path.basename(path), "mtime": st.st_mtime, "splits": {}, "names": [],
            "kpt_shape": None, "class_counts": None}
    if path.lower().endswith(".zip"):
        info.update(kind="zip", name=os.path.basename(path)[:-4], size=st.st_size)
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            yml = next((n for n in names if n.endswith(("/data.yaml", "data.yaml"))), None)
            data = _yaml_load(z.read(yml).decode("utf-8")) if yml else {}
            for split in ("train", "val", "valid", "test"):
                n = sum(1 for x in names if (f"images/{split}/" in x or f"{split}/images/" in x) and x.lower().endswith(IMG_EXT))
                if n:
                    info["splits"]["val" if split == "valid" else split] = n
            stats = next((n for n in names if n.endswith("stats.json")), None)
            if stats:
                info["class_counts"] = json.loads(z.read(stats)).get("seg")
            label = next((n for n in names if "/labels/" in n and n.endswith(".txt")), None)
            lines = z.read(label).decode("utf-8", "replace").splitlines() if label else []
    else:
        info["kind"] = "folder"
        data = {}
        cands = [os.path.join(path, n) for n in ("data.yaml", "seg.yaml", "pose.yaml")]
        if os.path.basename(path) == "pose":
            cands.append(os.path.join(os.path.dirname(path), "pose.yaml"))
            info["name"] = os.path.basename(os.path.dirname(path)) + " (pose)"
        for c in cands:
            if os.path.isfile(c):
                with open(c, encoding="utf-8") as f:
                    data = _yaml_load(f.read())
                break
        label_dirs = []
        for split in ("train", "val", "valid", "test"):
            for img_dir, lbl_dir in ((os.path.join(path, "images", split), os.path.join(path, "labels", split)),
                                     (os.path.join(path, split, "images"), os.path.join(path, split, "labels"))):
                if os.path.isdir(img_dir):
                    n = sum(1 for e in os.scandir(img_dir) if e.name.lower().endswith(IMG_EXT))
                    if n:
                        info["splits"]["val" if split == "valid" else split] = n
                        label_dirs.append(lbl_dir)
                    break
        for sp in (os.path.join(path, "stats.json"), os.path.join(os.path.dirname(path), "stats.json")):
            if os.path.isfile(sp):
                s = read_json(sp, {})
                info["class_counts"] = s.get("pose" if os.path.basename(path) == "pose" else "seg")
                break
        lines = []
        for d in label_dirs:
            try:
                first = next((e.path for e in os.scandir(d) if e.name.endswith(".txt") and e.stat().st_size), None)
            except OSError:
                first = None
            if first:
                with open(first, encoding="utf-8") as f:
                    lines = f.read().splitlines()
                break
        if detail and info["class_counts"] is None:
            info["class_counts"] = _count_classes(label_dirs, _names_list(data.get("names")))
        info["size"] = None
    info["names"] = _names_list(data.get("names"))
    info["kpt_shape"] = data.get("kpt_shape")
    info["task"] = _label_task(lines, info["kpt_shape"])
    info["images"] = sum(info["splits"].values())
    _ds_cache[key] = info
    return info


def _count_classes(label_dirs, names):
    counts = {}
    n = 0
    for d in label_dirs:
        for e in os.scandir(d):
            if not e.name.endswith(".txt"):
                continue
            n += 1
            if n > 30000:
                return None
            with open(e.path, encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        c = int(ln.split()[0])
                        name = names[c] if c < len(names) else str(c)
                        counts[name] = counts.get(name, 0) + 1
    return counts


# ============================================================================ training
AUG = [  # key, label, doc, min, max, step
    ("hsv_h", "Hue shift", "water tint: blue / green / teal", 0, 0.2, 0.005),
    ("hsv_s", "Saturation", "washed-out to saturated colours", 0, 1, 0.05),
    ("hsv_v", "Brightness", "dark to bright images", 0, 1, 0.05),
    ("degrees", "Rotation", "camera roll, degrees", 0, 45, 1),
    ("translate", "Shift", "object off-centre (fraction of the image)", 0, 0.5, 0.05),
    ("scale", "Zoom", "0.7 = objects 0.3x to 1.7x their size: far and close targets", 0, 0.9, 0.05),
    ("shear", "Shear", "slight skew, degrees", 0, 10, 0.5),
    ("perspective", "Perspective", "oblique views (above 0.001 looks unrealistic)", 0, 0.002, 0.0001),
    ("fliplr", "Mirror left-right", "chance of a mirrored image (pose: needs flip_idx)", 0, 1, 0.05),
    ("flipud", "Flip upside down", "keep 0 unless objects can be upside down", 0, 1, 0.05),
    ("mosaic", "Mosaic", "4 images in one: small targets, varied context", 0, 1, 0.05),
    ("mixup", "Mixup", "two images blended (murky views); above 0.2 hurts segmentation", 0, 0.5, 0.05),
    ("copy_paste", "Copy-paste", "segmentation: objects pasted onto other images", 0, 0.6, 0.05),
]
PRESETS = {
    "off": {"label": "Off", "doc": "no augmentation - only for checking the data",
            "values": {"hsv_h": 0, "hsv_s": 0, "hsv_v": 0, "degrees": 0, "translate": 0, "scale": 0, "shear": 0,
                       "perspective": 0, "fliplr": 0, "flipud": 0, "mosaic": 0, "mixup": 0, "copy_paste": 0},
            "albumentations": "none"},
    "light": {"label": "Light", "doc": "Ultralytics defaults: real, varied photos",
              "values": {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0, "translate": 0.1, "scale": 0.5,
                         "shear": 0, "perspective": 0, "fliplr": 0.5, "flipud": 0, "mosaic": 1, "mixup": 0,
                         "copy_paste": 0},
              "albumentations": "none"},
    "medium": {"label": "Medium", "doc": "synthetic data, moderate: stronger colour + geometry, light copy-paste",
               "values": {"hsv_h": 0.05, "hsv_s": 0.8, "hsv_v": 0.5, "degrees": 10, "translate": 0.15, "scale": 0.6,
                          "shear": 2, "perspective": 0.0005, "fliplr": 0.5, "flipud": 0, "mosaic": 1, "mixup": 0,
                          "copy_paste": 0.1},
               "albumentations": "none"},
    "strong": {"label": "Strong", "doc": "synthetic -> real: the Unreal recipe (unreal_config.yaml)",
               "values": {"hsv_h": 0.08, "hsv_s": 0.9, "hsv_v": 0.6, "degrees": 15, "translate": 0.2, "scale": 0.7,
                          "shear": 4, "perspective": 0.0008, "fliplr": 0.5, "flipud": 0, "mosaic": 1, "mixup": 0.15,
                          "copy_paste": 0.3},
               "albumentations": "underwater"},
}
VERSIONS = [
    {"id": "yolo11", "label": "YOLO11", "doc": "recommended: what this pipeline was tested with"},
    {"id": "yolo26", "label": "YOLO26", "doc": "newest Ultralytics models (8.4+)"},
    {"id": "yolov8", "label": "YOLOv8", "doc": "older, widely used"},
]
SIZES = [
    {"id": "n", "label": "n", "doc": "nano: fastest, least accurate - small robots / CPUs"},
    {"id": "s", "label": "s", "doc": "small: good balance (recommended)"},
    {"id": "m", "label": "m", "doc": "medium: more accurate, about 2x slower than s"},
    {"id": "l", "label": "l", "doc": "large: needs a strong GPU"},
    {"id": "x", "label": "x", "doc": "extra large: slowest, most accurate"},
]
TRAIN_DEFAULTS = {"version": "yolo11", "size": "s", "epochs": 150, "imgsz": 640, "batch": 16, "patience": 40,
                  "device": "auto", "workers": 6, "cache": "disk", "cos_lr": True, "close_mosaic": 15, "time": None,
                  "freeze": 0, "preset": "strong"}


def train_options(settings):
    repo = settings.get("trainer_repo") or ""
    return {"aug": [dict(zip(("key", "label", "doc", "min", "max", "step"), a)) for a in AUG], "presets": PRESETS,
            "versions": VERSIONS, "sizes": SIZES, "defaults": TRAIN_DEFAULTS,
            "repo": repo, "repo_ok": os.path.isfile(os.path.join(repo, "train.py")),
            "python": settings.get("trainer_python"), "python_ok": os.path.isfile(settings.get("trainer_python") or ""),
            "unreal_config": os.path.isfile(os.path.join(repo, "config", "unreal_config.yaml")),
            "real_data": settings.get("real_data", "")}


def _yaml_scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v))


def yaml_dump(cfg, header=""):
    lines = [header] if header else []
    for sec, vals in cfg.items():
        lines.append(f"{sec}:")
        for k, v in vals.items():
            lines.append(f"  {k}: {_yaml_scalar(v)}")
        lines.append("")
    return "\n".join(lines)


def _num(v, cast, lo=None, hi=None, name=""):
    try:
        x = cast(v)
    except (TypeError, ValueError):
        raise ApiError(f"{name}: not a number ({v!r})") from None
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise ApiError(f"{name}: must be between {lo} and {hi}")
    return x


def training_config(req):
    versions = {v["id"] for v in VERSIONS}
    if req.get("version") not in versions:
        raise ApiError("choose a model version")
    if req.get("size") not in "nsmlx" or not req.get("size"):
        raise ApiError("choose a model size")
    aug = {}
    for key, _label, _doc, lo, hi, _step in AUG:
        aug[key] = _num(req.get("aug", {}).get(key, PRESETS["strong"]["values"][key]), float, lo, hi, key)
    alb = req.get("albumentations", "none")
    if alb not in ("none", "underwater"):
        raise ApiError("albumentations: none or underwater")
    aug["erasing"] = 0.4
    aug["albumentations"] = alb
    time_h = req.get("time")
    train = {
        "epochs": _num(req.get("epochs", 150), int, 1, 5000, "epochs"),
        "batch": _num(req.get("batch", 16), int, -1, 1024, "batch"),
        "imgsz": _num(req.get("imgsz", 640), int, 32, 4096, "imgsz"),
        "device": str(req.get("device", "auto")),
        "workers": _num(req.get("workers", 6), int, 0, 64, "workers"),
        "patience": _num(req.get("patience", 40), int, 0, 5000, "patience"),
        "optimizer": "auto", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
        "warmup_epochs": 3, "cos_lr": bool(req.get("cos_lr", True)),
        "close_mosaic": _num(req.get("close_mosaic", 15), int, 0, 1000, "close_mosaic"),
        "cache": {"disk": "disk", "ram": "ram", "off": False}.get(req.get("cache", "disk"), "disk"),
        "amp": True, "seed": 0, "deterministic": True, "save_period": -1, "plots": True,
    }
    if train["batch"] == 0:
        raise ApiError("batch: -1 (auto) or a positive number")
    if time_h not in (None, "", 0):
        train["time"] = _num(time_h, float, 0.01, 1000, "time")
    freeze = _num(req.get("freeze", 0) or 0, int, 0, 100, "freeze")
    if freeze:
        train["freeze"] = freeze
    base = (req.get("base") or "").strip() or None
    if base and not os.path.isfile(base):
        raise ApiError(f"start-from model not found: {base}")
    return {
        "model": {"version": req["version"], "size": req["size"], "task": "auto", "base_weights": base,
                  "weights_dir": "weights"},
        "train": train, "augment": aug, "output": {"runs_dir": "runs"},
        "analyse": {"conf": 0.25, "iou": 0.7, "batch": 16, "val_rise_limit": 0.05, "late_window": 10, "smooth": 5,
                    "map_drop_sigmas": 2, "train_val_map_gap": 0.05},
    }


def _quote(a):
    a = str(a)
    return f'"{a}"' if (" " in a or not a) else a


def _registry_path():
    return os.path.join(STATE_DIR, "trainings.json")


_reg_lock = threading.Lock()


def registry():
    return read_json(_registry_path(), [])


def _save_registry(reg):
    os.makedirs(STATE_DIR, exist_ok=True)
    write_json(_registry_path(), reg[-200:])


def _launch_console(title, cmd, cwd, status_file=None):
    """Run cmd in its own console window that stays open at the end. Returns the process."""
    gui_dir = os.path.join(cwd, "runs", "_gui")
    os.makedirs(gui_dir, exist_ok=True)
    bat = os.path.join(gui_dir, f"{re.sub(r'[^A-Za-z0-9_-]', '_', title)}-{time.strftime('%Y%m%d-%H%M%S')}.bat")
    lines = ["@echo off", f"title {title}", f'cd /d "{cwd}"']
    if status_file:
        lines.append(f'> "{status_file}" echo running')
    lines += ["echo " + " ".join(_quote(a) for a in cmd).replace("%", "%%"), "echo.",
              " ".join(_quote(a) for a in cmd).replace("%", "%%"), "set RC=%ERRORLEVEL%"]
    if status_file:
        lines.append(f'> "{status_file}" echo exit %RC%')
    hidden = bool(os.environ.get("SDG_APP_HIDDEN"))     # tests: no window, no "press a key"
    if not hidden:
        lines += ["echo.", "echo Finished (exit code %RC%). Press a key to close this window.", "pause >nul"]
    with open(bat, "w", encoding="utf-8") as f:
        f.write("\r\n".join(lines) + "\r\n")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW if hidden else subprocess.CREATE_NEW_CONSOLE
    return subprocess.Popen(["cmd", "/c", bat], cwd=cwd, creationflags=flags), bat


def start_training(req, settings):
    repo, py = settings.get("trainer_repo") or "", settings.get("trainer_python") or ""
    if not os.path.isfile(os.path.join(repo, "train.py")):
        raise ApiError(f"yolo-training repo not found at {repo!r} - set it in Settings")
    if not os.path.isfile(py):
        raise ApiError(f"Python for training not found: {py!r} - set it in Settings")
    name = (req.get("model_name") or "").strip()
    if not NAME_RE.match(name):
        raise ApiError("model name: letters, digits, _ and - only")
    dataset = (req.get("dataset") or "").strip()
    if not dataset or not os.path.exists(dataset):
        raise ApiError("choose a dataset")
    model_path = os.path.join(repo, name + ".pt")
    if os.path.exists(model_path) and not req.get("overwrite"):
        return {"ok": False, "exists": True, "model": norm(model_path)}
    cfg = training_config(req)
    cfg_dir = os.path.join(repo, "config", "gui")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, name + ".yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(yaml_dump(cfg, f"# Written by the SDG app for model '{name}' ({time.strftime('%Y-%m-%d %H:%M')}).\n"
                               f"# Used as: train.py --config config/gui/{name}.yaml\n"))
    cmd = [py, "train.py", "--model", name + ".pt", "--dataset", dataset, "--config", cfg_path]
    if req.get("overwrite"):
        cmd.append("--overwrite")
    status = os.path.join(repo, "runs", "_gui", f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.status")
    proc, bat = _launch_console(f"Training {name}", cmd, repo, status)
    with _reg_lock:
        reg = registry()
        reg.append({"name": name, "model": norm(model_path), "dataset": norm(dataset), "config": norm(cfg_path),
                    "started": time.time(), "pid": proc.pid, "status_file": norm(status), "bat": norm(bat),
                    "run_dir": None, "epochs": cfg["train"]["epochs"]})
        _save_registry(reg)
    settings.remember_dataset(norm(dataset))
    return {"ok": True, "model": norm(model_path), "config": norm(cfg_path),
            "command": " ".join(_quote(a) for a in cmd)}


def process_alive(pid):
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259                   # STILL_ACTIVE


def _runs_dir(settings):
    return os.path.join(settings.get("trainer_repo") or "", "runs")


def _resolve_run_dir(entry, runs_dir):
    if entry.get("run_dir") and os.path.isdir(entry["run_dir"]):
        return entry["run_dir"]
    pat = re.compile(re.escape(entry["name"]) + r"\d*$")
    cands = []
    for e in os.scandir(runs_dir) if os.path.isdir(runs_dir) else []:
        if e.is_dir() and pat.match(e.name) and e.stat().st_ctime >= entry["started"] - 2:
            cands.append((e.stat().st_ctime, norm(e.path)))
    return min(cands)[1] if cands else None


def _read_results(run_dir):
    path = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(path):
        return [], []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    head = [h.strip() for h in rows[0]]
    data = []
    for r in rows[1:]:
        try:
            data.append([float(x) for x in r])
        except ValueError:
            continue
    return head, data


def _best_metric(head, data):
    for kind in ("M", "P", "B"):
        col = f"metrics/mAP50-95({kind})"
        if col in head and data:
            i = head.index(col)
            return {"kind": {"M": "mask", "P": "pose", "B": "box"}[kind], "value": max(r[i] for r in data),
                    "last": data[-1][i]}
    return None


def _guess_status(run_dir, head, data, epochs):
    """Status of a run the app didn't start: from how recently Ultralytics wrote results.csv."""
    if not data:
        return "unknown"
    if epochs and data[-1][0] >= epochs:
        return "finished"
    per_epoch = data[-1][head.index("time")] / max(data[-1][0], 1) if "time" in head else 300
    age = time.time() - os.path.getmtime(os.path.join(run_dir, "results.csv"))
    if age < max(600, 3 * per_epoch):
        return "running"
    return "ended"                                # early stop (patience) or interrupted


def _args(run_dir):
    p = os.path.join(run_dir, "args.yaml")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return _yaml_load(f.read())


def list_runs(settings):
    runs_dir = _runs_dir(settings)
    if not os.path.isdir(runs_dir):
        return []
    with _reg_lock:
        reg = registry()
        changed = False
        by_dir = {}
        for entry in reg:
            rd = _resolve_run_dir(entry, runs_dir)
            if rd and rd != entry.get("run_dir"):
                entry["run_dir"], changed = rd, True
            if rd:
                by_dir[rd] = entry
        if changed:
            _save_registry(reg)
    out = []
    for e in os.scandir(runs_dir):
        if not e.is_dir() or e.name.startswith("_"):
            continue
        rd = norm(e.path)
        head, data = _read_results(rd)
        args = _args(rd)
        entry = by_dir.get(rd)
        epochs = args.get("epochs") or (entry or {}).get("epochs")
        status = _guess_status(rd, head, data, epochs)
        if entry:
            st = _read_text(entry["status_file"]) or ""
            if st.startswith("exit"):
                status = "finished" if st.strip() == "exit 0" else "failed"
            elif process_alive(entry.get("pid")):
                status = "running"
            elif st == "running":
                status = "stopped"
        out.append({"name": e.name, "dir": rd, "status": status, "epoch": int(data[-1][0]) if data else 0,
                    "epochs": epochs, "best": _best_metric(head, data), "task": args.get("task"),
                    "mtime": os.path.getmtime(os.path.join(rd, "results.csv")) if data else e.stat().st_mtime,
                    "model": (entry or {}).get("model"), "dataset": (entry or {}).get("dataset"),
                    "elapsed": data[-1][head.index("time")] if data and "time" in head else None})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def run_detail(settings, name):
    if not NAME_RE.match(name or ""):
        raise ApiError("bad run name")
    rd = norm(os.path.join(_runs_dir(settings), name))
    if not os.path.isdir(rd):
        raise ApiError(f"run not found: {name}")
    head, data = _read_results(rd)
    run = next((r for r in list_runs(settings) if r["dir"] == rd), {"name": name, "dir": rd})
    args = _args(rd)
    run.update(columns=head, rows=data, args={k: args.get(k) for k in ("model", "data", "epochs", "imgsz", "batch", "task")},
               best_pt=norm(os.path.join(rd, "weights", "best.pt")) if os.path.isfile(os.path.join(rd, "weights", "best.pt")) else None,
               real_eval=os.path.isdir(os.path.join(rd, "real_eval")))
    return run


def model_for_run(settings, name):
    repo = settings.get("trainer_repo") or ""
    for entry in reversed(registry()):
        if entry.get("run_dir") and os.path.basename(entry["run_dir"]) == name and os.path.isfile(entry["model"]):
            return entry["model"]
    for cand in (os.path.join(repo, name + ".pt"), os.path.join(repo, "runs", name, "weights", "best.pt")):
        if os.path.isfile(cand):
            return norm(cand)
    raise ApiError(f"no model file for run {name}")


def launch_tool(settings, req):
    repo, py = settings.get("trainer_repo") or "", settings.get("trainer_python") or ""
    name = req.get("run", "")
    if not NAME_RE.match(name):
        raise ApiError("bad run name")
    model = model_for_run(settings, name)
    if req.get("tool") == "analyse":
        cmd = [py, "analyse_model.py", "--model", model]
    elif req.get("tool") == "evaluate":
        data = (req.get("data") or "").strip()
        if not os.path.isfile(data):
            raise ApiError("choose the real dataset's data.yaml")
        script = "evaluate_real.py"
        if not os.path.isfile(os.path.join(repo, script)):
            raise ApiError(f"{script} not found in {repo}")
        splits = (req.get("splits") or "valid test").split()
        cmd = [py, script, "--model", model, "--data", data, "--splits", *splits,
               "--out", norm(os.path.join(repo, "runs", name, "real_eval"))]
        settings.update(real_data=norm(data))
    else:
        raise ApiError("unknown tool")
    _launch_console(f"{req['tool']} {name}", cmd, repo)
    return {"ok": True, "command": " ".join(_quote(a) for a in cmd)}


# ============================================================================ desktop helpers
_dialog_lock = threading.Lock()


def browse(kind, initial=""):
    """Native file / folder dialog (this app runs on the same PC as the browser)."""
    import tkinter
    from tkinter import filedialog
    with _dialog_lock:
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        init = initial if initial and os.path.isdir(initial) else (os.path.dirname(initial) if initial else "")
        try:
            if kind == "folder":
                p = filedialog.askdirectory(parent=root, initialdir=init or None, title="Choose a folder")
            elif kind == "yaml":
                p = filedialog.askopenfilename(parent=root, initialdir=init or None, title="Choose data.yaml",
                                               filetypes=[("Dataset yaml", "*.yaml *.yml"), ("All files", "*.*")])
            elif kind == "model":
                p = filedialog.askopenfilename(parent=root, initialdir=init or None, title="Choose a model",
                                               filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")])
            elif kind == "python":
                p = filedialog.askopenfilename(parent=root, initialdir=init or None, title="Choose python.exe",
                                               filetypes=[("Python", "python*.exe"), ("All files", "*.*")])
            else:
                p = filedialog.askopenfilename(parent=root, initialdir=init or None, title="Choose a dataset zip or any file inside the dataset",
                                               filetypes=[("Dataset zip", "*.zip"), ("All files", "*.*")])
        finally:
            root.destroy()
    return norm(p) if p else ""


def open_path(path):
    if not path or not os.path.exists(path):
        raise ApiError(f"not found: {path}")
    if os.name != "nt":
        raise ApiError("only on Windows")
    if os.path.isdir(path):
        os.startfile(path)
    else:
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    return {"ok": True}


# ============================================================================ http
class ApiError(Exception):
    pass


class App:
    def __init__(self):
        self.settings = Settings()


CT = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
      ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon", ".json": "application/json"}


class Handler(BaseHTTPRequestHandler):
    server_version = "SDGApp/1.0"
    app: App = None

    def log_message(self, fmt, *args):             # quiet: only errors are printed
        pass

    # --- plumbing
    def _allowed(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return False                            # blocks DNS-rebinding style requests from web pages
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
            return False
        return True

    def _send(self, code, body, ctype="application/json", cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=str).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise ApiError("expected JSON")
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _file(self, path, cache=False):
        if not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        with open(path, "rb") as f:
            data = f.read()
        self._send(200, data, CT.get(os.path.splitext(path)[1].lower(), "application/octet-stream"), cache)

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def _route(self, method):
        if not self._allowed():
            return self._send(403, {"error": "forbidden"})
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if method == "GET":
                return self._get(u.path, q)
            return self._post(u.path, self._json_body())
        except ApiError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    # --- routes
    def _get(self, path, q):
        a = self.app
        if path in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"))
        if path.startswith("/static/"):
            name = path[len("/static/"):]
            if ".." in name or name.startswith(("/", "\\")):
                return self._send(404, {"error": "not found"})
            return self._file(os.path.join(STATIC, name))
        if path == "/api/ping":
            return self._send(200, {"app": "sdg", "pid": os.getpid()})
        if path == "/api/state":
            return self._send(200, {"unreal": unreal_status(), "configs": list_configs(), "settings": a.settings.data,
                                    "listener": LISTENER, "project": norm(PROJECT_DIR)})
        if path == "/api/schema":
            return self._send(200, schema())
        if path == "/api/config":
            return self._send(200, read_config(q.get("name", "")))
        if path == "/api/unreal/result":
            return self._send(200, unreal_result(q.get("id", "")))
        if path == "/api/run":
            return self._send(200, run_state(q.get("config", "")))
        if path == "/api/log":
            return self._send(200, log_tail(int(q.get("since", -1)), a.settings))
        if path == "/files/preview":
            cfg = load_cfg(q.get("config", ""))
            f = os.path.basename(q.get("f", ""))
            if not cfg or not f.lower().endswith(".jpg"):
                return self._send(404, {"error": "not found"})
            return self._file(os.path.join(cfg["output"]["dataset_dir"], cfg["name"], "previews", f), cache=True)
        if path == "/api/datasets":
            return self._send(200, discover_datasets(a.settings))
        if path == "/api/dataset":
            return self._send(200, dataset_info(q.get("path", ""), detail=True))
        if path == "/api/train/options":
            return self._send(200, train_options(a.settings))
        if path == "/api/train/runs":
            return self._send(200, list_runs(a.settings))
        if path == "/api/train/run":
            return self._send(200, run_detail(a.settings, q.get("name", "")))
        return self._send(404, {"error": f"no route {path}"})

    def _post(self, path, body):
        a = self.app
        if path == "/api/config/check":
            text, errors, summary = check_config(body.get("raw", {}))
            return self._send(200, {"ok": not errors, "errors": errors, "text": text, "summary": summary})
        if path == "/api/config/save":
            return self._send(200, save_config(body.get("name", ""), body.get("raw", {})))
        if path == "/api/toml/parse":
            v, err = parse_toml_value(body.get("text", ""))
            return self._send(200, {"ok": err is None, "value": v, "error": err})
        if path == "/api/unreal":
            return self._send(200, {"id": send_unreal(body.get("command", ""), body.get("args", []))})
        if path == "/api/train/start":
            return self._send(200, start_training(body, a.settings))
        if path == "/api/train/tool":
            return self._send(200, launch_tool(a.settings, body))
        if path == "/api/settings":
            a.settings.update(**{k: v for k, v in body.items() if k != "recent_datasets"})
            return self._send(200, {"ok": True, "settings": a.settings.data})
        if path == "/api/browse":
            return self._send(200, {"path": browse(body.get("kind", "file"), body.get("initial", ""))})
        if path == "/api/open":
            return self._send(200, open_path(body.get("path", "")))
        return self._send(404, {"error": f"no route {path}"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"     # on Windows SO_REUSEADDR lets a 2nd server share the port

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_server(port):
    Handler.app = App()
    return Server(("127.0.0.1", port), Handler)


def _edge():
    for base in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
        if base:
            p = os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe")
            if os.path.isfile(p):
                return p
    return None


def open_window(url):
    edge = _edge()
    if edge:                                      # a chrome-less app window
        subprocess.Popen([edge, f"--app={url}", "--window-size=1480,980"])
    else:
        webbrowser.open(url)


def _ours(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=2) as r:
            return json.load(r).get("app") == "sdg"
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--open", default="", help="page to open: recipe, render or train")
    ap.add_argument("--dataset", default="", help="with --open train: preselect this dataset")
    args = ap.parse_args()
    frag = args.open if args.open in ("recipe", "render", "train") else ""
    if frag and args.dataset:
        frag += "?dataset=" + quote(norm(os.path.abspath(args.dataset)))
    url = f"http://127.0.0.1:{args.port}/" + (f"#{frag}" if frag else "")
    if _ours(args.port):
        print(f"The SDG app is already running - opening {url}")
        if not args.no_browser:
            open_window(url)
        return
    try:
        server = make_server(args.port)
    except OSError:
        sys.exit(f"Port {args.port} is used by another program: start with --port <another number>")
    print(f"SDG app: {url}\n  Close this window to stop the app.")
    if not args.no_browser:
        threading.Timer(0.6, open_window, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
