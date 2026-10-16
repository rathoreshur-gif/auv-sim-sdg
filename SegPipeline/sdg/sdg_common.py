"""
Shared by both halves of the pipeline:
  - the Unreal side (editor Python 3.11, standard library only - no numpy)
  - the dataset side (SegPipeline venv)

Config loading and validation, seeded random sampling, and Unreal's coordinate conventions:
cm, left-handed, X forward, Y right, Z up. Rotators are (pitch, yaw, roll) in degrees; their
FRotationMatrix rows are the forward, right and up axes in world space.
"""
from __future__ import annotations

import copy
import fnmatch
import math
import os
import random
import re
import tomllib

SDG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SDG_DIR)             # SegPipeline/ - relative paths in configs start here

LABEL_PASS = "SDGLabels"                    # MRQ post-process pass name = EXR layer name suffix
MAX_STENCIL = 255
CAMERA_MODES = ("random_path", "teleport", "sequence")


class ConfigError(Exception):
    pass


# ============================================================================ config
DEFAULTS = {
    "seed": 0,
    "output": {
        "raw_dir": "C:/sdg/raw",            # MRQ EXRs land in <raw_dir>/<name>/<stage>/ (deleted once converted)
        "dataset_dir": "C:/sdg/datasets",   # dataset lands in <dataset_dir>/<name>/
        "val_fraction": 0.1,
        "jpg_quality": 95,
        "keep_exr": False,
        "previews": 30,                     # overlay images per stage
        "min_area": 60,                     # px: smaller visible parts get no label
        "epsilon": 1.0,                     # px: polygon simplification
        "empty_keep": 0.2,                  # fraction of label-less frames kept as backgrounds
        "max_label_depth": 0.0,             # cm: objects farther than this get no label (0 = off)
    },
    "render": {
        "resolution": [1280, 720],
        "executor": "pie",                  # pie = inside the editor | process = separate game process
        "fps": 30,                          # world time step per frame (water, caustics, particles)
        "anti_aliasing": "tsr",             # tsr | taa | fxaa | none
        "warmup_frames": 32,                # rendered but not saved at the start of each stage
        "texture_streaming": "fully_load",  # fully_load | disabled | none
        "exr_compression": "zip",           # zip | zips | none (the dataset reader handles these)
        "cinematic": False,                 # MRQ cinematic scalability (slower, prettier)
        "cvars": {},
    },
    "post": {
        "python": "",                       # interpreter with numpy + opencv; "" = default_python()
        "workers": 6,
        "train": "",                        # "", "seg", "pose", "both" or "quick": start training afterwards
                                            # (quick = seg, 30 epochs, scored on the real datasets set in the app)
        "train_args": "",
        "open_trainer": False,              # after a full render: open the SDG app's Train page with this dataset
    },
    "camera": {
        "mode": "random_path",
        "fov": [80.0, 80.0],                # horizontal FOV range, degrees
        "volume": None,                     # {min=[x,y,z], max=[x,y,z]} or {actor="Label*", margin=cm}
        "targets": "props",                 # "props" or a list of world points [[x,y,z], ...]
        "distance": [150.0, 800.0],         # cm from the target
        "elevation": [-20.0, 30.0],         # deg: camera height angle seen from the target
        "azimuth": None,                    # deg range around the target's `facing` side, e.g. [-70, 70]; None = all around
        "facing": "auto",                   # side azimuth 0 looks at: auto (thinnest axis, the face4 side), -auto,
                                            # +x/-x/+y/-y/+z/-z (actor axes); point targets: world +X
        "aim_jitter": 8.0,                  # deg: how far the target may sit off-centre
        "roll": [0.0, 0.0],
        "speed": [10.0, 40.0],              # random_path: cm per frame
        "turn": 60.0,                       # random_path: max change of view angle between waypoints (deg)
        "switch_target": 0.3,               # random_path: chance to fly to another target at each waypoint
        "hold": 1,                          # teleport: frames per pose; only the last one is kept
        "sequence": "",                     # sequence: level sequence with the planned camera
        "sample_fps": 0,                    # sequence: resample the path at this rate (0 = its display rate)
        "repeat": 1,                        # sequence: play the path this many times (new randomization each)
        "jitter_pos": 0.0,                  # sequence: smooth random offset, cm
        "jitter_rot": 0.0,                  # sequence: smooth random offset, deg
        "fov_every": 30,                    # sequence: frames between FOV changes (smooth)
    },
    "water_fog": {
        "enabled": False,                   # extra underwater haze over the whole image (SDG_PostProcess)
        "density": [0.0, 0.003],            # 1/cm; 0.002 halves the light every ~3.5 m
        "color": {"hsv": {"h": [0.45, 0.58], "s": [0.3, 0.8], "v": [0.05, 0.3]}},
        "every": 1,
        "smooth": True,
        "min_visibility": 0.15,             # objects dimmed below this fraction of their light get no label
    },
    "classes": {"seg": [], "pose": []},     # YOLO ids follow list order
    "props": [],
    "randomize": [],
    "stages": [],
}
TOP_KEYS = {"name", "map", "seed"} | set(DEFAULTS)

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
RANDOMIZER_TYPES = ("float", "double", "bool", "color", "vector", "transform", "visible", "mpc",
                    "material_param", "material", "postprocess", "image")
SDG_MPC = "/Game/SDG/MPC_SDG"
FACINGS = ("auto", "-auto", "+x", "-x", "+y", "-y", "+z", "-z")
FOG_DENSITY, FOG_COLOR = "water_fog_density", "water_fog_color"
POST_TRAIN = ("", "seg", "pose", "both", "quick")


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not _is_value_spec(v):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_path(p):
    """Config paths: absolute, or relative to the SegPipeline folder."""
    p = os.path.expandvars(os.path.expanduser(p))
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p)).replace("\\", "/")


def default_python():
    """Interpreter with numpy + opencv when post.python is empty: $SDG_PYTHON, else SegPipeline/.venv,
    else ~/seg-train/.venv. "" if none exists."""
    cands = [os.environ.get("SDG_PYTHON", ""),
             os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
             os.path.join(os.path.expanduser("~"), "seg-train", ".venv", "Scripts", "python.exe")]
    return next((c.replace("\\", "/") for c in cands if c and os.path.isfile(c)), "")


def app_settings():
    """The SDG app's settings (app/state/settings.json), {} if it was never saved."""
    path = os.path.join(os.environ.get("SDG_APP_STATE", os.path.join(ROOT, "app", "state")), "settings.json")
    try:
        import json
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def default_real_sets():
    """Real labelled datasets to score models on when none are set in the app: the Roboflow exports
    of the torpedo board in Downloads (unpacked folder or the zip)."""
    dl = os.path.join(os.path.expanduser("~"), "Downloads")
    out = []
    for stem in ("Torpedo_RS.v1i.yolov11", "Tropedo.v1i.yolov11"):
        folder, z = os.path.join(dl, stem, "data.yaml"), os.path.join(dl, stem + ".zip")
        p = folder if os.path.isfile(folder) else z if os.path.isfile(z) else None
        if p:
            out.append(p.replace("\\", "/"))
    return out


def merge_real_sets(sets, extra):
    """sets + extra (a path, or None) without duplicates (paths compared case- and slash-insensitively)."""
    out, seen = [], set()
    for p in list(sets or []) + ([extra] if extra else []):
        key = os.path.normcase(os.path.normpath(str(p)))
        if p and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def real_sets():
    """Real datasets from the app's settings (real_sets), else the defaults plus an older app's single real_data."""
    s = app_settings()
    sets = s.get("real_sets")
    if sets is None:
        sets = merge_real_sets(default_real_sets(), s.get("real_data"))
    return [p for p in sets if p and os.path.exists(p)]


def real_unstretch():
    """"W H" the Roboflow-stretched real images are resized back to before scoring ("" = off)."""
    return str(app_settings().get("real_unstretch", "1280 720") or "")


def find_config(arg):
    """Accept an absolute path, a path relative to cwd / SegPipeline / sdg, or a bare name."""
    cands = [arg, os.path.join(ROOT, arg), os.path.join(SDG_DIR, arg), os.path.join(SDG_DIR, "configs", arg),
             os.path.join(SDG_DIR, "configs", arg + ".toml")]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise ConfigError(f"config not found: {arg} (looked in {ROOT} and {SDG_DIR}/configs)")


def load_config(path):
    path = find_config(path)
    with open(path, encoding="utf-8-sig") as f:        # -sig: some Windows editors add a BOM
        try:
            user = tomllib.loads(f.read())
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{os.path.basename(path)}: {e}") from None
    check_keys(user)
    cfg = _merge(DEFAULTS, user)
    cfg["_path"] = path.replace("\\", "/")
    cfg["seg_classes"], cfg["pose_classes"] = list(cfg["classes"]["seg"]), list(cfg["classes"]["pose"])
    for sec, key in (("output", "raw_dir"), ("output", "dataset_dir"), ("post", "python")):
        if cfg[sec][key]:
            cfg[sec][key] = resolve_path(cfg[sec][key])
    if not cfg["post"]["python"]:
        cfg["post"]["python"] = default_python()
    for rz in cfg["randomize"]:             # "image" without a value: every image equally often
        if randomizer_type(rz) == "image" and "value" not in rz:
            prop = next((p for p in cfg["props"] if p.get("name") == rz.get("prop")), None)
            if prop is not None and isinstance(prop.get("images"), list):
                rz["value"] = {"choice": list(range(len(prop["images"]) + 1))}
    fog = cfg["water_fog"]
    if fog["enabled"]:                      # sugar for two MPC randomizers driving M_SDG_WaterFog
        base = {"mpc": SDG_MPC, "every": fog["every"], "smooth": fog["smooth"]}
        # default = what the editor shows outside a render: no haze
        cfg["randomize"] = [dict(base, name=FOG_DENSITY, param="FogDensity", value=fog["density"], default=0.0),
                            dict(base, name=FOG_COLOR, param="FogColor", value=fog["color"], default=[0.5, 0.5, 0.5])] + cfg["randomize"]
    validate(cfg)
    return cfg


def _err(where, msg):
    raise ConfigError(f"{where}: {msg}")


def check_keys(user):
    """Typos and misplaced keys fail loudly. (In TOML, a key written below a [table] header belongs
    to that table - a classic way to lose a setting without noticing.)"""
    for k in user:
        if k not in TOP_KEYS:
            _err(k, f"unknown top-level key (known: {', '.join(sorted(TOP_KEYS))})")
    for sec, defaults in DEFAULTS.items():
        if isinstance(defaults, dict) and isinstance(user.get(sec), dict):
            for k in user[sec]:
                if k not in defaults:
                    hint = " - top-level keys (name, map, seed) must come before the first [table]" if k in TOP_KEYS else ""
                    _err(f"[{sec}] {k}", f"unknown key{hint}; known: {', '.join(sorted(defaults))}")


def _range2(where, v, lo=None):
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(_is_num(x) for x in v)):
        _err(where, f"expected [min, max], got {v!r}")
    if v[0] > v[1]:
        _err(where, f"min > max in {v!r}")
    if lo is not None and v[0] < lo:
        _err(where, f"values must be >= {lo}, got {v!r}")


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(cfg):
    if not _NAME_RE.match(str(cfg.get("name", ""))):
        _err("name", "required; letters, digits, _ and - only")
    if not str(cfg.get("map", "")).startswith("/"):
        _err("map", "required, e.g. \"/Game/volume/water_volume\"")

    r = cfg["render"]
    if not (isinstance(r["resolution"], list) and len(r["resolution"]) == 2 and all(isinstance(v, int) and v >= 16 for v in r["resolution"])):
        _err("render.resolution", "expected [width, height]")
    if r["executor"] not in ("pie", "process"):
        _err("render.executor", "pie or process")
    if r["anti_aliasing"] not in ("tsr", "taa", "fxaa", "none"):
        _err("render.anti_aliasing", "tsr, taa, fxaa or none")
    if r["exr_compression"] not in ("zip", "zips", "none"):
        _err("render.exr_compression", "zip, zips or none (PIZ/DWA can't be read by the dataset builder)")

    if cfg["post"]["train"] not in POST_TRAIN:
        _err("post.train", f"one of {', '.join(repr(t) for t in POST_TRAIN)}")

    seg, pose = cfg["seg_classes"], cfg["pose_classes"]
    for key, names in (("seg_classes", seg), ("pose_classes", pose)):
        if len(set(names)) != len(names):
            _err(key, "duplicate class name")

    prop_names = set()
    for i, p in enumerate(cfg["props"]):
        where = f"props[{i}] ({p.get('name', '?')})"
        if not _NAME_RE.match(str(p.get("name", ""))):
            _err(where, "needs a \"name\" (letters, digits, _ -)")
        if p["name"] in prop_names:
            _err(where, "duplicate prop name")
        prop_names.add(p["name"])
        kind = p.get("type")
        if kind == "actor":
            if p.get("class") not in seg and p.get("class") is not None:
                _err(where, f"class {p.get('class')!r} is not in seg_classes")
            if not (p.get("actor_label") or p.get("actor_tag") or p.get("static_mesh")):
                _err(where, "give actor_label, actor_tag and/or static_mesh")
        elif kind == "texture_regions":
            if not p.get("texture") or not isinstance(p.get("regions"), dict) or not p["regions"]:
                _err(where, "needs \"texture\" and \"regions\" = { class = \"mask.png\" }")
            for cls, png in p["regions"].items():
                if cls not in seg:
                    _err(where, f"region class {cls!r} is not in seg_classes")
                if not os.path.isfile(resolve_path(png)):
                    _err(where, f"mask not found: {resolve_path(png)}")
            _check_images(where, p, seg)
        elif kind == "material_slots":
            if not isinstance(p.get("slots"), dict) or not p["slots"]:
                _err(where, "needs \"slots\" = { \"*Pattern*\" = \"class\" }")
            if not (p.get("static_mesh") or p.get("actor_label")):
                _err(where, "give static_mesh and/or actor_label")
            for cls in p["slots"].values():
                if cls not in seg:
                    _err(where, f"slot class {cls!r} is not in seg_classes")
        else:
            _err(where, "type must be actor, texture_regions or material_slots")
        if kind != "texture_regions" and "images" in p:
            _err(where, "images only works for texture_regions props")
        if p.get("body_class") is not None and p["body_class"] not in seg:
            _err(where, f"body_class {p['body_class']!r} is not in seg_classes")
        if p.get("pose_class") is not None:
            if p["pose_class"] not in pose:
                _err(where, f"pose_class {p['pose_class']!r} is not in pose_classes")
            if not p.get("keypoints"):
                _err(where, "pose_class needs keypoints (e.g. keypoints = \"face4\")")
            parse_keypoint_spec(where, p["keypoints"])

    rnames = set()
    for i, rz in enumerate(cfg["randomize"]):
        where = f"randomize[{i}] ({rz.get('name', '?')})"
        if not _NAME_RE.match(str(rz.get("name", ""))):
            _err(where, "needs a \"name\"")
        if rz["name"] in rnames:
            _err(where, "duplicate name")
        rnames.add(rz["name"])
        kind = randomizer_type(rz)
        if kind not in RANDOMIZER_TYPES:
            _err(where, f"unknown type {kind!r}; one of {', '.join(RANDOMIZER_TYPES)}")
        if kind == "mpc" and not (rz.get("mpc") and rz.get("param")):
            _err(where, "mpc randomizers need mpc = \"/Game/...\" and param = \"Name\"")
        if kind in ("material_param", "material") and "material_slot" not in rz:
            _err(where, "needs material_slot (index)")
        if kind in ("float", "double", "bool", "color", "material_param") and not rz.get("property") and not rz.get("param"):
            _err(where, "needs property = \"Name\"")
        if kind == "image":
            prop = next((p for p in cfg["props"] if p["name"] == rz.get("prop")), None)
            if prop is None or not prop.get("images"):
                _err(where, "needs prop = \"<name>\" of a texture_regions prop that has images = [...]")
            _check_image_value(where, rz.get("value"), len(prop["images"]) + 1)
        elif kind not in ("mpc", "postprocess") and not (rz.get("actor") or rz.get("tag")):
            _err(where, "needs actor = \"Label*\" (or tag = \"...\")")
        if kind == "transform" and not any(k in rz for k in ("location", "rotation", "scale")):
            _err(where, "transform needs location, rotation and/or scale")
        if kind not in ("transform", "visible", "image") and "value" not in rz:
            _err(where, "needs value = [min, max] (or another value spec)")
        v = rz.get("value")
        if isinstance(v, list) and len(v) == 2 and all(_is_num(x) for x in v) and v[0] > v[1]:
            _err(where, f"value: min > max in {v!r}")
        if not isinstance(rz.get("enabled", True), bool):
            _err(where, "enabled must be true or false")
        if int(rz.get("every", 1)) < 1:
            _err(where, "every must be >= 1")
        if rz.get("mode", "set") not in ("set", "add", "multiply"):
            _err(where, "mode must be set, add or multiply")
        if rz.get("mode", "set") != "set" and kind not in ("float", "double", "vector", "postprocess"):
            _err(where, "mode add/multiply works for float, double, vector and postprocess values")

    if not cfg["stages"]:
        _err("stages", "add at least one [[stages]] block")
    snames = set()
    for i, st in enumerate(cfg["stages"]):
        where = f"stages[{i}] ({st.get('name', '?')})"
        if not _NAME_RE.match(str(st.get("name", ""))):
            _err(where, "needs a \"name\"")
        if st["name"] in snames:
            _err(where, "duplicate stage name")
        snames.add(st["name"])
        for k in st:
            if k not in ("name", "frames", "camera", "override", "disable"):
                _err(where, f"unknown key {k!r} (name, frames, camera, override, disable)")
        for k in st.get("camera", {}):
            if k not in DEFAULTS["camera"]:
                _err(where, f"unknown camera key {k!r}; known: {', '.join(sorted(DEFAULTS['camera']))}")
        cam = stage_camera(cfg, st)
        az = cam.get("azimuth")
        if az is not None and not (isinstance(az, list) and len(az) == 2 and all(isinstance(v, (int, float)) for v in az)
                                   and az[0] <= az[1] and az[1] - az[0] <= 360):
            _err(where, "camera.azimuth: [min, max] in degrees, e.g. [-70, 70]")
        if cam["facing"] not in FACINGS:
            _err(where, f"camera.facing must be one of {', '.join(FACINGS)}")
        if cam["mode"] not in CAMERA_MODES:
            _err(where, f"camera.mode must be one of {', '.join(CAMERA_MODES)}")
        if cam["mode"] == "sequence":
            if not str(cam.get("sequence", "")).startswith("/"):
                _err(where, "camera.sequence = \"/Game/...\" is required in sequence mode")
        else:
            if not isinstance(st.get("frames"), int) or st["frames"] < 1:
                _err(where, "frames = <number of images> is required")
            _range2(where + ".camera.distance", cam["distance"], 1)
            _range2(where + ".camera.elevation", cam["elevation"])
            _range2(where + ".camera.speed", cam["speed"], 0)
            if cam["targets"] != "props" and not (isinstance(cam["targets"], list) and cam["targets"]):
                _err(where + ".camera.targets", "\"props\" or a list of [x, y, z] points")
        if cam["fov"] == "sequence":
            if cam["mode"] != "sequence":
                _err(where, "camera.fov = \"sequence\" only works in sequence mode")
        else:
            _range2(where + ".camera.fov", cam["fov"], 1)
        if int(cam.get("hold", 1)) < 1:
            _err(where, "camera.hold must be >= 1")
        for key in st.get("override", {}):
            if key not in rnames:
                _err(where, f"override {key!r} is not a randomizer name")
        for key in st.get("disable", []):
            if key not in rnames:
                _err(where, f"disable {key!r} is not a randomizer name")
        for rz in stage_randomizers(cfg, st):
            if randomizer_type(rz) == "image" and rz["name"] in st.get("override", {}):
                prop = next(p for p in cfg["props"] if p["name"] == rz["prop"])
                _check_image_value(f"{where} override {rz['name']}", rz.get("value"), len(prop["images"]) + 1)


def _check_images(where, p, seg):
    """images = ["props/<prop>/images/<name>/texture.png", ...]: one folder per image, its masks next to it."""
    if "images" not in p:
        return
    imgs = p["images"]
    if not isinstance(imgs, list) or not all(isinstance(x, str) for x in imgs):
        _err(where, "images = [\"props/<prop>/images/<name>/texture.png\", ...]")
    folders = set()
    for rel in imgs:
        path = resolve_path(rel)
        if not os.path.isfile(path):
            _err(where, f"image not found: {path}")
        folder = os.path.dirname(path)
        if folder in folders:
            _err(where, f"two images in {folder}: give every image its own folder (its T_Mask_<class>.png sit next to it)")
        folders.add(folder)
        if not any(os.path.isfile(f"{folder}/T_Mask_{c}.png") for c in seg):
            _err(where, f"no T_Mask_<class>.png next to {path} - make them with tools/prop_image.py")


def prop_images(prop, seg):
    """What a texture_regions prop can show, one entry per image number:
    0 = `texture` as it is in the level (masks from `regions`), 1.. = the `images` entries in order
    (masks = T_Mask_<class>.png next to each image; a class without a mask is not on that image).
    -> [{"image": png path or None for 0, "regions": {class: mask png path}}]"""
    out = [{"image": None, "regions": {c: resolve_path(m) for c, m in prop["regions"].items()}}]
    for rel in prop.get("images", []):
        path = resolve_path(rel)
        folder = os.path.dirname(path)
        out.append({"image": path, "regions": {c: f"{folder}/T_Mask_{c}.png" for c in seg
                                               if os.path.isfile(f"{folder}/T_Mask_{c}.png")}})
    return out


def _check_image_value(where, v, count):
    """Values of an "image" randomizer are image numbers 0 .. count-1."""
    msg = (f"value: image numbers 0..{count - 1} (0 = the prop's texture, 1.. = its images), "
           f"e.g. {{ choice = {list(range(count))} }} or {{ choice = [...], weights = [...] }}")
    if v is None:
        return
    if isinstance(v, dict) and "choice" in v:
        vals = v["choice"]
        if not isinstance(vals, list) or not vals:
            _err(where, msg)
        w = v.get("weights")
        if w is not None and (not isinstance(w, list) or len(w) != len(vals) or not all(_is_num(x) and x >= 0 for x in w)
                              or not any(x > 0 for x in w)):
            _err(where, "weights: one number >= 0 per choice")
    elif isinstance(v, dict) and "randint" in v:
        vals = v["randint"]
        if not (isinstance(vals, list) and len(vals) == 2 and vals[0] <= vals[1]):
            _err(where, msg)
    else:
        vals = [v]
    if not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x < count for x in vals):
        _err(where, msg)


def stage_camera(cfg, stage):
    return _merge(cfg["camera"], stage.get("camera", {}))


def stage_randomizers(cfg, stage):
    """Randomizers active in a stage, with the stage's overrides applied to their values.
    enabled = false switches a randomizer off everywhere; a stage's disable list only there."""
    out = []
    for rz in cfg["randomize"]:
        if rz.get("enabled", True) is False or rz["name"] in stage.get("disable", []):
            continue
        rz = copy.deepcopy(rz)
        ov = stage.get("override", {}).get(rz["name"])
        if ov is not None:
            if isinstance(ov, dict) and not _is_value_spec(ov):
                rz.update(ov)                     # e.g. { value = [..], every = 5 }
            else:
                rz["value"] = ov
        out.append(rz)
    return out


def _is_value_spec(d):
    return any(k in d for k in ("uniform", "normal", "loguniform", "choice", "randint", "hsv", "rgb", "kelvin"))


def randomizer_type(rz):
    if "type" in rz:
        return rz["type"]
    if "mpc" in rz:
        return "mpc"
    if "postprocess" in rz:
        return "postprocess"
    if "material_slot" in rz:
        return "material_param"
    return "float"


# ============================================================================ sampling
class Sampler:
    """Seeded sampling of config value specs.

    number / bool / string          constant
    [min, max]                      uniform
    {uniform = [a, b]}              uniform
    {normal = [mean, std], clip = [lo, hi]}
    {loguniform = [a, b]}
    {randint = [a, b]}              inclusive
    {choice = [...], weights = [...]}
    {hsv = {h = [..], s = [..], v = [..]}}    colour, returns (r, g, b)
    {rgb = [[r0, r1], [g0, g1], [b0, b1]]}    colour
    {kelvin = [k0, k1]}                        black-body colour
    """

    def __init__(self, seed):
        self.rng = random.Random(str(seed))

    def uniform(self, lo, hi):
        return self.rng.uniform(lo, hi)

    def range(self, v):
        """[a, b] -> uniform, number -> itself."""
        return self.rng.uniform(v[0], v[1]) if isinstance(v, (list, tuple)) else v

    def value(self, spec):
        rng = self.rng
        if isinstance(spec, (bool, int, float, str)):
            return spec
        if isinstance(spec, (list, tuple)):
            if len(spec) == 2 and all(_is_num(v) for v in spec):
                return rng.uniform(spec[0], spec[1])
            raise ConfigError(f"value {spec!r}: a list must be [min, max]")
        if isinstance(spec, dict):
            if "uniform" in spec:
                return rng.uniform(*spec["uniform"])
            if "normal" in spec:
                mean, std = spec["normal"]
                v = rng.gauss(mean, std)
                if "clip" in spec:
                    v = min(max(v, spec["clip"][0]), spec["clip"][1])
                return v
            if "loguniform" in spec:
                lo, hi = spec["loguniform"]
                return math.exp(rng.uniform(math.log(lo), math.log(hi)))
            if "randint" in spec:
                return rng.randint(*spec["randint"])
            if "choice" in spec:
                return rng.choices(spec["choice"], weights=spec.get("weights"))[0]
            if "hsv" in spec:
                h = spec["hsv"]
                return hsv_to_rgb(self.range(h.get("h", 0.0)) % 1.0, self.range(h.get("s", 0.0)), self.range(h.get("v", 1.0)))
            if "rgb" in spec:
                return tuple(self.range(c) for c in spec["rgb"])
            if "kelvin" in spec:
                return kelvin_to_rgb(self.range(spec["kelvin"]))
        raise ConfigError(f"can't sample value spec {spec!r}")

    def series(self, spec, n, every=1, smooth=False):
        """n values; a new sample every `every` steps, linearly blended in between if smooth."""
        every = max(1, int(every))
        anchors = [self.value(spec) for _ in range(n // every + 2)]
        out = []
        for i in range(n):
            k, f = divmod(i, every)
            a = anchors[k]
            out.append(lerp(a, anchors[k + 1], f / every) if smooth and every > 1 else a)
        return out


def lerp(a, b, t):
    if _is_num(a) and _is_num(b):
        return a + (b - a) * t
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(x + (y - x) * t for x, y in zip(a, b))
    return a                                  # bools, strings: no blending


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def kelvin_to_rgb(k):
    """Approximate black-body colour (Tanner Helland), normalised to max channel 1."""
    t = k / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    rgb = [min(max(c, 0.0), 255.0) / 255.0 for c in (r, g, b)]
    m = max(rgb) or 1.0
    return tuple(c / m for c in rgb)


def stable_fraction(name):
    """Deterministic [0, 1) from a string (FNV-1a) - same frame name always gets the same split."""
    h = 2166136261
    for ch in name.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return (h % 100000) / 100000.0


def wildcard(name, pattern):
    return fnmatch.fnmatchcase(name.lower(), pattern.lower())


# ============================================================================ unreal math
def rotator_axes(pitch, yaw, roll):
    """Rows of FRotationMatrix(FRotator(pitch, yaw, roll)): forward, right, up in world space."""
    p, y, r = math.radians(pitch), math.radians(yaw), math.radians(roll)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    sr, cr = math.sin(r), math.cos(r)
    fwd = (cp * cy, cp * sy, sp)
    right = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp)
    up = (-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp)
    return fwd, right, up


def look_at(direction):
    """Rotator (pitch, yaw, 0) whose forward axis points along `direction` (FRotationMatrix::MakeFromX)."""
    x, y, z = direction
    return (math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x)), 0.0)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def length(a):
    return math.sqrt(dot(a, a))


def transform_point(loc, rot, scl, p):
    """Actor/component space -> world (FTransform::TransformPosition)."""
    f, r, u = rotator_axes(*rot)
    x, y, z = p[0] * scl[0], p[1] * scl[1], p[2] * scl[2]
    return tuple(loc[i] + x * f[i] + y * r[i] + z * u[i] for i in range(3))


def inverse_transform_point(loc, rot, scl, w):
    f, r, u = rotator_axes(*rot)
    d = sub(w, loc)
    return (dot(d, f) / scl[0], dot(d, r) / scl[1], dot(d, u) / scl[2])


def focal_px(hfov_deg, width):
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def project(point, cam_loc, cam_rot, hfov_deg, width, height):
    """World point -> (u, v, depth) in pixels (u right, v down, pixel centres at +0.5), or None
    if behind the camera. Horizontal FOV spans the full width (AspectRatio_MaintainXFOV)."""
    f, r, u = rotator_axes(*cam_rot)
    d = sub(point, cam_loc)
    x, y, z = dot(d, f), dot(d, r), dot(d, u)
    if x <= 1e-3:
        return None
    fp = focal_px(hfov_deg, width)
    return (width / 2.0 + fp * y / x, height / 2.0 - fp * z / x, x)


def hfov_to_focal_mm(hfov_deg, sensor_width_mm):
    return sensor_width_mm / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def focal_mm_to_hfov(focal_mm, sensor_width_mm):
    return math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal_mm)))


def unwrap_degrees(values):
    """Remove 360 jumps so linearly keyed angles never spin the long way round."""
    out = []
    for v in values:
        if out:
            prev = out[-1]
            v = v + 360.0 * round((prev - v) / 360.0)
        out.append(v)
    return out


# ============================================================================ keypoints
def parse_keypoint_spec(where, spec):
    """keypoints = "face4" | "face4:-x" | "top4" | "bottom4" | "bounds8" | "center"
                   | { sockets = [...] } | { points = [[x, y, z], ...] }   (+ flip_idx = [...])"""
    if isinstance(spec, str):
        spec = {"preset": spec}
    if not isinstance(spec, dict):
        _err(where, f"keypoints: expected a preset name or a table, got {spec!r}")
    kinds = [k for k in ("preset", "sockets", "points") if k in spec]
    if len(kinds) != 1:
        _err(where, "keypoints: give exactly one of preset, sockets, points")
    if "preset" in spec:
        name, _, arg = spec["preset"].partition(":")
        if name not in ("face4", "top4", "bottom4", "bounds8", "center"):
            _err(where, f"keypoints: unknown preset {spec['preset']!r}")
        if arg and not re.match(r"^[+-]?[xyzXYZ]$", arg):
            _err(where, f"keypoints: face axis must look like +x / -y / z, got {arg!r}")
    return spec


def group_stacked(boxes, overlap=0.8, pad=1.0):
    """Group world boxes [(min3, max3), ...] that sit in the same place (e.g. an actor duplicated in
    place): two boxes join when their intersection covers `overlap` of the smaller one (boxes padded
    by `pad` cm so flat meshes have a volume). Returns lists of indices, in input order."""
    def vol(lo, hi):
        return max(hi[0] - lo[0], 0) * max(hi[1] - lo[1], 0) * max(hi[2] - lo[2], 0)
    bx = [([v - pad for v in lo], [v + pad for v in hi]) for lo, hi in boxes]
    parent = list(range(len(bx)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(bx)):
        for j in range(i + 1, len(bx)):
            lo = [max(bx[i][0][k], bx[j][0][k]) for k in range(3)]
            hi = [min(bx[i][1][k], bx[j][1][k]) for k in range(3)]
            small = min(vol(*bx[i]), vol(*bx[j]))
            if small > 0 and vol(lo, hi) >= overlap * small:
                parent[root(j)] = root(i)
    groups = {}
    for i in range(len(bx)):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def facing_axis(facing, bmin, bmax):
    """Actor-space unit vector of the side camera.azimuth is measured from. "auto" = the thinnest
    axis, + side (the side face4 picks), "-auto" the opposite side, or an explicit "+x" ... "-z"."""
    sign = -1.0 if facing.startswith("-") else 1.0
    if facing.lstrip("+-") == "auto":
        axis = min(range(3), key=lambda i: bmax[i] - bmin[i])
    else:
        axis = "xyz".index(facing[-1])
    n = [0.0, 0.0, 0.0]
    n[axis] = sign
    return tuple(n)


def facing_yaw(t, center, axis):
    """World yaw (deg) of an object's facing axis for transform t = [loc3, rot3, scale3], or None
    when that axis is (nearly) vertical and has no horizontal direction."""
    a = transform_point(t[0:3], t[3:6], t[6:9], center)
    b = transform_point(t[0:3], t[3:6], t[6:9], add(tuple(center), axis))
    d = sub(b, a)
    if math.hypot(d[0], d[1]) < 0.25 * max(length(d), 1e-9):
        return None
    return math.degrees(math.atan2(d[1], d[0]))


def preset_keypoints(preset, bmin, bmax):
    """Keypoints on an actor-space box -> (points, default flip_idx).

    face4/top4/bottom4 corners come in the order top-left, top-right, bottom-right, bottom-left as
    seen from outside the face, with local +Z as "up" (+X for the top and bottom faces), so a
    horizontal image flip swaps them as [1, 0, 3, 2]."""
    name, _, arg = preset.partition(":")
    c = [(bmin[i] + bmax[i]) / 2.0 for i in range(3)]
    half = [(bmax[i] - bmin[i]) / 2.0 for i in range(3)]
    if name == "center":
        return [tuple(c)], [0]
    if name == "bounds8":
        pts = []
        for z in (bmin[2], bmax[2]):
            for x, y in ((bmin[0], bmin[1]), (bmax[0], bmin[1]), (bmax[0], bmax[1]), (bmin[0], bmax[1])):
                pts.append((x, y, z))
        return pts, list(range(8))
    if name == "top4":
        axis, sign = 2, 1
    elif name == "bottom4":
        axis, sign = 2, -1
    elif arg:
        axis, sign = "xyz".index(arg[-1].lower()), (-1 if arg.startswith("-") else 1)
    else:
        axis, sign = min(range(3), key=lambda i: half[i]), 1      # thinnest axis = the face normal
    n = [0.0, 0.0, 0.0]
    n[axis] = float(sign)
    up = (0.0, 0.0, 1.0) if axis != 2 else (1.0, 0.0, 0.0)
    right = cross(up, scale(tuple(n), -1.0))                     # viewer looks along -normal
    centre = list(c)
    centre[axis] = bmax[axis] if sign > 0 else bmin[axis]
    ra = half[[abs(v) for v in right].index(1.0)]
    ua = half[[abs(v) for v in up].index(1.0)]
    tl = add(add(tuple(centre), scale(right, -ra)), scale(up, ua))
    tr = add(add(tuple(centre), scale(right, ra)), scale(up, ua))
    br = add(add(tuple(centre), scale(right, ra)), scale(up, -ua))
    bl = add(add(tuple(centre), scale(right, -ra)), scale(up, -ua))
    return [tl, tr, br, bl], [1, 0, 3, 2]
