"""
Minimal TOML writer for SDG configs (the standard library's tomllib only reads).

Top-level values first, then [tables] and [[arrays of tables]] in the order a hand-written config
uses. Nested dicts and lists are written inline. None means "not set" and is left out, so the
pipeline's default applies.
"""
import json
import math
import re

HEADER = "# SDG dataset recipe - written by the SDG app (SegPipeline/app)."
ORDER = ["name", "map", "seed", "output", "render", "post", "classes", "props", "camera", "randomize",
         "water_fog", "stages"]
NOTES = {
    "output": "where the dataset goes and what gets a label",
    "render": "Movie Render Queue settings",
    "post": "the dataset builder, and what happens after a full render",
    "classes": "YOLO ids follow list order: append new classes at the END",
    "props": "which objects in the level get labels",
    "camera": "where the camera may be and how it frames the props",
    "randomize": "one block per randomized variable",
    "water_fog": "extra underwater haze (M_SDG_WaterFog)",
    "stages": "batches rendered back to back into one dataset",
}
_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def key(k):
    return k if _BARE.match(k) else json.dumps(k, ensure_ascii=False)


def value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"{v} has no TOML form here")
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)      # JSON escapes are valid TOML basic-string escapes
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(value(x) for x in v) + "]"
    if isinstance(v, dict):
        items = [f"{key(str(k))} = {value(x)}" for k, x in v.items() if x is not None]
        return "{ " + ", ".join(items) + " }" if items else "{}"
    raise ValueError(f"can't write {type(v).__name__} {v!r} as TOML")


def _is_table_array(v):
    return isinstance(v, list) and v and all(isinstance(x, dict) for x in v)


def dumps(cfg, header=HEADER):
    lines = [header, ""] if header else []
    keys = [k for k in ORDER if k in cfg] + [k for k in cfg if k not in ORDER]
    scalars = [k for k in keys if cfg[k] is not None and not isinstance(cfg[k], dict)
               and not _is_table_array(cfg[k]) and cfg[k] != []]
    for k in scalars:
        lines.append(f"{key(k)} = {value(cfg[k])}")
    for k in keys:
        v = cfg[k]
        if isinstance(v, dict):
            lines += ["", f"# ---- {k}: {NOTES[k]}" if k in NOTES else f"# ---- {k}", f"[{key(k)}]"]
            lines += [f"{key(str(kk))} = {value(vv)}" for kk, vv in v.items() if vv is not None]
        elif _is_table_array(v):
            lines += ["", f"# ---- {k}: {NOTES[k]}" if k in NOTES else f"# ---- {k}"]
            for item in v:
                lines.append(f"[[{key(k)}]]")
                lines += [f"{key(str(kk))} = {value(vv)}" for kk, vv in item.items() if vv is not None]
                lines.append("")
            if lines[-1] == "":
                lines.pop()
    return "\n".join(lines) + "\n"
