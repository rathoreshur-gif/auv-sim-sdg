"""
Offline tests for the SDG app (no Unreal, no GPU):
    <venv python> app/tests/test_app.py

Everything runs in a temporary folder: configs, the Unreal bridge queue, app state, a fake
yolo-training repo whose train.py writes a results.csv like Ultralytics does.
"""
import copy
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import tomllib
import traceback
import types
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
REAL_CONFIGS = os.path.join(ROOT, "sdg", "configs")
TMP = tempfile.mkdtemp(prefix="sdg_app_test_")
os.environ.update(SDG_APP_CONFIGS=os.path.join(TMP, "configs"), SDG_BRIDGE_DIR=os.path.join(TMP, "bridge"),
                  SDG_APP_STATE=os.path.join(TMP, "state"), SDG_APP_HIDDEN="1")
os.makedirs(os.environ["SDG_APP_CONFIGS"])
sys.path[:0] = [APP, os.path.join(ROOT, "sdg")]

import server   # noqa: E402
import tomlw    # noqa: E402
import post     # noqa: E402

import cv2      # noqa: E402
import numpy as np  # noqa: E402


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_raw(name):
    with open(os.path.join(REAL_CONFIGS, name + ".toml"), encoding="utf-8-sig") as f:
        return f.read()


def expect_api_error(fn, *a):
    try:
        fn(*a)
    except server.ApiError:
        return
    raise AssertionError(f"{fn.__name__}{a} did not raise ApiError")


# ---------------------------------------------------------------------------- configs
def test_toml_roundtrip():
    for f in os.listdir(REAL_CONFIGS):
        if f.endswith(".toml"):
            raw = tomllib.loads(read_raw(f[:-5]))
            assert tomllib.loads(tomlw.dumps(raw)) == raw, f


def test_config_check_and_save():
    original = read_raw("torpedo")
    write(os.path.join(server.CONFIG_DIR, "torpedo.toml"), original)
    c = server.read_config("torpedo")
    assert not c["generated"]
    raw = c["raw"]
    text, errors, summary = server.check_config(raw)
    assert not errors and summary["images"] > 0, errors

    bad = copy.deepcopy(raw)
    bad["stages"][0].setdefault("camera", {})["distance"] = [900, 50]
    _, errors, _ = server.check_config(bad)
    assert errors and "min > max" in errors[0]["message"], errors
    bad = copy.deepcopy(raw)
    bad["randomize"][1]["value"] = [2.0, 0.5]
    _, errors, _ = server.check_config(bad)
    assert errors and "value: min > max" in errors[0]["message"], errors

    typed = copy.deepcopy(raw)
    typed["seed"] = {"$toml": "7"}
    typed["water_fog"]["color"] = {"$toml": "{ hsv = { h = [0.4, 0.5], s = [0.2, 0.3], v = [0.1, 0.2] } }"}
    text, errors, _ = server.check_config(typed)
    assert not errors and "seed = 7" in text and "h = [0.4, 0.5]" in text
    _, errors, _ = server.check_config({**typed, "seed": {"$toml": "[1, "}})
    assert errors and errors[0]["path"] == "seed"

    raw["seed"] = 3
    raw["randomize"][0]["enabled"] = False
    r = server.save_config("torpedo", raw)
    assert r["ok"] and r["backup"], r
    with open(r["backup"], encoding="utf-8") as f:
        assert f.read() == original                           # hand-written original kept, with comments
    again = server.read_config("torpedo")
    assert again["generated"] and again["raw"]["seed"] == 3 and again["raw"]["randomize"][0]["enabled"] is False
    r2 = server.save_config("torpedo", again["raw"])
    assert r2["ok"] and r2["backup"] is None                  # backup made once only
    expect_api_error(server.config_path, "../evil")
    assert server.list_configs() == ["torpedo"]


# ---------------------------------------------------------------------------- unreal bridge
def fake_unreal(cmdline="UnrealEditor.exe MyProject.uproject"):
    u = types.ModuleType("unreal")

    class Vector:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class Sub:
        saved = 0

        def get_editor_world(self):
            return None

        def get_level_viewport_camera_info(self):
            return (Vector(1.4, 2.6, -3.2), "rotator")

        def is_rendering(self):
            return False

        def get_active_executor(self):
            return None

        def save_current_level(self):
            Sub.saved += 1
            return True

    u.Vector = Vector
    u.UnrealEditorSubsystem = u.MoviePipelineQueueSubsystem = u.LevelEditorSubsystem = object
    u.get_editor_subsystem = lambda cls: Sub()
    u.log = u.log_warning = lambda msg: None
    u.Paths = types.SimpleNamespace(convert_relative_path_to_full=lambda p: p, get_project_file_path=lambda: "MyProject.uproject")
    u.SystemLibrary = types.SimpleNamespace(get_command_line=lambda: cmdline)
    u.register_slate_post_tick_callback = lambda fn: "handle"
    u.unregister_slate_post_tick_callback = lambda h: None
    u._Sub = Sub
    return u


def test_bridge_roundtrip():
    sys.modules["unreal"] = fake_unreal()
    sys.modules.pop("ue_listener", None)
    import ue_listener
    assert ue_listener.BRIDGE == server.BRIDGE
    assert server.unreal_status()["connected"] is False
    ue_listener._heartbeat()
    st = server.unreal_status()
    assert st["connected"] and not st["rendering"], st

    cid = server.send_unreal("where", [])
    assert server.unreal_result(cid)["status"] == "pending"
    ue_listener._process_inbox()
    res = server.unreal_result(cid)
    assert res["status"] == "done" and res["ok"] and res["location"] == [1, 3, -3], res

    calls = []

    def fake_run_path(path, run_name):
        calls.append((path, list(sys.argv)))
        return {"LAST_ERROR": "boom" if "bad" in sys.argv else None}
    ue_listener.runpy.run_path = fake_run_path
    cid = server.send_unreal("render", ["torpedo", "--quick", "20"])
    ue_listener._process_inbox()
    res = server.unreal_result(cid)
    assert res["ok"] and calls[-1][1][1:] == ["render", "torpedo", "--quick", "20"], (res, calls)
    assert calls[-1][0].endswith("sdg.py")
    res = ue_listener.execute({"command": "setup", "args": ["bad"]})
    assert not res["ok"] and res["error"] == "boom"
    assert ue_listener.execute({"command": "rm -rf"})["ok"] is False
    assert ue_listener.execute({"command": "save_level"})["ok"] and sys.modules["unreal"]._Sub.saved == 1
    assert ue_listener.execute({"command": "cancel"})["ok"] is False       # nothing rendering

    expect_api_error(server.send_unreal, "format_c", [])
    expect_api_error(server.send_unreal, "setup", ["no_such_config"])
    assert ue_listener.start() is True
    sys.modules["unreal"] = fake_unreal("UnrealEditor.exe MyProject.uproject -game -MoviePipelineLocalExecutorClass=x")
    sys.modules.pop("ue_listener", None)
    import ue_listener as ul2
    assert ul2.start() is False                               # MRQ's render process must not take commands
    del sys.modules["unreal"]
    sys.modules.pop("ue_listener", None)


# ---------------------------------------------------------------------------- render progress, log
def _test_config(name="apptest"):
    raw = tomllib.loads(read_raw("torpedo"))
    raw["name"] = name
    raw["output"]["raw_dir"] = os.path.join(TMP, "raw").replace("\\", "/")
    raw["output"]["dataset_dir"] = os.path.join(TMP, "datasets").replace("\\", "/")
    assert server.save_config(name, raw)["ok"]
    return server.load_cfg(name)


def test_run_state():
    cfg = _test_config()
    raw_run = os.path.join(cfg["output"]["raw_dir"], cfg["name"])
    stages = []
    for st in cfg["stages"]:
        d = os.path.join(raw_run, st["name"])
        os.makedirs(d)
        stages.append({"name": st["name"], "frames": 10, "seq_frames": 10, "exr_dir": d.replace("\\", "/")})
    write(os.path.join(stages[0]["exr_dir"], "x.000001.exr"), "")
    write(os.path.join(stages[0]["exr_dir"], "x.000002.exr"), "")
    write(os.path.join(raw_run, "run.json"), json.dumps({"created": "now", "quick": 10, "stages": stages}))
    time.sleep(0.05)
    write(os.path.join(raw_run, "progress.json"), json.dumps({"time": time.time(), "state": "converting",
                                                              "stages": [{"rendered": 3}], "counts": {"frames": 3}}))
    prev = os.path.join(cfg["output"]["dataset_dir"], cfg["name"], "previews")
    for i in range(3):
        cv2.imwrite(os.path.join(prev if os.makedirs(prev, exist_ok=True) is None else prev, f"s_{i:06d}.jpg"),
                    np.zeros((9, 16, 3), np.uint8))
    s = server.run_state("apptest")
    assert s["ok"] and s["planned"]["images"] == sum(st["frames"] for st in cfg["stages"])
    assert s["run"]["quick"] == 10 and s["run"]["stages"][0]["pending"] == 2
    assert s["progress"]["state"] == "converting" and s["previews_total"] == 3
    assert all("hold" in st for st in s["planned"]["stages"])
    write(os.path.join(raw_run, "_DONE"), "140 frames in 7.0 s")
    assert abs(server.run_state("apptest")["last_fps"] - 20.0) < 1e-6


def test_log_tail():
    log = os.path.join(TMP, "editor.log")
    write(log, "[2026.09.25-10.00.00:000][  1]LogTemp: noise\n"
               "[2026.09.25-10.00.01:000][  1]LogPython: [SDG] render torpedo_v1\n"
               "[2026.09.25-10.00.02:000][  1]LogPython: Error: [SDG] failed:\n"
               "Traceback (most recent call last):\n"
               "  File \"x.py\", line 1\n"
               "[2026.09.25-10.00.03:000][  1]LogTemp: more noise\n")
    st = types.SimpleNamespace(get=lambda k, d=None: log if k == "editor_log" else d)
    r = server.log_tail(-1, st)
    joined = "\n".join(r["lines"])
    assert "render torpedo_v1" in joined and "Traceback" in joined and "noise" not in joined, r["lines"]
    with open(log, "a", encoding="utf-8") as f:
        f.write("[2026.09.25-10.00.04:000][  1]LogPython: [SDG] render finished: 10 images\n")
    r2 = server.log_tail(r["offset"], st)
    assert len(r2["lines"]) == 1 and "render finished" in r2["lines"][0], r2


# ---------------------------------------------------------------------------- datasets
def _fake_dataset(ds, names, n=(6, 2)):
    for split, k in zip(("train", "val"), n):
        for i in range(k):
            os.makedirs(os.path.join(ds, "images", split), exist_ok=True)
            os.makedirs(os.path.join(ds, "labels", split), exist_ok=True)
            cv2.imwrite(os.path.join(ds, "images", split, f"a_{i:06d}.jpg"), np.full((32, 48, 3), 80, np.uint8))
            write(os.path.join(ds, "labels", split, f"a_{i:06d}.txt"), "1 0.1 0.1 0.5 0.1 0.5 0.5\n0 0.2 0.2 0.3 0.2 0.3 0.3\n")
    write(os.path.join(ds, "data.yaml"), "train: images/train\nval: images/val\nnames:\n" +
          "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    write(os.path.join(ds, "stats.json"), json.dumps({"seg": {n: 8 for n in names}}))


def test_datasets():
    cfg = server.load_cfg("apptest")
    names = ["ambulance", "blood", "fire", "firetruck"]
    ds = os.path.join(cfg["output"]["dataset_dir"], "demo_v1")
    _fake_dataset(ds, names)
    post.zip_dataset({"dataset": ds, "name": "demo_v1", "seg_classes": names})
    found = {d["path"].split("/")[-1]: d for d in server.discover_datasets(server.Settings())}
    assert "demo_v1" in found and "demo_v1_yolo_seg.zip" in found, list(found)
    for key in ("demo_v1", "demo_v1_yolo_seg.zip"):
        d = found[key]
        assert d["splits"] == {"train": 6, "val": 2} and d["names"] == names and d["task"] == "segment", d
    detail = server.dataset_info(ds, detail=True)
    assert detail["class_counts"]["ambulance"] == 8


# ---------------------------------------------------------------------------- training
FAKE_TRAIN = r'''
import argparse, os, sys, yaml
ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--dataset"); ap.add_argument("--config"); ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
assert cfg["augment"]["albumentations"] in ("none", "underwater") and os.path.exists(a.dataset)
name = os.path.splitext(os.path.basename(a.model))[0]
run = os.path.join("runs", name)
os.makedirs(os.path.join(run, "weights"), exist_ok=True)
with open(os.path.join(run, "args.yaml"), "w") as f:
    f.write(f"task: segment\nepochs: {cfg['train']['epochs']}\nimgsz: {cfg['train']['imgsz']}\n")
with open(os.path.join(run, "results.csv"), "w") as f:
    f.write("epoch,time,train/box_loss,train/seg_loss,metrics/mAP50(B),metrics/mAP50-95(B),metrics/mAP50(M),metrics/mAP50-95(M),val/box_loss,val/seg_loss\n")
    for e in range(1, cfg["train"]["epochs"] + 1):
        f.write(f"{e},{10.0 * e},{1.0 / e},{2.0 / e},0.5,{0.3 + 0.1 * e},0.6,{0.2 + 0.1 * e},{1.1 / e},{2.1 / e}\n")
open(os.path.join(run, "weights", "best.pt"), "wb").close()
open(a.model, "wb").close()
'''


def test_training_launch():
    repo = os.path.join(TMP, "yolo-training")
    write(os.path.join(repo, "train.py"), FAKE_TRAIN)
    write(os.path.join(repo, "analyse_model.py"), "print('ok')\n")
    settings = server.Settings()
    settings.update(trainer_repo=repo.replace("\\", "/"), trainer_python=sys.executable.replace("\\", "/"))
    ds = os.path.join(server.load_cfg("apptest")["output"]["dataset_dir"], "demo_v1")
    opts = server.train_options(settings)
    assert opts["repo_ok"] and opts["python_ok"] and set(opts["presets"]) == {"off", "light", "medium", "strong"}
    req = {"dataset": ds, "model_name": "demo_s", "version": "yolo11", "size": "s", "epochs": 3, "imgsz": 320,
           "batch": 4, "aug": opts["presets"]["strong"]["values"], "albumentations": "underwater", "cache": "off"}
    r = server.start_training(req, settings)
    assert r["ok"], r
    import yaml
    with open(r["config"], encoding="utf-8") as f:
        written = yaml.safe_load(f)
    assert written["augment"]["albumentations"] == "underwater" and written["train"]["cache"] is False
    assert written["model"] == {"version": "yolo11", "size": "s", "task": "auto", "base_weights": None, "weights_dir": "weights"}
    entry = server.registry()[-1]
    for _ in range(300):
        st = server._read_text(entry["status_file"]) or ""
        if st.startswith("exit"):
            break
        time.sleep(0.1)
    assert st.strip() == "exit 0", st
    runs = {x["name"]: x for x in server.list_runs(settings)}
    assert runs["demo_s"]["status"] == "finished" and runs["demo_s"]["epoch"] == 3, runs.get("demo_s")
    assert abs(runs["demo_s"]["best"]["value"] - 0.5) < 1e-9 and runs["demo_s"]["best"]["kind"] == "mask"
    d = server.run_detail(settings, "demo_s")
    assert len(d["rows"]) == 3 and "metrics/mAP50-95(M)" in d["columns"]
    assert server.start_training(req, settings).get("exists")          # model file there: ask first
    r = server.launch_tool(settings, {"tool": "analyse", "run": "demo_s"})
    assert r["ok"] and "analyse_model.py" in r["command"]
    for badreq in ({**req, "model_name": "bad name"}, {**req, "version": "yolo99"}, {**req, "epochs": 0},
                   {**req, "albumentations": "sepia"}, {**req, "dataset": os.path.join(TMP, "nope")}):
        expect_api_error(server.start_training, {**badreq, "overwrite": True}, settings)


# ---------------------------------------------------------------------------- http
def test_http():
    srv = server.make_server(0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def call(path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
    try:
        assert json.loads(call("/api/ping")[1])["app"] == "sdg"
        code, body = call("/")
        assert code == 200 and b"<title>SDG App</title>" in body
        assert call("/static/js/app.js")[0] == 200
        assert call("/static/../server.py")[0] == 404
        assert call("/api/state", headers={"Host": "evil.example"})[0] == 403
        assert call("/api/state", headers={"Origin": "http://evil.example"})[0] == 403
        req = urllib.request.Request(base + "/api/unreal", data=b'{"command":"ping"}', headers={"Content-Type": "text/plain"})
        try:
            urllib.request.urlopen(req)
            raise AssertionError("text/plain POST accepted")
        except urllib.error.HTTPError as e:
            assert e.code == 400
        schema = json.loads(call("/api/schema")[1])
        assert "camera" in schema["defaults"] and schema["hints"]["camera.volume"]["widget"] == "volume"
        raw = server.read_config("apptest")["raw"]
        res = json.loads(call("/api/config/check", {"raw": raw})[1])
        assert res["ok"], res
        res = json.loads(call("/api/toml/parse", {"text": "{ normal = [1, 0.2], clip = [0, 2] }"})[1])
        assert res["ok"] and res["value"] == {"normal": [1, 0.2], "clip": [0, 2]}
        code, body = call("/files/preview?config=apptest&f=s_000000.jpg")
        assert code == 200 and body[:2] == b"\xff\xd8"
        assert call("/files/preview?config=apptest&f=../../run.json")[0] == 404
    finally:
        srv.shutdown()


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as e:
            failed += 1
            traceback.print_exc()
            print(f"FAIL  {name}: {e}")
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
