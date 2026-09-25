"""
Bridge between the SDG app (SegPipeline/app) and this Unreal Editor.

The app drops command files into sdg/.bridge/inbox/. Between frames, this runs them on the editor's
main thread exactly like typing
    py ".../sdg.py" <command> <config> [options]
in the Output Log, and writes the outcome to sdg/.bridge/outbox/. A heartbeat file tells the app that
the editor is open, which level is loaded and whether Movie Render Queue is rendering.

Started automatically by Content/Python/init_unreal.py when the editor opens. To start it in an
editor that is already open, run once in the Output Log (Cmd):
    py "<project>/SegPipeline/sdg/ue_listener.py"
Running it again is harmless (it replaces the running copy).
"""
import glob
import json
import os
import runpy
import sys
import time
import traceback

import unreal

HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0]))
SDG = os.path.join(HERE, "sdg.py")
BRIDGE = os.environ.get("SDG_BRIDGE_DIR", os.path.join(HERE, ".bridge"))
INBOX, OUTBOX = os.path.join(BRIDGE, "inbox"), os.path.join(BRIDGE, "outbox")
HEARTBEAT = os.path.join(BRIDGE, "heartbeat.json")
COMMANDS = ("setup", "render", "bake", "preview", "where", "cancel", "save_level", "ping")
POLL_S, BEAT_S = 0.5, 2.0

# survives importlib.reload / re-running this file: only one tick callback may be registered
_state = sys.modules.setdefault("_sdg_listener_state", type(sys)("_sdg_listener_state"))


def log(msg):
    unreal.log(f"[SDG] {msg}")


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _mrq():
    try:
        return unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    except Exception:
        return None                    # Movie Render Queue plugin not loaded


def _heartbeat():
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    mrq = _mrq()
    _write_json(HEARTBEAT, {
        "time": time.time(), "pid": os.getpid(),
        "level": world.get_path_name().split(".")[0] if world else "",
        "rendering": bool(mrq and mrq.is_rendering()),
        "project": unreal.Paths.convert_relative_path_to_full(unreal.Paths.get_project_file_path()),
    })


def _viewport_location():
    res = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_level_viewport_camera_info()
    loc = next(v for v in (res if isinstance(res, tuple) else (res,)) if isinstance(v, unreal.Vector))
    return [round(loc.x), round(loc.y), round(loc.z)]


def _run_sdg(args):
    """Same as `py sdg.py <args>` in the Output Log. Returns sdg.py's error text, or None."""
    old = sys.argv
    sys.argv = [SDG] + list(args)
    try:
        g = runpy.run_path(SDG, run_name="__main__")
    finally:
        sys.argv = old
    return g.get("LAST_ERROR")


def execute(cmd):
    """Run one command dict {command, args}; returns a result dict (never raises)."""
    name, args = cmd.get("command"), [str(a) for a in cmd.get("args", [])]
    if name not in COMMANDS:
        return {"ok": False, "error": f"unknown command {name!r}"}
    try:
        if name == "ping":
            return {"ok": True}
        if name == "where":
            loc = _viewport_location()
            log(f"viewport camera at {loc}")
            return {"ok": True, "location": loc}
        if name == "save_level":
            unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
            log("level saved (SDG app)")
            return {"ok": True}
        if name == "cancel":
            mrq = _mrq()
            ex = mrq.get_active_executor() if mrq else None
            if ex is None:
                return {"ok": False, "error": "nothing is rendering"}
            ex.cancel_all_jobs()
            log("render cancelled from the SDG app")
            return {"ok": True}
        if name == "render" and (_mrq() and _mrq().is_rendering()):
            return {"ok": False, "error": "Movie Render Queue is already rendering"}
        err = _run_sdg([name] + args)
        return {"ok": err is None, "error": err}
    except Exception:
        return {"ok": False, "error": traceback.format_exc()}


def _process_inbox():
    files = sorted(glob.glob(os.path.join(INBOX, "*.json")), key=os.path.getmtime)
    if not files:
        return
    path = files[0]                    # one per poll: the editor stays responsive between commands
    cid = os.path.splitext(os.path.basename(path))[0]
    try:
        with open(path, encoding="utf-8") as f:
            cmd = json.load(f)
    except (OSError, ValueError):
        if time.time() - os.path.getmtime(path) < 5:
            return                     # probably still being written
        cmd = {"command": "invalid"}
    os.remove(path)
    started = time.time()
    res = execute(cmd)
    res.update(id=cid, command=cmd.get("command"), args=cmd.get("args", []), started=started, finished=time.time())
    _write_json(os.path.join(OUTBOX, cid + ".json"), res)
    for old in glob.glob(os.path.join(OUTBOX, "*.json")):          # keep the outbox small
        if time.time() - os.path.getmtime(old) > 3600:
            os.remove(old)


def _tick(delta):
    now = time.time()
    try:
        if now - getattr(_state, "t_beat", 0) >= BEAT_S:
            _state.t_beat = now
            _heartbeat()
        if now - getattr(_state, "t_poll", 0) >= POLL_S:
            _state.t_poll = now
            _process_inbox()
    except Exception:
        if now - getattr(_state, "t_err", 0) > 30:                 # don't flood the log
            _state.t_err = now
            unreal.log_warning("[SDG] app bridge: " + traceback.format_exc())


def start():
    cmdline = unreal.SystemLibrary.get_command_line().lower()
    if "-game" in cmdline or "-run=" in cmdline:
        return False                   # MRQ's render process / commandlets: the editor instance does this
    stop()
    os.makedirs(INBOX, exist_ok=True)
    os.makedirs(OUTBOX, exist_ok=True)
    _state.handle = unreal.register_slate_post_tick_callback(_tick)
    _state.t_beat = 0
    log("app bridge running - the SDG app can now drive this editor")
    return True


def stop():
    h = getattr(_state, "handle", None)
    if h is not None:
        unreal.unregister_slate_post_tick_callback(h)
        _state.handle = None


if __name__ == "__main__":
    # `py ue_listener.py` runs this file in the Output Log's shared namespace, where other scripts
    # could overwrite these functions later: run the importable module instead.
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import importlib
    import ue_listener as _mod
    importlib.reload(_mod).start()
