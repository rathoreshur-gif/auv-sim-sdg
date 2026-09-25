"""
Synthetic data generation - run from the UE Output Log (Cmd dropdown, then):

    py "<project>/SegPipeline/sdg/sdg.py" <command> <config> [options]

commands
    setup   <config>                  tag props, build M_SDG_Labels, add SDG_Camera + SDG_PostProcess.
                                      Once per level, and again after changing props. Then save the level.
    render  <config> [--quick N]      bake + render + dataset (the everyday command)
    bake    <config> [--quick N]      only write the sequences and manifests
    preview <config> <stage>          bake that stage and open its sequence in Sequencer to scrub it
    where                             print the viewport camera position (to fill in camera.volume)

options
    --quick N          at most N images per stage (smoke test before a big run)
    --stages a,b       only these stages
    --no-post          don't start the dataset builder (run <raw>/<name>/post.bat later)

<config> is a path, or a name in sdg/configs (e.g. "pool" for configs/pool.toml).
"""
import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0]))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sdg_common   # noqa: E402
import ue_setup     # noqa: E402
import ue_bake      # noqa: E402
import ue_render    # noqa: E402

for _m in (sdg_common, ue_setup, ue_bake, ue_render):      # pick up edits without restarting the editor
    importlib.reload(_m)

import unreal       # noqa: E402


def parse(argv):
    args = {"cmd": argv[0] if argv else "", "config": argv[1] if len(argv) > 1 else "", "quick": 0,
            "stages": None, "post": True, "rest": []}
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--quick":
            args["quick"] = int(argv[i + 1])
            i += 1
        elif a == "--stages":
            args["stages"] = [s for s in argv[i + 1].split(",") if s]
            i += 1
        elif a == "--no-post":
            args["post"] = False
        else:
            args["rest"].append(a)
        i += 1
    return args


def main(argv):
    if "-game" in unreal.SystemLibrary.get_command_line().lower():
        return                     # MRQ's render process inherits the editor's command line: do nothing there
    args = parse(argv)
    if args["cmd"] == "where":
        res = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_level_viewport_camera_info()
        loc = next(v for v in (res if isinstance(res, tuple) else (res,)) if isinstance(v, unreal.Vector))
        unreal.log(f"[SDG] viewport camera at [{loc.x:.0f}, {loc.y:.0f}, {loc.z:.0f}] - fly to two opposite corners of "
                   f"the space the camera may use and put them in camera.volume = {{ min = [..], max = [..] }}")
        return
    if args["cmd"] not in ("setup", "bake", "render", "preview") or not args["config"]:
        unreal.log(__doc__)
        return
    cfg = sdg_common.load_config(args["config"])
    unreal.log(f"[SDG] {args['cmd']} {cfg['name']}  ({cfg['_path']})")
    if args["cmd"] == "setup":
        ue_setup.run(cfg)
    elif args["cmd"] == "bake":
        ue_bake.run(cfg, args["quick"], args["stages"])
    elif args["cmd"] == "render":
        info = ue_bake.run(cfg, args["quick"], args["stages"])
        ue_render.run(cfg, info, args["post"])
    elif args["cmd"] == "preview":
        stage = args["rest"][0] if args["rest"] else cfg["stages"][0]["name"]
        info = ue_bake.run(cfg, args["quick"] or 300, [stage])
        seq = unreal.load_asset(info["stages"][0]["sequence"])
        unreal.LevelSequenceEditorBlueprintLibrary.open_level_sequence(seq)
        unreal.log("[SDG] sequence open: scrub the timeline; lock the viewport to the camera cut (camera icon) to look through SDG_Camera")


LAST_ERROR = None                  # read by ue_listener.py (the SDG app's bridge) after running this file
try:
    main(sys.argv[1:])
except sdg_common.ConfigError as e:
    LAST_ERROR = f"config error - {e}"
    unreal.log_error(f"[SDG] {LAST_ERROR}")
except Exception:
    LAST_ERROR = traceback.format_exc()
    unreal.log_error("[SDG] failed:\n" + LAST_ERROR)
