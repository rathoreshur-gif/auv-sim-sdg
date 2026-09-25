# Runs when the editor starts (Unreal runs every Content/Python/init_unreal.py).
# Starts the bridge that lets the SDG app (SegPipeline/app) run setup / renders in this editor.
import os
import sys

import unreal

_sdg = os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()), "SegPipeline", "sdg")
if os.path.isfile(os.path.join(_sdg, "ue_listener.py")):
    if _sdg not in sys.path:
        sys.path.insert(0, _sdg)
    try:
        import ue_listener
        ue_listener.start()
    except Exception as _e:
        unreal.log_warning(f"[SDG] app bridge not started: {_e}")
