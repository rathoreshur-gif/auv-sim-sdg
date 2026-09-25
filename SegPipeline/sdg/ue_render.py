"""
Render the baked stages with Movie Render Queue.

One job per stage, EXR only: FinalImage + the SDGLabels pass in one multilayer file per frame,
with the camera pose and FOV in its header. 1 spatial x 1 temporal sample, so MRQ skips its
accumulator (fast path) and never blends label ids across samples. The dataset builder
(post.py) is started in its own console first and converts frames while MRQ is still rendering.
"""
import glob
import os
import subprocess
import time

import unreal

import sdg_common as sc
import ue_setup as us

AA = {"tsr": "AAM_TSR", "taa": "AAM_TEMPORAL_AA", "fxaa": "AAM_FXAA", "none": "AAM_NONE"}
EXR = {"zip": "ZIP", "zips": "ZIPS", "none": "NONE"}
STREAMING = {"fully_load": "FULLY_LOAD", "disabled": "DISABLED", "none": "NONE"}
BASE_CVARS = {
    "r.MotionBlurQuality": 0,      # teleports/fast paths would smear
    "r.DepthOfFieldQuality": 0,
    "r.ScreenPercentage": 100,     # label pass reads scene textures: render at output resolution, no upscaling
}

_keep = {}          # MRQ runs asynchronously: keep the executor, queue and callbacks alive


def log(msg):
    unreal.log(f"[SDG] {msg}")


def warn(msg):
    unreal.log_warning(f"[SDG] {msg}")


def object_path(asset_path):
    return f"{asset_path}.{asset_path.rsplit('/', 1)[1]}"


def configure(job, cfg, stage, label_mat):
    c = job.get_configuration()
    for s in list(c.get_all_settings()):
        if isinstance(s, unreal.MoviePipelineOutputBase) and not isinstance(s, unreal.MoviePipelineImageSequenceOutput_EXR):
            c.remove_setting(s)                          # no PNG/JPG/video outputs: EXR carries everything
    r = cfg["render"]
    w, h = r["resolution"]

    out = c.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    folder = unreal.DirectoryPath()
    folder.set_editor_property("path", stage["exr_dir"])
    out.set_editor_property("output_directory", folder)
    out.set_editor_property("file_name_format", "{sequence_name}.{frame_number}")
    out.set_editor_property("output_resolution", unreal.IntPoint(w, h))
    out.set_editor_property("zero_pad_frame_numbers", 6)
    out.set_editor_property("override_existing_output", True)
    out.set_editor_property("use_custom_playback_range", False)
    out.set_editor_property("output_frame_step", 1)

    dp = c.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    pp = unreal.MoviePipelinePostProcessPass()
    pp.set_editor_property("enabled", True)
    pp.set_editor_property("name", sc.LABEL_PASS)
    pp.set_editor_property("material", label_mat)
    pp.set_editor_property("use_lossless_compression", True)
    dp.set_editor_property("additional_post_process_materials", [pp])
    dp.set_editor_property("disable_multisample_effects", False)

    aa = c.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.set_editor_property("spatial_sample_count", 1)
    aa.set_editor_property("temporal_sample_count", 1)
    aa.set_editor_property("override_anti_aliasing", True)
    aa.set_editor_property("anti_aliasing_method", getattr(unreal.AntiAliasingMethod, AA[r["anti_aliasing"]]))
    aa.set_editor_property("engine_warm_up_count", int(r["warmup_frames"]))
    aa.set_editor_property("render_warm_up_frames", True)     # warm-up frames are rendered: Lumen/TSR settle
    aa.set_editor_property("render_warm_up_count", 0)
    aa.set_editor_property("use_camera_cut_for_warm_up", False)

    ex = c.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_EXR)
    ex.set_editor_property("compression", getattr(unreal.EXRCompressionFormat, EXR[r["exr_compression"]]))
    ex.set_editor_property("multilayer", True)                # one file per frame, camera metadata in it

    cv = c.find_or_add_setting_by_class(unreal.MoviePipelineConsoleVariableSetting)
    for k, v in {**BASE_CVARS, **r["cvars"]}.items():
        cv.add_or_update_console_variable(k, float(v))

    g = c.find_or_add_setting_by_class(unreal.MoviePipelineGameOverrideSetting)
    g.set_editor_property("cinematic_quality_settings", bool(r.get("cinematic", False)))
    g.set_editor_property("texture_streaming", getattr(unreal.MoviePipelineTextureStreamingMethod, STREAMING[r["texture_streaming"]]))
    g.set_editor_property("disable_hlods", True)


def write_post_bat(cfg, run):
    """post.bat in the run folder: what the editor starts, and what you can double-click to redo it."""
    py = cfg["post"]["python"]
    bat = os.path.join(run["raw"], "post.bat")
    with open(bat, "w", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write(f"title SDG dataset - {cfg['name']}\n")
        f.write(f"\"{os.path.normpath(py)}\" \"{os.path.join(sc.SDG_DIR, 'post.py')}\" \"{run['raw']}\" %*\n")
    return bat


def launch_post(cfg, run):
    py = cfg["post"]["python"]
    if not py or not os.path.isfile(py):
        warn(f"post.python ({py!r}) not found - convert the frames yourself afterwards:\n"
             f"    <python with numpy+opencv> \"{os.path.join(sc.SDG_DIR, 'post.py')}\" \"{run['raw']}\"")
        return None
    bat = write_post_bat(cfg, run)
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    return subprocess.Popen(["cmd", "/k", bat, "--watch"], cwd=sc.SDG_DIR, creationflags=flags)


def _flag(name, text=""):
    run = _keep.get("run")
    if run:
        with open(os.path.join(run["raw"], name), "w", encoding="utf-8") as f:
            f.write(text)


def _frames_on_disk(run):
    """EXRs still waiting + images the dataset console already made from them."""
    n = sum(len(glob.glob(os.path.join(st["exr_dir"], "*.exr"))) for st in run["stages"])
    if _keep.get("post"):
        n += len(glob.glob(os.path.join(run["dataset"], "images", "*", "*.jpg")))
    return n


def _on_finished(executor, success):
    run, t0 = _keep.get("run"), _keep.get("t0", time.time())
    secs = time.time() - t0
    frames = sum(s["seq_frames"] for s in run["stages"]) if run else 0
    images = sum(s["frames"] for s in run["stages"]) if run else 0
    if success and run and _frames_on_disk(run) == 0:
        # the separate-process executor reports success even when its process crashes
        success = False
        warn("MRQ reported success but no frames were written - look for 'Process exited with non-success "
             "return code' in the Output Log, or try render.executor = \"pie\"")
    _flag("_DONE" if success else "_FAILED", f"{frames} frames in {secs:.1f} s")
    if success:
        log(f"render finished: {images} images ({frames} frames) in {secs:.0f} s = {images / max(secs, 1e-6):.1f} images/s "
            f"(includes loading the level and warm-up). The dataset console finishes the labels.")
    else:
        warn(f"render failed after {secs:.0f} s - see the Output Log (LogMovieRenderPipeline)")
    _keep.pop("executor", None)
    _keep.pop("queue", None)


def _on_error(executor, pipeline, is_fatal, error_text):
    warn(f"MRQ error (fatal={is_fatal}): {error_text}")
    if is_fatal:
        _flag("_FAILED", str(error_text))


def run(cfg, run_info, start_post=True):
    subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    if subsystem.is_rendering():
        raise RuntimeError("Movie Render Queue is already rendering")
    label_mat = unreal.load_asset("/Game/SDG/M_SDG_Labels")
    if label_mat is None:
        raise RuntimeError("/Game/SDG/M_SDG_Labels missing - run setup first")
    legacy = [a.get_actor_label() for a in us.eas().get_all_level_actors()
              if "BP_SynthetheticCamera" in a.get_class().get_name()]
    if legacy:
        raise RuntimeError(f"delete {legacy} from the level first: during the render its BeginPlay timer would "
                           f"randomize and capture on its own and finally Quit Game, which stops MRQ")

    if cfg["render"]["executor"] == "process":
        # the separate process loads the level from disk: SDG actors must be saved in it
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
        log("saved the level (render.executor = process renders what is on disk)")

    queue = unreal.MoviePipelineQueue()
    for st in run_info["stages"]:
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.set_editor_property("job_name", f"{cfg['name']}_{st['name']}")
        job.set_editor_property("author", "SDG")
        job.set_editor_property("map", unreal.SoftObjectPath(object_path(cfg["map"])))
        job.set_editor_property("sequence", unreal.SoftObjectPath(object_path(st["sequence"])))
        configure(job, cfg, st, label_mat)

    post = launch_post(cfg, run_info) if start_post else None
    executor_cls = unreal.MoviePipelinePIEExecutor if cfg["render"]["executor"] == "pie" else unreal.MoviePipelineNewProcessExecutor
    executor = executor_cls(subsystem)
    executor.on_executor_finished_delegate.add_callable_unique(_on_finished)
    executor.on_executor_errored_delegate.add_callable_unique(_on_error)
    _keep.update(executor=executor, queue=queue, run=run_info, post=post, t0=time.time())
    subsystem.render_queue_instance_with_executor_instance(queue, executor)
    total = sum(s["frames"] for s in run_info["stages"])
    log(f"rendering {len(run_info['stages'])} stage(s), {total} images -> {run_info['raw']}"
        + ("  (dataset console started)" if post else ""))
