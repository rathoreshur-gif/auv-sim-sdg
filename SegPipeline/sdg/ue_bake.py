"""
Bake: config -> one Level Sequence per stage with every randomized value keyed per frame, and a
manifest per stage that tells the dataset builder what each stencil id is and where every
keypoint sits.

Why sequences instead of Blueprint timers: MRQ evaluates the sequence before rendering each
frame, so a frame never mixes old and new values, nothing waits on a Delay node, and the same
seed gives the same images again.

Camera modes (per stage):
  random_path  the camera flies smoothly between random viewpoints around the targets. Every
               frame is saved and TSR / Lumen history stays valid, so this is the fastest mode.
  teleport     an independent random viewpoint per image. hold = N renders each pose N times and
               keeps the last one (the first N-1 let Lumen / TSR / auto exposure settle).
  sequence     your own Level Sequence supplies the camera path (resampled onto SDG_Camera, with
               optional smooth jitter); everything else is still randomized.
"""
import json
import math
import os
import shutil
import time

import unreal

import sdg_common as sc
import ue_setup as us

GENERATED = "/Game/SDG/Generated"
SENSOR_W = 36.0                          # SDG_Camera filmback width (ue_setup.ensure_camera)

at = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary
KI = unreal.MovieSceneKeyInterpolation


def log(msg):
    unreal.log(f"[SDG] {msg}")


def warn(msg):
    unreal.log_warning(f"[SDG] {msg}")


def vec(v):
    return (v.x, v.y, v.z)


def rot(r):
    return (r.pitch, r.yaw, r.roll)


def rounded(v, nd=4):
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, (tuple, list)):
        return [rounded(x, nd) for x in v]
    return v


# ============================================================================ scene
class Scene:
    """What setup left in the level: SDG actors, tagged objects, stencil ids, keypoints."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.actors = list(us.eas().get_all_level_actors())
        self.camera = next((a for a in self.actors if a.actor_has_tag("sdg.camera")), None)
        self.ppv = next((a for a in self.actors if a.actor_has_tag("sdg.postprocess")), None)
        if self.camera is None or self.ppv is None:
            raise RuntimeError("SDG_Camera / SDG_PostProcess not found - run `setup` first (and save the level)")
        props = {p["name"]: p for p in cfg["props"]}
        seg, pose = cfg["seg_classes"], cfg["pose_classes"]

        owners, copies = {}, {}
        self.instances = {}
        for a in self.actors:
            key = us.tag_value(a, "sdg.key:")
            if key is None:
                continue
            if "sdg.obj" in us.tags_of(a):
                owners.setdefault(key, []).append(a)      # several when copies of a prop were stacked in place
            if "sdg.copy" in us.tags_of(a):
                copies.setdefault(key, []).append(a)
            part = us.tag_value(a, "sdg.part:")
            sid = self._stencil(a)
            if part is None or not sid:
                continue
            prop = props.get(key.split("#")[0], {})
            if part == "__body__":
                cls = seg.index(prop["body_class"]) if prop.get("body_class") in seg else None
            else:
                cls = seg.index(part) if part in seg else None
            self.instances[sid] = {"object": key, "part": part, "seg_class": cls}

        self.objects, self.copies = {}, copies
        for key, group in sorted(owners.items()):
            owner = sorted(group, key=lambda a: a.get_name())[0]
            prop = props.get(key.split("#")[0])
            if prop is None:
                log(f"{owner.get_actor_label()} ({key}) is not a prop in this config: rendered, never labelled")
                continue
            lo, hi = local_bounds(owner)
            obj = {
                "prop": prop["name"], "actor": owner.get_actor_label(), "path": owner.get_path_name(),
                "stencils": sorted(s for s, i in self.instances.items() if i["object"] == key),
                "pose_class": pose.index(prop["pose_class"]) if prop.get("pose_class") in pose else None,
                "transform": list(vec(owner.get_actor_location()) + rot(owner.get_actor_rotation())
                                  + vec(owner.get_actor_scale3d())),
                "center": [(lo[i] + hi[i]) / 2 for i in range(3)],
                "keypoints": [], "flip_idx": None,
                "_lo": lo, "_hi": hi,
            }
            if obj["pose_class"] is not None:
                obj["keypoints"], obj["flip_idx"] = object_keypoints(owner, prop, lo, hi)
            obj["_actor"] = owner
            self.objects[key] = obj
        if not self.objects:
            warn("no tagged objects in the level - run setup (and check the props in the config match)")

    @staticmethod
    def _stencil(actor):
        for c in actor.get_components_by_class(unreal.PrimitiveComponent):
            if c.get_editor_property("render_custom_depth"):
                return int(c.get_editor_property("custom_depth_stencil_value"))
        return 0

    def find(self, rz):
        """(object, owner actor or None) pairs a randomizer applies to."""
        out = []
        pat, tag, comp = rz.get("actor"), rz.get("tag"), rz.get("component")
        for a in self.actors:
            label = a.get_actor_label()
            if label.startswith(us.COPY_PREFIX) or "sdg.copy" in us.tags_of(a):
                continue
            if pat and not sc.wildcard(label, pat):
                continue
            if tag and not a.actor_has_tag(tag):
                continue
            if comp:
                for c in a.get_components_by_class(unreal.ActorComponent):
                    if sc.wildcard(c.get_name(), comp) or sc.wildcard(c.get_class().get_name(), comp):
                        out.append((c, a))
            else:
                out.append((a, None))
        return out


def local_bounds(actor):
    """Actor-space box around all static meshes of the actor."""
    xf = actor.get_actor_transform()
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for c in actor.get_components_by_class(unreal.StaticMeshComponent):
        sm = c.get_editor_property("static_mesh")
        if sm is None:
            continue
        bb = sm.get_bounding_box()
        cw = c.get_world_transform()
        for x in (bb.min.x, bb.max.x):
            for y in (bb.min.y, bb.max.y):
                for z in (bb.min.z, bb.max.z):
                    p = xf.inverse_transform_location(cw.transform_location(unreal.Vector(x, y, z)))
                    for i, v in enumerate((p.x, p.y, p.z)):
                        lo[i], hi[i] = min(lo[i], v), max(hi[i], v)
    if lo[0] == math.inf:
        return [0.0] * 3, [0.0] * 3
    return lo, hi


def object_keypoints(owner, prop, lo, hi):
    spec = sc.parse_keypoint_spec(prop["name"], prop["keypoints"])
    flip = prop.get("flip_idx")
    if "preset" in spec:
        pts, default_flip = sc.preset_keypoints(spec["preset"], lo, hi)
        return [list(p) for p in pts], flip if flip is not None else default_flip
    if "points" in spec:
        return [list(map(float, p)) for p in spec["points"]], flip
    xf = owner.get_actor_transform()
    pts = []
    for name in spec["sockets"]:
        comp = next((c for c in owner.get_components_by_class(unreal.StaticMeshComponent)
                     if c.does_socket_exist(name)), None)
        if comp is None:
            raise RuntimeError(f"prop '{prop['name']}': no socket {name!r} on {owner.get_actor_label()}")
        p = xf.inverse_transform_location(comp.get_socket_location(name))
        pts.append([p.x, p.y, p.z])
    return pts, flip


# ============================================================================ sequence writing
class Writer:
    def __init__(self, path, n_frames, fps):
        folder, name = path.rsplit("/", 1)
        if eal.does_asset_exist(path):
            eal.delete_asset(path)
        self.seq = at.create_asset(name, folder, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
        self.seq.set_display_rate(unreal.FrameRate(int(fps), 1))
        self.seq.set_playback_start(0)
        self.seq.set_playback_end(n_frames)
        tick, disp = self.seq.get_tick_resolution(), self.seq.get_display_rate()
        self.ticks = (tick.numerator * disp.denominator) / (tick.denominator * disp.numerator)
        self.bindings = {}
        self.path = path

    def bind(self, obj, owner=None):
        key = obj.get_path_name()
        if key not in self.bindings:
            b = self.seq.add_possessable(obj)
            if owner is not None:
                b.set_parent(self.bind(owner))
            self.bindings[key] = b
        return self.bindings[key]

    @staticmethod
    def section(track):
        s = track.add_section()
        s.set_start_frame_bounded(False)
        s.set_end_frame_bounded(False)
        return s

    def property_section(self, obj, owner, track_cls, name, path=None):
        t = self.bind(obj, owner).add_track(track_cls)
        t.set_property_name_and_path(name, path or name)
        return self.section(t)

    def tick(self, frame):
        return unreal.FrameNumber(int(round(frame * self.ticks)))

    def save(self):
        eal.save_loaded_asset(self.seq)


def channels(section):
    return {str(c.get_editor_property("channel_name")): c for c in section.get_all_channels()}


def key_scalar(ch, keys, interp):
    for f, v in keys:
        ch.add_key(time=unreal.FrameNumber(f), new_value=float(v), interpolation=interp)


def key_bool(ch, keys):
    for f, v in keys:
        ch.add_key(time=unreal.FrameNumber(f), new_value=bool(v))


def key_color(section, keys, interp):
    def order(c):                                  # "Color.R" / "R" / "Red" -> 0..3
        return "RGBA".index(str(c.get_editor_property("channel_name")).split(".")[-1][0].upper())
    chs = sorted(section.get_all_channels(), key=order)
    for f, v in keys:
        rgba = tuple(v) + (1.0,) * (4 - len(v))
        for ch, x in zip(chs, rgba):
            ch.add_key(time=unreal.FrameNumber(f), new_value=float(x), interpolation=interp)


TRANSFORM_CHANNELS = ("Location.X", "Location.Y", "Location.Z", "Rotation.Y", "Rotation.Z", "Rotation.X",
                      "Scale.X", "Scale.Y", "Scale.Z")          # our order: loc, pitch, yaw, roll, scale


def key_transform(section, keys, interp):
    """keys: (frame, [x, y, z, pitch, yaw, roll, sx, sy, sz])."""
    chs = channels(section)
    for i, name in enumerate(TRANSFORM_CHANNELS):
        key_scalar(chs[name], [(f, t[i]) for f, t in keys], interp)


def series_keys(values, hold, every, smooth):
    """Per-sample values -> (sequence frame, value) keys: anchors if smooth, changes otherwise."""
    if smooth:
        idx = sorted(set(range(0, len(values), every)) | {len(values) - 1})
    else:
        idx = [i for i in range(len(values)) if i == 0 or values[i] != values[i - 1]]
    return [(i * hold, values[i]) for i in idx]


# ============================================================================ camera
def camera_volume(cam, scene):
    vol = cam.get("volume")
    if not vol:
        return None
    if "actor" in vol:
        a = next((a for a in scene.actors if sc.wildcard(a.get_actor_label(), vol["actor"])), None)
        if a is None:
            raise RuntimeError(f"camera.volume actor {vol['actor']!r} not found")
        origin, ext = a.get_actor_bounds(False)
        m = float(vol.get("margin", 0.0))
        return ([origin.x - ext.x + m, origin.y - ext.y + m, origin.z - ext.z + m],
                [origin.x + ext.x - m, origin.y + ext.y - m, origin.z + ext.z - m])
    a, b = vol["min"], vol["max"]
    return ([min(a[i], b[i]) for i in range(3)], [max(a[i], b[i]) for i in range(3)])


def inside(p, vol):
    return vol is None or all(vol[0][i] <= p[i] <= vol[1][i] for i in range(3))


def sph(az, el, d):
    a, e = math.radians(az), math.radians(el)
    return (d * math.cos(e) * math.cos(a), d * math.cos(e) * math.sin(a), d * math.sin(e))


def rel_azimuth(s, cam):
    az = cam.get("azimuth")
    return s.range(az) if az is not None else s.uniform(0, 360)


def clamp_azimuth(cam, rel):
    az = cam.get("azimuth")
    return rel if az is None else min(max(rel, az[0]), az[1])


class View:
    """A viewpoint around a target: spherical offset + aim jitter + roll + fov. target = (point, yaw of
    its facing side or None); the azimuth is kept relative to that side so a stage can stay in front."""

    def __init__(self, s, cam, target, az=None, el=None, dist=None):
        self.target, self.base = target[0], target[1]
        self.rel = rel_azimuth(s, cam) if az is None else az
        self.az = self.rel + (self.base or 0.0)
        self.el = s.range(cam["elevation"]) if el is None else el
        self.dist = s.range(cam["distance"]) if dist is None else dist
        j = cam["aim_jitter"]
        self.jp, self.jy = s.uniform(-j, j), s.uniform(-j, j)
        self.roll = s.range(cam["roll"])
        self.fov = s.range(cam["fov"])

    def loc(self):
        return sc.add(self.target, sph(self.az, self.el, self.dist))


def place(s, cam, target, vol, tries=200, **kw):
    v = None
    for _ in range(tries):
        v = View(s, cam, target, **kw)
        if inside(v.loc(), vol):
            return v, True
        kw = {}                                   # the proposal failed: fall back to fully random
    return v, False


def pose_of(loc, target, jp, jy, roll, fov):
    p, y, _ = sc.look_at(sc.sub(target, loc))
    return [loc[0], loc[1], loc[2], p + jp, y + jy, roll, fov]


def teleport_poses(s, cam, n, targets_at, vol):
    poses, misses = [], 0
    for i in range(n):
        targets = targets_at(i)
        v, ok = place(s, cam, targets[s.rng.randrange(len(targets))], vol)
        misses += not ok
        poses.append(pose_of(v.loc(), v.target, v.jp, v.jy, v.roll, v.fov))
    return poses, misses


def smoothstep(t):
    return t * t * (3 - 2 * t)


def random_path_poses(s, cam, n, targets_at, vol):
    """Smooth flight between viewpoints; every frame is a sample. Waypoints do a random walk in
    (azimuth, elevation, distance) around the current target and sometimes switch targets."""
    poses, misses = [], 0
    ti = s.rng.randrange(len(targets_at(0)))
    a, ok = place(s, cam, targets_at(0)[ti], vol)
    misses += not ok
    while len(poses) < n:
        f = len(poses)
        targets = targets_at(f)
        if len(targets) > 1 and s.rng.random() < cam["switch_target"]:
            ti = (ti + s.rng.randrange(1, len(targets))) % len(targets)
        ti = min(ti, len(targets) - 1)
        turn = cam["turn"]
        el = min(max(a.el + s.uniform(-turn / 3, turn / 3), cam["elevation"][0]), cam["elevation"][1])
        dist = min(max(a.dist * s.uniform(0.7, 1.4), cam["distance"][0]), cam["distance"][1])
        rel = clamp_azimuth(cam, a.rel + s.uniform(-turn, turn))     # relative to the (new) target's front
        b, ok = place(s, cam, targets[ti], vol, az=rel, el=el, dist=dist)
        misses += not ok
        pa, pb = a.loc(), b.loc()
        steps = max(2, int(round(sc.length(sc.sub(pb, pa)) / max(s.range(cam["speed"]), 1e-3))))
        for k in range(1, steps + 1):
            if len(poses) >= n:
                break
            t = smoothstep(k / steps)
            loc = sc.add(pa, sc.scale(sc.sub(pb, pa), t))
            tgt = sc.add(a.target, sc.scale(sc.sub(b.target, a.target), t))
            poses.append(pose_of(loc, tgt, sc.lerp(a.jp, b.jp, t), sc.lerp(a.jy, b.jy, t),
                                 sc.lerp(a.roll, b.roll, t), sc.lerp(a.fov, b.fov, t)))
        a = b
    return poses, misses


def planned_poses(s, cam, fps):
    """Camera path of an existing Level Sequence, sampled at `sample_fps` (or its display rate)."""
    seq = unreal.load_asset(cam["sequence"])
    if seq is None:
        raise RuntimeError(f"camera.sequence {cam['sequence']} not found")
    cuts = seq.find_tracks_by_type(unreal.MovieSceneCameraCutTrack)
    if not cuts:
        raise RuntimeError(f"{cam['sequence']} has no camera cut track")
    disp = seq.get_display_rate()
    rate = unreal.FrameRate(int(cam["sample_fps"]), 1) if cam["sample_fps"] else disp
    poses = []
    for cut in sorted(cuts[0].get_sections(), key=lambda x: x.get_start_frame()):
        binding = seq.resolve_binding_id(cut.get_camera_binding_id())
        tracks = binding.find_tracks_by_type(unreal.MovieScene3DTransformTrack)
        if not tracks:
            raise RuntimeError(f"{cam['sequence']}: camera {binding.get_display_name()} has no transform track")
        start, end = cut.get_start_frame(), cut.get_end_frame()
        rng = seq.make_range(start, end - start)
        chs = channels(tracks[0].get_sections()[0])
        cols = [chs[name].evaluate_keys(rng, rate) for name in TRANSFORM_CHANNELS[:6]]
        focal, sensor = None, None
        for child in binding.get_child_possessables():
            for t in child.find_tracks_by_type(unreal.MovieSceneFloatTrack):
                if str(t.get_property_name()) == "CurrentFocalLength":
                    focal = list(t.get_sections()[0].get_all_channels()[0].evaluate_keys(rng, rate))
        tmpl = binding.get_object_template()
        cam_actor = tmpl if tmpl else next(iter(seq.locate_bound_objects(binding, us.editor_world())), None)
        comp = cam_actor.get_component_by_class(unreal.CineCameraComponent) if cam_actor else None
        if comp:
            sensor = comp.get_editor_property("filmback").get_editor_property("sensor_width")
            if focal is None:
                focal = [comp.get_editor_property("current_focal_length")] * len(cols[0])
        for i in range(len(cols[0])):
            hfov = sc.focal_mm_to_hfov(focal[i], sensor) if focal and sensor else None
            poses.append([c[i] for c in cols] + [hfov])
    if not poses:
        raise RuntimeError(f"{cam['sequence']}: empty camera path")
    for i in (4, 5):                                  # yaw/roll may wrap in the source keys
        col = sc.unwrap_degrees([p[i] for p in poses])
        for p, v in zip(poses, col):
            p[i] = v
    return poses, disp.numerator / disp.denominator


def jitter(s, poses, cam):
    n = len(poses)
    jp, jr = float(cam["jitter_pos"]), float(cam["jitter_rot"])
    if jp <= 0 and jr <= 0:
        return poses
    every = 15
    offs = [s.series([-1.0, 1.0], n, every, True) for _ in range(6)]
    out = []
    for i, p in enumerate(poses):
        q = list(p)
        for k in range(3):
            q[k] += offs[k][i] * jp
            q[3 + k] += offs[3 + k][i] * jr
        out.append(q)
    return out


# ============================================================================ stage
def bake_stage(cfg, scene, si, st, quick, raw_run):
    t0 = time.time()
    name, seed = st["name"], cfg["seed"]
    cam = sc.stage_camera(cfg, st)
    mode = cam["mode"]
    fps = cfg["render"]["fps"]
    hold = int(cam["hold"]) if mode == "teleport" else 1
    vol = camera_volume(cam, scene)
    rzs = sc.stage_randomizers(cfg, st)

    planned = None
    if mode == "sequence":
        s_cam = sc.Sampler(f"{seed}:{name}:camera")
        path, _ = planned_poses(s_cam, cam, fps)
        planned = []
        for rep in range(int(cam["repeat"])):
            planned += jitter(sc.Sampler(f"{seed}:{name}:jitter:{rep}"), path, cam)
        n = len(planned)
    else:
        n = int(st["frames"])
    if quick:
        n = min(n, quick)
        planned = planned[:n] if planned else None
    n_frames = n * hold

    # --- object transform randomizers first: camera targets follow the moved objects
    obj_tf = {}                                              # key -> per-sample absolute transforms
    rz_values = {}
    s_all = {}
    for rz in rzs:
        kind = sc.randomizer_type(rz)
        targets = [(None, None)] if kind in ("mpc", "postprocess") else retarget(rz, kind, scene.find(rz))
        if not targets:
            warn(f"stage {name}: randomizer '{rz['name']}' matched nothing in the level - skipped")
            continue
        if kind == "transform":
            make_movable(targets)
        per_target = bool(rz.get("per_target", False))
        every = int(rz.get("every", 1))
        smooth = bool(rz.get("smooth", False)) and hold == 1
        vals = []
        for ti, tgt in enumerate(targets):
            s = sc.Sampler(f"{seed}:{name}:{rz['name']}:{ti if per_target else 0}")
            if kind == "transform":
                vals.append(transform_series(s, rz, tgt, n, every, smooth))
            elif kind == "vector":
                vals.append(vector_series(s, rz, tgt, n, every, smooth))
            elif kind == "visible":
                p = float(rz.get("probability", 0.5))
                vals.append(s.series({"choice": [True, False], "weights": [p, 1 - p]}, n, every, False))
            else:
                vals.append(apply_mode(rz, kind, tgt, scene, s.series(rz["value"], n, every, smooth)))
        rz_values[rz["name"]] = (rz, kind, targets, vals, every, smooth)
        s_all[rz["name"]] = vals[0]
        if kind == "transform":
            for (obj, owner), series in zip(targets, vals):
                if owner is None:
                    key = next((k for k, o in scene.objects.items() if o["path"] == obj.get_path_name()), None)
                    if key:
                        obj_tf[key] = world_series(obj, series)

    def obj_transform(key, i):
        return obj_tf[key][i] if key in obj_tf else scene.objects[key]["transform"]

    if cam["targets"] == "props":
        keys = list(scene.objects)
        if not keys and mode != "sequence":
            raise RuntimeError("camera.targets = \"props\" but there are no tagged props - run setup")

        axes = {k: sc.facing_axis(cam["facing"], scene.objects[k]["_lo"], scene.objects[k]["_hi"]) for k in keys}
        if cam.get("azimuth") is not None:
            for k in keys:
                if sc.facing_yaw(scene.objects[k]["transform"], scene.objects[k]["center"], axes[k]) is None:
                    warn(f"stage {name}: {k} faces up/down along {cam['facing']} - its azimuth is measured from world +X")

        def targets_at(i):
            i = min(i, n - 1)
            out = []
            for k in keys:
                t, c = obj_transform(k, i), scene.objects[k]["center"]
                out.append((sc.transform_point(t[0:3], t[3:6], t[6:9], c), sc.facing_yaw(t, c, axes[k])))
            return out
    else:
        pts = [(tuple(p), 0.0) for p in cam["targets"]]

        def targets_at(i):
            return pts

    s_cam = sc.Sampler(f"{seed}:{name}:camera")
    misses = 0
    if mode == "teleport":
        poses, misses = teleport_poses(s_cam, cam, n, targets_at, vol)
    elif mode == "random_path":
        poses, misses = random_path_poses(s_cam, cam, n, targets_at, vol)
    else:
        poses = planned
        if cam["fov"] != "sequence" or any(p[6] is None for p in poses):
            fov = cam["fov"] if cam["fov"] != "sequence" else [80.0, 80.0]
            fovs = s_cam.series(fov, n, max(1, int(cam["fov_every"])), True)
            poses = [p[:6] + [f] for p, f in zip(poses, fovs)]
    if misses:
        warn(f"stage {name}: {misses} viewpoints could not be placed inside camera.volume (used anyway)")
    if hold == 1:
        for k in (4, 5):
            col = sc.unwrap_degrees([p[k] for p in poses])
            for p, v in zip(poses, col):
                p[k] = v

    # --- write the sequence
    run = cfg["name"]
    w = Writer(f"{GENERATED}/{run}/SDG_{run}_{name}", n_frames, fps)
    interp = KI.LINEAR if hold == 1 else KI.CONSTANT
    cam_actor = scene.camera
    cam_b = w.bind(cam_actor)
    cut = w.seq.add_track(unreal.MovieSceneCameraCutTrack)
    cs = cut.add_section()
    cs.set_range(0, n_frames)
    cs.set_camera_binding_id(w.seq.get_binding_id(cam_b))
    tsec = Writer.section(cam_b.add_track(unreal.MovieScene3DTransformTrack))
    key_transform(tsec, [(i * hold, p[0:6] + [1.0, 1.0, 1.0]) for i, p in enumerate(poses)], interp)
    fsec = w.property_section(cam_actor.get_cine_camera_component(), cam_actor, unreal.MovieSceneFloatTrack,
                              "CurrentFocalLength")
    key_scalar(fsec.get_all_channels()[0], [(i * hold, sc.hfov_to_focal_mm(p[6], SENSOR_W)) for i, p in enumerate(poses)], interp)

    for rz_name, (rz, kind, targets, vals, every, smooth) in rz_values.items():
        ip = KI.LINEAR if smooth else KI.CONSTANT
        for (obj, owner), series in zip(targets, vals):
            write_randomizer(w, scene, rz, kind, obj, owner, series_keys(series, hold, every, smooth), ip)
    w.save()

    # --- manifest
    frames = []
    for i, p in enumerate(poses):
        fr = {"seq": i * hold + hold - 1, "cam": rounded(p, 4),
              "params": {k: rounded(v[i]) for k, v in s_all.items()}}
        if obj_tf:
            fr["objects"] = {k: rounded(obj_tf[k][i]) for k in obj_tf}
        frames.append(fr)
    exr_dir = os.path.join(raw_run, name).replace("\\", "/")
    if os.path.isdir(exr_dir):
        shutil.rmtree(exr_dir)
    os.makedirs(exr_dir)
    pose_objs = [o for o in scene.objects.values() if o["pose_class"] is not None]
    kmax = max((len(o["keypoints"]) for o in pose_objs), default=0)
    flips = [o["flip_idx"] for o in pose_objs if o["flip_idx"] is not None]
    flip = flips[0] if flips and all(f == flips[0] and len(f) == kmax for f in flips) and \
        len(flips) == len(pose_objs) else None
    man = {
        "version": 1, "run": run, "stage": name, "stage_index": si, "sequence": w.path,
        "resolution": cfg["render"]["resolution"], "mode": mode, "hold": hold, "frames": frames,
        "instances": {str(k): v for k, v in sorted(scene.instances.items())},
        "objects": {k: {kk: vv for kk, vv in o.items() if not kk.startswith("_")} for k, o in scene.objects.items()},
        "seg_classes": cfg["seg_classes"], "pose_classes": cfg["pose_classes"],
        "num_keypoints": kmax, "flip_idx": flip,
    }
    man_path = os.path.join(raw_run, f"{name}.manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(man, f)
    log(f"stage {name}: {mode}, {n} images ({n_frames} rendered frames), "
        f"{len(rz_values)} randomizers -> {w.path}  [{time.time() - t0:.1f} s]")
    return {"name": name, "sequence": w.path, "manifest": man_path.replace("\\", "/"), "exr_dir": exr_dir,
            "frames": n, "hold": hold, "seq_frames": n_frames}


def transform_series(s, rz, tgt, n, every, smooth):
    """Per-sample transforms [x, y, z, pitch, yaw, roll, sx, sy, sz] as keyed: relative to the attach
    parent (= world for unattached actors), like Sequencer's transform track."""
    obj, owner = tgt
    comp = obj.get_editor_property("root_component") if owner is None else obj
    base = list(vec(comp.get_editor_property("relative_location")) + rot(comp.get_editor_property("relative_rotation"))
                + vec(comp.get_editor_property("relative_scale3d")))
    offset = bool(rz.get("offset", True))
    cols = []
    groups = (("location", ("x", "y", "z"), 0), ("rotation", ("pitch", "yaw", "roll"), 3), ("scale", ("x", "y", "z"), 6))
    for key, axes, start in groups:
        spec = rz.get(key, {})
        uniform_scale = key == "scale" and not isinstance(spec, dict)
        u = s.series(spec, n, every, smooth) if uniform_scale else None
        for j, ax in enumerate(axes):
            b = base[start + j]
            if uniform_scale:
                cols.append([b * v if offset else v for v in u])
            elif ax in spec:
                vals = s.series(spec[ax], n, every, smooth)
                if key == "scale":
                    cols.append([b * v if offset else v for v in vals])
                else:
                    cols.append([b + v if offset else v for v in vals])
            else:
                cols.append([b] * n)
    return [[c[i] for c in cols] for i in range(n)]


def has_property(obj, prop):
    try:
        obj.get_editor_property(us.snake(prop))
        return True
    except Exception:
        return False


def retarget(rz, kind, targets):
    """Put tracks where Sequencer can apply them: material randomizers on the actor's mesh component,
    property randomizers on the component that actually has the property (so `component` is optional)."""
    out = []
    for obj, owner in targets:
        if owner is not None:
            out.append((obj, owner))
        elif kind in ("material_param", "material"):
            comps = list(obj.get_components_by_class(unreal.PrimitiveComponent))
            root = obj.get_editor_property("root_component")
            comp = root if root in comps else (comps[0] if comps else None)
            if comp is None:
                warn(f"randomizer '{rz['name']}': {obj.get_actor_label()} has no mesh component - skipped")
            else:
                out.append((comp, obj))
        elif kind in ("float", "double", "bool", "color", "vector") and not rz.get("path") \
                and not has_property(obj, rz["property"]):
            comp = next((c for c in obj.get_components_by_class(unreal.ActorComponent) if has_property(c, rz["property"])), None)
            if comp is None:
                warn(f"randomizer '{rz['name']}': {obj.get_actor_label()} has no property {rz['property']} - skipped")
            else:
                out.append((comp, obj))
        else:
            out.append((obj, owner))
    return out


def make_movable(targets):
    """Sequencer can't move Static components (props dragged into a level are Static)."""
    for obj, owner in targets:
        comp = obj.get_editor_property("root_component") if owner is None else obj
        if comp and comp.get_editor_property("mobility") != unreal.ComponentMobility.MOVABLE:
            comp.set_mobility(unreal.ComponentMobility.MOVABLE)
            log(f"set {(owner or obj).get_actor_label()}.{comp.get_name()} to Movable so it can be randomized (save the level)")


def world_series(actor, series):
    """Keyed (parent-relative) actor transforms -> world transforms, for keypoints and camera aim."""
    parent = actor.get_editor_property("root_component").get_attach_parent()
    if parent is None:
        return series
    pw = parent.get_world_transform()
    out = []
    for t in series:
        rel = unreal.Transform(unreal.Vector(*t[0:3]), unreal.Rotator(pitch=t[3], yaw=t[4], roll=t[5]), unreal.Vector(*t[6:9]))
        w = unreal.MathLibrary.compose_transforms(rel, pw)
        out.append(list(vec(w.translation) + rot(w.rotation.rotator()) + vec(w.scale3d)))
    return out


def current_value(rz, kind, tgt, scene):
    if kind == "postprocess":
        return scene.ppv.get_editor_property("settings").get_editor_property(us.snake(rz["postprocess"]))
    try:
        return tgt[0].get_editor_property(us.snake(rz["property"]))
    except Exception:
        raise RuntimeError(f"randomizer '{rz['name']}': can't read {rz['property']} from "
                           f"{tgt[0].get_name()} for mode = {rz.get('mode')}") from None


def apply_mode(rz, kind, tgt, scene, raw):
    """mode = set (default) | add | multiply - the latter two are relative to the value in the level."""
    mode = rz.get("mode", "set")
    if mode == "set":
        return raw
    base = float(current_value(rz, kind, tgt, scene))
    return [base + v if mode == "add" else base * v for v in raw]


def vector_series(s, rz, tgt, n, every, smooth):
    """FVector properties (e.g. LightFunctionScale): value = one spec for all axes or {x=.., y=.., z=..}."""
    b = current_value(rz, "vector", tgt, None)
    base = (b.x, b.y, b.z)
    mode, spec = rz.get("mode", "set"), rz["value"]
    if isinstance(spec, dict) and not sc._is_value_spec(spec):
        cols = [s.series(spec[ax], n, every, smooth) if ax in spec else None for ax in "xyz"]
    else:
        u = s.series(spec, n, every, smooth)
        cols = [u, u, u]
    out = []
    for i in range(n):
        v = []
        for a in range(3):
            c = cols[a]
            v.append(base[a] if c is None else base[a] + c[i] if mode == "add" else base[a] * c[i] if mode == "multiply" else c[i])
        out.append(tuple(v))
    return out


def write_randomizer(w, scene, rz, kind, obj, owner, keys, ip):
    if kind == "transform":
        sec = Writer.section(w.bind(obj, owner).add_track(unreal.MovieScene3DTransformTrack))
        key_transform(sec, keys, ip)
    elif kind == "vector":
        cls = unreal.MovieSceneFloatVectorTrack if rz.get("precision") == "float" else unreal.MovieSceneDoubleVectorTrack
        t = w.bind(obj, owner).add_track(cls)
        t.set_num_channels_used(3)
        t.set_property_name_and_path(rz["property"], rz.get("path") or rz["property"])
        sec = Writer.section(t)
        chs = sorted(sec.get_all_channels(),
                     key=lambda c: "XYZW".index(str(c.get_editor_property("channel_name")).split(".")[-1][0].upper()))
        for a in range(3):
            key_scalar(chs[a], [(f, v[a]) for f, v in keys], ip)
    elif kind in ("float", "double"):
        cls = unreal.MovieSceneFloatTrack if kind == "float" else unreal.MovieSceneDoubleTrack
        sec = w.property_section(obj, owner, cls, rz["property"], rz.get("path"))
        key_scalar(sec.get_all_channels()[0], keys, ip)
    elif kind == "bool":
        sec = w.property_section(obj, owner, unreal.MovieSceneBoolTrack, rz["property"], rz.get("path"))
        key_bool(sec.get_all_channels()[0], keys)
    elif kind == "color":
        sec = w.property_section(obj, owner, unreal.MovieSceneColorTrack, rz["property"], rz.get("path"))
        key_color(sec, keys, ip)
    elif kind == "visible":
        hidden = [(f, not v) for f, v in keys]
        targets = [(obj, owner)]
        actor = obj if isinstance(obj, unreal.Actor) else obj.get_owner()
        key = us.tag_value(actor, "sdg.key:") if actor else None
        targets += [(c, None) for c in scene.copies.get(key, [])]          # its label copies hide with it
        for o, ow in targets:
            sec = w.property_section(o, ow, unreal.MovieSceneVisibilityTrack, "bHiddenInGame")
            key_bool(sec.get_all_channels()[0], hidden)
    elif kind == "postprocess":
        setting = rz["postprocess"]
        is_color = isinstance(keys[0][1], tuple)
        cls = unreal.MovieSceneColorTrack if is_color else unreal.MovieSceneFloatTrack
        sec = w.property_section(scene.ppv, None, cls, setting, f"Settings.{setting}")
        key_color(sec, keys, ip) if is_color else key_scalar(sec.get_all_channels()[0], keys, ip)
    elif kind == "mpc":
        mpc = unreal.load_asset(rz["mpc"])
        if mpc is None:
            raise RuntimeError(f"randomizer '{rz['name']}': {rz['mpc']} not found - run setup")
        track = next((t for t in w.seq.find_tracks_by_type(unreal.MovieSceneMaterialParameterCollectionTrack)
                      if t.get_editor_property("mpc") == mpc), None)
        if track is None:
            track = w.seq.add_track(unreal.MovieSceneMaterialParameterCollectionTrack)
            track.set_editor_property("mpc", mpc)
            track.add_section()
        sec = track.get_sections()[0]
        for f, v in keys:
            if isinstance(v, tuple):
                sec.add_color_parameter_key(rz["param"], w.tick(f), unreal.LinearColor(*(tuple(v) + (1.0,))[:4]), ip)
            else:
                sec.add_scalar_parameter_key(rz["param"], w.tick(f), float(v), ip)
    elif kind == "material_param":
        track = w.bind(obj, owner).add_track(unreal.MovieSceneComponentMaterialTrack)
        info = unreal.ComponentMaterialInfo()
        info.set_editor_property("material_slot_index", int(rz["material_slot"]))
        info.set_editor_property("material_type", unreal.ComponentMaterialType.INDEXED_MATERIAL)
        track.set_material_info(info)
        sec = track.add_section()
        sec.set_start_frame_bounded(False)
        sec.set_end_frame_bounded(False)
        pinfo = unreal.MaterialParameterInfo()
        pinfo.set_editor_property("name", rz.get("param") or rz["property"])
        for f, v in keys:
            if isinstance(v, tuple):
                sec.add_color_parameter_key(pinfo, w.tick(f), unreal.LinearColor(*(tuple(v) + (1.0,))[:4]), "", "",
                                            unreal.ParameterChannelNames(), ip)
            else:
                sec.add_scalar_parameter_key(pinfo, w.tick(f), float(v), "", "", ip)
    elif kind == "material":
        track = w.bind(obj, owner).add_track(unreal.MovieScenePrimitiveMaterialTrack)
        track.set_material_index(int(rz["material_slot"]))
        sec = Writer.section(track)
        ch = sec.get_all_channels()[0]
        for f, v in keys:
            mat = unreal.load_asset(v)
            if mat is None:
                raise RuntimeError(f"randomizer '{rz['name']}': material {v} not found")
            ch.add_key(time=unreal.FrameNumber(f), new_value=mat)
    else:
        raise RuntimeError(f"randomizer '{rz['name']}': unsupported type {kind}")


# ============================================================================ entry
def run(cfg, quick=0, only=None):
    t0 = time.time()
    world = us.editor_world()
    level = world.get_path_name().split(".")[0]
    if level != cfg["map"]:
        raise RuntimeError(f"open {cfg['map']} first (the open level is {level}); sequences bind to its actors")
    unreal.LevelSequenceEditorBlueprintLibrary.close_level_sequence()
    scene = Scene(cfg)
    us.ensure_postprocess(cfg)                        # override flags for the randomized settings
    raw_run = os.path.join(cfg["output"]["raw_dir"], cfg["name"]).replace("\\", "/")
    os.makedirs(raw_run, exist_ok=True)
    for flag in ("_DONE", "_FAILED"):
        if os.path.exists(os.path.join(raw_run, flag)):
            os.remove(os.path.join(raw_run, flag))

    stages = []
    for si, st in enumerate(cfg["stages"]):
        if only and st["name"] not in only:
            continue
        stages.append(bake_stage(cfg, scene, si, st, quick, raw_run))

    dataset = os.path.join(cfg["output"]["dataset_dir"], cfg["name"]).replace("\\", "/")
    run_info = {
        "name": cfg["name"], "config": cfg["_path"], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "map": cfg["map"], "dataset": dataset, "raw": raw_run, "stages": stages, "quick": int(quick or 0),
        "output": cfg["output"], "post": cfg["post"], "render": cfg["render"], "water_fog": cfg["water_fog"],
        "seg_classes": cfg["seg_classes"], "pose_classes": cfg["pose_classes"],
    }
    with open(os.path.join(raw_run, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=1)
    total = sum(s["frames"] for s in stages)
    log(f"baked {len(stages)} stage(s), {total} images in {time.time() - t0:.1f} s -> {raw_run}/run.json")
    return run_info
