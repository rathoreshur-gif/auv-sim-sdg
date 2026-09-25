"""
Set up segmentation for the open level from SegPipeline/seg_config.json.
Run inside the editor (Output Log -> Cmd):

    py "<project>/SegPipeline/tools/setup_segmentation.py"

What it does (safe to re-run after editing the config):
  - Custom Depth-Stencil on (r.CustomDepth 3)
  - Class i in "classes" gets stencil i+1 and YOLO id i
  - M_SegMask: post-process material that writes  stencil * stencil_step  as grey,
    only where the tagged surface is actually visible (occluders cut it out)
  - RT_SegRGB / RT_SegMask render targets for the two scene captures
  - For each prop:
      "texture_regions": imports the region masks and adds one invisible copy of every
                         actor using that texture per region (Render in Main Pass off,
                         CustomDepth on). Copies are attached to the original.
      "actor":           sets CustomDepth + stencil directly on matching actors
                         (by "actor_label" wildcard and/or "actor_tag")
      "material_slots":  tags only the mesh parts whose material/slot name matches a pattern
                         (e.g. the yellow marking on a bin); invisible attached copies, no masks
                         needed. Actors matched by "static_mesh" path and/or "actor_label".
The level is NOT saved - check it (View Mode > Buffer Visualization > Custom Stencil), then save.
"""
import fnmatch
import json
import os
import sys

import unreal

# ------------------------------------------------------------------ config
_here = globals().get("__file__") or (sys.argv[0] if sys.argv and sys.argv[0] else "")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(_here))) if _here else \
    os.path.join(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_dir()), "SegPipeline")
with open(os.path.join(ROOT, "seg_config.json"), encoding="utf-8") as f:
    CFG = json.load(f)

CONTENT = CFG.get("content_path", "/Game/SegPipeline")
STEP = int(CFG.get("stencil_step", 10))
CLASSES = CFG["classes"]
STENCIL = {name: i + 1 for i, name in enumerate(CLASSES)}
CAP_W = int(CFG.get("capture", {}).get("width", 1280))
CAP_H = int(CFG.get("capture", {}).get("height", 720))
ACTOR_PREFIX = "Seg_"
LEGACY_PREFIXES = ("Stencil_",)          # copies made by the first one-off script
DEPTH_TOL_CM = 5.0                       # how far behind the visible surface still counts as visible

if len(CLASSES) * STEP > 255:
    raise RuntimeError(f"{len(CLASSES)} classes x stencil_step {STEP} > 255 - lower stencil_step")

at = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def log(msg):
    unreal.log(f"[SegPipeline] {msg}")


def warn(msg):
    unreal.log_warning(f"[SegPipeline] {msg}")


def get_or_create(folder, name, cls, factory):
    path = f"{folder}/{name}"
    if eal.does_asset_exist(path):
        return unreal.load_asset(path)
    return at.create_asset(name, folder, cls, factory)


def safe_set(obj, prop, value):
    try:
        obj.set_editor_property(prop, value)
    except Exception as e:  # property names shift a little between engine versions
        warn(f"  skipped {prop}: {e}")


def expr(mat, cls, x, y, **props):
    e = mel.create_material_expression(mat, cls, x, y)
    for k, v in props.items():
        e.set_editor_property(k, v)
    return e


def link(src, src_out, dst, dst_in):
    if not mel.connect_material_expressions(src, src_out, dst, dst_in):
        raise RuntimeError(f"could not connect {src.get_name()}.{src_out} -> {dst.get_name()}.{dst_in}")


# ------------------------------------------------------- 1. custom depth on
unreal.SystemLibrary.execute_console_command(world, "r.CustomDepth 3")

# ------------------------------------------------------------ 2. M_SegMask
def build_seg_mask():
    m = get_or_create(CONTENT, "M_SegMask", unreal.Material, unreal.MaterialFactoryNew())
    mel.delete_all_material_expressions(m)
    m.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    loc = None
    for n in ("BL_SCENE_COLOR_AFTER_TONEMAPPING", "BL_AFTER_TONEMAPPING"):
        loc = getattr(unreal.BlendableLocation, n, None)
        if loc is not None:
            break
    m.set_editor_property("blendable_location", loc)

    def scene_tex(tex_id, x, y):
        st = expr(m, unreal.MaterialExpressionSceneTexture, x, y, scene_texture_id=tex_id)
        r = expr(m, unreal.MaterialExpressionComponentMask, x + 250, y, r=True, g=False, b=False, a=False)
        link(st, "Color", r, "")
        return r

    stencil = scene_tex(unreal.SceneTextureId.PPI_CUSTOM_STENCIL, -1400, -250)
    cdepth = scene_tex(unreal.SceneTextureId.PPI_CUSTOM_DEPTH, -1400, 50)
    sdepth = scene_tex(unreal.SceneTextureId.PPI_SCENE_DEPTH, -1400, 250)

    # visible = saturate((SceneDepth + tol - CustomDepth) * 100)
    add = expr(m, unreal.MaterialExpressionAdd, -900, 250, const_b=DEPTH_TOL_CM)
    link(sdepth, "", add, "A")
    sub = expr(m, unreal.MaterialExpressionSubtract, -700, 150)
    link(add, "", sub, "A")
    link(cdepth, "", sub, "B")
    amp = expr(m, unreal.MaterialExpressionMultiply, -500, 150, const_b=100.0)
    link(sub, "", amp, "A")
    vis = expr(m, unreal.MaterialExpressionSaturate, -300, 150)
    link(amp, "", vis, "")

    # grey = stencil * step / 255
    grey = expr(m, unreal.MaterialExpressionMultiply, -700, -250, const_b=STEP / 255.0)
    link(stencil, "", grey, "A")
    out = expr(m, unreal.MaterialExpressionMultiply, -100, -100)
    link(grey, "", out, "A")
    link(vis, "", out, "B")
    mel.connect_material_property(out, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

    mel.recompile_material(m)
    eal.save_loaded_asset(m)
    return m


seg_mask = build_seg_mask()
log(f"M_SegMask ready (grey value = stencil x {STEP})")

# ----------------------------------------------------------- 3. render targets
def make_rt(name):
    rt = get_or_create(CONTENT, name, unreal.TextureRenderTarget2D, unreal.TextureRenderTargetFactoryNew())
    rt.set_editor_property("size_x", CAP_W)
    rt.set_editor_property("size_y", CAP_H)
    rt.set_editor_property("render_target_format", unreal.TextureRenderTargetFormat.RTF_RGBA8)
    rt.set_editor_property("clear_color", unreal.LinearColor(0, 0, 0, 1))
    eal.save_loaded_asset(rt)
    return rt


make_rt("RT_SegRGB")
make_rt("RT_SegMask")
log(f"render targets RT_SegRGB / RT_SegMask at {CAP_W}x{CAP_H}")

# ---------------------------------------------------- 4. cutout materials
def masked_unlit(mat):
    mel.delete_all_material_expressions(mat)
    mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    mat.set_editor_property("two_sided", True)


_cutout = []


def get_cutout(default_mask):
    """M_SegCutout needs a default texture on its parameter or it fails to compile (and then
    falls back to an opaque material that tags the whole surface), so it is built lazily with
    the first imported mask as the default."""
    if _cutout:
        return _cutout[0]
    m = get_or_create(CONTENT, "M_SegCutout", unreal.Material, unreal.MaterialFactoryNew())
    masked_unlit(m)
    tp = expr(m, unreal.MaterialExpressionTextureSampleParameter2D, -400, 0,
              parameter_name="Mask", sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_MASKS,
              texture=default_mask)
    mel.connect_material_property(tp, "R", unreal.MaterialProperty.MP_OPACITY_MASK)
    mel.recompile_material(m)
    eal.save_loaded_asset(m)
    _cutout.append(m)
    return m


nothing = get_or_create(CONTENT, "M_SegNone", unreal.Material, unreal.MaterialFactoryNew())
masked_unlit(nothing)
zero = expr(nothing, unreal.MaterialExpressionConstant, -300, 0, r=0.0)
mel.connect_material_property(zero, "", unreal.MaterialProperty.MP_OPACITY_MASK)
mel.recompile_material(nothing)
eal.save_loaded_asset(nothing)

# ------------------------------------------------------------- 5. props
removed = 0
for a in list(eas.get_all_level_actors()):
    if a.get_actor_label().startswith((ACTOR_PREFIX,) + LEGACY_PREFIXES):
        eas.destroy_actor(a)
        removed += 1
if removed:
    log(f"removed {removed} old segmentation copies")


def check_class(name, prop_name):
    if name not in STENCIL:
        raise RuntimeError(f"prop '{prop_name}' uses class '{name}' which is not in \"classes\"")
    return STENCIL[name]


def material_uses(mat, tex):
    if mat is None:
        return False
    if isinstance(mat, unreal.MaterialInstance):
        for p in mat.get_editor_property("texture_parameter_values"):
            if p.get_editor_property("parameter_value") == tex:
                return True
    base = mat.get_base_material()
    return base is not None and tex in mel.get_used_textures(base)


def tag_component(comp, stencil):
    safe_set(comp, "render_custom_depth", True)
    safe_set(comp, "custom_depth_stencil_value", stencil)


def setup_texture_regions(prop):
    name = prop["name"]
    folder = f"{CONTENT}/{name}"
    tex = unreal.load_asset(prop["texture"])
    if tex is None:
        raise RuntimeError(f"prop '{name}': texture not found {prop['texture']}")

    regions = prop["regions"]
    tasks = []
    for cls, rel in regions.items():
        check_class(cls, name)
        t = unreal.AssetImportTask()
        t.filename = os.path.join(ROOT, rel).replace("\\", "/")
        t.destination_path = folder
        t.destination_name = f"T_SegMask_{name}_{cls}"
        t.automated = True
        t.replace_existing = True
        t.save = False
        tasks.append(t)
    at.import_asset_tasks(tasks)

    instances = {}
    for cls in regions:
        mtex = unreal.load_asset(f"{folder}/T_SegMask_{name}_{cls}")
        if mtex is None:
            raise RuntimeError(f"prop '{name}': import failed for {regions[cls]}")
        mtex.set_editor_property("srgb", False)
        mtex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
        eal.save_loaded_asset(mtex)

        mi = get_or_create(folder, f"MI_Seg_{name}_{cls}", unreal.MaterialInstanceConstant,
                           unreal.MaterialInstanceConstantFactoryNew())
        mel.set_material_instance_parent(mi, get_cutout(mtex))
        mel.set_material_instance_texture_parameter_value(mi, "Mask", mtex)
        mel.update_material_instance(mi)
        eal.save_loaded_asset(mi)
        instances[cls] = mi

    owners = []
    for a in eas.get_all_level_actors():
        for comp in a.get_components_by_class(unreal.StaticMeshComponent):
            if any(material_uses(comp.get_material(i), tex) for i in range(comp.get_num_materials())):
                owners.append(a)
                break
    if not owners:
        warn(f"prop '{name}': no actor in this level uses {prop['texture']}")
        return

    for owner in owners:
        slots = {c.get_name(): [material_uses(c.get_material(i), tex) for i in range(c.get_num_materials())]
                 for c in owner.get_components_by_class(unreal.StaticMeshComponent)}
        for cls, mi in instances.items():
            make_stencil_copy(owner, f"{name}_{cls}", STENCIL[cls],
                              lambda comp, i, used=slots: mi if used.get(comp.get_name(), [False] * 99)[i] else nothing)
    log(f"prop '{name}': {len(owners)} actor(s) x {len(instances)} regions "
        f"-> {len(owners) * len(instances)} invisible copies")


def make_stencil_copy(owner, tag, stencil, pick_material):
    """Invisible duplicate of `owner` that only writes custom depth/stencil.
    pick_material(copy_component, slot_index) -> material for that slot (M_SegNone = write nothing).
    Attached to the owner so it follows it when the prop is moved or randomized."""
    copy = eas.duplicate_actor(owner)
    copy.set_actor_label(f"{ACTOR_PREFIX}{tag}_{owner.get_actor_label()}")
    copy.set_actor_transform(owner.get_actor_transform(), False, False)
    copy.attach_to_actor(owner, "", unreal.AttachmentRule.KEEP_WORLD,
                         unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD, False)
    for comp in copy.get_components_by_class(unreal.StaticMeshComponent):
        for i in range(comp.get_num_materials()):
            comp.set_material(i, pick_material(comp, i))
        safe_set(comp, "render_in_main_pass", False)
        tag_component(comp, stencil)
        safe_set(comp, "cast_shadow", False)
        safe_set(comp, "affect_dynamic_indirect_lighting", False)
        safe_set(comp, "affect_distance_field_lighting", False)
        safe_set(comp, "visible_in_ray_tracing", False)
        safe_set(comp, "visible_in_reflection_captures", False)
        safe_set(comp, "visible_in_real_time_sky_captures", False)
        comp.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    return copy


_solid = []


def get_solid():
    """M_SegSolid: writes stencil over the whole slot (used by material_slots props)."""
    if not _solid:
        m = get_or_create(CONTENT, "M_SegSolid", unreal.Material, unreal.MaterialFactoryNew())
        masked_unlit(m)
        one = expr(m, unreal.MaterialExpressionConstant, -300, 0, r=1.0)
        mel.connect_material_property(one, "", unreal.MaterialProperty.MP_OPACITY_MASK)
        mel.recompile_material(m)
        eal.save_loaded_asset(m)
        _solid.append(m)
    return _solid[0]


def slot_names(comp):
    """(slot name, material asset name) per material slot of a static mesh component."""
    try:
        names = [str(n) for n in comp.get_material_slot_names()]
    except Exception:
        names = []
    out = []
    for i in range(comp.get_num_materials()):
        mat = comp.get_material(i)
        out.append((names[i] if i < len(names) else "", mat.get_name() if mat else ""))
    return out


def setup_material_slots(prop):
    """Tag the parts of a mesh whose MATERIAL matches a pattern - no mask textures needed.
    Actors are matched by static mesh asset ("static_mesh", wildcards ok) and/or "actor_label"."""
    name = prop["name"]
    mesh_pat, label_pat = prop.get("static_mesh"), prop.get("actor_label")
    if not mesh_pat and not label_pat:
        raise RuntimeError(f"prop '{name}': give \"static_mesh\" and/or \"actor_label\"")
    rules = prop["slots"]                              # {"*Yellow*": "bin_o", ...}
    for cls in rules.values():
        check_class(cls, name)
    solid = get_solid()

    def mesh_path(comp):
        sm = comp.get_editor_property("static_mesh")
        return sm.get_path_name().split(".")[0] if sm else ""

    owners = []
    for a in eas.get_all_level_actors():
        if a.get_actor_label().startswith(ACTOR_PREFIX):
            continue
        if label_pat and not fnmatch.fnmatch(a.get_actor_label(), label_pat):
            continue
        comps = a.get_components_by_class(unreal.StaticMeshComponent)
        if mesh_pat and not any(fnmatch.fnmatch(mesh_path(c), mesh_pat) for c in comps):
            continue
        if comps:
            owners.append(a)
    if not owners:
        warn(f"prop '{name}': no actor matched static_mesh={mesh_pat!r} label={label_pat!r}")
        return

    made = 0
    for owner in owners:
        # class of each slot, decided on the ORIGINAL (the copy's materials get replaced)
        slot_cls = {}
        for c in owner.get_components_by_class(unreal.StaticMeshComponent):
            per = []
            for slot, mat in slot_names(c):
                hit = next((cls for pat, cls in rules.items()
                            if fnmatch.fnmatch(slot, pat) or fnmatch.fnmatch(mat, pat)), None)
                per.append(hit)
            slot_cls[c.get_name()] = per
        found = {cls for per in slot_cls.values() for cls in per if cls}
        if not found:
            warn(f"prop '{name}': {owner.get_actor_label()} has no slot matching {list(rules)} "
                 f"(slots: {[slot_names(c) for c in owner.get_components_by_class(unreal.StaticMeshComponent)]})")
        for cls in sorted(found):
            make_stencil_copy(owner, f"{name}_{cls}", STENCIL[cls],
                              lambda comp, i, cls=cls: solid
                              if (slot_cls.get(comp.get_name(), []) + [None] * 99)[i] == cls else nothing)
            made += 1
    log(f"prop '{name}': {len(owners)} actor(s) -> {made} invisible copies")


def setup_actor(prop):
    name, cls = prop["name"], prop["class"]
    stencil = check_class(cls, name)
    label_pat, tag = prop.get("actor_label"), prop.get("actor_tag")
    if not label_pat and not tag:
        raise RuntimeError(f"prop '{name}': give \"actor_label\" and/or \"actor_tag\"")
    hits = []
    for a in eas.get_all_level_actors():
        if label_pat and not fnmatch.fnmatch(a.get_actor_label(), label_pat):
            continue
        if tag and not a.actor_has_tag(tag):
            continue
        comps = a.get_components_by_class(unreal.PrimitiveComponent)
        for c in comps:
            tag_component(c, stencil)
        if comps:
            hits.append(a.get_actor_label())
    if hits:
        log(f"prop '{name}': class '{cls}' (stencil {stencil}) on {hits}")
    else:
        warn(f"prop '{name}': no actor matched label={label_pat!r} tag={tag!r}")


for prop in CFG.get("props", []):
    kind = prop.get("type")
    if kind == "texture_regions":
        setup_texture_regions(prop)
    elif kind == "actor":
        setup_actor(prop)
    elif kind == "material_slots":
        setup_material_slots(prop)
    else:
        raise RuntimeError(f"prop '{prop.get('name')}': unknown type {kind!r}")

log("class table (YOLO id / stencil / grey value in mask):")
for i, c in enumerate(CLASSES):
    log(f"   {i}  {c:<14} stencil {i + 1:<3} grey {(i + 1) * STEP}")
log("done - check View Mode > Buffer Visualization > Custom Stencil, then save the level.")
