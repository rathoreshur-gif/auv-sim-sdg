"""
One-time scene setup (re-run after changing props in the config or in the level):

  - every prop instance and every labelled part of it gets its OWN stencil id (1..255), so two
    boards or two bins that touch still come out as separate YOLO instances
  - "texture_regions" / "material_slots" props get invisible copies that only write custom
    depth: one per class region (as before) plus a "body" copy for the whole object, which gives
    the pose bounding box. The body copy is pushed 0.5 cm away from the camera, so regions win on top of it.
  - texture_regions props with `images`: each image is imported next to the masks with a material
    instance that shows it instead of the texture, plus one cutout material per class mask of every
    image. The "image" randomizer (ue_bake) swaps both on the same frames.
  - M_SDG_Labels: post-process material rendered by MRQ as an extra pass.
    R = stencil id where that surface is the visible one (0 elsewhere), G = scene depth in cm
  - SDG_Camera (CineCamera, filmback matched to the output aspect ratio, no depth of field) and
    SDG_PostProcess (unbound volume that carries the randomized post-process settings)
  - material parameter collections named by "mpc" randomizers, with any missing parameters

The level is changed, not saved - save it afterwards (Ctrl+S).
"""
import os

import unreal

import sdg_common as sc

CONTENT = "/Game/SDG"
COPY_PREFIX = "SDG_Seg_"
OLD_COPY_PREFIXES = ("Seg_", "Stencil_")      # copies made by setup_segmentation.py / the first script
DEPTH_TOL_CM = 5.0                             # surfaces this far behind the visible one still count
BODY_OFFSET_CM = 0.5                           # body copy is pushed back by this much

at = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary


def log(msg):
    unreal.log(f"[SDG] {msg}")


def warn(msg):
    unreal.log_warning(f"[SDG] {msg}")


def eas():
    return unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def editor_world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def get_or_create(folder, name, cls, factory):
    path = f"{folder}/{name}"
    if eal.does_asset_exist(path):
        return unreal.load_asset(path)
    return at.create_asset(name, folder, cls, factory)


def safe_set(obj, prop, value):
    try:
        obj.set_editor_property(prop, value)
    except Exception as e:                        # property names shift a little between versions
        warn(f"  skipped {prop}: {e}")


def expr(mat, cls, x, y, **props):
    e = mel.create_material_expression(mat, cls, x, y)
    for k, v in props.items():
        e.set_editor_property(k, v)
    return e


def link(src, src_out, dst, dst_in):
    if not mel.connect_material_expressions(src, src_out, dst, dst_in):
        raise RuntimeError(f"could not connect {src.get_name()}.{src_out} -> {dst.get_name()}.{dst_in}")


def tags_of(actor):
    return [str(t) for t in actor.get_editor_property("tags")]


def set_tags(actor, tags):
    actor.set_editor_property("tags", [unreal.Name(t) for t in tags])


def tag_value(actor, prefix):
    for t in tags_of(actor):
        if t.startswith(prefix):
            return t[len(prefix):]
    return None


def is_sdg_helper(actor):
    label = actor.get_actor_label()
    return label.startswith(("SDG_",) + OLD_COPY_PREFIXES) or "SDG" in tags_of(actor)


# ============================================================================ materials
def after_tonemapping():
    for n in ("BL_SCENE_COLOR_AFTER_TONEMAPPING", "BL_AFTER_TONEMAPPING"):
        loc = getattr(unreal.BlendableLocation, n, None)
        if loc is not None:
            return loc
    raise RuntimeError("no after-tonemapping blendable location in this engine version")


def build_label_material():
    """R = stencil id if the tagged surface is the visible one, G = scene depth (cm)."""
    m = get_or_create(CONTENT, "M_SDG_Labels", unreal.Material, unreal.MaterialFactoryNew())
    mel.delete_all_material_expressions(m)
    m.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    m.set_editor_property("blendable_location", after_tonemapping())

    def scene_tex(tex_id, x, y):
        st = expr(m, unreal.MaterialExpressionSceneTexture, x, y, scene_texture_id=tex_id)
        r = expr(m, unreal.MaterialExpressionComponentMask, x + 250, y, r=True, g=False, b=False, a=False)
        link(st, "Color", r, "")
        return r

    stencil = scene_tex(unreal.SceneTextureId.PPI_CUSTOM_STENCIL, -1500, -300)
    cdepth = scene_tex(unreal.SceneTextureId.PPI_CUSTOM_DEPTH, -1500, 0)
    sdepth = scene_tex(unreal.SceneTextureId.PPI_SCENE_DEPTH, -1500, 250)

    # visible = saturate((SceneDepth + tol - CustomDepth) * 100)
    add = expr(m, unreal.MaterialExpressionAdd, -1000, 150, const_b=DEPTH_TOL_CM)
    link(sdepth, "", add, "A")
    sub = expr(m, unreal.MaterialExpressionSubtract, -800, 50)
    link(add, "", sub, "A")
    link(cdepth, "", sub, "B")
    amp = expr(m, unreal.MaterialExpressionMultiply, -600, 50, const_b=100.0)
    link(sub, "", amp, "A")
    vis = expr(m, unreal.MaterialExpressionSaturate, -400, 50)
    link(amp, "", vis, "")
    sid = expr(m, unreal.MaterialExpressionMultiply, -250, -250)
    link(stencil, "", sid, "A")
    link(vis, "", sid, "B")

    cap = expr(m, unreal.MaterialExpressionConstant, -1000, 400, r=60000.0)     # stay inside float16
    dmin = expr(m, unreal.MaterialExpressionMin, -800, 300)
    link(sdepth, "", dmin, "A")
    link(cap, "", dmin, "B")

    rg = expr(m, unreal.MaterialExpressionAppendVector, -50, -100)
    link(sid, "", rg, "A")
    link(dmin, "", rg, "B")
    zero = expr(m, unreal.MaterialExpressionConstant, -250, 150, r=0.0)
    rgb = expr(m, unreal.MaterialExpressionAppendVector, 150, -50)
    link(rg, "", rgb, "A")
    link(zero, "", rgb, "B")
    mel.connect_material_property(rgb, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(m)
    eal.save_loaded_asset(m)
    return m


def masked_unlit(mat):
    mel.delete_all_material_expressions(mat)
    mat.set_editor_property("blend_mode", unreal.BlendMode.BLEND_MASKED)
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    mat.set_editor_property("two_sided", True)


def build_constant_material(name, opacity, depth_offset=0.0):
    m = get_or_create(CONTENT, name, unreal.Material, unreal.MaterialFactoryNew())
    masked_unlit(m)
    c = expr(m, unreal.MaterialExpressionConstant, -300, 0, r=opacity)
    mel.connect_material_property(c, "", unreal.MaterialProperty.MP_OPACITY_MASK)
    if depth_offset:
        # push every vertex `depth_offset` cm away from the camera along its view ray (Pixel Depth
        # Offset would do the same, but it is hidden from Python)
        cam = expr(m, unreal.MaterialExpressionCameraPositionWS, -900, 150)
        wp = expr(m, unreal.MaterialExpressionWorldPosition, -900, 300)
        away = expr(m, unreal.MaterialExpressionSubtract, -700, 220)
        link(wp, "", away, "A")
        link(cam, "", away, "B")
        unit = expr(m, unreal.MaterialExpressionNormalize, -500, 220)
        link(away, "", unit, "")
        push = expr(m, unreal.MaterialExpressionMultiply, -300, 220, const_b=depth_offset)
        link(unit, "", push, "A")
        mel.connect_material_property(push, "", unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)
    mel.recompile_material(m)
    eal.save_loaded_asset(m)
    return m


_cutout = []


def get_cutout(default_mask, rebuild=True):
    """Masked material with a "Mask" texture parameter. Built lazily: it needs a default
    texture on the parameter or it fails to compile (and then tags the whole surface).
    rebuild=False (bake) reuses the one setup made."""
    if not _cutout and not rebuild and eal.does_asset_exist(f"{CONTENT}/M_SDG_Cutout"):
        _cutout.append(unreal.load_asset(f"{CONTENT}/M_SDG_Cutout"))
    if not _cutout:
        m = get_or_create(CONTENT, "M_SDG_Cutout", unreal.Material, unreal.MaterialFactoryNew())
        masked_unlit(m)
        tp = expr(m, unreal.MaterialExpressionTextureSampleParameter2D, -400, 0,
                  parameter_name="Mask", sampler_type=unreal.MaterialSamplerType.SAMPLERTYPE_MASKS,
                  texture=default_mask)
        mel.connect_material_property(tp, "R", unreal.MaterialProperty.MP_OPACITY_MASK)
        mel.recompile_material(m)
        eal.save_loaded_asset(m)
        _cutout.append(m)
    return _cutout[0]


# ============================================================================ matching
def material_uses(mat, tex):
    if mat is None:
        return False
    if isinstance(mat, unreal.MaterialInstance):
        for p in mat.get_editor_property("texture_parameter_values"):
            if p.get_editor_property("parameter_value") == tex:
                return True
    base = mat.get_base_material()
    return base is not None and tex in mel.get_used_textures(base)


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


def mesh_path(comp):
    sm = comp.get_editor_property("static_mesh")
    return sm.get_path_name().split(".")[0] if sm else ""


def find_owners(prop, actors):
    """Level actors a prop entry applies to, in a stable order (so stencil ids don't shuffle)."""
    kind = prop["type"]
    hits = []
    tex = None
    if kind == "texture_regions":
        tex = unreal.load_asset(prop["texture"])
        if tex is None:
            raise RuntimeError(f"prop '{prop['name']}': texture not found {prop['texture']}")
    for a in actors:
        if is_sdg_helper(a):
            continue
        label = a.get_actor_label()
        if prop.get("actor_label") and not sc.wildcard(label, prop["actor_label"]):
            continue
        if prop.get("actor_tag") and not a.actor_has_tag(prop["actor_tag"]):
            continue
        comps = a.get_components_by_class(unreal.StaticMeshComponent)
        if prop.get("static_mesh") and not any(sc.wildcard(mesh_path(c), prop["static_mesh"]) for c in comps):
            continue
        if kind == "texture_regions":
            if not any(material_uses(c.get_material(i), tex) for c in comps for i in range(c.get_num_materials())):
                continue
        elif kind == "actor":
            comps = a.get_components_by_class(unreal.PrimitiveComponent)
        if comps:
            hits.append(a)
    return sorted(hits, key=lambda a: a.get_name())


# ============================================================================ tagging
class Allocator:
    def __init__(self):
        self.next = 1
        self.table = []

    def take(self, key, part, owner):
        if self.next > sc.MAX_STENCIL:
            raise RuntimeError("more than 255 labelled parts - custom stencil is 8 bit. Remove props or regions.")
        sid = self.next
        self.next += 1
        self.table.append((sid, key, part, owner.get_actor_label()))
        return sid


def tag_component(comp, stencil):
    safe_set(comp, "render_custom_depth", True)
    safe_set(comp, "custom_depth_stencil_value", stencil)


def make_copy(owner, key, part, stencil, pick_material):
    """Invisible duplicate of `owner` that only writes custom depth/stencil.
    pick_material(copy_component, slot_index) -> material for that slot. Attached to the owner so
    it follows it when the prop is moved or randomized."""
    copy = eas().duplicate_actor(owner)
    copy.set_actor_label(f"{COPY_PREFIX}{key.replace('#', '_')}_{part}")
    copy.set_actor_transform(owner.get_actor_transform(), False, False)
    copy.attach_to_actor(owner, "", unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD,
                         unreal.AttachmentRule.KEEP_WORLD, False)
    set_tags(copy, ["SDG", "sdg.copy", f"sdg.key:{key}", f"sdg.part:{part}"])
    for comp in copy.get_components_by_class(unreal.StaticMeshComponent):
        comp.set_mobility(unreal.ComponentMobility.MOVABLE)     # must follow the prop when it is moved
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


def clear_previous(actors):
    removed = 0
    for a in actors:
        if a.get_actor_label().startswith((COPY_PREFIX,) + OLD_COPY_PREFIXES) or "sdg.copy" in tags_of(a):
            eas().destroy_actor(a)
            removed += 1
        elif tag_value(a, "sdg.key:") is not None:              # an original we tagged last time
            set_tags(a, [t for t in tags_of(a) if not t.startswith("sdg.")])
            for c in a.get_components_by_class(unreal.PrimitiveComponent):
                tag_component(c, 0)
                safe_set(c, "render_custom_depth", False)
    if removed:
        log(f"removed {removed} old segmentation copies")


def import_textures(prop, jobs, rebuild):
    """jobs: [(png path, folder, asset name)] -> (texture assets, names imported now). rebuild=False only
    imports what is missing (bake); True re-imports everything (setup: picks up edited PNGs)."""
    tasks, fresh = [], set()
    for src, folder, name in jobs:
        if not rebuild and eal.does_asset_exist(f"{folder}/{name}"):
            continue
        fresh.add(name)
        t = unreal.AssetImportTask()
        t.filename = src
        t.destination_path = folder
        t.destination_name = name
        t.automated = True
        t.replace_existing = True
        t.save = False
        tasks.append(t)
    if tasks:
        at.import_asset_tasks(tasks)
    out = []
    for src, folder, name in jobs:
        tex = unreal.load_asset(f"{folder}/{name}")
        if tex is None:
            raise RuntimeError(f"prop '{prop['name']}': import failed for {src}")
        out.append(tex)
    return out, fresh


def region_materials(prop, folder, idx, regions, rebuild):
    """Cutout material instance per class mask of one image: {class: MI}.
    Image 0 (the texture in the level) keeps the asset names it always had."""
    sfx = "" if idx == 0 else f"_img{idx}"
    classes = list(regions)
    names = [f"T_SDGMask_{prop['name']}_{cls}{sfx}" for cls in classes]
    texs, fresh = import_textures(prop, [(regions[c], folder, n) for c, n in zip(classes, names)], rebuild)
    out = {}
    for cls, name, tex in zip(classes, names, texs):
        if name in fresh:
            tex.set_editor_property("srgb", False)
            tex.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_MASKS)
            eal.save_loaded_asset(tex)
        path = f"{folder}/MI_SDG_{prop['name']}_{cls}{sfx}"
        if not rebuild and name not in fresh and eal.does_asset_exist(path):
            out[cls] = unreal.load_asset(path)
            continue
        mi = get_or_create(folder, f"MI_SDG_{prop['name']}_{cls}{sfx}", unreal.MaterialInstanceConstant,
                           unreal.MaterialInstanceConstantFactoryNew())
        mel.set_material_instance_parent(mi, get_cutout(tex, rebuild))
        mel.set_material_instance_texture_parameter_value(mi, "Mask", tex)
        mel.update_material_instance(mi)
        eal.save_loaded_asset(mi)
        out[cls] = mi
    return out


TEXTURE_SETTINGS = ("srgb", "compression_settings", "lod_group", "mip_gen_settings", "filter",
                    "address_x", "address_y", "never_stream")


def image_textures(prop, folder, tex, images, rebuild):
    """The prop's other images as textures with the same settings as its texture: [tex, image 1, ...]."""
    jobs = [(img["image"], folder, f"T_SDGImage_{prop['name']}_{i}") for i, img in enumerate(images) if i > 0]
    texs, fresh = import_textures(prop, jobs, rebuild)
    for (src, _, name), t in zip(jobs, texs):
        if name not in fresh:
            continue
        for p in TEXTURE_SETTINGS:
            try:
                t.set_editor_property(p, tex.get_editor_property(p))
            except Exception:
                pass
        eal.save_loaded_asset(t)
        try:
            a, b = (tex.blueprint_get_size_x(), tex.blueprint_get_size_y()), (t.blueprint_get_size_x(), t.blueprint_get_size_y())
            if a != b:
                warn(f"prop '{prop['name']}': {os.path.basename(src)} is {b[0]}x{b[1]} but the texture is {a[0]}x{a[1]} "
                     f"- the image must have the same layout (UVs) as the texture")
        except Exception:
            pass
    return [tex] + texs


def texture_params(mat, tex):
    """Names of mat's texture parameters that currently hold tex."""
    try:
        names = mel.get_texture_parameter_names(mat)
    except Exception:
        return []
    out = []
    for n in names:
        try:
            if isinstance(mat, unreal.MaterialInstance):
                v = mel.get_material_instance_texture_parameter_value(mat, n)
            else:
                v = mel.get_material_default_texture_parameter_value(mat, n)
        except Exception:
            continue
        if v == tex:
            out.append(n)
    return out


def image_material(prop, folder, mat, tex, idx, image, rebuild):
    """Instance of `mat` that shows image `idx` wherever mat shows the prop's texture."""
    name = f"MI_SDGImage_{prop['name']}_{idx}_{mat.get_name()}"
    if not rebuild and eal.does_asset_exist(f"{folder}/{name}"):
        return unreal.load_asset(f"{folder}/{name}")
    params = texture_params(mat, tex)
    if not params:
        raise RuntimeError(
            f"prop '{prop['name']}': {mat.get_path_name()} uses {tex.get_name()} without a texture parameter, so "
            f"the image can't be swapped. In that material right-click the Texture Sample -> Convert to Parameter, "
            f"save, and run setup again.")
    mi = get_or_create(folder, name, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(mi, mat)
    for p in params:
        mel.set_material_instance_texture_parameter_value(mi, p, image)
    mel.update_material_instance(mi)
    eal.save_loaded_asset(mi)
    return mi


def textured_slots(actor, tex):
    """{static mesh component name: [slot indices whose material shows tex]}. Label copies are
    duplicates of their owner, so the same names and indices apply to them."""
    return {c.get_name(): [i for i in range(c.get_num_materials()) if material_uses(c.get_material(i), tex)]
            for c in actor.get_components_by_class(unreal.StaticMeshComponent)}


def prepare_images(prop, owners, seg, rebuild):
    """Everything a texture_regions prop needs, per image number (0 = its texture as in the level,
    1.. = its `images`):
        labels     {class: [cutout MI or None (class not on that image)] per image}   -> label copies
        materials  {path of a material showing the texture: [material per image]}  -> the prop's own meshes
    rebuild=True (setup) re-imports every PNG; False (bake) only creates what is missing."""
    folder = f"{CONTENT}/{prop['name']}"
    tex = unreal.load_asset(prop["texture"])
    if tex is None:
        raise RuntimeError(f"prop '{prop['name']}': texture not found {prop['texture']}")
    images = sc.prop_images(prop, seg)
    labels = {}
    for idx, img in enumerate(images):
        for cls, mi in region_materials(prop, folder, idx, img["regions"], rebuild).items():
            labels.setdefault(cls, [None] * len(images))[idx] = mi
    materials = {}
    if len(images) > 1:
        textures = image_textures(prop, folder, tex, images, rebuild)
        for owner in owners:
            for c in owner.get_components_by_class(unreal.StaticMeshComponent):
                for i in range(c.get_num_materials()):
                    mat = c.get_material(i)
                    if mat is None or mat.get_path_name() in materials or not material_uses(mat, tex):
                        continue
                    materials[mat.get_path_name()] = [mat] + [image_material(prop, folder, mat, tex, k, textures[k], rebuild)
                                                              for k in range(1, len(images))]
    return {"count": len(images), "texture": tex, "labels": labels, "materials": materials}


def setup_props(cfg, alloc):
    actors = list(eas().get_all_level_actors())
    clear_previous(actors)
    actors = list(eas().get_all_level_actors())
    none_mat = build_constant_material("M_SDG_None", 0.0)
    solid = build_constant_material("M_SDG_Solid", 1.0)
    body = build_constant_material("M_SDG_Body", 1.0, BODY_OFFSET_CM)

    for prop in cfg["props"]:
        owners = find_owners(prop, actors)
        if not owners:
            warn(f"prop '{prop['name']}': nothing in this level matched it")
            continue
        kind = prop["type"]
        regions, tex = {}, None
        if kind == "texture_regions":
            prep = prepare_images(prop, owners, cfg["seg_classes"], rebuild=True)
            regions, tex = prep["labels"], prep["texture"]
            if prep["count"] > 1:
                log(f"prop '{prop['name']}': {prep['count']} images (texture + {prep['count'] - 1}), "
                    f"classes {sorted(regions)}")
        groups = stacked_groups(owners)
        for k, group in enumerate(groups):
            key = f"{prop['name']}#{k}"
            if len(group) > 1:
                warn(f"prop '{prop['name']}': {[o.get_actor_label() for o in group]} sit in the same place "
                     f"(duplicated in place?) - labelled as ONE object. Delete the extra copies if they are accidental.")
            sids = {}

            def take(part, owner):          # stacked copies share the ids: whichever wins a pixel, the label is the same
                if part not in sids:
                    sids[part] = alloc.take(key, part, owner)
                return sids[part]

            for owner in group:
                label_owner(prop, kind, key, owner, regions, tex, take, none_mat, solid, body)
        log(f"prop '{prop['name']}' ({kind}): {len(owners)} actor(s) -> {len(groups)} object(s)")


def stacked_groups(owners):
    boxes = []
    for a in owners:
        origin, ext = a.get_actor_bounds(False)
        o, e = (origin.x, origin.y, origin.z), (ext.x, ext.y, ext.z)
        boxes.append(([o[i] - e[i] for i in range(3)], [o[i] + e[i] for i in range(3)]))
    return [[owners[i] for i in g] for g in sc.group_stacked(boxes)]


def label_owner(prop, kind, key, owner, regions, tex, take, none_mat, solid, body):
    keep = [t for t in tags_of(owner) if not t.startswith("sdg.")]
    set_tags(owner, keep + ["sdg.obj", f"sdg.key:{key}", f"sdg.prop:{prop['name']}"])

    if kind == "actor":
        part = prop.get("class") or "__body__"
        sid = take(part, owner)
        for c in owner.get_components_by_class(unreal.PrimitiveComponent):
            tag_component(c, sid)
        set_tags(owner, tags_of(owner) + [f"sdg.part:{part}"])
        return

    make_copy(owner, key, "__body__", take("__body__", owner), lambda comp, i: body)
    if kind == "texture_regions":
        # one copy per class on ANY of the prop's images; it shows the level texture's mask (image 0),
        # and the "image" randomizer swaps in the mask of the image being shown
        used = textured_slots(owner, tex)
        for cls, per_image in regions.items():
            mi = per_image[0] or none_mat
            make_copy(owner, key, cls, take(cls, owner),
                      lambda comp, i, mi=mi: mi if i in used.get(comp.get_name(), []) else none_mat)
    else:                                                   # material_slots
        slot_cls = {}
        for c in owner.get_components_by_class(unreal.StaticMeshComponent):
            slot_cls[c.get_name()] = [
                next((cls for pat, cls in prop["slots"].items()
                      if sc.wildcard(slot, pat) or sc.wildcard(mat, pat)), None)
                for slot, mat in slot_names(c)]
        found = sorted({cls for per in slot_cls.values() for cls in per if cls})
        if not found:
            warn(f"prop '{prop['name']}': {owner.get_actor_label()} has no slot matching {list(prop['slots'])} "
                 f"(slots: {[slot_names(c) for c in owner.get_components_by_class(unreal.StaticMeshComponent)]})")
        for cls in found:
            make_copy(owner, key, cls, take(cls, owner),
                      lambda comp, i, cls=cls: solid if (slot_cls.get(comp.get_name(), []) + [None] * 99)[i] == cls else none_mat)


# ============================================================================ helper actors
def find_tagged(tag):
    return next((a for a in eas().get_all_level_actors() if a.actor_has_tag(tag)), None)


def ensure_camera(cfg):
    cam = find_tagged("sdg.camera")
    if cam is None:
        cam = eas().spawn_actor_from_class(unreal.CineCameraActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        cam.set_actor_label("SDG_Camera")
        set_tags(cam, ["SDG", "sdg.camera"])
        log("created SDG_Camera")
    comp = cam.get_cine_camera_component()
    w, h = cfg["render"]["resolution"]
    fb = comp.get_editor_property("filmback")
    fb.set_editor_property("sensor_width", 36.0)
    fb.set_editor_property("sensor_height", 36.0 * h / w)
    comp.set_editor_property("filmback", fb)
    lens = comp.get_editor_property("lens_settings")
    lens.set_editor_property("min_focal_length", 1.0)
    lens.set_editor_property("max_focal_length", 2000.0)
    comp.set_editor_property("lens_settings", lens)
    focus = comp.get_editor_property("focus_settings")
    focus.set_editor_property("focus_method", unreal.CameraFocusMethod.DISABLE)       # no depth of field
    comp.set_editor_property("focus_settings", focus)
    safe_set(comp, "constrain_aspect_ratio", False)
    return cam


def snake(name):
    """C++ property name -> Python name: AutoExposureBias -> auto_exposure_bias, bUseTemperature -> use_temperature."""
    if len(name) > 1 and name[0] == "b" and name[1].isupper():
        name = name[1:]
    out = ""
    for i, ch in enumerate(name):
        if ch.isupper() and i and (not name[i - 1].isupper() or (i + 1 < len(name) and name[i + 1].islower())):
            out += "_"
        out += ch.lower()
    return out


def ensure_postprocess(cfg):
    ppv = find_tagged("sdg.postprocess")
    if ppv is None:
        ppv = eas().spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        ppv.set_actor_label("SDG_PostProcess")
        set_tags(ppv, ["SDG", "sdg.postprocess"])
        log("created SDG_PostProcess")
    ppv.set_editor_property("unbound", True)
    ppv.set_editor_property("priority", 1000.0)
    ppv.set_editor_property("blend_weight", 1.0)
    s = ppv.get_editor_property("settings")
    s.set_editor_property("override_motion_blur_amount", True)
    s.set_editor_property("motion_blur_amount", 0.0)
    for rz in cfg["randomize"]:
        if sc.randomizer_type(rz) == "postprocess":
            try:        # a switched-off randomizer leaves the level's own value alone
                s.set_editor_property("override_" + snake(rz["postprocess"]), rz.get("enabled", True) is not False)
            except Exception:
                raise RuntimeError(f"randomizer '{rz['name']}': post-process setting {rz['postprocess']!r} "
                                   f"has no override flag - use the C++ name, e.g. AutoExposureBias") from None
    fog = unreal.load_asset(f"{CONTENT}/M_SDG_WaterFog")
    wb = s.get_editor_property("weighted_blendables")
    arr = [b for b in wb.get_editor_property("array") if fog is None or b.get_editor_property("object") != fog]
    if cfg["water_fog"]["enabled"]:
        if fog is None:
            fog = build_water_fog_material()
        b = unreal.WeightedBlendable()
        b.set_editor_property("weight", 1.0)
        b.set_editor_property("object", fog)
        arr.append(b)
    wb.set_editor_property("array", arr)
    s.set_editor_property("weighted_blendables", wb)
    ppv.set_editor_property("settings", s)
    return ppv


def build_water_fog_material():
    """Underwater haze on top of the scene: lerp(FogColor, scene, exp(-FogDensity * depth)).
    Both come from MPC_SDG, so the sequence can change them every frame."""
    mpc = unreal.load_asset(sc.SDG_MPC)
    if mpc is None:
        raise RuntimeError(f"{sc.SDG_MPC} missing - run setup")
    m = get_or_create(CONTENT, "M_SDG_WaterFog", unreal.Material, unreal.MaterialFactoryNew())
    mel.delete_all_material_expressions(m)
    m.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    m.set_editor_property("blendable_location", after_tonemapping())
    depth = expr(m, unreal.MaterialExpressionSceneTexture, -900, 0, scene_texture_id=unreal.SceneTextureId.PPI_SCENE_DEPTH)
    dmask = expr(m, unreal.MaterialExpressionComponentMask, -650, 0, r=True, g=False, b=False, a=False)
    link(depth, "Color", dmask, "")
    scene = expr(m, unreal.MaterialExpressionSceneTexture, -900, 250, scene_texture_id=unreal.SceneTextureId.PPI_POST_PROCESS_INPUT0)
    smask = expr(m, unreal.MaterialExpressionComponentMask, -650, 250, r=True, g=True, b=True, a=False)
    link(scene, "Color", smask, "")
    dens = expr(m, unreal.MaterialExpressionCollectionParameter, -650, -200, collection=mpc, parameter_name="FogDensity")
    color = expr(m, unreal.MaterialExpressionCollectionParameter, -900, -350, collection=mpc, parameter_name="FogColor")
    cmask = expr(m, unreal.MaterialExpressionComponentMask, -650, -350, r=True, g=True, b=True, a=False)
    link(color, "", cmask, "")
    inputs = []
    for n in ("Fog", "Scene", "D", "Z"):
        ci = unreal.CustomInput()
        ci.set_editor_property("input_name", n)
        inputs.append(ci)
    code = expr(m, unreal.MaterialExpressionCustom, -300, 0,
                code="return lerp(Fog, Scene, exp(-D * Z));",
                output_type=unreal.CustomMaterialOutputType.CMOT_FLOAT3, inputs=inputs)
    link(cmask, "", code, "Fog")
    link(smask, "", code, "Scene")
    link(dens, "", code, "D")
    link(dmask, "", code, "Z")
    mel.connect_material_property(code, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(m)
    eal.save_loaded_asset(m)
    return m


def ensure_mpcs(cfg):
    """Create the MPC params named in `mpc` randomizers. Their default is what the editor (and any frame
    outside a render) uses: the randomizer's `default`, else 1 / white (neutral for multipliers)."""
    wanted = {}
    for rz in cfg["randomize"]:
        if sc.randomizer_type(rz) == "mpc":
            is_color = isinstance(sc.Sampler(0).value(rz["value"]), tuple)
            wanted.setdefault(rz["mpc"], {})[rz["param"]] = (is_color, rz.get("default"))
    for path, params in wanted.items():
        folder, name = path.rsplit("/", 1)
        mpc = unreal.load_asset(path)
        if mpc is None:
            mpc = at.create_asset(name, folder, unreal.MaterialParameterCollection,
                                  unreal.MaterialParameterCollectionFactoryNew())
            log(f"created {path}")
        scalars = list(mpc.get_editor_property("scalar_parameters"))
        vectors = list(mpc.get_editor_property("vector_parameters"))
        have = {str(p.get_editor_property("parameter_name")): p for p in scalars + vectors}
        added, changed = [], False
        for pname, (is_color, default) in params.items():
            p = have.get(pname)
            if p is None:
                if is_color:
                    p = unreal.CollectionVectorParameter()
                    vectors.append(p)
                else:
                    p = unreal.CollectionScalarParameter()
                    scalars.append(p)
                p.set_editor_property("parameter_name", pname)
                added.append(pname)
                if default is None:
                    default = [1, 1, 1] if is_color else 1.0
            if default is None:
                continue                      # existing param without an explicit default: leave the user's value
            if is_color:
                c = list(default) + [1.0] * (4 - len(default))
                p.set_editor_property("default_value", unreal.LinearColor(*c[:4]))
            else:
                p.set_editor_property("default_value", float(default))
            changed = True
        if changed:
            mpc.set_editor_property("scalar_parameters", scalars)
            mpc.set_editor_property("vector_parameters", vectors)
            eal.save_loaded_asset(mpc)
        if added:
            log(f"{path}: added {added} - read them in your materials with a Collection Parameter node")


# ============================================================================ entry
def run(cfg):
    world = editor_world()
    level = world.get_path_name().split(".")[0] if world else "?"
    if level != cfg["map"]:
        warn(f"the open level is {level} but the config renders {cfg['map']} - open that level first")
    unreal.SystemLibrary.execute_console_command(world, "r.CustomDepth 3")

    build_label_material()
    alloc = Allocator()
    setup_props(cfg, alloc)
    ensure_camera(cfg)
    ensure_mpcs(cfg)
    if cfg["water_fog"]["enabled"]:
        build_water_fog_material()
    ensure_postprocess(cfg)

    seg = cfg["seg_classes"]
    log("stencil table (id / object / part):")
    for sid, key, part, label in alloc.table:
        cls = f"seg {seg.index(part)}" if part in seg else "body"
        log(f"   {sid:>3}  {key:<22} {part:<14} {cls:<7} ({label})")
    legacy = [a.get_actor_label() for a in eas().get_all_level_actors()
              if "BP_SynthetheticCamera" in a.get_class().get_name()]
    if legacy:
        warn(f"remove {legacy} from the level before rendering: its BeginPlay timer would teleport, "
             f"capture and finally Quit Game in the middle of the MRQ render")
    log("setup done. Check View Mode > Buffer Visualization > Custom Stencil, then SAVE THE LEVEL.")
