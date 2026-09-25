"""
Diagnostic post-process material for when captured masks come out wrong.

    py "<project>/SegPipeline/tools/make_debug_mask_material.py"

Creates /Game/SegPipeline/M_SegMask_Debug. Put it on CapMask instead of M_SegMask, capture a
few frames, and read the mask PNG channels:
    R = custom stencil * 60   (raw, no occlusion test)   -> 0 means nothing wrote stencil
    G = scene depth  / 20 m   (0..255)
    B = custom depth / 20 m   (0..255, 255 = nothing in custom depth)
Where R > 0, G and B should be (almost) equal; if B > G there, the occlusion test in
M_SegMask is rejecting the pixels. Swap M_SegMask back afterwards.
"""
import unreal

CONTENT = "/Game/SegPipeline"
at = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary

path = f"{CONTENT}/M_SegMask_Debug"
m = unreal.load_asset(path) if eal.does_asset_exist(path) else \
    at.create_asset("M_SegMask_Debug", CONTENT, unreal.Material, unreal.MaterialFactoryNew())
mel.delete_all_material_expressions(m)
m.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
for n in ("BL_SCENE_COLOR_AFTER_TONEMAPPING", "BL_AFTER_TONEMAPPING"):
    loc = getattr(unreal.BlendableLocation, n, None)
    if loc is not None:
        m.set_editor_property("blendable_location", loc)
        break


def node(cls, x, y, **props):
    e = mel.create_material_expression(m, cls, x, y)
    for k, v in props.items():
        e.set_editor_property(k, v)
    return e


def link(src, dst, dst_in=""):
    if not mel.connect_material_expressions(src, "", dst, dst_in):
        raise RuntimeError(f"could not connect {src.get_name()} -> {dst.get_name()}.{dst_in}")


def channel(tex_id, scale, y):
    st = node(unreal.MaterialExpressionSceneTexture, -1200, y, scene_texture_id=tex_id)
    r = node(unreal.MaterialExpressionComponentMask, -950, y, r=True, g=False, b=False, a=False)
    if not mel.connect_material_expressions(st, "Color", r, ""):
        raise RuntimeError("could not connect SceneTexture.Color")
    mul = node(unreal.MaterialExpressionMultiply, -750, y, const_b=scale)
    link(r, mul, "A")
    sat = node(unreal.MaterialExpressionSaturate, -550, y)
    link(mul, sat)
    return sat


red = channel(unreal.SceneTextureId.PPI_CUSTOM_STENCIL, 60.0 / 255.0, -300)
green = channel(unreal.SceneTextureId.PPI_SCENE_DEPTH, 1.0 / 2000.0, 0)
blue = channel(unreal.SceneTextureId.PPI_CUSTOM_DEPTH, 1.0 / 2000.0, 300)

rg = node(unreal.MaterialExpressionAppendVector, -300, -150)
link(red, rg, "A")
link(green, rg, "B")
rgb = node(unreal.MaterialExpressionAppendVector, -100, 0)
link(rg, rgb, "A")
link(blue, rgb, "B")
mel.connect_material_property(rgb, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)

mel.recompile_material(m)
eal.save_loaded_asset(m)
unreal.log("[SegPipeline] M_SegMask_Debug ready - put it on CapMask in place of M_SegMask")
