"""
Export a texture asset to PNG so masks can be made from it.

In the editor (Output Log -> Cmd):
    py "<project>/SegPipeline/tools/export_texture.py" /Game/Path/To/Texture "C:/out/texture.png"
"""
import sys
import unreal

if len(sys.argv) < 3:
    raise SystemExit("usage: export_texture.py <texture asset path> <output .png>")
src, out = sys.argv[1], sys.argv[2]

tex = unreal.load_asset(src)
if tex is None:
    raise SystemExit(f"texture not found: {src}")

t = unreal.AssetExportTask()
t.object = tex
t.filename = out
t.automated = True
t.prompt = False
t.replace_identical = True
ok = unreal.Exporter.run_asset_export_task(t)
unreal.log(f"[SegPipeline] export {src} -> {out}: {'ok' if ok else 'FAILED'} "
           f"({tex.blueprint_get_size_x()}x{tex.blueprint_get_size_y()})")
