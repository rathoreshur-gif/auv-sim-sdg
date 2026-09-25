"""
Import mesh files (glb / gltf / fbx / obj) into the project, one Content folder per file.

In the editor (Output Log -> Cmd):
    py ".../SegPipeline/tools/import_mesh.py" /Game/Props "C:/path/SM_Thing.glb" "C:/path/Other.fbx"

Each file goes to <dest>/<file name without extension>, so props that share material names
(e.g. two bins both using M_Bin_Yellow) don't overwrite each other. Prints the static mesh
paths - use those as "static_mesh" in seg_config.json.
"""
import os
import re
import sys

import unreal

if len(sys.argv) < 3:
    raise SystemExit("usage: import_mesh.py <dest content folder> <file> [file ...]")
dest_root = sys.argv[1].rstrip("/")
files = sys.argv[2:]

at = unreal.AssetToolsHelpers.get_asset_tools()
ar = unreal.AssetRegistryHelpers.get_asset_registry()
for f in files:
    if not os.path.exists(f):
        unreal.log_error(f"[SegPipeline] not found: {f}")
        continue
    stem = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(f))[0]).strip("_")
    stem = re.sub(r"_+", "_", stem)
    folder = f"{dest_root}/{stem}"
    t = unreal.AssetImportTask()
    t.filename = f
    t.destination_path = folder
    t.automated = True
    t.replace_existing = True
    t.save = True
    at.import_asset_tasks([t])

    meshes = [a.get_asset().get_path_name().split(".")[0]
              for a in ar.get_assets_by_path(folder, recursive=True)
              if a.asset_class_path.asset_name == "StaticMesh"]
    unreal.log(f"[SegPipeline] {os.path.basename(f)} -> {folder}  static meshes: {meshes}")
