# SegPipeline: synthetic segmentation data from UE5

> **Superseded by `sdg/`** (see `sdg/README.md`): Movie Render Queue instead of the Blueprint timer,
> randomization and stages in one config file, per-instance stencil ids, YOLO seg + pose labels
> written during the render. Running `sdg.py setup` replaces the `Seg_*` copies made by
> `tools/setup_segmentation.py`, so the Blueprint capture below stops working after that.

```
seg_config.json      <- the only file you edit per prop / class
tools/
  export_texture.py        (UE)  texture asset -> PNG
  make_texture_masks.ps1   (PC)  PNG with drawings -> one mask per drawing
  setup_segmentation.py    (UE)  config -> stencil tagging, M_SegMask, render targets
  mask_to_yolo.ps1         (PC)  captured masks -> YOLO-seg dataset + data.yaml
props/<prop>/              masks + previews per prop
```

How it works: every class gets a stencil value (class i -> stencil i+1). A second scene
capture renders `M_SegMask`, which writes `stencil x stencil_step` as grey but only where
the object is actually visible. `mask_to_yolo.ps1` turns those greys back into polygons.

UE commands run from **Output Log -> Cmd**: `py "<path to script>" [args]`.
PC commands: `powershell -ExecutionPolicy Bypass -File <script> ...`

---

## Adding a prop

**A. The whole object is one class** (buoy, gate, bin, marker...)

Add to `props` in `seg_config.json`. Match actors by label wildcard and/or tag
(Details -> Actor -> Tags), and add the class name to `classes` if it's new:
```json
{ "name": "gate", "type": "actor", "actor_label": "Gate*", "class": "gate" }
```

**B. Drawings printed on a texture** (like the torpedo board)

1. Export the texture:
   `py ".../tools/export_texture.py" /Game/Path/To/Texture "C:/.../SegPipeline/props/<prop>/texture.png"`
2. Make masks. Give the class names **in reading order** (top row left to right, then the next row):
   `make_texture_masks.ps1 -Texture ..\props\<prop>\texture.png -Classes a,b,c`
   Open `props/<prop>/masks_preview.png` and check every label sits on the right drawing.
   (Options: `-WhiteTol` if the background isn't pure white, `-KeepPureRed` if a drawing is pure red.)
3. Add to `props`:
```json
{ "name": "<prop>", "type": "texture_regions", "texture": "/Game/Path/To/Texture",
  "regions": { "a": "props/<prop>/T_Mask_a.png", "b": "props/<prop>/T_Mask_b.png" } }
```

**C. Only part of a mesh is the class, and that part has its own material** (e.g. the yellow X/O on a bin)

No masks needed. Import the mesh (one folder per file):
`py ".../tools/import_mesh.py" /Game/Props "C:/path/SM_Bin_O.glb" "C:/path/SM_Bin_X.glb"`
Place it in the level, then add:
```json
{ "name": "bin_o", "type": "material_slots", "static_mesh": "/Game/Props/SM_Bin_O*",
  "slots": { "*Yellow*": "bin_o" } }
```
`slots` patterns match the material slot name or the material asset name. Every other slot is ignored.

**Then (all cases)**, with the level open:
`py ".../tools/setup_segmentation.py"`
Check Viewport **Lit -> Buffer Visualization -> Custom Stencil**, then save the level.

Class order matters: YOLO ids come from the order of `classes`. Add new classes at the **end**
so existing datasets keep their ids. With `stencil_step` 10 you can have up to 25 classes.

---

## Capture Blueprint (one-time, in BP_SynthetheticCamera)

HighResShot only saves the RGB view, so the Blueprint uses two scene captures that
render the same pose in the same tick.

**Components.** Add both as children of `CineCamera`, with relative transform zero:

| Component | Setting |
|---|---|
| `CapRGB` (SceneCaptureComponent2D) | Texture Target `RT_SegRGB` · Capture Source *Final Color (LDR) in RGB* · Capture Every Frame **off** · Capture on Movement **off** |
| `CapMask` (SceneCaptureComponent2D) | same, but Texture Target `RT_SegMask` · Post Process Settings -> Rendering Features -> Post Process Materials -> add `M_SegMask` (weight 1) |

Both `RT_` assets and `M_SegMask` are in `/Game/SegPipeline` (made by setup_segmentation.py).
- **FOV:** on BeginPlay, set both captures' `FOV Angle` to `CineCamera -> Get Horizontal Field of View`.
- **Underwater look:** if your water post-process material (DPI_water) is on the CineCamera,
  also add it to **CapRGB** (`Add or Update Blendable` on CapRGB with DPI_water). Scene captures don't use the camera's post-process settings.

**Variables:** `FrameIndex` (int), `NumFrames` (int, Instance Editable, e.g. 2000),
`Target` (Actor, Instance Editable; pick the board in the level's Details panel).

**CaptureFrame** (replace the HighResShot node):
```
CaptureFrame
 -> RandomizeScene
 -> Set Actor Location (random, as now)
 -> Set Actor Rotation ( Find Look at Rotation( GetActorLocation(self), GetActorLocation(Target) ) )
 -> Delay 0.1                      (let lighting settle after the jump)
 -> CapRGB  -> Capture Scene
 -> CapMask -> Capture Scene
 -> Export Render Target (RT_SegRGB,  Dir + "images/", Name)
 -> Export Render Target (RT_SegMask, Dir + "masks/",  Name)
 -> FrameIndex = FrameIndex + 1
 -> Branch (FrameIndex >= NumFrames) -> True: Quit Game
```
- `Dir  = Append( Get Project Saved Directory, "SegCapture/" )`
- `Name = Append( "frame_", ToString(FrameIndex + 100000), ".png" )`. The +100000 keeps the names zero-padded and sorted.

The timer (0.5 s, looping) stays as it is. Output lands in `MyProject/Saved/SegCapture/`.

---

## Building the dataset

```
powershell -ExecutionPolicy Bypass -File tools\mask_to_yolo.ps1 -Capture "..\Saved\SegCapture" -Inspect
powershell -ExecutionPolicy Bypass -File tools\mask_to_yolo.ps1 -Capture "..\Saved\SegCapture" -Out "D:\datasets\torpedo_seg"
```
- `-Inspect` prints the grey levels in the first masks. They should be multiples of
  `stencil_step` (0, 10, 20...). Run it once after your first capture.
- The output has `images/{train,val}`, `labels/{train,val}`, `data.yaml`, and `previews/`, which shows the polygons drawn on RGB. **Look at the previews before training.**
- Options: `-Val 0.1`, `-MinArea 60` (px), `-Epsilon 1.5` (polygon simplification, px).

## Training (local GPU)

Environment: `SegPipeline\.venv` (Python 3.12, PyTorch CUDA 12.8, Ultralytics, Albumentations).
Create it once from the `SegPipeline` folder:
```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
(or `uv venv --python 3.12` then `uv pip install -r requirements.txt`). `SDG App.bat` and the
render pipeline find it on their own. To use a venv somewhere else, set `SDG_PYTHON` to its
`python.exe`. An existing `%USERPROFILE%\seg-train\.venv` is also picked up.

```
.venv\Scripts\python.exe train\train_seg.py --data <dataset>\data.yaml --name <run name>
```
`train/train_seg.py` applies heavy augmentation aimed at generalization: colour/hue shifts,
grayscale, blur/defocus, noise, JPEG, low-res, fog; plus scale, rotation, perspective, mosaic
and copy-paste. Options: `--model` (default `yolo11s-seg.pt`), `--epochs`, `--imgsz`, `--batch`,
`--freeze N` (for fine-tuning on a few real images).
Runs go to `SegPipeline\runs\<name>` (`weights\best.pt`, `results.png`, ...); `--project <dir>` changes that.
Trained weights and `runs/` are git-ignored. Only the stock pretrained `yolo*.pt` files are committed.

The val mAP is computed on synthetic frames, so it will look optimistic. Judge the model on real images.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Custom Stencil view is all black | `r.CustomDepth 3` in Cmd; `r.CustomDepth=3` must be in Config/DefaultEngine.ini |
| `-Inspect` greys are not multiples of the step | RT_SegMask must be RGBA8 with sRGB off; M_SegMask Blendable Location must be *After Tonemapping* |
| Mask shows objects that are hidden behind something | occlusion test tolerance: `DEPTH_TOL_CM` in setup_segmentation.py (default 5 cm) |
| RGB frames too dark / washed out | toggle **Force Linear Gamma** on RT_SegRGB |
| Many frames with "no objects" | camera isn't aimed at `Target`, or it's too far away / outside the pool |
| Some other actor lights up in Custom Stencil | it has Render CustomDepth on for something else; turn it off or give it stencil 0 |
