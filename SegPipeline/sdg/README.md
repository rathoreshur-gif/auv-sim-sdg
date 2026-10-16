# SDG: synthetic data generation (segmentation + keypoints) with Movie Render Queue

One config file describes a dataset: which props are labelled, what gets randomized and within
which ranges, and a list of stages (camera mode, image count, per-stage ranges). One command in
the editor bakes it into Level Sequences, renders it with Movie Render Queue, and writes
ready-to-train YOLO-seg and YOLO-pose datasets while the render is still running.

```
configs/pool.toml   <- the file you edit (copy it for a new recipe)
sdg.py              UE entry point: setup | render | bake | preview | where
ue_setup.py         (UE) stencil ids, label copies, M_SDG_Labels, SDG_Camera, SDG_PostProcess, MPCs
ue_bake.py          (UE) config -> one Level Sequence per stage + label manifest
ue_render.py        (UE) MRQ jobs (EXR only), starts post.py
post.py             (SegPipeline venv) EXR -> jpg + YOLO seg/pose labels, previews, data yamls
exr.py              minimal EXR reader (numpy only)
sdg_common.py       config, seeded sampling, Unreal coordinate math (shared)
tests/              offline tests: python tests/test_pipeline.py
ue_listener.py      (UE) bridge for the SDG App (started by Content/Python/init_unreal.py)
../app/             the SDG App (below); ../SDG App.bat starts it
../train/train_yolo.py   train seg or pose with the recipe in train/recipe.py (--quick, --eval real sets)
../tools/prop_image.py   another image for a texture_regions prop (see "Printed images")
```

## The SDG App (no command line)

Double-click **`SegPipeline/SDG App.bat`**. It opens a window with three pages:

| Page | What you do there |
|---|---|
| **1 Recipe** | Create / edit configs with forms: classes, props, camera (the volume corners come from the Unreal viewport with one click), randomizers (switch each on/off, templates for lights, post-process, materials, props), haze, stages. Every change is checked like the pipeline does; Save writes `configs/<name>.toml` (the first save keeps your hand-written file as `<name>.toml.orig`). |
| **2 Render** | Run setup (and save the level), a quick test and the full render in the open editor. Live progress per stage, images/s, time left, the preview images and the pipeline's log lines. |
| **3 Train** | **Quick train** a dataset: `train_yolo.py --quick` (30 epochs, fast recipe), then scored on the real labelled datasets from Settings. Or set up a full training: YOLO version and size, epochs / image size / batch ..., augmentation strength (Off / Light / Medium / Strong, the underwater effects, or each value), in the yolo-training repo. Follows every run in `SegPipeline/runs` (quick trains, `post.train`) and the repo's `runs/`: mAP and loss, real-image scores per dataset with the prediction images, `analyse_model.py`, Evaluate on real images. |

The app drives the editor through `ue_listener.py`, which starts by itself when the editor opens
(`Content/Python/init_unreal.py`). In an editor that was already open, run once in the Output Log:
`py "<project>/SegPipeline/sdg/ue_listener.py"`.
With `post.open_trainer = true` in a recipe, the Train page opens by itself when a full render is
done, with the new dataset selected. Settings (bottom left): yolo-training folder, its Python, the
real dataset for evaluation. Tests: `<venv python> app/tests/test_app.py`.

## Every run (command line)

In the editor, with `/Game/volume/water_volume` open, **Output Log -> Cmd**:

```
py "<project>/SegPipeline/sdg/sdg.py" render pool --quick 20
py "<project>/SegPipeline/sdg/sdg.py" render pool
```

`--quick 20` renders 20 images per stage: use it after every config change and look at
`Saved/SDG/datasets/pool_v1/previews/` before starting a big run. A console window opens and
converts frames as MRQ writes them; when it says `dataset:` the data is ready:

```
Saved/SDG/datasets/pool_v1/
  seg.yaml   images/{train,val}   labels/{train,val}          YOLO-seg (polygons)
  pose.yaml  pose/images (hard links)  pose/labels             YOLO-pose (box + keypoints)
  previews/  meta/<stage>.jsonl (sampled values per image)  stats.json
```

With `post.train = ""` (off), a full run also packs `Saved/SDG/datasets/<name>_yolo_seg.zip`: just the
YOLO-seg dataset (images, labels, a `data.yaml` without absolute paths) to share. Unzip anywhere and train
with `data=<folder>/data.yaml`. Re-pack by hand: `post.bat --zip` in the raw run folder.

Train in the SDG app (Train page, **Quick train**), or set `post.train = "quick"` (30 epochs) / `"seg"` /
`"both"` to start right after the render instead of zipping. Segmentation runs started this way are scored
on the real labelled datasets set in the app's Settings when they finish (`<run>/real_eval/`). By hand:

```
..\.venv\Scripts\python.exe ..\train\train_yolo.py --task seg  --data <dataset>\seg.yaml --quick --eval <real>\data.yaml
..\.venv\Scripts\python.exe ..\train\train_yolo.py --task pose --data <dataset>\pose.yaml
```

## One-time setup

1. **Restart the editor** once: Movie Render Queue is now enabled in `MyProject.uproject`.
2. **Remove `BP_SynthetheticCamera` from the level.** Its BeginPlay timer would teleport, capture
   and finally Quit Game in the middle of an MRQ render. (MRQ + SDG_Camera replace it.)
3. Set `camera.volume` in the config: fly the viewport to two opposite corners of the water (inside
   the walls, under the surface) and run `py ".../sdg.py" where` at each.
4. `py ".../sdg.py" setup pool`, check **View Mode > Buffer Visualization > Custom Stencil**, save the level.

Re-run `setup` after adding/removing props in the level or in the config. It replaces the old
`Seg_*` copies from `tools/setup_segmentation.py` (the old Blueprint capture stops working after that).

## How the pieces fit

| Old pain | Now |
|---|---|
| 0.3 s Delay per image | No timers. MRQ renders frame after frame as fast as the GPU goes; every frame's values are keyed in the sequence, so a frame never mixes old and new values. |
| 4-5 Blueprint nodes per randomized variable | One `[[randomize]]` block: target actor/component, property, range. |
| Masks, then a second script | Labels come straight out of each rendered frame (`post.py` runs during the render). |
| One-shot recording | `[[stages]]`: any number of batches with different camera modes, counts and ranges, all in one run and one dataset. Same config + seed = same images. |
| New prop = hand work | `[[props]]` entries (`actor`, `texture_regions`, `material_slots`), then `setup`. |
| Segmentation only | Every prop instance gets its own stencil ids, so touching boards are separate instances, and `keypoints` gives YOLO-pose labels (box from the visible pixels, keypoints projected with the exact camera from the EXR header, visibility from the depth pass). |

Per frame, MRQ writes one multilayer EXR: `FinalImage` (the lit render) and `SDGLabels`
(R = stencil id of the visible tagged surface, G = scene depth in cm), with the camera location,
rotation and FOV in the header. `post.py` converts that into a jpg and the label files, then
deletes the EXR.

**Why EXR and not PNG:** MRQ adds random dither when it writes 8-bit files, so a PNG label pass
would not hold exact ids. EXR keeps them exact, and only multilayer EXRs carry the camera metadata.

**Why one sample per frame:** with 1 spatial x 1 temporal sample MRQ skips its accumulator
(its fast path) and never averages label ids of neighbouring objects at the edges.

### Camera modes

| mode | what | when |
|---|---|---|
| `random_path` | smooth flight between random viewpoints around the props; every frame is an image | default, fastest: Lumen/TSR history stays valid |
| `teleport` | independent random viewpoint per image; `hold = N` renders each pose N times and keeps the last | maximum variety; hold 2-3 lets Lumen, TSR and auto exposure settle after the jump (MRQ only tells the renderer about camera cuts at the start of a stage) |
| `sequence` | **your** Level Sequence supplies the camera path (camera cut track + transform keys), resampled onto SDG_Camera with optional smooth jitter; lights, fog, colours, props are still randomized | planned approaches (e.g. the AUV's path to the board) |

Viewpoints are placed around a target (a prop's centre by default): `distance`, `elevation`,
`aim_jitter`, `roll`, `fov`, and they must fall inside `camera.volume`. `azimuth = [-70, 70]` keeps
the camera within 70 deg of the target's front; `facing` says which side that is (`auto` = the
thinnest axis, the side `face4` uses; `-auto` = the other side; or `+x` ... `-z`).

### Randomizers

```toml
[[randomize]]
name = "sun_intensity"          # also the key for stage overrides and in meta/*.jsonl
actor = "SunSky*"               # actor label wildcard (or tag = "...")
component = "DirectionalLight*" # component name or class wildcard; omit to target the actor
property = "Intensity"          # C++ property name
mode = "multiply"               # set (default) | add | multiply (relative to the level's value)
value = [0.25, 1.6]
every = 1                       # new value every N images
smooth = false                  # blend between values (random_path / sequence)
per_target = false              # true: each matched actor gets its own value
```

`type`: `float` (default), `double`, `bool`, `color`, `vector`, `transform`
(`location`/`rotation`/`scale`, offsets unless `offset = false`), `visible` (`probability`),
`postprocess` (a Post Process Volume setting on SDG_PostProcess, e.g. `AutoExposureBias`, `WhiteTemp`,
`FilmGrainIntensity`, `SceneFringeIntensity`, `VignetteIntensity`, `SceneColorTint`),
`mpc` (`mpc` + `param`: material parameter collection, for any material you wire it into),
`material_param` (`material_slot` + `param` on a mesh's material), `material` (`material_slot`,
`value = { choice = [material paths] }` swaps materials), `image` (`prop`: which of a texture_regions
prop's images it shows, see below).

Values: `[a, b]` uniform, `{ normal = [mean, std], clip = [lo, hi] }`, `{ loguniform = [a, b] }`,
`{ choice = [...], weights = [...] }`, `{ randint = [a, b] }`, a constant, or colours
`{ hsv = { h = [..], s = [..], v = [..] } }`, `{ rgb = [[..],[..],[..]] }`, `{ kelvin = [k0, k1] }`.

Stage overrides: `override = { sun_intensity = [0.05, 0.3] }` or a table with `value`, `every`,
`smooth`; `disable = ["film_grain"]` turns randomizers off for a stage.

**Materials** (caustic strength, pool tile colour, ...): add a *Collection Parameter* node reading
`MPC_SDG` (setup creates it and adds every `param` named in `mpc` randomizers) to the material,
then randomize it with an `mpc` block. Outside a render the editor shows the parameter's default:
1 / white, or `default = ...` in the block (setup writes it into MPC_SDG).

**Printed images** (e.g. the torpedo board layouts): list the prop's other images in its `images`,
one folder per image with its masks next to it, and add an `image` randomizer:

```toml
[[props]]
name = "torpedo_board"
type = "texture_regions"
texture = "/Game/..."                   # image 0: what the level shows, masks in `regions`
regions = { blood = "props/torpedo_board/T_Mask_blood.png", ... }
images = ["props/torpedo_board/images/task4_ver1/texture.png", "props/torpedo_board/images/task4_ver2/texture.png"]

[[randomize]]
name = "board_image"
type = "image"
prop = "torpedo_board"
value = { choice = [0, 1, 2], weights = [1, 2, 1] }   # optional; left out = every image equally often
every = 60
```

Make a new image's folder with `tools/prop_image.py` (copies the image, cuts the same see-through holes as
the texture into its red rings with `--holes-like`, writes `T_Mask_<class>.png` + `masks_preview.png`;
classes in reading order, like `make_texture_masks.ps1`). The image must have the texture's size and
layout (same UVs). `setup` imports the images and makes a material instance per image (the texture has to
reach the material through a texture parameter, as imported glTF materials do) and a cutout per mask;
the label copies swap masks on the same frames as the picture, so labels always match what is printed.
A class that is missing from an image simply gets no label while that image is shown. Re-run `setup`
after changing `images` (a `bake` also creates missing assets, e.g. when `setup` ran with another config).
In flying stages keep `every` large (TSR carries the old picture over for a frame or two after a swap);
teleport stages with `hold >= 2` can use `every = 1` (the swap lands on the frame that is not kept).

**`[water_fog]`** adds a randomized haze over the whole image (colour + density per frame, applied
after your water material). Objects dimmed below `min_visibility` by the haze get no label, so
the model is never asked to find an object that is invisible in the image.

### Keypoints

`keypoints = "face4"` (4 corners of the face across the thinnest axis; `"face4:-x"` picks a side),
`"top4"`, `"bottom4"`, `"bounds8"`, `"center"`, `{ sockets = ["KP_TL", ...] }` (static mesh
sockets), or `{ points = [[x, y, z], ...] }` (actor space, cm). Corner presets come in the order
top-left, top-right, bottom-right, bottom-left and set `flip_idx = [1, 0, 3, 2]` for horizontal
flips. Visibility per keypoint: 2 = visible, 1 = inside the image but hidden (behind something or
on the far side), 0 = outside the image. All pose classes share the largest keypoint count.

## What has been verified

Before this was handed over, the whole chain ran in a real UE 5.7 editor on a small test level
(textured board with two drawings, two cones, an occluder, a planned camera sequence), through
`sdg.py setup` / `sdg.py render` with both executors:
- every randomizer type baked onto the right Sequencer binding and rendered; MRQ wrote exactly the
  expected frames for random_path, teleport (hold 2) and sequence stages
- the EXR layout (`FinalImage` + `FinalImageSDGLabels`, ZIP) and camera metadata matched the reader
- seg polygons vs the label pass: IoU 0.98-0.99; board corner keypoints landed on the rendered
  board corners; keypoints behind the occluder / behind the object were marked occluded
- same config + seed gave identical labels with the `pie` and `process` executors
- both datasets load in Ultralytics 8.4 with 0 corrupt labels, and 1-epoch seg and pose runs train
- `tests/test_pipeline.py` covers the EXR reader, polygons, Unreal math, config, and a synthetic
  frame through the full labelling path

Not verified on your level: the numbers below and the look of your scene under these ranges -
start with `--quick 20` and the previews.

## Speed

Measured on the test level (RTX A4000, 640x360): ~40-50 frames/s through MRQ once shaders are
warm; the dataset builder needs ~115 ms per 1280x720 frame per worker (~50 frames/s with 6
workers), so it keeps up with the renderer. Your level (Lumen, water, 1280x720) will be slower;
the Output Log prints images/s at the end of every render. Levers, biggest first:
- resolution (`[render] resolution`); 960x540 is ~1.8x faster than 1280x720
- `random_path` / `sequence` over `teleport` with `hold > 1`
- `executor = "process"` renders in a separate game process (no editor overhead; saves the level first)
- `cinematic = false` (default) keeps the level's scalability instead of MRQ's cinematic preset

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Movie Render Queue` classes missing | enable the plugin (setup step 1) |
| previews: labels on objects you can barely see | raise `water_fog.min_visibility`, or set `output.max_label_depth` (cm) |
| stencil ids with `?` in `post.py --check` | something else in the level has Render CustomDepth on; turn it off or re-run setup |
| keypoints on the wrong face | `face4:-x` / `face4:+y` etc., or sockets |
| a randomizer "matched nothing" | check the actor label / component name in the Outliner (wildcards allowed) |
| a transform randomizer does nothing | the actor was Static; bake sets it Movable - save the level |
| camera outside the pool / above water | set `camera.volume` |
| `post.py` didn't start | `post.python` path; or double-click `<raw>/<name>/post.bat` |
| "MRQ reported success but no frames were written" | the separate render process crashed; see its log, or use `executor = "pie"` |
| config error on line 1 | the file was saved as something other than UTF-8 text; re-save it as UTF-8 |

Automation: `UnrealEditor.exe MyProject.uproject -ExecCmds="py <path>/sdg.py render pool"` opens
the editor and starts a render. (`-ExecutePythonScript` does not work for this: the editor quits
as soon as the script returns, killing the render that was just started.)
