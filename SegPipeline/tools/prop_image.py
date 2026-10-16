"""
Get another image of a texture_regions prop ready for the "image" randomizer (sdg/README.md):

  - copies it to <out>/texture.png
  - optionally cuts the same see-through holes as the prop's current texture (torpedo board: the
    circles are holes in the board), centred on this image's own red rings
  - makes one mask per drawing, <out>/T_Mask_<class>.png, and <out>/masks_preview.png
    (same rules as make_texture_masks.ps1)

From the SegPipeline folder, with the venv (numpy + opencv):

  .venv\\Scripts\\python.exe tools\\prop_image.py C:\\Users\\me\\Downloads\\Task4_ver1.png ^
      --out props\\torpedo_board\\images\\task4_ver1 --classes fire,firetruck,blood,ambulance ^
      --holes-like props\\torpedo_board\\board_texture.png

--classes in READING ORDER (top row left to right, then the next row). Open masks_preview.png and check
every label sits on the right drawing, then add "<out>/texture.png" to the prop's `images` list.
"""
import argparse
import os
import sys

import cv2
import numpy as np


def read_rgba(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"can't read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8)
    return img                                             # B, G, R, A


def pure_red(img):
    b, g, r = (img[..., i].astype(int) for i in range(3))
    return (r > 220) & (g < 40) & (b < 40)


# ---------------------------------------------------------------------------- holes
def fill_holes(m):
    """Pixels not reachable from the image edge through background (4-connected) -> set."""
    n, lab = cv2.connectedComponents((~m).astype(np.uint8), connectivity=4)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    outside = np.isin(lab, edge[edge > 0]) if n > 1 else np.zeros_like(m)
    return ~outside


def rings(img, min_area=2000):
    """Red rings (a red blob with a hole in it): [(cx, cy, outer radius)]."""
    red = pure_red(img) & (img[..., 3] > 127)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=4)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = stats[k, :4]
        m = lab[y:y + h, x:x + w] == k
        filled = fill_holes(m)
        if filled.sum() < 1.3 * m.sum():                   # solid red shape, not a ring
            continue
        ys, xs = np.nonzero(filled)
        out.append((x + xs.mean(), y + ys.mean(), float(np.sqrt(filled.sum() / np.pi))))
    return out


def hole_sizes(ref):
    """From an image that has see-through holes inside red rings: [(ring outer radius, hole radius)]."""
    holes = (ref[..., 3] < 128).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(holes, connectivity=4)
    rs = rings(ref)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 500:
            continue
        cx, cy = cents[k]
        ring = min(rs, key=lambda r: (r[0] - cx) ** 2 + (r[1] - cy) ** 2, default=None)
        if ring is None or (ring[0] - cx) ** 2 + (ring[1] - cy) ** 2 > ring[2] ** 2:
            continue                                        # a transparent area that isn't inside a ring
        out.append((ring[2], float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi))))
    return out


def cut_holes(img, sizes):
    """Transparent disc (anti-aliased edge) inside every ring of img, sized like the reference ring
    with the closest outer radius. Returns the number of holes cut."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    alpha = img[..., 3].astype(np.float32) / 255.0
    cut = 0
    for cx, cy, ro in rings(img):
        ref_ro, ref_rh = min(sizes, key=lambda s: abs(s[0] - ro))
        if abs(ref_ro - ro) > 0.1 * ref_ro:
            print(f"  ring at ({cx:.0f}, {cy:.0f}) r={ro:.0f} px: no reference ring of that size - left solid")
            continue
        rh = ref_rh * ro / ref_ro
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        alpha = np.minimum(alpha, np.clip(d - rh + 0.5, 0.0, 1.0))
        print(f"  hole at ({cx:.0f}, {cy:.0f}) r={rh:.0f} px (ring r={ro:.0f})")
        cut += 1
    img[..., 3] = np.round(alpha * 255).astype(np.uint8)
    return cut


# ---------------------------------------------------------------------------- masks (make_texture_masks.ps1)
def morph(m, r, dilate):
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    op = cv2.dilate if dilate else cv2.erode               # outside the image counts as background
    return op(m.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def find_drawings(img, n, white_tol=14, border=24, exclude_red=True, row_tol=0.15):
    h, w = img.shape[:2]
    b, g, r, a = (img[..., i].astype(int) for i in range(4))
    dist = np.maximum(255 - r, np.maximum(255 - g, 255 - b))
    m = (a > 127) & (dist > white_tol)
    if exclude_red:
        m &= ~pure_red(img)
    inner = np.zeros_like(m)
    inner[border:h - border, border:w - border] = True
    m &= inner
    m = morph(morph(m, 3, True), 3, False)                 # close small gaps inside drawings
    m = morph(morph(m, 3, False), 3, True)                 # drop thin anti-aliased edges and specks
    count, lab, stats, cents = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    blobs = sorted(range(1, count), key=lambda k: -stats[k, cv2.CC_STAT_AREA])
    if len(blobs) < n:
        sys.exit(f"found only {len(blobs)} drawings, expected {n}")
    blobs = sorted(blobs[:n], key=lambda k: cents[k][1])
    ordered, s = [], 0
    while s < len(blobs):                                   # reading order: rows by centre y, then x
        e = s + 1
        while e < len(blobs) and cents[blobs[e]][1] - cents[blobs[s]][1] < row_tol * h:
            e += 1
        ordered += sorted(blobs[s:e], key=lambda k: cents[k][0])
        s = e
    out = []
    for k in ordered:
        x, y, bw, bh = stats[k, :4]
        out.append({"mask": fill_holes(lab == k), "centre": tuple(cents[k]), "box": (x, y, x + bw - 1, y + bh - 1)})
    return out


def save_mask(mask, path):
    v = np.where(mask, 255, 0).astype(np.uint8)
    cv2.imwrite(path, cv2.merge([v, v, v]))


def save_preview(img, drawings, names, path):
    a = img[..., 3:4].astype(np.float32) / 255.0
    o = img[..., :3].astype(np.float32) * a + 60.0 * (1.0 - a)
    cols = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 165, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0)]
    for k, d in enumerate(drawings):
        o[d["mask"]] = (o[d["mask"]] + np.array(cols[k % len(cols)], np.float32)) / 2.0
    o = o.astype(np.uint8)
    scale = max(0.6, img.shape[1] / 900.0)
    for k, (d, name) in enumerate(zip(drawings, names)):
        x0, y0, x1, y1 = d["box"]
        cv2.rectangle(o, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), 3)
        label = f"{k}: {name}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        ly = int(max(0, y0 - th - base - 4))
        cv2.rectangle(o, (int(x0), ly), (int(x0) + tw + 8, ly + th + base + 4), (0, 0, 0), -1)
        cv2.putText(o, label, (int(x0) + 4, ly + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(path, o)


# ---------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("image")
    ap.add_argument("--out", required=True, help="folder for texture.png + masks (one folder per image)")
    ap.add_argument("--classes", required=True, help="drawing classes in reading order, e.g. fire,firetruck,blood,ambulance")
    ap.add_argument("--holes-like", default="", help="the prop's current texture: cut its see-through holes into this image's rings")
    ap.add_argument("--white-tol", type=int, default=14)
    ap.add_argument("--border", type=int, default=24)
    ap.add_argument("--row-tol", type=float, default=0.15)
    ap.add_argument("--keep-pure-red", action="store_true", help="count pure red (255,0,0) as drawing")
    ap.add_argument("--masks-only", action="store_true", help="make masks for the image as it is (no texture.png)")
    a = ap.parse_args(argv)

    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    os.makedirs(a.out, exist_ok=True)
    img = read_rgba(a.image)
    h, w = img.shape[:2]
    print(f"{a.image}: {w}x{h}")
    if a.holes_like:
        ref = read_rgba(a.holes_like)
        if ref.shape[:2] != img.shape[:2]:
            print(f"  WARNING: {a.holes_like} is {ref.shape[1]}x{ref.shape[0]}; the prop's UVs expect the same size")
        sizes = hole_sizes(ref)
        if not sizes:
            sys.exit(f"{a.holes_like} has no see-through holes inside red rings")
        cut_holes(img, sizes)
    if not a.masks_only:
        cv2.imwrite(os.path.join(a.out, "texture.png"), img)

    drawings = find_drawings(img, len(classes), a.white_tol, a.border, not a.keep_pure_red, a.row_tol)
    for k, (d, name) in enumerate(zip(drawings, classes)):
        path = os.path.join(a.out, f"T_Mask_{name}.png")
        save_mask(d["mask"], path)
        cx, cy = d["centre"]
        print(f"{k}: {name:<12} centre ({cx:.0f},{cy:.0f}) {int(d['mask'].sum()):>8} px  -> {os.path.basename(path)}")
    save_preview(img, drawings, classes, os.path.join(a.out, "masks_preview.png"))
    print(f"preview -> {os.path.join(a.out, 'masks_preview.png')}   (check every label sits on the right drawing)")
    if not a.masks_only:
        print(f'add to the prop\'s images: "{os.path.join(a.out, "texture.png").replace(os.sep, "/")}"')


if __name__ == "__main__":
    main()
