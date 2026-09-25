# Turn captured stencil masks into a YOLO-seg dataset.
#
# Input  (-Capture):  images\<frame>.png   RGB from the capture camera
#                     masks\<frame>.png    grey mask from M_SegMask (grey = stencil * stencil_step)
# Output (-Out):      images\{train,val}\<frame>.png
#                     labels\{train,val}\<frame>.txt   "cls x1 y1 x2 y2 ..." (normalised)
#                     data.yaml, previews\ (polygons drawn on RGB for checking)
#
#   powershell -ExecutionPolicy Bypass -File mask_to_yolo.ps1 -Capture <capture dir> -Out <dataset dir>
#   powershell -ExecutionPolicy Bypass -File mask_to_yolo.ps1 -Capture <capture dir> -Inspect
#
# Classes and stencil_step come from seg_config.json, so this never needs editing for a new prop.
param(
    [Parameter(Mandatory = $true)][string]$Capture,
    [string]$Out,
    [string]$Config,             # default: ..\seg_config.json next to this tools folder
    [double]$Val = 0.1,          # fraction of frames for validation (stable per file name)
    [int]$MinArea = 60,          # drop blobs smaller than this many pixels
    [double]$Epsilon = 1.5,      # polygon simplification tolerance in pixels
    [int]$Previews = 25,         # how many preview overlays to write
    [switch]$Inspect             # just print the grey levels found in a few masks
)

Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;

public static class YoloSeg
{
    // Raw R channel. Read with LockBits (no drawing): UE scene captures write alpha = 0,
    // and drawing/compositing would turn every pixel black.
    public static byte[] ReadGray(string path, out int w, out int h)
    {
        using (var src = new Bitmap(path))
        {
            w = src.Width; h = src.Height;
            var d = src.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var row = new byte[w * 4];
            var gray = new byte[w * h];
            for (int y = 0; y < h; y++)
            {
                Marshal.Copy(d.Scan0 + y * d.Stride, row, 0, w * 4);
                for (int x = 0; x < w; x++) gray[y * w + x] = row[x * 4 + 2];   // R channel
            }
            src.UnlockBits(d);
            return gray;
        }
    }

    // Fraction of pixels that are neither background nor a valid class grey (+-2).
    // A broken capture (e.g. solid white) scores near 1; a good mask near 0.
    public static double InvalidFraction(byte[] g, int nClasses, int step)
    {
        int bad = 0;
        foreach (byte v in g)
        {
            int s = (int)Math.Round(v / (double)step);
            if (v == 0 || (s >= 1 && s <= nClasses && Math.Abs(v - s * step) <= 2)) continue;
            bad++;
        }
        return bad / (double)g.Length;
    }

    public static string Histogram(byte[] g)
    {
        var n = new int[256];
        foreach (byte v in g) n[v]++;
        var sb = new StringBuilder();
        for (int v = 0; v < 256; v++) if (n[v] > 0) sb.AppendFormat("{0}:{1}  ", v, n[v]);
        return sb.ToString();
    }

    // Polygons per object: cls, then list of (x,y) pixel points
    public class Poly { public int Cls; public List<int[]> Pts; }

    public static List<Poly> Extract(byte[] g, int w, int h, int nClasses, int step, int minArea, double eps)
    {
        // decode grey -> class index (0-based), -1 = background
        var cls = new int[w * h];
        for (int i = 0; i < g.Length; i++)
        {
            int s = (int)Math.Round(g[i] / (double)step);
            cls[i] = (s >= 1 && s <= nClasses) ? s - 1 : -1;
        }

        // 8-connected components of equal class
        var lab = new int[w * h];
        var polys = new List<Poly>();
        var q = new Queue<int>();
        int next = 0;
        for (int s = 0; s < cls.Length; s++)
        {
            if (cls[s] < 0 || lab[s] != 0) continue;
            int id = ++next, c = cls[s], area = 0;
            lab[s] = id; q.Enqueue(s);
            while (q.Count > 0)
            {
                int i = q.Dequeue(), x = i % w, y = i / w; area++;
                for (int d = 0; d < 8; d++)
                {
                    int nx = x + DX[d], ny = y + DY[d];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    int j = ny * w + nx;
                    if (lab[j] == 0 && cls[j] == c) { lab[j] = id; q.Enqueue(j); }
                }
            }
            if (area < minArea) continue;
            // s is the first pixel of this blob in raster order -> its west neighbour is outside
            var pts = Simplify(Trace(lab, w, h, id, s % w, s / w), eps);
            if (pts.Count >= 3) polys.Add(new Poly { Cls = c, Pts = pts });
        }
        return polys;
    }

    // Clockwise neighbours in image coords (y down): E, SE, S, SW, W, NW, N, NE
    static readonly int[] DX = { 1, 1, 0, -1, -1, -1, 0, 1 };
    static readonly int[] DY = { 0, 1, 1, 1, 0, -1, -1, -1 };

    static int DirOf(int dx, int dy)
    {
        for (int d = 0; d < 8; d++) if (DX[d] == dx && DY[d] == dy) return d;
        return -1;
    }

    // Moore-neighbour tracing of the outer boundary of blob "id"
    static List<int[]> Trace(int[] lab, int w, int h, int id, int sx, int sy)
    {
        var pts = new List<int[]>();
        pts.Add(new[] { sx, sy });
        int cx = sx, cy = sy, back = 4, firstDir = -1;
        int cap = 4 * (w + h) * 64;
        for (int step = 0; step < cap; step++)
        {
            int found = -1;
            for (int k = 1; k <= 8; k++)
            {
                int d = (back + k) % 8, nx = cx + DX[d], ny = cy + DY[d];
                if (nx >= 0 && ny >= 0 && nx < w && ny < h && lab[ny * w + nx] == id) { found = d; break; }
            }
            if (found < 0) break;                                   // single pixel
            if (cx == sx && cy == sy)
            {
                if (firstDir < 0) firstDir = found;
                else if (found == firstDir) break;                  // back at start, same way out
            }
            int pd = (found + 7) % 8;                               // last background neighbour checked
            int bx = cx + DX[pd], by = cy + DY[pd];
            cx += DX[found]; cy += DY[found];
            back = DirOf(bx - cx, by - cy);
            pts.Add(new[] { cx, cy });
        }
        if (pts.Count > 1 && pts[pts.Count - 1][0] == sx && pts[pts.Count - 1][1] == sy) pts.RemoveAt(pts.Count - 1);
        return pts;
    }

    // Douglas-Peucker on a closed polygon
    static List<int[]> Simplify(List<int[]> p, double eps)
    {
        if (p.Count < 4) return p;
        int far = 0; double best = -1;
        for (int i = 1; i < p.Count; i++)
        {
            double dx = p[i][0] - p[0][0], dy = p[i][1] - p[0][1], d = dx * dx + dy * dy;
            if (d > best) { best = d; far = i; }
        }
        var q = new List<int[]>(p); q.Add(p[0]);
        var keep = new bool[q.Count];
        keep[0] = true; keep[far] = true;
        DP(q, 0, far, eps, keep);
        DP(q, far, q.Count - 1, eps, keep);
        var r = new List<int[]>();
        for (int i = 0; i < q.Count - 1; i++) if (keep[i]) r.Add(q[i]);
        return r;
    }

    static void DP(List<int[]> p, int s, int e, double eps, bool[] keep)
    {
        var st = new Stack<int[]>();
        st.Push(new[] { s, e });
        while (st.Count > 0)
        {
            var se = st.Pop(); int a = se[0], b = se[1];
            if (b <= a + 1) continue;
            double maxd = -1; int idx = -1;
            for (int i = a + 1; i < b; i++)
            {
                double d = Dist(p[i], p[a], p[b]);
                if (d > maxd) { maxd = d; idx = i; }
            }
            if (maxd > eps) { keep[idx] = true; st.Push(new[] { a, idx }); st.Push(new[] { idx, b }); }
        }
    }

    static double Dist(int[] pt, int[] a, int[] b)
    {
        double dx = b[0] - a[0], dy = b[1] - a[1], len = Math.Sqrt(dx * dx + dy * dy);
        if (len < 1e-9) { double ex = pt[0] - a[0], ey = pt[1] - a[1]; return Math.Sqrt(ex * ex + ey * ey); }
        return Math.Abs(dy * (pt[0] - a[0]) - dx * (pt[1] - a[1])) / len;
    }

    public static string ToYolo(List<Poly> polys, int w, int h)
    {
        var sb = new StringBuilder();
        var inv = CultureInfo.InvariantCulture;
        foreach (var p in polys)
        {
            sb.Append(p.Cls);
            foreach (var pt in p.Pts)
            {
                sb.Append(' ').Append(((pt[0] + 0.5) / w).ToString("0.######", inv));
                sb.Append(' ').Append(((pt[1] + 0.5) / h).ToString("0.######", inv));
            }
            sb.Append('\n');
        }
        return sb.ToString();
    }

    public static void Preview(string rgbPath, List<Poly> polys, string[] names, string outPath)
    {
        Color[] cols = { Color.Lime, Color.Cyan, Color.Magenta, Color.Yellow, Color.Orange, Color.Red, Color.DeepSkyBlue, Color.White };
        using (var src = new Bitmap(rgbPath))
        using (var bmp = new Bitmap(src.Width, src.Height, PixelFormat.Format24bppRgb))
        {
            using (var g = Graphics.FromImage(bmp))
            using (var font = new Font("Arial", Math.Max(10, bmp.Width / 70), FontStyle.Bold))
            {
                g.DrawImage(src, 0, 0, src.Width, src.Height);
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                foreach (var p in polys)
                {
                    var col = cols[p.Cls % cols.Length];
                    var pts = new PointF[p.Pts.Count];
                    for (int i = 0; i < pts.Length; i++) pts[i] = new PointF(p.Pts[i][0] + 0.5f, p.Pts[i][1] + 0.5f);
                    using (var fill = new SolidBrush(Color.FromArgb(70, col))) g.FillPolygon(fill, pts);
                    using (var pen = new Pen(col, 2)) g.DrawPolygon(pen, pts);
                    g.DrawString(names[p.Cls], font, Brushes.Black, pts[0].X + 1, pts[0].Y - font.Height + 1);
                    using (var b = new SolidBrush(col)) g.DrawString(names[p.Cls], font, b, pts[0].X, pts[0].Y - font.Height);
                }
            }
            bmp.Save(outPath, ImageFormat.Png);
        }
    }

    // Stable split: same file name always lands in the same split
    public static bool IsVal(string name, double frac)
    {
        uint hsh = 2166136261;
        foreach (char ch in name) { hsh ^= ch; hsh *= 16777619; }
        return (hsh % 10000) < frac * 10000;
    }
}
"@

if (-not $Config) { $Config = Join-Path (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)) "seg_config.json" }
$cfg = Get-Content -Raw -Encoding UTF8 $Config | ConvertFrom-Json
[string[]]$names = @($cfg.classes)
$step = [int]$cfg.stencil_step
$maskDir = Join-Path $Capture "masks"
$imgDir = Join-Path $Capture "images"
$masks = @(Get-ChildItem $maskDir -File | Where-Object { $_.Extension -in '.png', '.bmp' } | Sort-Object Name)
if ($masks.Count -eq 0) { throw "no masks in $maskDir" }

Write-Host "classes (YOLO id / grey):"
for ($i = 0; $i -lt $names.Count; $i++) { Write-Host ("  {0}  {1,-14} grey {2}" -f $i, $names[$i], (($i + 1) * $step)) }

if ($Inspect) {
    # Scan every mask: empty frames are normal (target out of view), so show levels from non-empty ones
    $emptyNames = @(); $shown = 0
    foreach ($m in $masks) {
        $w = 0; $h = 0
        $g = [YoloSeg]::ReadGray($m.FullName, [ref]$w, [ref]$h)
        $hist = [YoloSeg]::Histogram($g)
        if ($hist -match '^0:\d+\s*$') { $emptyNames += $m.Name; continue }
        if ($shown -lt 5) { Write-Host "$($m.Name) (${w}x${h})  grey:count  $hist"; $shown++ }
    }
    Write-Host "$($masks.Count) masks, $($masks.Count - $emptyNames.Count) with objects, $($emptyNames.Count) empty"
    if ($emptyNames.Count -gt 0) { Write-Host "  empty: $(($emptyNames | Select-Object -First 10) -join ', ')$(if ($emptyNames.Count -gt 10) { ' ...' })" }
    Write-Host "Expected greys are multiples of $step. Anything else (except a few edge pixels) means the mask capture is altering values - see README."
    return
}

if (-not $Out) { throw "give -Out <dataset folder> (or -Inspect)" }
foreach ($d in 'images\train', 'images\val', 'labels\train', 'labels\val', 'previews') {
    New-Item -ItemType Directory -Force (Join-Path $Out $d) | Out-Null
}
$Out = (Resolve-Path $Out).Path

$perClass = New-Object 'int[]' $names.Count
$empty = 0; $missing = 0; $nVal = 0; $done = 0; $bad = 0
foreach ($m in $masks) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($m.Name)
    $img = Get-ChildItem $imgDir -File -Filter "$stem.*" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $img) { $missing++; continue }

    # Any failure skips the frame - never fall through with the previous frame's mask/polygons
    $w = 0; $h = 0; $g = $null; $polys = $null
    if ($m.Length -eq 0) { Write-Warning "skipping $($m.Name): empty file (capture stopped mid-write?)"; $bad++; continue }
    try {
        $g = [YoloSeg]::ReadGray($m.FullName, [ref]$w, [ref]$h)
    } catch {
        Write-Warning "skipping $($m.Name): unreadable mask: $($_.Exception.InnerException.Message)"
        $bad++; continue
    }
    if ([YoloSeg]::InvalidFraction($g, $names.Count, $step) -gt 0.01) {
        Write-Warning "skipping $($m.Name): mask is not a valid class mask (broken capture?)"
        $bad++; continue
    }
    try {
        $polys = [YoloSeg]::Extract($g, $w, $h, $names.Count, $step, $MinArea, $Epsilon)
    } catch {
        Write-Warning "skipping $($m.Name): polygon extraction failed: $($_.Exception.InnerException)"
        $bad++; continue
    }
    foreach ($p in $polys) { $perClass[$p.Cls]++ }
    if ($polys.Count -eq 0) { $empty++ }

    $split = if ([YoloSeg]::IsVal($stem, $Val)) { 'val' } else { 'train' }
    if ($split -eq 'val') { $nVal++ }
    Copy-Item $img.FullName (Join-Path $Out "images\$split\$($img.Name)") -Force
    [IO.File]::WriteAllText((Join-Path $Out "labels\$split\$stem.txt"), [YoloSeg]::ToYolo($polys, $w, $h))

    if ($done -lt $Previews) { [YoloSeg]::Preview($img.FullName, $polys, $names, (Join-Path $Out "previews\$stem.png")) }
    $done++
    if ($done % 100 -eq 0) { Write-Host "  $done / $($masks.Count)" }
}

$yaml = "path: $($Out -replace '\\', '/')`ntrain: images/train`nval: images/val`nnames:`n"
for ($i = 0; $i -lt $names.Count; $i++) { $yaml += "  ${i}: $($names[$i])`n" }
[IO.File]::WriteAllText((Join-Path $Out "data.yaml"), $yaml)

Write-Host ""
Write-Host "frames: $done  (train $($done - $nVal), val $nVal)   no objects: $empty   mask without image: $missing   broken masks skipped: $bad"
for ($i = 0; $i -lt $names.Count; $i++) { Write-Host ("  {0,-14} {1} polygons" -f $names[$i], $perClass[$i]) }
Write-Host "dataset -> $Out   (check previews\ before training)"
