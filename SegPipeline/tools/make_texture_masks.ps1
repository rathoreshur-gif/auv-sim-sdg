# Make one opacity mask per drawing on a texture (e.g. images printed on a board).
# White = object, black = everything else; same size/UVs as the texture.
#
#   powershell -ExecutionPolicy Bypass -File make_texture_masks.ps1 `
#       -Texture ..\props\torpedo_board\board_texture.png `
#       -Classes blood,ambulance,fire,firetruck `
#       -OutDir  ..\props\torpedo_board
#
# A pixel counts as "drawing" if it is opaque, not near-white and not an excluded
# colour (pure red by default - the torpedo rings). The N largest blobs (N = number
# of classes) are named in reading order: top row first, left to right.
# ALWAYS check masks_preview.png - it labels every blob with the name it got.
param(
    [Parameter(Mandatory = $true)][string]$Texture,
    [Parameter(Mandatory = $true)][string[]]$Classes,
    [string]$OutDir = (Split-Path -Parent $Texture),
    [int]$WhiteTol = 14,        # how far from pure white a pixel must be to count as drawing
    [int]$Border = 24,          # ignore this many px at the texture edge (frame lines)
    [double]$RowTol = 0.15,     # blobs whose centres differ by less than this * height are one row
    [switch]$KeepPureRed        # set to count pure red (255,0,0) as drawing
)

Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public static class TexMasks
{
    public static byte[] ReadBgra(string path, out int w, out int h)
    {
        using (var src = new Bitmap(path))
        using (var bmp = new Bitmap(src.Width, src.Height, PixelFormat.Format32bppArgb))
        {
            using (var g = Graphics.FromImage(bmp))
            {
                g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                g.DrawImage(src, 0, 0, src.Width, src.Height);
            }
            w = bmp.Width; h = bmp.Height;
            var d = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var buf = new byte[w * h * 4];
            for (int y = 0; y < h; y++)
                Marshal.Copy(d.Scan0 + y * d.Stride, buf, y * w * 4, w * 4);
            bmp.UnlockBits(d);
            return buf; // B, G, R, A
        }
    }

    public class Blob { public int Id, N, X0, Y0, X1, Y1; public double Cx, Cy; }

    // Returns n masks in reading order, plus the blobs (for the preview)
    public static byte[][] Find(byte[] px, int w, int h, int n, int whiteTol, int border,
                                bool excludeRed, double rowTol, out List<Blob> blobs)
    {
        var m = new bool[w * h];
        for (int y = border; y < h - border; y++)
            for (int x = border; x < w - border; x++)
            {
                int i = (y * w + x) * 4;
                int b = px[i], g = px[i + 1], r = px[i + 2], a = px[i + 3];
                bool red = excludeRed && r > 220 && g < 40 && b < 40;
                int dist = Math.Max(255 - r, Math.Max(255 - g, 255 - b));
                m[y * w + x] = a > 127 && dist > whiteTol && !red;
            }

        m = Erode(Dilate(m, w, h, 3), w, h, 3);   // close small gaps inside drawings
        m = Dilate(Erode(m, w, h, 3), w, h, 3);   // drop thin anti-aliased edges and specks

        int[] lab; List<int> sizes;
        Label(m, w, h, out lab, out sizes);

        var all = new List<Blob>();
        for (int k = 1; k < sizes.Count; k++) all.Add(new Blob { Id = k, N = sizes[k], X0 = w, Y0 = h });
        all.Sort((a, b) => b.N.CompareTo(a.N));
        if (all.Count < n) throw new Exception("found only " + all.Count + " blobs, expected " + n);
        blobs = all.GetRange(0, n);

        var byId = new Dictionary<int, Blob>();
        foreach (var bl in blobs) byId[bl.Id] = bl;
        var sx = new Dictionary<int, long>(); var sy = new Dictionary<int, long>();
        foreach (var bl in blobs) { sx[bl.Id] = 0; sy[bl.Id] = 0; }
        for (int i = 0; i < lab.Length; i++)
        {
            Blob bl;
            if (lab[i] == 0 || !byId.TryGetValue(lab[i], out bl)) continue;
            int x = i % w, y = i / w;
            sx[bl.Id] += x; sy[bl.Id] += y;
            bl.X0 = Math.Min(bl.X0, x); bl.Y0 = Math.Min(bl.Y0, y);
            bl.X1 = Math.Max(bl.X1, x); bl.Y1 = Math.Max(bl.Y1, y);
        }
        foreach (var bl in blobs) { bl.Cx = (double)sx[bl.Id] / bl.N; bl.Cy = (double)sy[bl.Id] / bl.N; }

        // Reading order: group into rows by centre Y, then left to right
        blobs.Sort((a, b) => a.Cy.CompareTo(b.Cy));
        var ordered = new List<Blob>();
        int s = 0;
        while (s < blobs.Count)
        {
            int e = s + 1;
            while (e < blobs.Count && blobs[e].Cy - blobs[s].Cy < rowTol * h) e++;
            var row = blobs.GetRange(s, e - s);
            row.Sort((a, b) => a.Cx.CompareTo(b.Cx));
            ordered.AddRange(row);
            s = e;
        }
        blobs = ordered;

        var result = new byte[n][];
        for (int k = 0; k < n; k++)
        {
            var one = new bool[w * h];
            for (int i = 0; i < lab.Length; i++) one[i] = lab[i] == blobs[k].Id;
            one = FillHoles(one, w, h);
            var outp = new byte[w * h];
            for (int i = 0; i < one.Length; i++) if (one[i]) outp[i] = 255;
            result[k] = outp;
        }
        return result;
    }

    static bool[] Dilate(bool[] m, int w, int h, int r) { return Morph(m, w, h, r, true); }
    static bool[] Erode(bool[] m, int w, int h, int r)  { return Morph(m, w, h, r, false); }

    // Square structuring element, done as two 1-D passes
    static bool[] Morph(bool[] m, int w, int h, int r, bool dilate)
    {
        var tmp = new bool[w * h]; var outp = new bool[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                bool v = !dilate;
                for (int k = -r; k <= r; k++)
                {
                    int xx = x + k; bool p = (xx >= 0 && xx < w) ? m[y * w + xx] : false;
                    if (dilate && p) { v = true; break; }
                    if (!dilate && !p) { v = false; break; }
                }
                tmp[y * w + x] = v;
            }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                bool v = !dilate;
                for (int k = -r; k <= r; k++)
                {
                    int yy = y + k; bool p = (yy >= 0 && yy < h) ? tmp[yy * w + x] : false;
                    if (dilate && p) { v = true; break; }
                    if (!dilate && !p) { v = false; break; }
                }
                outp[y * w + x] = v;
            }
        return outp;
    }

    static void Label(bool[] m, int w, int h, out int[] lab, out List<int> sizes)
    {
        lab = new int[w * h]; sizes = new List<int> { 0 };
        var q = new Queue<int>();
        for (int s = 0; s < m.Length; s++)
        {
            if (!m[s] || lab[s] != 0) continue;
            int id = sizes.Count, n = 0; lab[s] = id; q.Enqueue(s);
            while (q.Count > 0)
            {
                int i = q.Dequeue(), x = i % w, y = i / w; n++;
                if (x > 0     && m[i - 1] && lab[i - 1] == 0) { lab[i - 1] = id; q.Enqueue(i - 1); }
                if (x < w - 1 && m[i + 1] && lab[i + 1] == 0) { lab[i + 1] = id; q.Enqueue(i + 1); }
                if (y > 0     && m[i - w] && lab[i - w] == 0) { lab[i - w] = id; q.Enqueue(i - w); }
                if (y < h - 1 && m[i + w] && lab[i + w] == 0) { lab[i + w] = id; q.Enqueue(i + w); }
            }
            sizes.Add(n);
        }
    }

    // Anything not reachable from the image edge through background is a hole -> fill it
    static bool[] FillHoles(bool[] m, int w, int h)
    {
        var outside = new bool[w * h];
        var q = new Queue<int>();
        for (int x = 0; x < w; x++) { Seed(m, outside, q, x); Seed(m, outside, q, (h - 1) * w + x); }
        for (int y = 0; y < h; y++) { Seed(m, outside, q, y * w); Seed(m, outside, q, y * w + w - 1); }
        while (q.Count > 0)
        {
            int i = q.Dequeue(), x = i % w, y = i / w;
            if (x > 0) Seed(m, outside, q, i - 1);
            if (x < w - 1) Seed(m, outside, q, i + 1);
            if (y > 0) Seed(m, outside, q, i - w);
            if (y < h - 1) Seed(m, outside, q, i + w);
        }
        var r = new bool[w * h];
        for (int i = 0; i < r.Length; i++) r[i] = !outside[i];
        return r;
    }

    static void Seed(bool[] m, bool[] seen, Queue<int> q, int i)
    {
        if (!m[i] && !seen[i]) { seen[i] = true; q.Enqueue(i); }
    }

    public static int Count(byte[] m) { int n = 0; foreach (byte b in m) if (b > 0) n++; return n; }

    public static void SaveGray(byte[] mask, int w, int h, string path)
    {
        using (var bmp = new Bitmap(w, h, PixelFormat.Format24bppRgb))
        {
            var d = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            var row = new byte[w * 3];
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++) { byte v = mask[y * w + x]; row[x * 3] = v; row[x * 3 + 1] = v; row[x * 3 + 2] = v; }
                Marshal.Copy(row, 0, d.Scan0 + y * d.Stride, w * 3);
            }
            bmp.UnlockBits(d);
            bmp.Save(path, ImageFormat.Png);
        }
    }

    // Texture (transparent = dark grey) with each mask tinted and labelled with its class name
    public static void SavePreview(byte[] px, int w, int h, byte[][] masks, List<Blob> blobs, string[] names, string path)
    {
        Color[] cols = { Color.Blue, Color.Lime, Color.Cyan, Color.Magenta, Color.Orange, Color.Yellow, Color.Purple, Color.Teal };
        var o = new byte[w * h * 3];
        for (int i = 0; i < w * h; i++)
        {
            int a = px[i * 4 + 3];
            for (int c = 0; c < 3; c++) o[i * 3 + c] = (byte)((px[i * 4 + c] * a + 60 * (255 - a)) / 255);
        }
        for (int k = 0; k < masks.Length; k++)
        {
            var col = cols[k % cols.Length];
            for (int i = 0; i < w * h; i++)
                if (masks[k][i] > 0)
                {
                    o[i * 3]     = (byte)((o[i * 3]     + col.B) / 2);
                    o[i * 3 + 1] = (byte)((o[i * 3 + 1] + col.G) / 2);
                    o[i * 3 + 2] = (byte)((o[i * 3 + 2] + col.R) / 2);
                }
        }
        using (var bmp = new Bitmap(w, h, PixelFormat.Format24bppRgb))
        {
            var d = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            for (int y = 0; y < h; y++) Marshal.Copy(o, y * w * 3, d.Scan0 + y * d.Stride, w * 3);
            bmp.UnlockBits(d);
            using (var g = Graphics.FromImage(bmp))
            using (var font = new Font("Arial", Math.Max(12, w / 40), FontStyle.Bold))
            {
                g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
                for (int k = 0; k < blobs.Count; k++)
                {
                    var bl = blobs[k];
                    g.DrawRectangle(new Pen(Color.Black, 3), bl.X0, bl.Y0, bl.X1 - bl.X0, bl.Y1 - bl.Y0);
                    string label = k + ": " + names[k];
                    var sz = g.MeasureString(label, font);
                    float ly = Math.Max(0, bl.Y0 - sz.Height);   // inside the box if no room above
                    g.FillRectangle(Brushes.Black, bl.X0, ly, sz.Width, sz.Height);
                    g.DrawString(label, font, Brushes.White, bl.X0, ly);
                }
            }
            bmp.Save(path, ImageFormat.Png);
        }
    }
}
"@

# Allow "-Classes a,b,c" as well as "-Classes 'a,b,c'" (the latter arrives as one string via -File)
$Classes = @($Classes | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Texture = (Resolve-Path $Texture).Path
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force $OutDir | Out-Null }
$OutDir = (Resolve-Path $OutDir).Path

$w = 0; $h = 0
$px = [TexMasks]::ReadBgra($Texture, [ref]$w, [ref]$h)
Write-Host "texture: ${w}x${h}, looking for $($Classes.Count) drawings"

$blobs = $null
$masks = [TexMasks]::Find($px, $w, $h, $Classes.Count, $WhiteTol, $Border, (-not $KeepPureRed), $RowTol, [ref]$blobs)

for ($k = 0; $k -lt $Classes.Count; $k++) {
    $b = $blobs[$k]
    $out = Join-Path $OutDir "T_Mask_$($Classes[$k]).png"
    [TexMasks]::SaveGray($masks[$k], $w, $h, $out)
    Write-Host ("{0}: {1,-12} centre ({2:F0},{3:F0})  {4,8} px  -> {5}" -f $k, $Classes[$k], $b.Cx, $b.Cy, [TexMasks]::Count($masks[$k]), (Split-Path $out -Leaf))
}

$preview = Join-Path $OutDir "masks_preview.png"
[TexMasks]::SavePreview($px, $w, $h, $masks, $blobs, [string[]]$Classes, $preview)
Write-Host "preview -> $preview   (check every label sits on the right drawing)"
