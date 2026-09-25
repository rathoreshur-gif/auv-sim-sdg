# Delete captured frames whose mask contains no labelled pixels (image + mask pair).
#
#   powershell -ExecutionPolicy Bypass -File remove_empty_frames.ps1 -Capture <capture dir> -DryRun
#   powershell -ExecutionPolicy Bypass -File remove_empty_frames.ps1 -Capture <capture dir>
#
# -DryRun only lists what would be deleted. -KeepFraction 0.1 keeps ~10% of empty frames
# (chosen evenly) as "nothing here" negatives instead of deleting them all.
param(
    [Parameter(Mandatory = $true)][string]$Capture,
    [switch]$DryRun,
    [double]$KeepFraction = 0.0
)

Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @"
using System; using System.Drawing; using System.Drawing.Imaging; using System.Runtime.InteropServices;
public static class EmptyMask {
    // True if every pixel's R channel is 0. Reads raw values (captures have alpha = 0).
    public static bool IsEmpty(string path) {
        using (var b = new Bitmap(path)) {
            var d = b.LockBits(new Rectangle(0, 0, b.Width, b.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var row = new byte[b.Width * 4];
            try {
                for (int y = 0; y < b.Height; y++) {
                    Marshal.Copy(d.Scan0 + y * d.Stride, row, 0, row.Length);
                    for (int x = 2; x < row.Length; x += 4) if (row[x] != 0) return false;
                }
                return true;
            } finally { b.UnlockBits(d); }
        }
    }
}
"@

$maskDir = Join-Path $Capture "masks"
$imgDir = Join-Path $Capture "images"
$masks = @(Get-ChildItem $maskDir -File | Where-Object { $_.Extension -in '.png', '.bmp' } | Sort-Object Name)
$empty = @($masks | Where-Object { [EmptyMask]::IsEmpty($_.FullName) })

$keepEvery = if ($KeepFraction -gt 0) { [math]::Max(1, [math]::Round(1 / $KeepFraction)) } else { 0 }
$toDelete = @()
for ($i = 0; $i -lt $empty.Count; $i++) {
    if ($keepEvery -and ($i % $keepEvery -eq 0)) { continue }
    $toDelete += $empty[$i]
}

Write-Host "$($masks.Count) frames, $($empty.Count) empty, deleting $($toDelete.Count)$(if ($keepEvery) { " (keeping $($empty.Count - $toDelete.Count) as negatives)" })"
foreach ($m in $toDelete) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($m.Name)
    $imgs = @(Get-ChildItem $imgDir -File -Filter "$stem.*" -ErrorAction SilentlyContinue)
    if ($DryRun) { Write-Host "  would delete $($m.Name) (+ $($imgs.Count) image)"; continue }
    Remove-Item $m.FullName
    $imgs | Remove-Item
}
if ($DryRun) { Write-Host "dry run - nothing deleted" } else { Write-Host "done" }
