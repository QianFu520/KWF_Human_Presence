<#
check_zips.ps1: checks clip zips without unzipping them.

For every .zip in -ZipDir, counts the 3-second clips, checks each is exactly
288,044 bytes (144,000 samples at 48 kHz, 16-bit mono, plus the WAV header),
and counts clips per recorder, day and hour.

Usage:
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\check_zips.ps1 -ZipDir "C:\BirdNET ONNX\zips\recut" -OutDir "C:\BirdNET ONNX\hp_audit\recut_check"

Writes to -OutDir:
  zip_summary.csv         one row per zip
  coverage_by_day.csv     clips per recorder per day (a full day is 28,800)
  coverage_by_hour.csv    clips per recorder per day per hour (a full hour is 1,200)
  wrong_length_clips.csv  any clip that isn't 288,044 bytes
#>
param(
    [Parameter(Mandatory = $true)][string]$ZipDir,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [long]$ExpectedBytes = 288044
)

Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force $OutDir | Out-Null

$re = '^(Audio_Moth_\d)_(\d{8})_(\d{2})\d{4}\.wav$'
$summary = New-Object System.Collections.Generic.List[object]
$bad = New-Object System.Collections.Generic.List[object]
$hours = @{}

$zips = Get-ChildItem -Path $ZipDir -Filter *.zip | Sort-Object Name
if (-not $zips) { Write-Host "No .zip files found in $ZipDir"; exit 1 }

foreach ($z in $zips) {
    Write-Host "Checking $($z.Name) ..."
    $zip = [System.IO.Compression.ZipFile]::OpenRead($z.FullName)
    try {
        $n = 0; $ok = 0; $wrong = 0; $other = 0
        foreach ($e in $zip.Entries) {
            if ($e.FullName.EndsWith('/')) { continue }   # folder entry
            $m = [regex]::Match($e.Name, $re)
            if (-not $m.Success) { $other++; continue }
            $n++
            if ($e.Length -eq $ExpectedBytes) { $ok++ }
            else {
                $wrong++
                $bad.Add([pscustomobject]@{ zip = $z.Name; clip = $e.Name; bytes = $e.Length })
            }
            $key = "$($m.Groups[1].Value)|$($m.Groups[2].Value)|$($m.Groups[3].Value)"
            if ($hours.ContainsKey($key)) { $hours[$key]++ } else { $hours[$key] = 1 }
        }
        $summary.Add([pscustomobject]@{ zip = $z.Name; clips = $n; correct_length = $ok; wrong_length = $wrong; other_files = $other })
    }
    finally { $zip.Dispose() }
}

$hourRows = foreach ($k in $hours.Keys) {
    $p = $k.Split('|')
    [pscustomobject]@{ recorder = $p[0]; date = $p[1]; hour = [int]$p[2]; clips = $hours[$k] }
}
$hourRows = $hourRows | Sort-Object recorder, date, hour
$dayRows = $hourRows | Group-Object recorder, date | ForEach-Object {
    [pscustomobject]@{
        recorder = $_.Group[0].recorder
        date     = $_.Group[0].date
        hours    = $_.Count
        clips    = ($_.Group | Measure-Object clips -Sum).Sum
    }
} | Sort-Object recorder, date

$summary  | Export-Csv (Join-Path $OutDir 'zip_summary.csv') -NoTypeInformation
$dayRows  | Export-Csv (Join-Path $OutDir 'coverage_by_day.csv') -NoTypeInformation
$hourRows | Export-Csv (Join-Path $OutDir 'coverage_by_hour.csv') -NoTypeInformation
$bad      | Export-Csv (Join-Path $OutDir 'wrong_length_clips.csv') -NoTypeInformation

Write-Host "`n== Per zip =="
$summary | Format-Table -AutoSize
Write-Host "== Per recorder per day (full day = 28,800) =="
$dayRows | Format-Table -AutoSize
$tot = ($summary | Measure-Object clips -Sum).Sum
$tw  = ($summary | Measure-Object wrong_length -Sum).Sum
Write-Host "Total clips: $tot | wrong length: $tw"
Write-Host "Wrote CSVs to $OutDir"
