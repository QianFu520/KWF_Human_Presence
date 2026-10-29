<#
check_dupes.ps1: finds clips that appear in more than one zip, and counts UNIQUE
clips per recorder per day.

Usage:
  .\check_dupes.ps1 -ZipDir "C:\BirdNET ONNX\zips\recut" -OutDir "C:\BirdNET ONNX\hp_audit\recut_check"

Writes to -OutDir:
  unique_by_day.csv     unique clips per recorder per day (a full day is 28,800)
  duplicate_pairs.csv   which zips share clips, and how many
#>
param(
    [Parameter(Mandatory = $true)][string]$ZipDir,
    [Parameter(Mandatory = $true)][string]$OutDir
)

Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force $OutDir | Out-Null
$re = '^(Audio_Moth_\d)_(\d{8})_\d{6}\.wav$'

# clip name -> list of zips it appears in
$seen = @{}
foreach ($z in (Get-ChildItem -Path $ZipDir -Filter *.zip | Sort-Object Name)) {
    Write-Host "Reading $($z.Name) ..."
    $zip = [System.IO.Compression.ZipFile]::OpenRead($z.FullName)
    try {
        foreach ($e in $zip.Entries) {
            if (-not [regex]::IsMatch($e.Name, $re)) { continue }
            if ($seen.ContainsKey($e.Name)) { $seen[$e.Name].Add($z.Name) }
            else { $l = New-Object System.Collections.Generic.List[string]; $l.Add($z.Name); $seen[$e.Name] = $l }
        }
    }
    finally { $zip.Dispose() }
}

$days = @{}
$pairs = @{}
foreach ($name in $seen.Keys) {
    $m = [regex]::Match($name, $re)
    $key = "$($m.Groups[1].Value)|$($m.Groups[2].Value)"
    if ($days.ContainsKey($key)) { $days[$key]++ } else { $days[$key] = 1 }
    $zs = $seen[$name]
    if ($zs.Count -gt 1) {
        $pk = ($zs | Sort-Object) -join ' + '
        if ($pairs.ContainsKey($pk)) { $pairs[$pk]++ } else { $pairs[$pk] = 1 }
    }
}

$dayRows = foreach ($k in $days.Keys) {
    $p = $k.Split('|')
    [pscustomobject]@{ recorder = $p[0]; date = $p[1]; unique_clips = $days[$k] }
}
$dayRows = $dayRows | Sort-Object recorder, date
$pairRows = foreach ($k in $pairs.Keys) { [pscustomobject]@{ zips = $k; shared_clips = $pairs[$k] } }
$pairRows = $pairRows | Sort-Object zips

$dayRows  | Export-Csv (Join-Path $OutDir 'unique_by_day.csv') -NoTypeInformation
$pairRows | Export-Csv (Join-Path $OutDir 'duplicate_pairs.csv') -NoTypeInformation

Write-Host "`n== Unique clips per recorder per day (full day = 28,800) =="
$dayRows | Format-Table -AutoSize
Write-Host "== Zips that share clips =="
if ($pairRows) { $pairRows | Format-Table -AutoSize } else { Write-Host "None" }
$byRec = $dayRows | Group-Object recorder | ForEach-Object { [pscustomobject]@{ recorder = $_.Name; unique_clips = ($_.Group | Measure-Object unique_clips -Sum).Sum } }
Write-Host "== Unique clips per recorder =="
$byRec | Format-Table -AutoSize
Write-Host "Total unique clips: $($seen.Count)"
