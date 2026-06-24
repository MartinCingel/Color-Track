param(
    [Parameter(Mandatory = $true)]
    [string]$FfmpegRoot
)

$ErrorActionPreference = 'Stop'
$source = Join-Path $FfmpegRoot 'bin'
$destination = Join-Path $PSScriptRoot 'bin'
$required = @('avformat-62.dll', 'avcodec-62.dll', 'avutil-60.dll', 'swresample-6.dll')

if (-not (Test-Path $source)) {
    throw "FFmpeg bin directory was not found: $source"
}
New-Item -ItemType Directory -Force -Path $destination | Out-Null
foreach ($name in $required) {
    $file = Join-Path $source $name
    if (-not (Test-Path $file)) {
        throw "Required FFmpeg runtime DLL is missing: $file"
    }
    Copy-Item -LiteralPath $file -Destination (Join-Path $destination $name) -Force
}
Write-Output "Staged FFmpeg runtime DLLs in $destination"
