param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$source = Join-Path $PSScriptRoot "d3d11_workspace_decoder.cpp"
$outputDirectory = Join-Path $PSScriptRoot "bin"
$output = Join-Path $outputDirectory "NativeD3DDecoder.dll"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

if (-not (Test-Path $vswhere)) {
    throw "Visual Studio Build Tools were not found."
}
$installation = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) {
    throw "C++ build tools are not installed."
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$ffmpegRoot = $env:FFMPEG_SHARED_ROOT
if (-not $ffmpegRoot) {
    $ffmpegRoot = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory -Filter "Gyan.FFmpeg.Shared*" -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ChildItem $_.FullName -Directory -Filter "ffmpeg-*-full_build-shared" | Select-Object -First 1 } |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ffmpegRoot -or -not (Test-Path (Join-Path $ffmpegRoot "include\libavcodec\avcodec.h"))) {
    throw "Set FFMPEG_SHARED_ROOT to an FFmpeg shared development build."
}
$devCmd = Join-Path $installation "Common7\Tools\VsDevCmd.bat"
$command = "call `"$devCmd`" -arch=x64 -host_arch=x64 && cl /nologo /std:c++17 /EHsc /O2 /MT /LD /I `"$ffmpegRoot\include`" `"$source`" /Fe:`"$output`" /link /LIBPATH:`"$ffmpegRoot\lib`" avformat.lib avcodec.lib avutil.lib d3d11.lib d3dcompiler.lib dxgi.lib mfplat.lib mfreadwrite.lib mfuuid.lib ole32.lib"
cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    throw "Native D3D11 decoder build failed."
}
Write-Output "Built $output"
