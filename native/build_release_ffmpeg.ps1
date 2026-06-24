param(
    [string]$SourceDirectory = (Join-Path $PSScriptRoot "third_party\\ffmpeg-8.1.1"),
    [string]$InstallDirectory = (Join-Path $PSScriptRoot "bin"),
    [switch]$FetchSource
)

$ErrorActionPreference = 'Stop'

# Exact FFmpeg commit used by the 8.1.1 Gyan release that seeded development.
$commit = '239f2c733de417201d7ad3b3b8b0d9b63285b2b1'
$bash = 'C:\msys64\usr\bin\bash.exe'

if (-not (Test-Path $bash)) {
    throw 'MSYS2 bash was not found at C:\msys64\usr\bin\bash.exe.'
}
if ($FetchSource -and -not (Test-Path $SourceDirectory)) {
    git clone https://github.com/FFmpeg/FFmpeg.git $SourceDirectory
}
if (-not (Test-Path (Join-Path $SourceDirectory '.git'))) {
    throw "FFmpeg source checkout is required: $SourceDirectory"
}

Push-Location $SourceDirectory
try {
    git cat-file -e "$commit^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        git fetch --depth 1 origin $commit
    }
    git checkout --detach $commit
} finally {
    Pop-Location
}

$sourceUnix = ('/' + ($SourceDirectory -replace ':', '' -replace '\\', '/')).ToLowerInvariant()
# FFmpeg sources its generated config shell script while building. Keep the
# build prefix free of spaces/parentheses, then copy the runtime DLLs back.
$installUnix = '/tmp/colortrack-ffmpeg-runtime'
$installWindows = 'C:\msys64\tmp\colortrack-ffmpeg-runtime'
$configure = @(
    'export PATH=/mingw64/bin:/usr/bin',
    "cd '$sourceUnix'",
    'make distclean >/dev/null 2>&1 || true',
    "./configure --arch=x86_64 --target-os=mingw32 --disable-x86asm --enable-shared --disable-static --disable-programs --disable-doc --disable-network --disable-autodetect --disable-everything --enable-protocol=file --enable-demuxer=mov,matroska,avi,mpegts,flv --enable-decoder=h264,hevc,av1,vp9,mpeg4,mjpeg --enable-parser=h264,hevc,av1,vp9,mpeg4video,mjpeg --enable-hwaccel=h264_d3d11va,h264_d3d11va2,hevc_d3d11va,hevc_d3d11va2,av1_d3d11va,av1_d3d11va2 --enable-d3d11va --enable-dxva2 --prefix='$installUnix'",
    'make -j$(nproc)',
    'make install'
) -join ' && '

& $bash -lc $configure
if ($LASTEXITCODE -ne 0) {
    throw 'Minimal FFmpeg build failed.'
}

New-Item -ItemType Directory -Force -Path $InstallDirectory | Out-Null
foreach ($name in @('avcodec-62.dll', 'avformat-62.dll', 'avutil-60.dll', 'swresample-6.dll')) {
    $built = Join-Path $installWindows "bin\\$name"
    if (-not (Test-Path $built)) {
        throw "Minimal FFmpeg runtime is missing $name"
    }
    Copy-Item -LiteralPath $built -Destination (Join-Path $InstallDirectory $name) -Force
}
$winpthread = 'C:\msys64\mingw64\bin\libwinpthread-1.dll'
if (-not (Test-Path $winpthread)) {
    throw 'MinGW-w64 libwinpthread runtime is required by this FFmpeg build.'
}
Copy-Item -LiteralPath $winpthread -Destination (Join-Path $InstallDirectory 'libwinpthread-1.dll') -Force

Write-Output "Built reproducible minimal FFmpeg runtime from $commit"
