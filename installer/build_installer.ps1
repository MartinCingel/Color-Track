param(
    [string]$Version = '0.1.0-beta'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root 'build\pyinstaller'
$dist = Join-Path $root 'dist'
$release = Join-Path $root 'release'
$iss = Join-Path $PSScriptRoot 'ColorTrack.iss'

Push-Location $root
try {
    Remove-Item -LiteralPath $build,$dist -Recurse -Force -ErrorAction SilentlyContinue
    # The CPU Point Fast and D3D11 decoder paths do not require the optional
    # CUDA/ML packages installed in this development environment. Excluding
    # them prevents PyInstaller from adding several gigabytes of unused DLLs.
    python -m PyInstaller --noconfirm --clean --windowed --onedir --name ColorTrack --icon 'assets\ColorTrack.ico' `
        --workpath $build --distpath $dist `
        --exclude-module torch --exclude-module torchvision --exclude-module tensorflow `
        --exclude-module cupy --exclude-module cupy_backends `
        --exclude-module tracking.trackers --exclude-module tracking.point_fast_old --exclude-module tracking.point_fast_old_2 `
        --exclude-module gpu.fft_utils --exclude-module gpu.hessian `
        --add-binary 'native\bin\NativeD3DDecoder.dll;native\bin' `
        --add-binary 'native\bin\avcodec-62.dll;native\bin' `
        --add-binary 'native\bin\avformat-62.dll;native\bin' `
        --add-binary 'native\bin\avutil-60.dll;native\bin' `
        --add-binary 'native\bin\swresample-6.dll;native\bin' `
        --add-binary 'native\bin\libwinpthread-1.dll;native\bin' `
        --add-data 'LICENSE;licenses' `
        --add-data 'COPYRIGHT;licenses' `
        --add-data 'THIRD_PARTY_NOTICES.md;licenses' `
        --add-data 'PATENT_AND_DECODER_NOTICE.md;licenses' `
        --add-data 'FFMPEG_PROVENANCE.md;licenses' `
        --add-data 'native\FFMPEG_BUILD_CONFIGURATION.txt;licenses\native' `
        --add-data 'native\FFMPEG_RELEASE_SOURCE.md;licenses\native' `
        --add-data 'native\licenses;licenses\native' `
        --add-data 'assets\ColorTrack.ico;assets' `
        --collect-all PyQt6 main.py

    $iscc = Get-ChildItem 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe','C:\Program Files\Inno Setup 6\ISCC.exe',"$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    if (-not $iscc) { throw 'Inno Setup 6 ISCC.exe was not found.' }
    New-Item -ItemType Directory -Force -Path $release | Out-Null
    & $iscc "/DAppVersion=$Version" "/DSourceDir=$dist\ColorTrack" "/DOutputDir=$release" $iss
    if ($LASTEXITCODE -ne 0) { throw 'Inno Setup compilation failed.' }
} finally {
    Pop-Location
}
