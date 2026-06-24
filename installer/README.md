# Windows Installer Build

Run from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File installer\build_installer.ps1
```

The build produces a one-folder PyInstaller application and an Inno Setup
installer in `release`. The installer contains Color Track's GPLv3 license,
third-party notices, FFmpeg provenance/build configuration, and the relevant
runtime license files under the installed `licenses` folder.

Before sharing a build, run the smoke tests and native runtime verification.
