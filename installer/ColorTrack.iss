#define AppName "Color Track"
#ifndef AppVersion
  #define AppVersion "0.1.0-beta"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\ColorTrack"
#endif
#ifndef OutputDir
  #define OutputDir "..\release"
#endif

[Setup]
AppId={{9A714A5A-47F1-4C4A-A10B-7BCE2A55D8BA}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Martin Cingel
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\Programs\Color Track
DefaultGroupName=Color Track
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
SetupIconFile=..\assets\ColorTrack.ico
OutputDir={#OutputDir}
OutputBaseFilename=ColorTrack-{#AppVersion}-Windows-x64-Setup
Compression=zip
SolidCompression=no
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayName=Color Track
UninstallDisplayIcon={app}\ColorTrack.exe

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\Color Track"; Filename: "{app}\ColorTrack.exe"; IconFilename: "{app}\ColorTrack.exe"
Name: "{autodesktop}\Color Track"; Filename: "{app}\ColorTrack.exe"; IconFilename: "{app}\ColorTrack.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\ColorTrack.exe"; Description: "Launch Color Track"; Flags: nowait postinstall skipifsilent
