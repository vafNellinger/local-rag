; NSIS-Installer für local-rag (Windows).
;
; Kompilieren (erzeugt dist\local-rag-setup.exe):
;   makensis packaging\local-rag.nsi           ; auf Windows oder per Linux-NSIS
;
; Der Installer legt den Code unter %LOCALAPPDATA%\Programs\local-rag ab, macht
; eine Startmenü- und Desktop-Verknüpfung und bringt einen Deinstallierer mit.
; Die eigentliche Python-Umgebung entsteht beim ersten Start (siehe
; windows-launcher.bat) — deshalb braucht der Installer keine Admin-Rechte und
; bündelt keine Gigabyte an Modellen.

Unicode true
!define APPNAME "local-rag"
!define VERSION "0.1.0"

Name "${APPNAME}"
OutFile "..\dist\local-rag-setup.exe"
RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\Programs\${APPNAME}"

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section "Install"
  SetOutPath "$INSTDIR"
  ; Anwendungscode, relativ zum Ort dieser .nsi-Datei (packaging\).
  ; Übersetzten Bytecode aus der Entwicklungsumgebung ausschließen — er wird
  ; pro Zielumgebung neu erzeugt und gehört nicht in den Installer.
  File /r /x "__pycache__" /x "*.pyc" "..\rag"
  File /r /x "__pycache__" /x "*.pyc" "..\config"
  File "..\pyproject.toml"
  File "..\README.md"
  File "windows-launcher.bat"
  File "detect_gpu.py"

  ; Verknüpfungen auf den Launcher.
  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\windows-launcher.bat"
  CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\windows-launcher.bat"

  ; Deinstallation.
  WriteUninstaller "$INSTDIR\uninstall.exe"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\Deinstallieren.lnk" "$INSTDIR\uninstall.exe"
SectionEnd

Section "Uninstall"
  Delete "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk"
  Delete "$SMPROGRAMS\${APPNAME}\Deinstallieren.lnk"
  RMDir "$SMPROGRAMS\${APPNAME}"
  Delete "$DESKTOP\${APPNAME}.lnk"
  RMDir /r "$INSTDIR"
  ; Umgebung und Modelle unter %LOCALAPPDATA%\local-rag bleiben bewusst stehen
  ; (mehrere GB); zum vollständigen Entfernen diesen Ordner von Hand löschen.
SectionEnd
