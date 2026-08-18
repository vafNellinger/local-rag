; NSIS-Installer für local-rag (Windows).
;
; Per-User-Installation ohne Admin-Rechte: das Bundle landet unter
; %LOCALAPPDATA%\Programs\local-rag, Verknüpfungen im Startmenü und auf dem
; Desktop, Eintrag unter "Apps & Features" samt Deinstaller. Damit ist die
; Installation ein gewöhnlicher Klick-Wizard, keine Admin-Anhebung nötig.
;
; Cross-Bau auf Linux/CI:
;   makensis -DVERSION=0.1.0 -DSRC=dist/local-rag -DOUTFILE=local-rag-setup.exe \
;            packaging/nsis/local-rag.nsi
; Quellpfade mit / (vom Host gelesen), Zielpfade sind Windows-Laufzeitpfade.

Unicode true
!include "MUI2.nsh"

!ifndef VERSION
  !define VERSION "0.1.0"
!endif
!ifndef SRC
  !define SRC "../../dist/local-rag"
!endif
!ifndef OUTFILE
  !define OUTFILE "local-rag-setup-${VERSION}.exe"
!endif

!define APPNAME "local-rag"
!define PUBLISHER "von Affenfels"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}"

Name "${APPNAME} ${VERSION}"
OutFile "${OUTFILE}"
RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\Programs\${APPNAME}"
InstallDirRegKey HKCU "Software\${APPNAME}" "InstallDir"
SetCompressor /SOLID lzma

; --- Wizard-Seiten ----------------------------------------------------------
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\${APPNAME}.exe"
!define MUI_FINISHPAGE_RUN_TEXT "local-rag jetzt starten"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Deutsch zuerst (Vorgabe), Englisch als Ausweichsprache.
!insertmacro MUI_LANGUAGE "German"
!insertmacro MUI_LANGUAGE "English"

; --- Installation -----------------------------------------------------------
Section "local-rag" SecMain
  SectionIn RO
  SetOutPath "$INSTDIR"
  ; Backslash-Pfad: Windows-makensis findet mit / keine Dateien. Der Aufrufer
  ; übergibt SRC als absoluten Windows-Pfad (cygpath -w).
  File /r "${SRC}\*"

  CreateDirectory "$SMPROGRAMS\${APPNAME}"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk" "$INSTDIR\${APPNAME}.exe"
  CreateShortcut "$SMPROGRAMS\${APPNAME}\${APPNAME} deinstallieren.lnk" "$INSTDIR\uninstall.exe"
  CreateShortcut "$DESKTOP\${APPNAME}.lnk" "$INSTDIR\${APPNAME}.exe"

  WriteUninstaller "$INSTDIR\uninstall.exe"

  WriteRegStr HKCU "Software\${APPNAME}" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${APPNAME}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "${PUBLISHER}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\${APPNAME}.exe"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
SectionEnd

; --- Deinstallation ---------------------------------------------------------
Section "Uninstall"
  Delete "$SMPROGRAMS\${APPNAME}\${APPNAME}.lnk"
  Delete "$SMPROGRAMS\${APPNAME}\${APPNAME} deinstallieren.lnk"
  RMDir "$SMPROGRAMS\${APPNAME}"
  Delete "$DESKTOP\${APPNAME}.lnk"
  RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "Software\${APPNAME}"
  DeleteRegKey HKCU "${UNINST_KEY}"
SectionEnd
