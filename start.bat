@echo off
rem Ein-Klick-Start fuer local-rag (Windows). Doppelklick genuegt.
rem
rem Erster Start richtet eine isolierte Umgebung ein; jeder weitere Start
rem oeffnet nur die Oberflaeche im Browser. uv holt bei Bedarf Python selbst,
rem und llama-cpp-python kommt als vorkompiliertes CPU-Wheel — deshalb sind
rem WEDER ein vorhandenes Python NOCH die Visual-Studio-Build-Tools noetig,
rem nur beim ersten Mal eine Internetverbindung.
setlocal
title local-rag

rem Ins Projektverzeichnis wechseln — unabhaengig davon, von wo gestartet wurde.
cd /d "%~dp0"

set "VENV=.venv"
set "MARKER=%VENV%\.setup-done"
rem Vorkompilierte CPU-Wheels von llama-cpp-python — spart den Compiler. Bei
rem einer NVIDIA-Karte waehlt packaging\detect_gpu.py ein CUDA-Wheel.
set "CPU_WHEELS=https://abetlen.github.io/llama-cpp-python/whl/cpu"
rem uv und seine Shims (auch whichllm) landen hier, ohne Admin-Rechte.
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

rem ── uv sicherstellen ─────────────────────────────────────────────────────
where uv >nul 2>nul
if not errorlevel 1 goto :haveuv
echo uv wird installiert (einmalig)...
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 goto :fehler
:haveuv

rem ── Ersteinrichtung ──────────────────────────────────────────────────────
if exist "%MARKER%" goto :run
echo Ersteinrichtung - das dauert beim ersten Mal einige Minuten...
rem Nur anlegen, wenn nichts da ist: ein bestehendes .venv (etwa mit einem
rem selbst kompilierten GPU-Build) soll nicht ueberschrieben werden. Die
rem pip-Schritte darunter sind idempotent — schon Vorhandenes bleibt.
if exist "%VENV%" goto :haveenv
uv venv --python 3.12
if errorlevel 1 goto :fehler
:haveenv
rem GPU automatisch waehlen: NVIDIA -> passendes CUDA-Wheel, sonst CPU. Fuer die
rem Flotte ueberschreibbar mit LOCAL_RAG_GPU_INDEX.
set "WHEELS=%LOCAL_RAG_GPU_INDEX%"
if not "%WHEELS%"=="" goto :gotwheels
for /f "usebackq delims=" %%i in (`"%VENV%\Scripts\python.exe" packaging\detect_gpu.py 2^>nul`) do set "WHEELS=%%i"
:gotwheels
if "%WHEELS%"=="" set "WHEELS=%CPU_WHEELS%"
if /i "%WHEELS%"=="%CPU_WHEELS%" (echo GPU: keine passende erkannt - laeuft auf CPU.) else (echo NVIDIA erkannt - installiere CUDA-Build %WHEELS%.)
rem Primaerindex ist der Wheel-Server, die uebrigen Abhaengigkeiten von PyPI.
uv pip install --index-url "%WHEELS%" --extra-index-url https://pypi.org/simple llama-cpp-python
if errorlevel 1 goto :fehler
rem GPU-Build gegenpruefen; laedt er nicht oder kann er nicht auslagern, hart
rem auf CPU zurueck - lieber langsam als kaputt.
if /i "%WHEELS%"=="%CPU_WHEELS%" goto :llamaok
"%VENV%\Scripts\python.exe" -c "import llama_cpp,sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)"
if not errorlevel 1 goto :llamaok
echo GPU-Build nicht nutzbar - Rueckfall auf CPU.
uv pip install --index-url "%CPU_WHEELS%" --extra-index-url https://pypi.org/simple --reinstall-package llama-cpp-python llama-cpp-python
if errorlevel 1 goto :fehler
:llamaok
uv pip install -e ".[ingest,generate,gui,native]"
if errorlevel 1 goto :fehler
rem whichllm isoliert wie per pipx — nicht ins RAG-venv, sonst kollidieren seine
rem Abhaengigkeiten mit torch/docling. rag ruft es als Subprozess ueber PATH auf.
uv tool install whichllm
type nul > "%MARKER%"
echo Einrichtung fertig.

:run
call "%VENV%\Scripts\activate.bat"
echo Oberflaeche startet als eigenstaendiges Fenster (Rueckfall: Browser auf http://127.0.0.1:8080).
rag gui --native
goto :eof

:fehler
echo.
echo FEHLER bei der Einrichtung. Bitte die Meldung oben pruefen.
pause
exit /b 1
