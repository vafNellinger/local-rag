@echo off
rem Installierter Starter fuer local-rag (Windows). Vom Startmenue-Eintrag
rem aufgerufen. Der Code liegt im Installationsverzeichnis (%~dp0); die
rem Python-Umgebung und die Modelle entstehen unter %LOCALAPPDATA%\local-rag.
setlocal
title local-rag

rem Installationsverzeichnis ohne abschliessenden Backslash.
set "APP=%~dp0"
if "%APP:~-1%"=="\" set "APP=%APP:~0,-1%"

set "DATA=%LOCALAPPDATA%\local-rag"
set "VENV=%DATA%\venv"
set "MARKER=%VENV%\.setup-done"
rem CPU-Wheels von llama-cpp-python; bei NVIDIA waehlt detect_gpu.py ein CUDA-Wheel.
set "CPU_WHEELS=https://abetlen.github.io/llama-cpp-python/whl/cpu"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if not exist "%DATA%" mkdir "%DATA%"

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
if exist "%VENV%" goto :haveenv
uv venv --python 3.12 "%VENV%"
if errorlevel 1 goto :fehler
:haveenv
rem GPU automatisch waehlen: NVIDIA -> passendes CUDA-Wheel, sonst CPU. Fuer die
rem Flotte ueberschreibbar mit LOCAL_RAG_GPU_INDEX.
set "WHEELS=%LOCAL_RAG_GPU_INDEX%"
if not "%WHEELS%"=="" goto :gotwheels
for /f "usebackq delims=" %%i in (`"%VENV%\Scripts\python.exe" "%APP%\detect_gpu.py" 2^>nul`) do set "WHEELS=%%i"
:gotwheels
if "%WHEELS%"=="" set "WHEELS=%CPU_WHEELS%"
if /i "%WHEELS%"=="%CPU_WHEELS%" (echo GPU: keine passende erkannt - laeuft auf CPU.) else (echo NVIDIA erkannt - installiere CUDA-Build %WHEELS%.)
rem llama-cpp-python vorkompiliert (kein Compiler noetig).
uv pip install --python "%VENV%" --index-url "%WHEELS%" --extra-index-url https://pypi.org/simple llama-cpp-python
if errorlevel 1 goto :fehler
rem GPU-Build gegenpruefen; laedt er nicht oder kann er nicht auslagern, hart
rem auf CPU zurueck - lieber langsam als kaputt.
if /i "%WHEELS%"=="%CPU_WHEELS%" goto :llamaok
"%VENV%\Scripts\python.exe" -c "import llama_cpp,sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)"
if not errorlevel 1 goto :llamaok
echo GPU-Build nicht nutzbar - Rueckfall auf CPU.
uv pip install --python "%VENV%" --index-url "%CPU_WHEELS%" --extra-index-url https://pypi.org/simple --reinstall-package llama-cpp-python llama-cpp-python
if errorlevel 1 goto :fehler
:llamaok
rem Editierbar aus dem Installationsordner, damit config\ gefunden wird.
uv pip install --python "%VENV%" -e "%APP%[ingest,generate,gui,native]"
if errorlevel 1 goto :fehler
uv tool install whichllm
type nul > "%MARKER%"
echo Einrichtung fertig.

:run
echo Oberflaeche startet als eigenstaendiges Fenster (Rueckfall: Browser auf http://127.0.0.1:8080).
"%VENV%\Scripts\rag.exe" gui --native
goto :eof

:fehler
echo.
echo FEHLER bei der Einrichtung. Bitte die Meldung oben pruefen.
pause
exit /b 1
