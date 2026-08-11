#!/usr/bin/env bash
# Installierter Starter für local-rag (per .deb nach /usr/bin/local-rag).
#
# Der Anwendungscode liegt schreibgeschützt unter /opt/local-rag; die
# Python-Umgebung und die Modelle entstehen im Benutzerprofil, damit kein
# Root nötig ist. Der erste Start richtet ein, jeder weitere startet nur die
# Oberfläche. uv holt bei Bedarf Python selbst, llama-cpp-python kommt als
# vorkompiliertes CPU-Wheel — kein Compiler nötig, nur beim ersten Mal Netz.
set -euo pipefail

# /opt schreibgeschützt: deshalb editierbare Installation (sonst landet das
# rag-Paket in site-packages und findet config/platforms.toml nicht mehr, das
# relativ zum Paket gesucht wird).
APP="/opt/local-rag"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/local-rag"
VENV="$DATA/venv"
MARKER="$VENV/.setup-done"
# Vorkompilierte CPU-Wheels von llama-cpp-python — spart den Compiler. Bei einer
# NVIDIA-Karte wählt detect_gpu.py (neben dem Code in /opt) ein CUDA-Wheel.
CPU_WHEELS="https://abetlen.github.io/llama-cpp-python/whl/cpu"

fehler() {
    echo >&2
    echo "FEHLER: $*" >&2
    echo "Zum Schließen Enter drücken." >&2
    read -r _ || true
    exit 1
}

# uv und seine Shims (auch whichllm) liegen in ~/.local/bin, ohne Root.
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$DATA"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv wird installiert (einmalig)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh || fehler "uv-Installation fehlgeschlagen."
fi

if [ ! -f "$MARKER" ]; then
    echo "Ersteinrichtung — das dauert beim ersten Mal einige Minuten…"
    if [ ! -d "$VENV" ]; then
        # Für das eigenständige Fenster braucht pywebview die System-WebKitGTK
        # (per .deb als Abhängigkeit installiert), die an die System-Python-
        # Version gebunden ist — also den venv aus dem System-Python bauen und
        # ihm dessen Pakete (gi) zeigen. Sonst uv-eigenes Python; dann öffnet
        # die Oberfläche im Browser.
        if command -v python3 >/dev/null 2>&1 &&
            python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)'; then
            uv venv --python "$(command -v python3)" --system-site-packages "$VENV" ||
                fehler "Umgebung ließ sich nicht anlegen."
        else
            uv venv --python 3.12 "$VENV" || fehler "Umgebung ließ sich nicht anlegen."
        fi
    fi
    # GPU automatisch wählen: NVIDIA → passendes CUDA-Wheel, sonst CPU. Für die
    # Flotte überschreibbar mit LOCAL_RAG_GPU_INDEX.
    WHEELS="${LOCAL_RAG_GPU_INDEX:-$("$VENV/bin/python" "$APP/detect_gpu.py" 2>/dev/null || echo "$CPU_WHEELS")}"
    [ -n "$WHEELS" ] || WHEELS="$CPU_WHEELS"
    case "$WHEELS" in
        */cpu) echo "GPU: keine passende erkannt — läuft auf CPU." ;;
        *) echo "NVIDIA erkannt — installiere CUDA-Build ($WHEELS)." ;;
    esac
    # Primärindex ist der Wheel-Server, die übrigen Abhängigkeiten kommen von
    # PyPI. So wird nichts kompiliert.
    uv pip install --python "$VENV" --index-url "$WHEELS" \
        --extra-index-url https://pypi.org/simple llama-cpp-python ||
        fehler "llama-cpp-python ließ sich nicht installieren."
    # GPU-Build gegenprüfen; lädt er nicht oder kann er nicht auslagern, hart
    # auf CPU zurück — lieber langsam als kaputt.
    if [ "$WHEELS" != "$CPU_WHEELS" ] &&
        ! "$VENV/bin/python" -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" 2>/dev/null; then
        echo "GPU-Build nicht nutzbar — Rückfall auf CPU."
        uv pip install --python "$VENV" --index-url "$CPU_WHEELS" \
            --extra-index-url https://pypi.org/simple \
            --reinstall-package llama-cpp-python llama-cpp-python ||
            fehler "CPU-Rückfall fehlgeschlagen."
    fi
    # Editierbar aus /opt, damit config/ neben dem Paket gefunden wird.
    uv pip install --python "$VENV" -e "${APP}[ingest,generate,gui,native]" ||
        fehler "Abhängigkeiten ließen sich nicht installieren."
    # whichllm isoliert wie per pipx — nicht in die RAG-Umgebung, sonst
    # kollidieren seine Abhängigkeiten mit torch/docling. rag ruft es als
    # Subprozess über PATH auf, die Shim liegt in ~/.local/bin.
    uv tool install whichllm ||
        echo "Hinweis: whichllm nicht installiert — Modellplanung eingeschränkt."
    touch "$MARKER"
    echo "Einrichtung fertig."
fi

exec "$VENV/bin/rag" gui --native
