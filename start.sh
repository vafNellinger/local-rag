#!/usr/bin/env bash
# Ein-Klick-Start für local-rag (Linux).
#
# Erster Start richtet eine isolierte Umgebung ein und legt einen Menü-Eintrag
# an; jeder weitere Start öffnet nur die Oberfläche im Browser. uv holt bei
# Bedarf sogar Python selbst, und llama-cpp-python kommt als vorkompiliertes
# CPU-Wheel — deshalb braucht diese Maschine weder ein vorhandenes Python noch
# einen C++-Compiler, nur beim ersten Mal eine Internetverbindung.
#
# Doppelklick im Dateimanager: „Im Terminal ausführen“ wählen (oder den beim
# ersten Lauf angelegten Menü-Eintrag „local-rag“ benutzen). Auf der Konsole:
# ./start.sh
set -euo pipefail

# Ins Projektverzeichnis wechseln — unabhängig davon, von wo gestartet wurde.
cd "$(dirname "$(readlink -f "$0")")"

VENV=".venv"
MARKER="$VENV/.setup-done"
# Vorkompilierte CPU-Wheels von llama-cpp-python — spart den Compiler. Bei einer
# NVIDIA-Karte wählt packaging/detect_gpu.py stattdessen ein CUDA-Wheel.
CPU_WHEELS="https://abetlen.github.io/llama-cpp-python/whl/cpu"

fehler() {
    echo >&2
    echo "FEHLER: $*" >&2
    echo "Zum Schließen Enter drücken." >&2
    read -r _ || true
    exit 1
}

# Legt einen Startmenü-Eintrag an, der dieses Skript in einem Terminal ausführt
# — so wird der Doppelklick-Start unter Linux verlässlich (das Terminal zeigt
# Fortschritt und Protokoll).
desktop_eintrag_anlegen() {
    local ziel="$HOME/.local/share/applications/local-rag.desktop"
    local skript
    skript="$(readlink -f "$0")"
    mkdir -p "$(dirname "$ziel")"
    cat >"$ziel" <<EOF
[Desktop Entry]
Type=Application
Name=local-rag
Comment=Lokales RAG-System
Exec=$skript
Icon=utilities-terminal
Terminal=true
Categories=Utility;
EOF
    echo "Menü-Eintrag angelegt: $ziel"
}

# ── uv sicherstellen ─────────────────────────────────────────────────────────
# uv installiert sich (und seine Shims) nach ~/.local/bin, ohne Root-Rechte.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv wird installiert (einmalig)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh || fehler "uv-Installation fehlgeschlagen."
fi

# ── Ersteinrichtung ──────────────────────────────────────────────────────────
if [ ! -f "$MARKER" ]; then
    echo "Ersteinrichtung — das dauert beim ersten Mal einige Minuten…"
    # Nur anlegen, wenn nichts da ist: ein bestehendes .venv (etwa mit einem
    # selbst kompilierten GPU-Build) soll nicht überschrieben werden. Die
    # pip-Schritte darunter sind idempotent — schon Vorhandenes bleibt, weil
    # llama-cpp-python ohne Versionsvorgabe steht und uv es dann nicht ersetzt.
    if [ ! -d "$VENV" ]; then
        # Für das eigenständige Fenster braucht pywebview die System-WebKitGTK,
        # die an die System-Python-Version gebunden ist — also den venv aus dem
        # System-Python bauen und ihm dessen Pakete (gi) zeigen, sofern es neu
        # genug ist. Sonst uv-eigenes Python; dann öffnet die Oberfläche im
        # Browser statt als Fenster.
        if command -v python3 >/dev/null 2>&1 &&
            python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)'; then
            uv venv --python "$(command -v python3)" --system-site-packages ||
                fehler "Konnte keine Python-Umgebung anlegen."
        else
            uv venv --python 3.12 || fehler "Konnte keine Python-Umgebung anlegen."
        fi
    fi
    # GPU automatisch wählen: NVIDIA → passendes CUDA-Wheel, sonst CPU. Für die
    # Flotte überschreibbar mit LOCAL_RAG_GPU_INDEX (z. B. um GPU zu erzwingen
    # oder auszuschließen).
    WHEELS="${LOCAL_RAG_GPU_INDEX:-$("$VENV/bin/python" packaging/detect_gpu.py 2>/dev/null || echo "$CPU_WHEELS")}"
    [ -n "$WHEELS" ] || WHEELS="$CPU_WHEELS"
    case "$WHEELS" in
        */cpu) echo "GPU: keine passende erkannt — läuft auf CPU." ;;
        *) echo "NVIDIA erkannt — installiere CUDA-Build ($WHEELS)." ;;
    esac
    # Primärindex ist der Wheel-Server, dort liegt llama-cpp-python vorkompiliert;
    # die übrigen Abhängigkeiten kommen von PyPI. So wird nichts kompiliert.
    uv pip install --index-url "$WHEELS" --extra-index-url https://pypi.org/simple \
        llama-cpp-python || fehler "llama-cpp-python ließ sich nicht installieren."
    # GPU-Build gegenprüfen: lädt er nicht oder kann er nicht auslagern, hart auf
    # CPU zurück — lieber langsam als kaputt.
    if [ "$WHEELS" != "$CPU_WHEELS" ] &&
        ! "$VENV/bin/python" -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" 2>/dev/null; then
        echo "GPU-Build nicht nutzbar — Rückfall auf CPU."
        uv pip install --index-url "$CPU_WHEELS" --extra-index-url https://pypi.org/simple \
            --reinstall-package llama-cpp-python llama-cpp-python ||
            fehler "CPU-Rückfall fehlgeschlagen."
    fi
    uv pip install -e ".[ingest,generate,gui,native]" ||
        fehler "Abhängigkeiten ließen sich nicht installieren."
    # whichllm isoliert wie per pipx — nicht ins RAG-venv, sonst kollidieren
    # seine Abhängigkeiten mit torch/docling. rag ruft es als Subprozess über
    # PATH auf, und die uv-Shim liegt in ~/.local/bin (oben auf PATH gesetzt).
    uv tool install whichllm ||
        echo "Hinweis: whichllm nicht installiert — Modellplanung eingeschränkt."
    desktop_eintrag_anlegen
    touch "$MARKER"
    echo "Einrichtung fertig."
fi

# ── Starten ──────────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$VENV/bin/activate"
echo "Oberfläche startet als eigenständiges Fenster (Rückfall: Browser auf http://127.0.0.1:8080)."
exec rag gui --native
