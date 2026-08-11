#!/usr/bin/env bash
# Baut ein installierbares .deb aus dem Projekt.
#
# Läuft auf jeder Debian-/Ubuntu-Maschine mit dpkg-deb; braucht kein Root.
# Ergebnis: dist/local-rag_<version>_all.deb
#
# Installieren:   sudo apt install ./dist/local-rag_<version>_all.deb
# Starten:        Menü-Eintrag „local-rag“ oder Befehl  local-rag
# Entfernen:      sudo apt remove local-rag
set -euo pipefail

# Immer aus der Projektwurzel arbeiten, egal von wo aufgerufen.
cd "$(dirname "$(readlink -f "$0")")/.."

PKG="local-rag"
ARCH="all"
VERSION="$(sed -nE 's/^version *= *"([^"]+)".*/\1/p' pyproject.toml | head -1)"
[ -n "$VERSION" ] || { echo "Version nicht aus pyproject.toml lesbar" >&2; exit 1; }

BUILD="$(mktemp -d)"
ROOT="$BUILD/$PKG"
trap 'rm -rf "$BUILD"' EXIT

# ── Dateibaum ────────────────────────────────────────────────────────────────
install -d "$ROOT/opt/local-rag" "$ROOT/usr/bin" \
    "$ROOT/usr/share/applications" "$ROOT/DEBIAN"

# Anwendungscode nach /opt (nur, was zur Laufzeit gebraucht wird — keine Tests,
# Testdaten, Tools, kein venv oder Git).
cp -r rag config pyproject.toml README.md "$ROOT/opt/local-rag/"
# GPU-Erkennung neben den Code: der Launcher ruft sie vor der Installation auf.
install -m0644 packaging/detect_gpu.py "$ROOT/opt/local-rag/detect_gpu.py"
# Übersetzten Python-Bytecode nicht mitliefern — er wird pro Umgebung neu
# erzeugt und würde nur das Paket aufblähen.
find "$ROOT/opt/local-rag" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$ROOT/opt/local-rag" -type f -name '*.pyc' -delete

install -m0755 packaging/local-rag-launcher.sh "$ROOT/usr/bin/local-rag"
install -m0644 packaging/local-rag.desktop \
    "$ROOT/usr/share/applications/local-rag.desktop"

# ── Steuerdatei ──────────────────────────────────────────────────────────────
SIZE="$(du -sk "$ROOT/opt" "$ROOT/usr" | awk '{s+=$1} END {print s}')"
cat >"$ROOT/DEBIAN/control" <<EOF
Package: $PKG
Version: $VERSION
Section: utils
Priority: optional
Architecture: $ARCH
Depends: curl, ca-certificates, python3 (>= 3.11), python3-gi, gir1.2-webkit2-4.1 | gir1.2-webkit2-4.0, libwebkit2gtk-4.1-0 | libwebkit2gtk-4.0-37
Installed-Size: $SIZE
Maintainer: Niklas Nellinger <niklas.nellinger@vonaffenfels.de>
Description: Lokales RAG-System mit plattformabhängiger Modellauswahl
 Startet eine lokale Oberfläche zum Fragen an eigene Dokumente. Beim ersten
 Start richtet das Werkzeug eine isolierte Python-Umgebung im Benutzerprofil
 ein (uv, vorkompiliertes llama-cpp-python-CPU-Wheel) — dafür ist einmalig
 eine Internetverbindung nötig. Modelle und Index liegen im Benutzerprofil,
 nicht im Paket.
EOF

# ── Bauen ────────────────────────────────────────────────────────────────────
# --root-owner-group: Dateien gehören root:root ohne fakeroot.
mkdir -p dist
DEB="dist/${PKG}_${VERSION}_${ARCH}.deb"
dpkg-deb --build --root-owner-group "$ROOT" "$DEB"

echo
echo "Fertig: $DEB"
echo "Installieren: sudo apt install ./$DEB"
