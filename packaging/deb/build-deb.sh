#!/usr/bin/env bash
# Baut ein installierbares .deb aus dem fertigen PyInstaller-Bundle.
#
# Das Bundle (dist/local-rag/) landet unter /opt/local-rag, dazu ein
# Start-Wrapper in /usr/bin und ein Menü-Eintrag. Die schweren GTK/WebKit-
# Bibliotheken für das native Fenster werden NICHT mitgepackt, sondern als
# Depends deklariert — apt zieht sie beim Installieren nach. Fehlt WebKit,
# fällt die Oberfläche ohnehin sauber auf den Browser zurück.
#
# Aufruf: packaging/deb/build-deb.sh <version> [dist-dir] [out-dir]
set -euo pipefail

VERSION="${1:-0.1.0}"
DIST="${2:-dist/local-rag}"
OUT="${3:-.}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$DIST" ]; then
  echo "Bundle-Verzeichnis fehlt: $DIST" >&2
  exit 1
fi

ARCH="$(dpkg --print-architecture)"
PKG="local-rag_${VERSION}_${ARCH}"
ROOT="$(mktemp -d)"
trap 'rm -rf "$ROOT"' EXIT

# --- Bundle nach /opt/local-rag ---------------------------------------------
install -d "$ROOT/opt/local-rag"
cp -a "$DIST/." "$ROOT/opt/local-rag/"

# --- Start-Wrapper in /usr/bin ----------------------------------------------
install -d "$ROOT/usr/bin"
cat > "$ROOT/usr/bin/local-rag" <<'EOF'
#!/bin/sh
exec /opt/local-rag/local-rag "$@"
EOF
chmod 0755 "$ROOT/usr/bin/local-rag"

# --- Menü-Eintrag + optionales Icon -----------------------------------------
install -d "$ROOT/usr/share/applications"
install -m 0644 "$HERE/local-rag.desktop" \
  "$ROOT/usr/share/applications/local-rag.desktop"
if [ -f "$HERE/../icons/local-rag.png" ]; then
  install -d "$ROOT/usr/share/icons/hicolor/256x256/apps"
  install -m 0644 "$HERE/../icons/local-rag.png" \
    "$ROOT/usr/share/icons/hicolor/256x256/apps/local-rag.png"
fi

# --- Steuerdatei ------------------------------------------------------------
INSTALLED_KB="$(du -sk "$ROOT" | cut -f1)"
install -d "$ROOT/DEBIAN"
cat > "$ROOT/DEBIAN/control" <<EOF
Package: local-rag
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Depends: libwebkit2gtk-4.1-0 | libwebkit2gtk-4.0-37, libgtk-3-0, libglib2.0-0, libgirepository-1.0-1
Recommends: gir1.2-webkit2-4.1 | gir1.2-webkit2-4.0
Installed-Size: ${INSTALLED_KB}
Maintainer: Niklas Nellinger <niklas.nellinger@vonaffenfels.de>
Description: Lokales RAG-System mit GPU-Beschleunigung
 Durchsucht lokale Dokumente und beantwortet Fragen dazu, vollständig
 offline. Extraktion mit docling, Einbettung und Reranking über
 bge-Modelle, Antworten über ein lokales LLM. Native Oberfläche ohne
 Browser; fehlt die System-WebKit, öffnet die Oberfläche im Browser.
EOF

# postinst/postrm: Icon- und Desktop-Datenbank auffrischen (best effort).
cat > "$ROOT/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database -q /usr/share/applications || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -q /usr/share/icons/hicolor || true
fi
EOF
chmod 0755 "$ROOT/DEBIAN/postinst"

mkdir -p "$OUT"
fakeroot dpkg-deb --build --root-owner-group "$ROOT" "$OUT/${PKG}.deb"
echo "gebaut: $OUT/${PKG}.deb"
