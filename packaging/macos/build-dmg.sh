#!/usr/bin/env bash
# Baut ein .dmg aus dem PyInstaller-.app-Bundle. Nur auf macOS (hdiutil).
#
# Ergebnis ist das übliche "nach Programme ziehen"-Fenster: die App und ein
# Verweis auf /Applications nebeneinander. Signierung/Notarisierung passieren
# davor am .app (siehe docs/standalone-app/06-installer-signing.md) — ein
# unsigniertes .dmg lässt Gatekeeper nur nach Rechtsklick > Öffnen zu.
#
# Aufruf: packaging/macos/build-dmg.sh <version> [app-path] [out-dir]
set -euo pipefail

VERSION="${1:-0.1.0}"
APP="${2:-dist/local-rag.app}"
OUT="${3:-.}"
DMG="${OUT%/}/local-rag-${VERSION}.dmg"

if [ ! -d "$APP" ]; then
  echo "App-Bundle fehlt: $APP" >&2
  exit 1
fi

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

mkdir -p "$OUT"
hdiutil create \
  -volname "local-rag" \
  -srcfolder "$STAGING" \
  -ov -format UDZO \
  "$DMG"
echo "gebaut: $DMG"
