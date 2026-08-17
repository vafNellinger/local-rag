"""Plattformkorrekte Speicherorte für Index, Cache und Einstellungen.

Eine einzige Stelle statt hartcodierter ``~/.cache``/``~/.config``-Pfade: die
sind der XDG-Stil und auf Windows (``%LOCALAPPDATA%``) und macOS (``~/Library``)
der falsche Ort. ``platformdirs`` liefert je Plattform das Richtige — die
Voraussetzung dafür, dieselbe App als Bundle auf allen drei Systemen
auszuliefern.

Drei Rollen mit XDG-Semantik:

- **data** (``user_data_dir``): was der Nutzer nicht verlieren will — der Index
  und hochgeladene Dokumente.
- **cache** (``user_cache_dir``): jederzeit neu herstellbar — Extraktions-Cache,
  ONNX-Modelle, Erkennungs-Caches, GUI-Log.
- **config** (``user_config_dir``): die Einstellungen.

Beim Import wird ein Altbestand vom früheren Ort einmalig migriert, damit eine
bestehende Installation ihren Index nicht verliert: er lag unter ``~/.cache``
(dem damaligen Sammelort) und gehört in den data-Ordner.
"""

from __future__ import annotations

import logging
from pathlib import Path

import platformdirs

logger = logging.getLogger(__name__)

APPNAME = "local-rag"

DATA_DIR = Path(platformdirs.user_data_dir(APPNAME))
CACHE_DIR = Path(platformdirs.user_cache_dir(APPNAME))
CONFIG_DIR = Path(platformdirs.user_config_dir(APPNAME))

# ── Abgeleitete Standardpfade ────────────────────────────────────────────────
# data: bleibt erhalten, wandert nicht in den Papierkorb eines Cache-Cleaners.
DEFAULT_INDEX_PATH = DATA_DIR / "index.db"
UPLOAD_DIR = DATA_DIR / "dokumente"
# config: die Einstellungen.
SETTINGS_PATH = CONFIG_DIR / "settings.json"
# cache: wegwerfbar, jederzeit neu herstellbar.
EXTRACT_CACHE_DIR = CACHE_DIR / "extract"
ONNX_DIR = CACHE_DIR / "onnx"
HF_LANGUAGE_CACHE = CACHE_DIR / "hf-language.json"
LOG_PATH = CACHE_DIR / "gui.log"


def _migrate_legacy(alt: Path | None = None, ziel: Path | None = None) -> None:
    """Den Index vom früheren cache-Ort in den data-Ordner holen.

    Bis zur Umstellung lag der Index unter ``~/.cache/local-rag/index.db`` (dem
    damaligen Sammelort). Er gehört zu den Daten, nicht zum Cache — also wandert
    er einmalig, samt seiner WAL-/SHM-Begleitdateien. Nur wenn am Zielort noch
    nichts liegt; Fehler dürfen den Start nicht verhindern. Die Pfade sind
    parametrisierbar, damit die Migration testbar bleibt.
    """
    alt = alt if alt is not None else Path.home() / ".cache" / "local-rag" / "index.db"
    ziel = ziel if ziel is not None else DEFAULT_INDEX_PATH
    if not alt.exists() or ziel.exists() or alt == ziel:
        return
    try:
        ziel.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            quelle = alt.parent / (alt.name + suffix)
            if quelle.exists():
                quelle.rename(ziel.parent / quelle.name)
        logger.info("Index vom Alt-Ort nach %s migriert", ziel)
    except OSError as exc:
        logger.warning("Index-Migration übersprungen (%s)", exc)


_migrate_legacy()
