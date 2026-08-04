"""Gemeinsame Vorkehrungen für alle Tests.

Der Hauptzweck ist ein Sicherheitsnetz: **kein Test darf in die echten
Verzeichnisse des Anwenders schreiben.** Das war keine theoretische Sorge —
nach dem Einbau des Extraktions-Caches lagen sechs Testeinträge in
``~/.cache/local-rag/extract/``, weil nur die Tests des neuen Moduls den Pfad
umgelenkt hatten und alle älteren weiter den echten benutzten.

Betroffen sind vier Orte, die das Programm im Heimverzeichnis anlegt:
Extraktions-Cache, Upload-Ablage, Protokoll und Einstellungen. Sie werden hier
pauschal auf ein temporäres Verzeichnis gelegt, für jeden Test einzeln.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def keine_echten_nutzerpfade(tmp_path, monkeypatch):
    """Alle Schreibziele im Heimverzeichnis auf tmp_path umbiegen.

    ``autouse`` mit Absicht: eine Fixture, die man anfordern muss, wird
    vergessen — und genau das ist passiert. Einzelne Tests dürfen weiter
    gezielt umlenken, das gewinnt dann gegen diese Vorgabe.
    """
    from rag import extract, pipeline, store

    monkeypatch.setattr(extract, "EXTRACT_CACHE_DIR", tmp_path / "extract-cache")
    monkeypatch.setattr(pipeline, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(
        pipeline, "DEFAULT_INDEX_PATH", tmp_path / "index" / store.DEFAULT_INDEX_NAME
    )

    # Die Oberfläche nur umlenken, wenn NiceGUI installiert ist — die übrigen
    # Tests sollen ohne das Extra laufen.
    try:
        from rag import ui
    except ImportError:  # pragma: no cover
        return
    monkeypatch.setattr(ui, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(ui, "LOG_PATH", tmp_path / "gui.log")
