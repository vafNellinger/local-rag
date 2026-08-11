"""Tests für das Offline-Laden und den gedämpften HF-Hinweis.

Ohne Netz und ohne Modell: geprüft wird die Fallback-Logik (offline zuerst,
sonst online) und dass genau der „unauthenticated"-Hinweis gefiltert wird —
und nichts sonst.
"""

from __future__ import annotations

import logging

from rag.hfload import _DropUnauthenticatedHint, load_offline_first


class TestOfflineFirst:
    def test_offline_erfolg_laedt_nicht_nach(self):
        aufrufe = []

        def build(offline):
            aufrufe.append(offline)
            return "modell"

        assert load_offline_first(build, was="X") == "modell"
        # Nur der Offline-Versuch, kein Netz-Fallback.
        assert aufrufe == [True]

    def test_offline_fehlschlag_faellt_auf_netz(self):
        aufrufe = []

        def build(offline):
            aufrufe.append(offline)
            if offline:
                raise OSError("nicht im Cache")
            return "geladen"

        assert load_offline_first(build, was="X") == "geladen"
        # Erst offline (True), dann mit Netz (False).
        assert aufrufe == [True, False]

    def test_netzfehler_schlaegt_durch(self):
        import pytest

        def build(offline):
            raise RuntimeError("offline" if offline else "auch online kaputt")

        # Die Online-Ausnahme ist die aussagekräftige und muss durchschlagen.
        with pytest.raises(RuntimeError, match="auch online kaputt"):
            load_offline_first(build, was="X")

    def test_offline_versuch_erzwingt_offline_schalter(self, monkeypatch):
        import os

        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        gesehen: dict[bool, dict[str, str | None]] = {}

        def build(offline):
            gesehen[offline] = {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            }
            if offline:
                raise OSError("nicht im Cache")
            return "geladen"

        load_offline_first(build, was="X")
        # Der Offline-Versuch sah beide Schalter gesetzt …
        assert gesehen[True] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        # … der Netz-Fallback keinen, sonst käme kein Download zustande.
        assert gesehen[False] == {"HF_HUB_OFFLINE": None, "TRANSFORMERS_OFFLINE": None}

    def test_schalter_werden_nach_dem_laden_zurueckgesetzt(self, monkeypatch):
        import os

        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        load_offline_first(lambda offline: "m", was="X")
        assert "HF_HUB_OFFLINE" not in os.environ

    def test_vorhandener_schalter_bleibt_unveraendert(self, monkeypatch):
        import os

        # Ein bewusst gesetztes Offline-Flag darf nicht verloren gehen.
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        load_offline_first(lambda offline: "m", was="X")
        assert os.environ["HF_HUB_OFFLINE"] == "0"


class TestHintFilter:
    def _record(self, msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            "huggingface_hub.utils._http", logging.WARNING, __file__, 0, msg, (), None
        )

    def test_hinweis_wird_gefiltert(self):
        f = _DropUnauthenticatedHint()
        record = self._record(
            "Warning: You are sending unauthenticated requests to the HF Hub. "
            "Please set a HF_TOKEN to enable higher rate limits and faster downloads."
        )
        # filter() False heißt: Satz fällt weg.
        assert f.filter(record) is False

    def test_andere_meldungen_bleiben(self):
        f = _DropUnauthenticatedHint()
        assert f.filter(self._record("429 Too Many Requests")) is True
        assert f.filter(self._record("Repository not found")) is True

    def test_filter_ist_am_hf_logger_installiert(self):
        # Der Import von rag.hfload installiert ihn — hier nur bestätigen.
        import rag.hfload  # noqa: F401

        hf_logger = logging.getLogger("huggingface_hub.utils._http")
        assert any(
            isinstance(f, _DropUnauthenticatedHint) for f in hf_logger.filters
        )
