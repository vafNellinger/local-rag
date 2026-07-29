"""Tests für die Auswahllogik.

Bewusst ohne whichllm-Aufruf: getestet wird die Entscheidungslogik, die wir
selbst verantworten. Die whichllm-Anbindung wird über einen Beispiel-Kandidaten
abgedeckt, wie er real aus ``--json`` kommt.
"""

from __future__ import annotations

import pytest

from rag.detect import classify
from rag.resolve import (
    ResolutionError,
    _context_truncation_warning,
    _quant_acceptable,
    parse_context_length,
)


class TestClassify:
    def test_keine_gpu(self):
        assert classify(0, shared_memory=False) == "cpu_only"

    def test_igpu_wird_ueber_shared_memory_erkannt(self):
        # Der Kern der Klassifizierung: geteilter Speicher schlägt die reine
        # VRAM-Zahl. Eine APU mit nominell viel VRAM konkurriert trotzdem mit
        # dem System-RAM.
        big_but_shared = 24 * 1024**3
        assert classify(big_but_shared, shared_memory=True) == "igpu_shared"

    def test_kleine_dedizierte_gpu(self):
        assert classify(8 * 1024**3, shared_memory=False) == "igpu_shared"
        assert classify(16 * 1024**3, shared_memory=False) == "dgpu_small"

    def test_grosse_dedizierte_gpu(self):
        assert classify(24 * 1024**3, shared_memory=False) == "dgpu_large"


class TestParseContextLength:
    @pytest.mark.parametrize(
        "value,expected",
        [("32k", 32768), ("8k", 8192), ("4096", 4096), (65536, 65536), ("1.5k", 1536)],
    )
    def test_varianten(self, value, expected):
        assert parse_context_length(value) == expected


class TestQuantAcceptable:
    def test_besser_als_grenze_wird_akzeptiert(self):
        assert _quant_acceptable("Q5_K_M", "Q4_K_S")
        assert _quant_acceptable("Q8_0", "Q4_K_S")

    def test_grenze_selbst_wird_akzeptiert(self):
        assert _quant_acceptable("Q4_K_S", "Q4_K_S")

    def test_schlechter_als_grenze_wird_abgelehnt(self):
        # Der Fall, der die Auswahl auf dieser Hardware real verändert:
        # whichllm rankt Qwen3-8B Q3_K_M nach oben, für RAG ist es zu grob.
        assert not _quant_acceptable("Q3_K_M", "Q4_K_S")
        assert not _quant_acceptable("Q2_K", "Q4_K_S")

    def test_unbekannte_quantisierung_wird_abgelehnt(self):
        assert not _quant_acceptable("SOMETHING_NEW", "Q4_K_S")

    def test_fehlende_quantisierung_wird_abgelehnt(self):
        assert not _quant_acceptable(None, "Q4_K_S")

    def test_unbekannte_grenze_ist_konfigurationsfehler(self):
        with pytest.raises(ResolutionError, match="unbekannt"):
            _quant_acceptable("Q5_K_M", "Q9_ULTRA")


class TestContextTruncationWarning:
    def test_erkennt_whichllm_warnung(self):
        # Wortlaut aus engine/compatibility.py:231.
        candidate = {
            "warnings": [
                "Model max context 8192 < requested 32768; "
                "runtime will truncate or reject"
            ]
        }
        assert _context_truncation_warning(candidate) is not None

    def test_ignoriert_andere_warnungen(self):
        candidate = {
            "warnings": [
                "Large context (32768) increases VRAM usage significantly",
                "Will run on CPU only (much slower)",
            ]
        }
        assert _context_truncation_warning(candidate) is None

    def test_ohne_warnungen(self):
        assert _context_truncation_warning({}) is None
