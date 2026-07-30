"""Tests für die Extraktion, Schwerpunkt Seitenklassifizierung.

Der Kern ist die Unterscheidung Scan / leer: beide liefern keinen Text, aber
nur bei einem hilft OCR. Ein Fehlalarm hier kostet später echte Rechenzeit,
weil OCR auf leere Seiten losgelassen wird.
"""

from __future__ import annotations

import pytest

from rag.extract import (
    MIN_SCAN_IMAGE_PIXELS,
    TEXT_LAYER_MIN_CHARS,
    Document,
    ExtractionError,
    Page,
    _classify_page,
    _replacement_ratio,
    extract,
)

VIEL_TEXT = "Die Ordnungsgemäßheit der Buchführung wird bestätigt. " * 5
SCAN_BILD = MIN_SCAN_IMAGE_PIXELS * 20  # seitenfüllend
LOGO = 5_000  # Kopfzeilen-Logo


class TestClassifyPage:
    def test_brauchbarer_text_ohne_bild(self):
        status, _ = _classify_page(VIEL_TEXT, 0)
        assert status == "text"

    def test_scan_ist_wenig_text_plus_grosses_bild(self):
        status, reason = _classify_page("", SCAN_BILD)
        assert status == "scan"
        assert "Bild" in reason

    def test_leere_seite_ohne_bild_ist_nicht_scan(self):
        # Der reale Fehlalarm: eine Schlussformel-Seite mit 60 Zeichen und
        # ohne Bild wurde als Scan gewertet. OCR haette dort nichts gefunden.
        status, reason = _classify_page("Mit freundlichen Grüßen", 0)
        assert status == "sparse"
        assert "kein Bild" in reason

    def test_logo_macht_aus_leerer_seite_keinen_scan(self):
        status, _ = _classify_page("Seite 4", LOGO)
        assert status == "sparse"

    def test_schwelle_gilt_ab_erreichen(self):
        status, _ = _classify_page("x" * TEXT_LAYER_MIN_CHARS, 0)
        assert status == "text"
        status, _ = _classify_page("x" * (TEXT_LAYER_MIN_CHARS - 1), 0)
        assert status == "sparse"

    def test_kaputte_kodierung_mit_bild_geht_ins_ocr(self):
        status, _ = _classify_page("�" * 200, SCAN_BILD)
        assert status == "scan"

    def test_kaputte_kodierung_ohne_bild_ist_fehler(self):
        # OCR kann hier nicht helfen, aber der Text ist unbrauchbar — das
        # muss sichtbar bleiben statt still durchzurutschen.
        status, reason = _classify_page("�" * 200, 0)
        assert status == "error"
        assert "Kodierung" in reason

    def test_vereinzelte_ersatzzeichen_sind_ok(self):
        status, _ = _classify_page(VIEL_TEXT + "�", 0)
        assert status == "text"


class TestReplacementRatio:
    def test_sauberer_text(self):
        assert _replacement_ratio("Völlig normaler Text mit Umlauten äöüß") == 0.0

    def test_leerer_text(self):
        assert _replacement_ratio("   ") == 0.0

    def test_umlaute_zaehlen_nicht_als_defekt(self):
        # Wichtig fuer deutsche Dokumente: Umlaute sind gueltige Zeichen.
        assert _replacement_ratio("äöüßÄÖÜ") == 0.0

    def test_ersatzzeichen_werden_gezaehlt(self):
        assert _replacement_ratio("ab��") == pytest.approx(0.5)

    def test_zeilenumbrueche_sind_kein_defekt(self):
        assert _replacement_ratio("Zeile eins\nZeile zwei\tTab") == 0.0


class TestPageStatus:
    def test_needs_ocr_nur_bei_scan(self):
        assert Page(1, "", status="scan").needs_ocr
        assert not Page(1, "", status="sparse").needs_ocr
        assert not Page(1, "text", status="text").needs_ocr

    def test_bereits_ocr_verarbeitete_seite_ist_erledigt(self):
        page = Page(1, "erkannt", status="scan", ocr_applied=True)
        assert not page.needs_ocr


class TestDocument:
    def test_ocr_ratio_zaehlt_nur_scans(self):
        doc = Document(path=None, format="pdf")
        doc.pages = [
            Page(1, VIEL_TEXT, status="text"),
            Page(2, "", status="scan"),
            Page(3, "", status="sparse"),
            Page(4, "", status="scan"),
        ]
        assert doc.ocr_ratio == 0.5
        assert len(doc.sparse_pages) == 1
        assert len(doc.pages_needing_ocr) == 2

    def test_leeres_dokument_teilt_nicht_durch_null(self):
        assert Document(path=None, format="pdf").ocr_ratio == 0.0


class TestExtractDispatch:
    def test_fehlende_datei(self, tmp_path):
        with pytest.raises(ExtractionError, match="nicht gefunden"):
            extract(tmp_path / "gibtsnicht.pdf")

    def test_altes_word_format_nennt_den_weg(self, tmp_path):
        legacy = tmp_path / "alt.doc"
        legacy.write_bytes(b"egal")
        with pytest.raises(ExtractionError, match="libreoffice"):
            extract(legacy)

    def test_unbekanntes_format(self, tmp_path):
        weird = tmp_path / "datei.xyz"
        weird.write_text("inhalt")
        with pytest.raises(ExtractionError, match="nicht unterstützt"):
            extract(weird)

    def test_markdown_wird_gelesen(self, tmp_path):
        md = tmp_path / "notiz.md"
        md.write_text("# Überschrift\n\nInhalt mit Umlauten: äöü", encoding="utf-8")
        doc = extract(md)
        assert doc.format == "markdown"
        assert "Überschrift" in doc.text

    def test_latin1_fallback(self, tmp_path):
        txt = tmp_path / "alt.txt"
        txt.write_bytes("Grüße aus München".encode("latin-1"))
        doc = extract(txt)
        assert "Grüße" in doc.text
        assert any("latin-1" in w for w in doc.warnings)
