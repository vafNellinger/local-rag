"""Tests für den Datei-Upload der Oberfläche.

Der Upload war kaputt und niemand hat es gemerkt: der Handler griff auf
``event.name`` und ``event.content`` zu — eine ältere NiceGUI-Schnittstelle.
In Version 3 steckt die Datei in ``event.file`` und ``save()`` ist eine
Coroutine. Der Fehler landete als ``AttributeError`` im Serverprozess, wo ihn
ohne eingerichtetes Protokoll niemand sah.

Diese Tests halten beides fest: die erwartete Schnittstelle und das Verhalten
des Handlers. Sie brauchen keinen Browser — ``on_upload`` bekommt ein Objekt
mit ``file``, und genau das wird hier gestellt.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest

from rag import ui as rag_ui


class FakeUpload:
    """Ein ``FileUpload`` mit dem Verhalten, auf das der Handler baut."""

    def __init__(self, name: str, content: bytes = b"Inhalt mit Umlaut: \xc3\xa4") -> None:
        self.name = name
        self.content_type = "application/octet-stream"
        self._content = content
        self.saved_to: Path | None = None

    async def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self._content)
        self.saved_to = target

    def size(self) -> int:
        return len(self._content)


class FakeEvent:
    def __init__(self, upload: FakeUpload) -> None:
        self.file = upload


class FakeLabel:
    """Steht für das Statuslabel; NiceGUI wird hier nicht gebraucht."""

    def __init__(self) -> None:
        self.text = ""


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    """UPLOAD_DIR umlenken, damit kein echtes Verzeichnis beschrieben wird."""
    ziel = tmp_path / "dokumente"
    monkeypatch.setattr(rag_ui, "UPLOAD_DIR", ziel)
    return ziel


def handle(event, label) -> None:
    asyncio.run(rag_ui._handle_upload(event, label))


class TestNiceGuiInterface:
    """Die Annahmen über die Fremdschnittstelle, an denen der Bug hing."""

    def test_event_traegt_die_datei_unter_file(self):
        from nicegui.events import UploadEventArguments

        felder = {f.name for f in dataclasses.fields(UploadEventArguments)}
        assert "file" in felder
        # Der alte Zugriffsweg darf nicht zurückkehren.
        assert "name" not in felder
        assert "content" not in felder

    def test_fileupload_hat_name_und_asynchrones_save(self):
        from nicegui.elements.upload_files import FileUpload

        assert "name" in FileUpload.__annotations__
        assert asyncio.iscoroutinefunction(FileUpload.save)

    def test_handler_ist_eine_coroutine(self):
        # Ohne das könnte er das asynchrone save() nicht abwarten.
        assert asyncio.iscoroutinefunction(rag_ui._handle_upload)


class TestErlaubteFormate:
    @pytest.mark.parametrize("name", ["akte.pdf", "vertrag.docx", "notiz.md", "text.txt"])
    def test_unterstuetzte_formate_landen_im_verzeichnis(self, name, upload_dir):
        upload = FakeUpload(name)
        label = FakeLabel()
        handle(FakeEvent(upload), label)
        assert (upload_dir / name).exists()
        assert name in label.text
        assert "gespeichert" in label.text

    def test_groesse_wird_gemeldet(self, upload_dir):
        label = FakeLabel()
        handle(FakeEvent(FakeUpload("a.md", b"x" * 4096)), label)
        assert "KB" in label.text

    def test_hinweis_nennt_den_naechsten_schritt(self, upload_dir):
        # Ohne den Hinweis bleibt unklar, dass Hochladen noch nicht Indizieren ist.
        label = FakeLabel()
        handle(FakeEvent(FakeUpload("a.md")), label)
        assert "aufnehmen" in label.text.lower()


class TestAbgelehnteFormate:
    @pytest.mark.parametrize("name", ["bild.png", "tabelle.xlsx", "programm.exe", "ohneendung"])
    def test_fremde_formate_werden_nicht_gespeichert(self, name, upload_dir):
        # Das accept-Attribut im Browser ist nur ein Vorschlag — abgelehnt
        # werden muss serverseitig.
        label = FakeLabel()
        handle(FakeEvent(FakeUpload(name)), label)
        assert not (upload_dir / name).exists()
        assert "nicht unterstützt" in label.text

    def test_meldung_nennt_die_moeglichen_formate(self, upload_dir):
        label = FakeLabel()
        handle(FakeEvent(FakeUpload("bild.png")), label)
        assert ".pdf" in label.text


class TestPfadsicherheit:
    def test_pfadanteile_im_namen_werden_entfernt(self, upload_dir):
        # Ein Name wie "../../.bashrc" darf nicht aus dem Zielverzeichnis
        # herausführen.
        label = FakeLabel()
        handle(FakeEvent(FakeUpload("../../entkommen.md")), label)
        assert (upload_dir / "entkommen.md").exists()
        assert not (upload_dir.parent.parent / "entkommen.md").exists()

    def test_absoluter_pfad_wird_entschaerft(self, upload_dir):
        label = FakeLabel()
        handle(FakeEvent(FakeUpload("/etc/passwd.md")), label)
        assert (upload_dir / "passwd.md").exists()

    def test_leerer_name_wird_abgelehnt(self, upload_dir):
        label = FakeLabel()
        handle(FakeEvent(FakeUpload("")), label)
        assert "ohne Namen" in label.text
        assert not upload_dir.exists() or not list(upload_dir.iterdir())


class TestFehlerbehandlung:
    def test_schreibfehler_wird_gemeldet_statt_zu_werfen(self, upload_dir, monkeypatch):
        class Sperrig(FakeUpload):
            async def save(self, path):
                raise OSError("Kein Platz auf dem Gerät")

        label = FakeLabel()
        # Darf nicht durchschlagen: ein Upload-Fehler beendet die Oberfläche nicht.
        handle(FakeEvent(Sperrig("a.md")), label)
        assert "Fehler bei a.md" in label.text
        assert "Kein Platz" in label.text


class TestLogging:
    def test_setup_legt_die_datei_an(self, tmp_path):
        ziel = tmp_path / "unterordner" / "gui.log"
        ergebnis = rag_ui.setup_logging(log_path=ziel)
        assert ergebnis == ziel
        assert ziel.parent.exists()

    def test_meldungen_landen_in_der_datei(self, tmp_path):
        ziel = tmp_path / "gui.log"
        rag_ui.setup_logging(log_path=ziel)
        import logging

        logging.getLogger("rag.test").warning("Testmeldung 4711")
        for handler in logging.getLogger("rag").handlers:
            handler.flush()
        assert "Testmeldung 4711" in ziel.read_text(encoding="utf-8")

    def test_wiederholter_aufruf_verdoppelt_die_handler_nicht(self, tmp_path):
        import logging

        ziel = tmp_path / "gui.log"
        rag_ui.setup_logging(log_path=ziel)
        erste = len(logging.getLogger("rag").handlers)
        rag_ui.setup_logging(log_path=ziel)
        assert len(logging.getLogger("rag").handlers) == erste

    def test_upload_fehler_steht_im_protokoll(self, tmp_path, upload_dir):
        # Der eigentliche Punkt der ganzen Übung: ein abgelehnter Upload muss
        # nachlesbar sein, ohne den Serverprozess zu beobachten.
        import logging

        ziel = tmp_path / "gui.log"
        rag_ui.setup_logging(log_path=ziel)
        handle(FakeEvent(FakeUpload("bild.png")), FakeLabel())
        for handler in logging.getLogger("rag").handlers:
            handler.flush()
        assert "bild.png" in ziel.read_text(encoding="utf-8")
