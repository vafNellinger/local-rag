"""Tests für den Index.

Schwerpunkte sind die beiden Stellen, an denen ein Fehler still bleibt: ein
Modellwechsel unter einem bestehenden Index und verwaiste Vektoren nach dem
Löschen eines Dokuments. Beide liefern weiterhin Suchergebnisse, nur falsche.

Die Vektoren sind hier handgeschrieben — der Index braucht kein Modell, und
diese Tests sollen ohne Download laufen.
"""

from __future__ import annotations

import math

import pytest

from rag.chunk import Chunk
from rag.store import IndexStore, SearchHit, StoreError, file_sha256

DIMENSIONS = 4


def unit(*values: float) -> list[float]:
    """Normalisierter Vektor — der Index rechnet mit Cosine."""
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


X = unit(1, 0, 0, 0)
Y = unit(0, 1, 0, 0)
XY = unit(1, 1, 0, 0)


@pytest.fixture
def store(tmp_path):
    with IndexStore(
        tmp_path / "index.db", embedder="test/modell", dimensions=DIMENSIONS
    ) as opened:
        yield opened


def chunks(*texts: str, heading: tuple[str, ...] = ()) -> list[Chunk]:
    return [
        Chunk(ordinal=i, text=text, heading_path=heading, token_count=len(text.split()))
        for i, text in enumerate(texts)
    ]


class TestSchema:
    def test_leerer_index_hat_keine_dokumente(self, store):
        assert store.stats()["documents"] == 0
        assert store.documents() == []

    def test_meta_wird_beim_anlegen_gesetzt(self, store):
        stats = store.stats()
        assert stats["embedder"] == "test/modell"
        assert stats["dimensions"] == DIMENSIONS

    def test_wiederoeffnen_mit_gleichem_modell(self, tmp_path):
        pfad = tmp_path / "index.db"
        with IndexStore(pfad, embedder="a/b", dimensions=DIMENSIONS) as first:
            first.replace_document(
                tmp_path / "d.md",
                sha256="hash",
                format="markdown",
                chunks=chunks("Inhalt"),
                embeddings=[X],
            )
        with IndexStore(pfad, embedder="a/b", dimensions=DIMENSIONS) as second:
            assert second.stats()["documents"] == 1

    def test_datei_wird_angelegt(self, tmp_path):
        pfad = tmp_path / "tief" / "index.db"
        with IndexStore(pfad, embedder="a/b", dimensions=DIMENSIONS):
            pass
        assert pfad.exists()


class TestCompatibility:
    def test_modellwechsel_wird_abgelehnt(self, tmp_path):
        # Der stille Fehler: Vektoren zweier Modelle im selben Raum liefern
        # weiter Treffer, nur falsche.
        pfad = tmp_path / "index.db"
        with IndexStore(pfad, embedder="modell/a", dimensions=DIMENSIONS):
            pass
        with pytest.raises(StoreError, match="nicht vergleichbar"):
            IndexStore(pfad, embedder="modell/b", dimensions=DIMENSIONS).open()

    def test_dimensionswechsel_wird_abgelehnt(self, tmp_path):
        pfad = tmp_path / "index.db"
        with IndexStore(pfad, embedder="modell/a", dimensions=DIMENSIONS):
            pass
        with pytest.raises(StoreError, match="Dimensionen"):
            IndexStore(pfad, embedder="modell/a", dimensions=8).open()

    def test_fehlermeldung_nennt_den_pfad(self, tmp_path):
        pfad = tmp_path / "index.db"
        with IndexStore(pfad, embedder="modell/a", dimensions=DIMENSIONS):
            pass
        with pytest.raises(StoreError, match=str(pfad)):
            IndexStore(pfad, embedder="modell/b", dimensions=DIMENSIONS).open()

    def test_geschlossener_index_wirft(self, tmp_path):
        store = IndexStore(tmp_path / "i.db", embedder="a/b", dimensions=DIMENSIONS)
        with pytest.raises(StoreError, match="nicht geöffnet"):
            store.stats()


class TestWriting:
    def test_dokument_schreiben(self, store, tmp_path):
        geschrieben = store.replace_document(
            tmp_path / "akte.md",
            sha256="abc",
            format="markdown",
            chunks=chunks("Erster Chunk", "Zweiter Chunk"),
            embeddings=[X, Y],
            page_count=2,
            char_count=100,
        )
        assert geschrieben == 2
        stats = store.stats()
        assert stats["documents"] == 1
        assert stats["chunks"] == 2
        assert stats["vectors"] == 2

    def test_chunkzahl_und_vektorzahl_bleiben_gleich(self, store, tmp_path):
        store.replace_document(
            tmp_path / "a.md",
            sha256="1",
            format="markdown",
            chunks=chunks("Eins", "Zwei", "Drei"),
            embeddings=[X, Y, XY],
        )
        stats = store.stats()
        assert stats["chunks"] == stats["vectors"]

    def test_ungleiche_laengen_werden_abgelehnt(self, store, tmp_path):
        with pytest.raises(StoreError, match="auseinanderlaufen"):
            store.replace_document(
                tmp_path / "a.md",
                sha256="1",
                format="markdown",
                chunks=chunks("Eins", "Zwei"),
                embeddings=[X],
            )

    def test_falsche_vektordimension_wird_abgelehnt(self, store, tmp_path):
        with pytest.raises(StoreError, match="Dimensionen"):
            store.replace_document(
                tmp_path / "a.md",
                sha256="1",
                format="markdown",
                chunks=chunks("Eins"),
                embeddings=[[1.0, 0.0]],
            )

    def test_heading_path_ueberlebt_die_runde(self, store, tmp_path):
        store.replace_document(
            tmp_path / "a.md",
            sha256="1",
            format="markdown",
            chunks=chunks("Die Frist beträgt 14 Tage.", heading=("Kündigung", "Fristen")),
            embeddings=[X],
        )
        treffer = store.search(X, limit=1)[0]
        assert treffer.heading_path == ("Kündigung", "Fristen")
        assert treffer.heading == "Kündigung > Fristen"

    def test_umlaute_im_heading_path(self, store, tmp_path):
        store.replace_document(
            tmp_path / "a.md",
            sha256="1",
            format="markdown",
            chunks=chunks("Text", heading=("Größe", "Prüfung")),
            embeddings=[X],
        )
        assert store.search(X, limit=1)[0].heading_path == ("Größe", "Prüfung")


class TestReplacement:
    def test_erneutes_schreiben_ersetzt_statt_zu_ergaenzen(self, store, tmp_path):
        pfad = tmp_path / "a.md"
        store.replace_document(
            pfad,
            sha256="1",
            format="markdown",
            chunks=chunks("Alt eins", "Alt zwei"),
            embeddings=[X, Y],
        )
        store.replace_document(
            pfad,
            sha256="2",
            format="markdown",
            chunks=chunks("Neu"),
            embeddings=[XY],
        )
        stats = store.stats()
        assert stats["documents"] == 1
        assert stats["chunks"] == 1
        # Der eigentliche Punkt: keine verwaisten Vektoren. Sonst zeigt die
        # Suche auf Chunk-IDs, die es nicht mehr gibt.
        assert stats["vectors"] == 1
        assert "Alt" not in store.search(XY, limit=5)[0].text

    def test_keine_verwaisten_vektoren_nach_ersetzen(self, store, tmp_path):
        pfad = tmp_path / "a.md"
        for runde in range(3):
            store.replace_document(
                pfad,
                sha256=str(runde),
                format="markdown",
                chunks=chunks(f"Runde {runde} A", f"Runde {runde} B"),
                embeddings=[X, Y],
            )
        stats = store.stats()
        assert stats["chunks"] == 2 and stats["vectors"] == 2
        # Jeder Treffer muss einen echten Chunk haben — ein verwaister Vektor
        # wuerde beim JOIN wegfallen und die Trefferzahl senken.
        assert len(store.search(X, limit=10)) == 2


class TestIdempotence:
    def test_unveraenderte_datei_wird_erkannt(self, store, tmp_path):
        pfad = tmp_path / "a.md"
        store.replace_document(
            pfad, sha256="abc", format="markdown", chunks=chunks("X"), embeddings=[X]
        )
        assert store.is_current(pfad, "abc")
        assert not store.is_current(pfad, "anders")

    def test_unbekannte_datei_ist_nicht_aktuell(self, store, tmp_path):
        assert not store.is_current(tmp_path / "gibtsnicht.md", "abc")

    def test_relativer_und_absoluter_pfad_sind_dasselbe_dokument(self, store, tmp_path):
        # Ohne Normalisierung landet dieselbe Datei zweimal im Index.
        datei = tmp_path / "a.md"
        datei.write_text("Inhalt")
        store.replace_document(
            datei, sha256="1", format="markdown", chunks=chunks("X"), embeddings=[X]
        )
        store.replace_document(
            tmp_path / "." / "a.md",
            sha256="1",
            format="markdown",
            chunks=chunks("X"),
            embeddings=[X],
        )
        assert store.stats()["documents"] == 1

    def test_document_record_liefert_hash(self, store, tmp_path):
        pfad = tmp_path / "a.md"
        store.replace_document(
            pfad,
            sha256="deadbeef",
            format="markdown",
            chunks=chunks("X", "Y"),
            embeddings=[X, Y],
        )
        record = store.document_record(pfad)
        assert record.sha256 == "deadbeef"
        assert record.chunk_count == 2


class TestSearch:
    def test_naechster_treffer_zuerst(self, store, tmp_path):
        store.replace_document(
            tmp_path / "a.md",
            sha256="1",
            format="markdown",
            chunks=chunks("Passend", "Unpassend"),
            embeddings=[X, Y],
        )
        treffer = store.search(X, limit=2)
        assert treffer[0].text == "Passend"
        assert treffer[0].distance < treffer[1].distance

    def test_limit_wird_beachtet(self, store, tmp_path):
        store.replace_document(
            tmp_path / "a.md",
            sha256="1",
            format="markdown",
            chunks=chunks("A", "B", "C"),
            embeddings=[X, Y, XY],
        )
        assert len(store.search(X, limit=2)) == 2

    def test_similarity_ist_gegenstueck_zur_distanz(self, store, tmp_path):
        store.replace_document(
            tmp_path / "a.md",
            sha256="1",
            format="markdown",
            chunks=chunks("Identisch"),
            embeddings=[X],
        )
        treffer = store.search(X, limit=1)[0]
        assert treffer.similarity == pytest.approx(1.0, abs=1e-5)

    def test_orthogonaler_vektor_hat_distanz_eins(self, store, tmp_path):
        store.replace_document(
            tmp_path / "a.md",
            sha256="1",
            format="markdown",
            chunks=chunks("Anderes Thema"),
            embeddings=[Y],
        )
        assert store.search(X, limit=1)[0].distance == pytest.approx(1.0, abs=1e-5)

    def test_suche_ueber_mehrere_dokumente(self, store, tmp_path):
        store.replace_document(
            tmp_path / "a.md",
            sha256="1",
            format="markdown",
            chunks=chunks("Aus A"),
            embeddings=[X],
        )
        store.replace_document(
            tmp_path / "b.md",
            sha256="2",
            format="markdown",
            chunks=chunks("Aus B"),
            embeddings=[XY],
        )
        pfade = {h.document_path for h in store.search(X, limit=5)}
        assert len(pfade) == 2

    def test_falsche_anfragedimension_wird_abgelehnt(self, store):
        with pytest.raises(StoreError, match="Anfragevektor"):
            store.search([1.0, 0.0])

    def test_leerer_index_liefert_keine_treffer(self, store):
        assert store.search(X, limit=5) == []


class TestForgetMissing:
    def test_verschwundene_datei_wird_entfernt(self, store, tmp_path):
        weg = tmp_path / "geloescht.md"
        store.replace_document(
            weg, sha256="1", format="markdown", chunks=chunks("X"), embeddings=[X]
        )
        entfernt = store.forget_missing([])
        assert len(entfernt) == 1
        assert store.stats()["documents"] == 0
        # Auch hier: keine verwaisten Vektoren zurücklassen.
        assert store.stats()["vectors"] == 0

    def test_vorhandene_datei_bleibt(self, store, tmp_path):
        da = tmp_path / "da.md"
        da.write_text("Inhalt")
        store.replace_document(
            da, sha256="1", format="markdown", chunks=chunks("X"), embeddings=[X]
        )
        assert store.forget_missing([]) == []
        assert store.stats()["documents"] == 1

    def test_uebergebene_pfade_bleiben_auch_ohne_datei(self, store, tmp_path):
        # Beim Ingest kann eine Datei gerade ersetzt werden — was der Aufrufer
        # als vorhanden meldet, wird nicht angetastet.
        pfad = tmp_path / "temporaer.md"
        store.replace_document(
            pfad, sha256="1", format="markdown", chunks=chunks("X"), embeddings=[X]
        )
        assert store.forget_missing([pfad]) == []


class TestCitation:
    def test_mit_ueberschrift(self):
        hit = SearchHit(
            chunk_id=1,
            document_path="/pfad/zur/Akte 2026.pdf",
            ordinal=0,
            text="Text",
            heading_path=("Kündigung", "Fristen"),
            kind="prosa",
            distance=0.1,
        )
        assert hit.citation == "Akte 2026.pdf — Kündigung > Fristen"

    def test_ohne_ueberschrift_nur_dateiname(self):
        hit = SearchHit(
            chunk_id=1,
            document_path="/pfad/notiz.md",
            ordinal=0,
            text="Text",
            heading_path=(),
            kind="prosa",
            distance=0.1,
        )
        assert hit.citation == "notiz.md"


class TestFileHash:
    def test_gleicher_inhalt_gleicher_hash(self, tmp_path):
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text("Identischer Inhalt")
        b.write_text("Identischer Inhalt")
        assert file_sha256(a) == file_sha256(b)

    def test_andere_inhalte_andere_hashes(self, tmp_path):
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_text("Eins")
        b.write_text("Zwei")
        assert file_sha256(a) != file_sha256(b)

    def test_grosse_datei_wird_stueckweise_gelesen(self, tmp_path):
        gross = tmp_path / "gross.bin"
        gross.write_bytes(b"x" * (3 * 1024 * 1024))
        assert len(file_sha256(gross)) == 64
