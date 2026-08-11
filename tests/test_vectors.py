"""Tests für die austauschbaren Vektor-Backends.

Der Zweck ist Gleichheit, nicht Abdeckung: derselbe Ablauf läuft gegen jedes
Backend und muss überall dasselbe Ergebnis liefern. Was hier schiefgehen kann,
geht still schief — ein Backend mit umgekehrter Distanzkonvention dreht die
Trefferreihenfolge, ohne dass etwas fehlschlägt, und ein Backend, das beim
Löschen nichts tut, liefert Treffer auf gelöschte Dokumente.

Die Vektoren sind handgeschrieben und vierdimensional; keiner dieser Tests
lädt ein Modell.
"""

from __future__ import annotations

import importlib.util
import math

import pytest

from rag.chunk import Chunk
from rag.store import IndexStore, StoreError
from rag.vectors import BACKENDS, DEFAULT_BACKEND, sidecar_for

DIMENSIONS = 4


def unit(*values: float) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


X = unit(1, 0, 0, 0)
Y = unit(0, 1, 0, 0)
XY = unit(1, 1, 0, 0)


def chunks(*texts: str) -> list[Chunk]:
    return [
        Chunk(ordinal=i, text=text, heading_path=(), token_count=len(text.split()))
        for i, text in enumerate(texts)
    ]


_MODUL = {
    DEFAULT_BACKEND: "sqlite_vec",
    "lancedb": "lancedb",
    "chromadb": "chromadb",
    "qdrant": "qdrant_client",
}


def verfuegbar(backend: str) -> bool:
    """Ist die Bibliothek dieses Backends installiert?

    Die Backends sind optionale Extras. Ein fehlendes soll den Testlauf
    überspringen und nicht rot machen — sonst müsste jeder Beitragende drei
    Vektordatenbanken installieren.
    """
    return importlib.util.find_spec(_MODUL[backend]) is not None


@pytest.fixture(params=BACKENDS)
def backend(request):
    if not verfuegbar(request.param):
        pytest.skip(f"{request.param} ist nicht installiert")
    return request.param


@pytest.fixture
def store(tmp_path, backend):
    with IndexStore(
        tmp_path / "index.db",
        embedder="test/modell",
        dimensions=DIMENSIONS,
        vector_backend=backend,
    ) as opened:
        yield opened


class TestGleichesVerhalten:
    """Derselbe Ablauf, jedes Backend, dasselbe Ergebnis."""

    def test_schreiben_und_finden(self, store):
        store.replace_document(
            "/tmp/a.txt",
            sha256="a" * 64,
            format="txt",
            chunks=chunks("erster", "zweiter"),
            embeddings=[X, Y],
        )
        treffer = store.search(X, limit=2)
        assert [hit.text for hit in treffer] == ["erster", "zweiter"]

    def test_distanz_ist_klein_bei_naehe(self, store):
        """Die Konvention, an der alles hängt: klein heißt näher.

        Qdrant liefert von sich aus die Ähnlichkeit. Fällt die Umrechnung weg,
        besteht dieser Test nicht mehr — und nur er.
        """
        store.replace_document(
            "/tmp/a.txt",
            sha256="a" * 64,
            format="txt",
            chunks=chunks("nah", "fern"),
            embeddings=[X, Y],
        )
        treffer = store.search(X, limit=2)
        assert treffer[0].text == "nah"
        assert treffer[0].distance < treffer[1].distance
        # X gegen sich selbst: Distanz praktisch null.
        assert treffer[0].distance == pytest.approx(0.0, abs=1e-5)
        # X gegen Y ist orthogonal, also Distanz 1.
        assert treffer[1].distance == pytest.approx(1.0, abs=1e-5)

    def test_reihenfolge_bei_zwischenwinkel(self, store):
        store.replace_document(
            "/tmp/a.txt",
            sha256="a" * 64,
            format="txt",
            chunks=chunks("orthogonal", "halb", "gleich"),
            embeddings=[Y, XY, X],
        )
        treffer = store.search(X, limit=3)
        assert [hit.text for hit in treffer] == ["gleich", "halb", "orthogonal"]

    def test_limit_wird_eingehalten(self, store):
        store.replace_document(
            "/tmp/a.txt",
            sha256="a" * 64,
            format="txt",
            chunks=chunks("eins", "zwei", "drei"),
            embeddings=[X, XY, Y],
        )
        assert len(store.search(X, limit=2)) == 2

    def test_ersetzen_laesst_keine_alten_treffer_zurueck(self, store):
        """Der stille Fehler: Vektoren des alten Stands überleben das Ersetzen.

        Bei sqlite-vec hängt die vec0-Tabelle an keinem Fremdschlüssel, bei den
        externen Backends gibt es gar keinen. Ohne das explizite Löschen liefert
        die Suche Text, den es nicht mehr gibt.
        """
        store.replace_document(
            "/tmp/a.txt",
            sha256="a" * 64,
            format="txt",
            chunks=chunks("alter text"),
            embeddings=[X],
        )
        store.replace_document(
            "/tmp/a.txt",
            sha256="b" * 64,
            format="txt",
            chunks=chunks("neuer text"),
            embeddings=[X],
        )
        treffer = store.search(X, limit=10)
        assert [hit.text for hit in treffer] == ["neuer text"]
        assert store.stats()["vectors"] == 1

    def test_vergessen_entfernt_vektoren(self, store, tmp_path):
        datei = tmp_path / "weg.txt"
        datei.write_text("inhalt", encoding="utf-8")
        store.replace_document(
            datei,
            sha256="a" * 64,
            format="txt",
            chunks=chunks("verschwindet"),
            embeddings=[X],
        )
        datei.unlink()

        entfernt = store.forget_missing([])
        assert len(entfernt) == 1
        assert store.search(X, limit=10) == []
        assert store.stats()["vectors"] == 0

    def test_leerer_index_findet_nichts(self, store):
        assert store.search(X, limit=5) == []

    def test_vektorzahl_folgt_der_chunkzahl(self, store):
        store.replace_document(
            "/tmp/a.txt",
            sha256="a" * 64,
            format="txt",
            chunks=chunks("eins", "zwei", "drei"),
            embeddings=[X, XY, Y],
        )
        stats = store.stats()
        assert stats["vectors"] == stats["chunks"] == 3

    def test_backend_steht_im_metadatum(self, store, backend):
        assert store.stats()["vector_backend"] == backend


class TestWechsel:
    def test_anderes_backend_wird_abgelehnt(self, tmp_path, backend):
        """Ein Backendwechsel ohne Neuaufbau muss auffallen.

        Sonst zeigt die Suche einen leeren Vektorspeicher und wirkt wie ein
        kaputter Index — die Chunks sind ja alle noch da.
        """
        if backend == DEFAULT_BACKEND:
            anderes = "lancedb"
        else:
            anderes = DEFAULT_BACKEND

        pfad = tmp_path / "index.db"
        with IndexStore(
            pfad,
            embedder="a/b",
            dimensions=DIMENSIONS,
            vector_backend=backend,
        ):
            pass

        with pytest.raises(StoreError, match="Vektor-Backend"):
            IndexStore(
                pfad,
                embedder="a/b",
                dimensions=DIMENSIONS,
                vector_backend=anderes,
            ).open()

    def test_gleiches_backend_oeffnet_wieder(self, tmp_path, backend):
        pfad = tmp_path / "index.db"
        with IndexStore(
            pfad, embedder="a/b", dimensions=DIMENSIONS, vector_backend=backend
        ) as erster:
            erster.replace_document(
                "/tmp/a.txt",
                sha256="a" * 64,
                format="txt",
                chunks=chunks("bleibt"),
                embeddings=[X],
            )

        with IndexStore(
            pfad, embedder="a/b", dimensions=DIMENSIONS, vector_backend=backend
        ) as zweiter:
            assert [hit.text for hit in zweiter.search(X, limit=1)] == ["bleibt"]


class TestAblage:
    def test_sqlite_vec_hat_kein_nebenverzeichnis(self, tmp_path):
        assert sidecar_for(tmp_path / "index.db", DEFAULT_BACKEND) is None

    @pytest.mark.parametrize(
        "backend,erwartet",
        [
            ("lancedb", "index.lance"),
            ("chromadb", "index.chroma"),
            ("qdrant", "index.qdrant"),
        ],
    )
    def test_nebenverzeichnis_neben_der_indexdatei(self, tmp_path, backend, erwartet):
        pfad = sidecar_for(tmp_path / "index.db", backend)
        assert pfad is not None
        assert pfad.name == erwartet
        assert pfad.parent == tmp_path

    def test_externes_backend_legt_sein_verzeichnis_an(self, tmp_path, backend):
        if backend == DEFAULT_BACKEND:
            pytest.skip("sqlite-vec wohnt in der Index-Datei")
        pfad = tmp_path / "index.db"
        with IndexStore(
            pfad, embedder="a/b", dimensions=DIMENSIONS, vector_backend=backend
        ):
            assert sidecar_for(pfad, backend).exists()


class TestIds:
    def test_chunk_ids_werden_nicht_wiederverwendet(self, store):
        """AUTOINCREMENT, damit ein Rest im Backend nie auf neuen Text trifft.

        Ohne das vergibt SQLite nach dem Löschen der höchsten Zeile dieselbe
        ID erneut. Ein abgebrochener Ingest könnte dann im externen Backend
        einen alten Vektor unter einer ID hinterlassen, die inzwischen anderen
        Text trägt — falsche Treffer statt fehlender.
        """
        store.replace_document(
            "/tmp/a.txt",
            sha256="a" * 64,
            format="txt",
            chunks=chunks("erst"),
            embeddings=[X],
        )
        erste = store.search(X, limit=1)[0].chunk_id

        store.replace_document(
            "/tmp/a.txt",
            sha256="b" * 64,
            format="txt",
            chunks=chunks("dann"),
            embeddings=[X],
        )
        zweite = store.search(X, limit=1)[0].chunk_id

        assert zweite > erste
