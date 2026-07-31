"""Tests für die Ingest-Orchestrierung.

Ohne echtes Modell: der Embedder wird durch einen Stub ersetzt, der aus dem
Text deterministische Vektoren ableitet. Getestet wird die Steuerung —
Dateiauswahl, Idempotenz, Fehlerbehandlung —, nicht die Embedding-Qualität.
"""

from __future__ import annotations

import hashlib
import math

import pytest

from rag.embed import EmbedderConfig
from rag.ingest import (
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_NEW,
    STATUS_SKIPPED,
    STATUS_UPDATED,
    collect_files,
    ingest_paths,
    search_index,
)
from rag.store import IndexStore

DIMENSIONS = 8


class StubEmbedder:
    """Deterministische Vektoren ohne Modell.

    Gleicher Text ergibt gleichen Vektor, ähnlicher Text ergibt keinen
    ähnlichen — für die Tests hier reicht das, weil sie die Steuerung prüfen
    und nicht die Retrieval-Qualität.
    """

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self.config = EmbedderConfig(
            model_id="stub/embedder",
            dimensions=dimensions,
            max_seq_length=512,
        )
        self.device = "cpu"
        self.dimensions = dimensions
        self.passage_calls = 0
        self.query_calls = 0

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[i] / 255.0 + 0.01 for i in range(self.dimensions)]
        norm = math.sqrt(sum(v * v for v in raw))
        return [v / norm for v in raw]

    def embed_passages(self, texts, *, progress: bool = False):
        self.passage_calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str):
        self.query_calls += 1
        return self._vector(text)

    def token_counter(self):
        # Wörter statt Subtokens — exakt genug für die Chunk-Steuerung im Test.
        return lambda text: max(1, len(text.split()))

    def check_lengths(self, texts):
        return []


@pytest.fixture
def stub():
    return StubEmbedder()


@pytest.fixture
def store(tmp_path):
    with IndexStore(
        tmp_path / "index.db", embedder="stub/embedder", dimensions=DIMENSIONS
    ) as opened:
        yield opened


@pytest.fixture
def docs(tmp_path):
    """Zwei Markdown-Dateien mit deutschem Inhalt."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "mietvertrag.md").write_text(
        "# Mietvertrag\n\n## Kündigung\n\n"
        "Die Kündigungsfrist beträgt sechs Monate zum Quartalsende.\n",
        encoding="utf-8",
    )
    (folder / "urlaub.md").write_text(
        "# Urlaub\n\nDer Jahresurlaub beträgt 30 Arbeitstage.\n",
        encoding="utf-8",
    )
    return folder


class TestCollectFiles:
    def test_verzeichnis_wird_aufgeklappt(self, docs):
        assert len(collect_files([docs])) == 2

    def test_unbekannte_formate_fallen_aus_verzeichnissen(self, docs):
        (docs / "bild.png").write_bytes(b"nicht text")
        (docs / "tabelle.xlsx").write_bytes(b"auch nicht")
        namen = {p.name for p in collect_files([docs])}
        assert namen == {"mietvertrag.md", "urlaub.md"}

    def test_direkt_benannte_datei_bleibt_trotz_format(self, tmp_path):
        # Wer eine Datei ausdrücklich nennt, soll die Fehlermeldung sehen und
        # nicht rätseln, warum nichts passiert ist.
        seltsam = tmp_path / "datei.xyz"
        seltsam.write_text("Inhalt")
        assert collect_files([seltsam]) == [seltsam]

    def test_doppelte_angabe_ergibt_eine_datei(self, docs):
        datei = docs / "urlaub.md"
        assert len(collect_files([datei, datei, docs / "." / "urlaub.md"])) == 1

    def test_verzeichnis_und_einzeldatei_ueberschneiden_sich_nicht(self, docs):
        assert len(collect_files([docs, docs / "urlaub.md"])) == 2

    def test_rekursiv_in_unterverzeichnisse(self, docs):
        tief = docs / "unterordner" / "noch tiefer"
        tief.mkdir(parents=True)
        (tief / "notiz.md").write_text("# Notiz\n\nInhalt hier.")
        assert len(collect_files([docs])) == 3

    def test_leere_eingabe(self):
        assert collect_files([]) == []


class TestIngest:
    def test_neue_dateien_werden_aufgenommen(self, docs, store, stub):
        report = ingest_paths([docs], store=store, embedder=stub)
        assert len(report.results) == 2
        assert all(r.status == STATUS_NEW for r in report.results)
        assert report.chunk_count > 0
        assert store.stats()["documents"] == 2

    def test_zweiter_lauf_ueberspringt_alles(self, docs, store, stub):
        ingest_paths([docs], store=store, embedder=stub)
        aufrufe_nach_erstem = stub.passage_calls
        report = ingest_paths([docs], store=store, embedder=stub)
        assert all(r.status == STATUS_SKIPPED for r in report.results)
        # Der eigentliche Punkt: kein erneutes Embedden.
        assert stub.passage_calls == aufrufe_nach_erstem

    def test_geaenderte_datei_wird_aktualisiert(self, docs, store, stub):
        ingest_paths([docs], store=store, embedder=stub)
        (docs / "urlaub.md").write_text(
            "# Urlaub\n\nDer Jahresurlaub beträgt jetzt 32 Arbeitstage.\n",
            encoding="utf-8",
        )
        report = ingest_paths([docs], store=store, embedder=stub)
        stati = {r.path.name: r.status for r in report.results}
        assert stati["urlaub.md"] == STATUS_UPDATED
        assert stati["mietvertrag.md"] == STATUS_SKIPPED

    def test_force_liest_alles_neu(self, docs, store, stub):
        ingest_paths([docs], store=store, embedder=stub)
        report = ingest_paths([docs], store=store, embedder=stub, force=True)
        assert all(r.status == STATUS_UPDATED for r in report.results)

    def test_chunkzahl_bleibt_nach_aktualisierung_konsistent(self, docs, store, stub):
        ingest_paths([docs], store=store, embedder=stub)
        vorher = store.stats()["chunks"]
        ingest_paths([docs], store=store, embedder=stub, force=True)
        stats = store.stats()
        assert stats["chunks"] == vorher
        assert stats["vectors"] == stats["chunks"]

    def test_leere_datei_wird_als_leer_gemeldet(self, tmp_path, store, stub):
        leer = tmp_path / "leer.md"
        leer.write_text("   \n\n  \n")
        report = ingest_paths([leer], store=store, embedder=stub)
        assert report.results[0].status == STATUS_EMPTY
        assert store.stats()["documents"] == 0

    def test_nur_ueberschriften_ist_leer(self, tmp_path, store, stub):
        md = tmp_path / "gerippe.md"
        md.write_text("# Titel\n\n## Abschnitt\n")
        assert ingest_paths([md], store=store, embedder=stub).results[0].status == (
            STATUS_EMPTY
        )

    def test_fehlende_datei_bricht_den_lauf_nicht_ab(self, docs, store, stub):
        report = ingest_paths(
            [docs / "gibtsnicht.md", docs], store=store, embedder=stub
        )
        assert len(report.failed) == 1
        # Die anderen beiden müssen trotzdem durchgelaufen sein.
        assert len(report.changed) == 2

    def test_unbekanntes_format_meldet_fehler(self, tmp_path, store, stub):
        seltsam = tmp_path / "datei.xyz"
        seltsam.write_text("Inhalt")
        report = ingest_paths([seltsam], store=store, embedder=stub)
        assert report.results[0].status == STATUS_FAILED
        assert "nicht unterstützt" in report.results[0].error

    def test_altformat_nennt_den_konvertierungsweg(self, tmp_path, store, stub):
        alt = tmp_path / "alt.doc"
        alt.write_bytes(b"egal")
        report = ingest_paths([alt], store=store, embedder=stub)
        assert "libreoffice" in report.results[0].error

    def test_ueberschriften_landen_im_index(self, docs, store, stub):
        ingest_paths([docs], store=store, embedder=stub)
        treffer = store.search(stub.embed_query("Kündigungsfrist"), limit=10)
        pfade = {h.heading for h in treffer}
        assert "Mietvertrag > Kündigung" in pfade

    def test_report_summiert_richtig(self, docs, store, stub):
        report = ingest_paths([docs], store=store, embedder=stub)
        assert report.chunk_count == store.stats()["chunks"]
        assert report.duration_seconds >= 0


class TestProgress:
    def test_fortschritt_wird_gemeldet(self, docs, store, stub):
        meldungen: list[tuple[str, int, int, str]] = []

        def sammeln(path, position, total, phase):
            meldungen.append((path.name, position, total, phase))

        ingest_paths([docs], store=store, embedder=stub, progress=sammeln)
        phasen = {m[3] for m in meldungen}
        assert {"extrahieren", "chunken", "embedden", "schreiben"} <= phasen
        assert all(m[2] == 2 for m in meldungen)

    def test_uebersprungene_datei_meldet_ihren_status(self, docs, store, stub):
        ingest_paths([docs], store=store, embedder=stub)
        phasen: list[str] = []
        ingest_paths(
            [docs],
            store=store,
            embedder=stub,
            progress=lambda p, i, t, phase: phasen.append(phase),
        )
        assert phasen == [STATUS_SKIPPED, STATUS_SKIPPED]


class TestPrune:
    def test_verschwundene_datei_wird_entfernt(self, docs, store, stub):
        ingest_paths([docs], store=store, embedder=stub)
        (docs / "urlaub.md").unlink()
        report = ingest_paths([docs], store=store, embedder=stub, prune=True)
        assert len(report.removed) == 1
        assert store.stats()["documents"] == 1

    def test_ohne_prune_bleibt_der_eintrag(self, docs, store, stub):
        ingest_paths([docs], store=store, embedder=stub)
        (docs / "urlaub.md").unlink()
        report = ingest_paths([docs], store=store, embedder=stub)
        assert report.removed == []
        assert store.stats()["documents"] == 2


class TestSearchIndex:
    def test_identischer_text_wird_gefunden(self, tmp_path, store, stub):
        # Der Stub bildet gleichen Text auf gleichen Vektor ab, also muss der
        # Chunk mit Distanz nahe null zurueckkommen.
        md = tmp_path / "a.md"
        md.write_text("# Kapitel\n\nDie Frist beträgt sechs Monate.\n")
        ingest_paths([md], store=store, embedder=stub)
        chunk_text = store.search(stub.embed_query("irgendwas"), limit=1)[0]
        treffer = search_index(
            f"Kapitel\n\n{chunk_text.text}", store=store, embedder=stub, limit=1
        )
        assert treffer[0].similarity == pytest.approx(1.0, abs=1e-5)

    def test_limit_wird_durchgereicht(self, docs, store, stub):
        ingest_paths([docs], store=store, embedder=stub)
        assert len(search_index("Frist", store=store, embedder=stub, limit=1)) == 1

    def test_suche_im_leeren_index(self, store, stub):
        assert search_index("Frist", store=store, embedder=stub) == []
