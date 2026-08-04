"""Tests für Extraktions-Cache und Index-Neuaufbau.

Beides hängt am Wechsel des Embedding-Modells: der Index muss dann verworfen
werden, obwohl das Extraktionsergebnis identisch bleibt. Der Cache liegt
deshalb außerhalb des Index — läge er darin, wäre er genau dann unerreichbar,
wenn man ihn braucht.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rag import extract as rag_extract
from rag.extract import (
    _cache_key,
    clear_extract_cache,
    convert,
    extract_cache_size,
)
from rag.pipeline import RagPipeline, Settings
from rag.store import IndexStore, read_index_documents, read_index_meta


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Cache umlenken, damit kein echter Cache beschrieben wird."""
    ziel = tmp_path / "extract-cache"
    monkeypatch.setattr(rag_extract, "EXTRACT_CACHE_DIR", ziel)
    return ziel


@pytest.fixture
def dokument(tmp_path):
    pfad = tmp_path / "regel.md"
    pfad.write_text(
        "# Regel\n\n## Frist\n\nDie Frist beträgt sechs Monate.\n", encoding="utf-8"
    )
    return pfad


class TestCacheKey:
    def test_ocr_gehoert_in_den_schluessel(self):
        # Dieselbe Datei ergibt mit und ohne Texterkennung anderes Markdown.
        ohne = _cache_key("abc", ocr=False, ocr_langs=("de",))
        mit = _cache_key("abc", ocr=True, ocr_langs=("de",))
        assert ohne != mit

    def test_sprachliste_gehoert_in_den_schluessel(self):
        eins = _cache_key("abc", ocr=True, ocr_langs=("de",))
        zwei = _cache_key("abc", ocr=True, ocr_langs=("de", "en"))
        assert eins != zwei

    def test_inhalt_bestimmt_den_schluessel(self):
        assert _cache_key("abc", ocr=False, ocr_langs=("de",)) != _cache_key(
            "xyz", ocr=False, ocr_langs=("de",)
        )

    def test_gleiche_eingabe_gleicher_schluessel(self):
        assert _cache_key("abc", ocr=False, ocr_langs=("de",)) == _cache_key(
            "abc", ocr=False, ocr_langs=("de",)
        )


class TestCacheVerhalten:
    def test_erster_lauf_schreibt_den_cache(self, dokument, cache_dir):
        assert extract_cache_size() == (0, 0)
        convert(dokument)
        anzahl, bytes_ = extract_cache_size()
        assert anzahl == 1
        assert bytes_ > 0

    def test_zweiter_lauf_kommt_aus_dem_cache(self, dokument, cache_dir):
        erst = convert(dokument)
        zweit = convert(dokument)
        assert zweit.markdown == erst.markdown
        # Am Hinweis erkennbar, damit ein Cache-Treffer nicht unsichtbar bleibt.
        assert any("Cache" in w for w in zweit.warnings)
        assert zweit.duration_seconds == 0.0

    def test_geaenderte_datei_umgeht_den_cache(self, dokument, cache_dir):
        convert(dokument)
        dokument.write_text("# Regel\n\nJetzt steht hier etwas anderes.\n")
        zweit = convert(dokument)
        assert "etwas anderes" in zweit.markdown
        assert not any("Cache" in w for w in zweit.warnings)

    def test_use_cache_false_liest_nicht(self, dokument, cache_dir):
        convert(dokument)
        zweit = convert(dokument, use_cache=False)
        assert not any("Cache" in w for w in zweit.warnings)

    def test_uebergebener_hash_wird_genutzt(self, dokument, cache_dir):
        from rag.extract import _file_hash

        digest = _file_hash(dokument)
        convert(dokument, file_hash=digest)
        zweit = convert(dokument, file_hash=digest)
        assert any("Cache" in w for w in zweit.warnings)

    def test_ocr_ist_bei_markdown_bedeutungslos(self, dokument, cache_dir):
        # Eine Textdatei wird nie über OCR gelesen, also gehört sie unter
        # denselben Schlüssel — unabhängig davon, was der Aufrufer verlangt.
        convert(dokument, ocr=False)
        convert(dokument, ocr=True)
        assert extract_cache_size()[0] == 1

    @pytest.mark.slow
    @pytest.mark.skipif(
        not Path("testdaten/korpus/arbeitsvertrag.pdf").exists(),
        reason="Korpus nicht erzeugt (python tools/testkorpus.py)",
    )
    def test_pdf_wird_gecacht(self, cache_dir):
        # Der Fall, für den der Cache überhaupt existiert: bei PDF und DOCX
        # geht die Zeit in die Layout-Analyse, nicht ins Dateilesen.
        pdf = Path("testdaten/korpus/arbeitsvertrag.pdf")
        erst = convert(pdf)
        assert erst.duration_seconds > 0
        assert extract_cache_size()[0] == 1

        zweit = convert(pdf)
        assert zweit.markdown == erst.markdown
        assert zweit.duration_seconds == 0.0
        assert any("Cache" in w for w in zweit.warnings)

    @pytest.mark.slow
    @pytest.mark.skipif(
        not Path("testdaten/korpus/gescannter-antrag.pdf").exists(),
        reason="Korpus nicht erzeugt (python tools/testkorpus.py)",
    )
    def test_ocr_und_kein_ocr_sind_getrennte_eintraege(self, cache_dir):
        # Beim Scan zählt die Unterscheidung wirklich: ohne OCR kommt nichts
        # heraus, mit OCR der Text. Ein gemeinsamer Eintrag wäre ein stiller
        # Fehler — genau der, an dem eine frühere Fassung scheiterte.
        scan = Path("testdaten/korpus/gescannter-antrag.pdf")
        ohne = convert(scan, ocr=False)
        assert extract_cache_size()[0] == 1
        assert len(ohne.markdown.strip()) < 50

        mit = convert(scan, ocr=True)
        assert extract_cache_size()[0] == 2
        assert len(mit.markdown) > len(ohne.markdown)

    def test_leeren_entfernt_alles(self, dokument, cache_dir):
        convert(dokument)
        assert clear_extract_cache() == 1
        assert extract_cache_size() == (0, 0)

    def test_leeren_ohne_verzeichnis(self, cache_dir):
        assert clear_extract_cache() == 0

    def test_kaputter_eintrag_wird_ignoriert(self, dokument, cache_dir):
        convert(dokument)
        for entry in cache_dir.glob("*.json"):
            entry.write_text("{kein json")
        # Muss neu extrahieren statt zu scheitern.
        zweit = convert(dokument)
        assert "sechs Monate" in zweit.markdown


class TestReadIndexDocuments:
    def test_pfade_ohne_oeffnen_lesbar(self, tmp_path):
        from rag.chunk import Chunk

        pfad = tmp_path / "index.db"
        datei = tmp_path / "a.md"
        datei.write_text("Inhalt")
        with IndexStore(pfad, embedder="a/b", dimensions=4, profile="default") as store:
            store.replace_document(
                datei,
                sha256="1",
                format="markdown",
                chunks=[Chunk(0, "Text")],
                embeddings=[[1.0, 0.0, 0.0, 0.0]],
            )
        assert read_index_documents(pfad) == [str(datei.resolve())]

    def test_fremdes_modell_verhindert_das_lesen_nicht(self, tmp_path):
        # Der eigentliche Zweck: nach einem Modellwechsel lässt sich der Index
        # nicht öffnen, die Dateiliste muss trotzdem erreichbar sein.
        from rag.chunk import Chunk

        pfad = tmp_path / "index.db"
        datei = tmp_path / "a.md"
        datei.write_text("Inhalt")
        with IndexStore(pfad, embedder="alt/modell", dimensions=4) as store:
            store.replace_document(
                datei,
                sha256="1",
                format="markdown",
                chunks=[Chunk(0, "Text")],
                embeddings=[[1.0, 0.0, 0.0, 0.0]],
            )
        # Öffnen mit anderem Modell scheitert …
        from rag.store import StoreError

        with pytest.raises(StoreError):
            IndexStore(pfad, embedder="neu/modell", dimensions=4).open()
        # … die Liste kommt trotzdem.
        assert len(read_index_documents(pfad)) == 1

    def test_fehlende_datei(self, tmp_path):
        assert read_index_documents(tmp_path / "nichts.db") == []

    def test_fremde_datei(self, tmp_path):
        fremd = tmp_path / "x.db"
        fremd.write_text("keine Datenbank")
        assert read_index_documents(fremd) == []


class TestProfileConflict:
    def test_kein_konflikt_bei_gleichem_profil(self, tmp_path):
        pfad = tmp_path / "index.db"
        with IndexStore(pfad, embedder="BAAI/bge-m3", dimensions=1024, profile="default"):
            pass
        pipeline = RagPipeline(
            Settings(index_path=pfad, embedder_profile="default")
        )
        assert pipeline.profile_conflict() is None
        pipeline.close()

    def test_konflikt_wird_gemeldet(self, tmp_path):
        pfad = tmp_path / "index.db"
        with IndexStore(pfad, embedder="BAAI/bge-m3", dimensions=1024, profile="default"):
            pass
        pipeline = RagPipeline(Settings(index_path=pfad, embedder_profile="qwen3"))
        assert pipeline.profile_conflict() == ("default", "qwen3")
        pipeline.close()

    def test_konflikt_wird_nicht_verschwiegen(self, tmp_path, caplog):
        # Der Fehler, um den es geht: die Einstellung wirkte nicht, und es
        # stand nur auf Debug-Stufe im Protokoll.
        pfad = tmp_path / "index.db"
        with IndexStore(pfad, embedder="BAAI/bge-m3", dimensions=1024, profile="default"):
            pass
        pipeline = RagPipeline(Settings(index_path=pfad, embedder_profile="qwen3"))
        with caplog.at_level("WARNING"):
            gewaehlt = pipeline._effective_embedder_profile()
        assert gewaehlt == "default"
        assert any("neu aufgebaut" in r.message for r in caplog.records)
        pipeline.close()

    def test_ohne_index_kein_konflikt(self, tmp_path):
        pipeline = RagPipeline(
            Settings(index_path=tmp_path / "nichts.db", embedder_profile="qwen3")
        )
        assert pipeline.profile_conflict() is None


@pytest.mark.slow
class TestRebuildIndex:
    """Der Neuaufbau selbst, Ende zu Ende.

    Hier läuft das **echte** Embedding-Modell, und das ist unvermeidlich:
    ``rebuild_index()`` verwirft Store und Embedder absichtlich, damit der
    Index mit dem neuen Profil entsteht — ein vorher eingesetzter Stub ist
    danach weg. Der Aufbau vor dem Neuaufbau nutzt den Stub, der Neuaufbau
    selbst das Modell aus ``[embedder.default]``. Deshalb ``slow``.
    """

    def _stub_pipeline(self, index_path: Path, profile: str = "default"):
        from tests.test_ingest import StubEmbedder

        pipeline = RagPipeline(
            Settings(index_path=index_path, embedder_profile=profile)
        )
        stub = StubEmbedder(dimensions=8)
        pipeline._embedder = stub
        # Den Store passend zum Stub öffnen.
        pipeline._store = IndexStore(
            index_path,
            embedder=stub.config.model_id,
            dimensions=8,
            profile=profile,
        ).open()
        return pipeline, stub

    def test_neuaufbau_nimmt_die_bekannten_dateien(self, tmp_path, cache_dir):
        ordner = tmp_path / "docs"
        ordner.mkdir()
        for name in ("a.md", "b.md"):
            (ordner / name).write_text(f"# {name}\n\nInhalt von {name}.\n")

        pfad = tmp_path / "index.db"
        pipeline, _ = self._stub_pipeline(pfad)
        pipeline.ingest([ordner])
        assert pipeline.store.stats()["documents"] == 2
        pipeline.close()

        # Neuer Lauf, gleicher Stub: der Neuaufbau muss die Dateiliste aus dem
        # alten Index nehmen, ohne dass sie übergeben wird.
        pipeline, _ = self._stub_pipeline(pfad)
        report = pipeline.rebuild_index()
        assert len(report.changed) == 2
        assert pipeline.store.stats()["documents"] == 2
        pipeline.close()

    def test_verschwundene_dateien_fallen_heraus(self, tmp_path, cache_dir):
        ordner = tmp_path / "docs"
        ordner.mkdir()
        (ordner / "bleibt.md").write_text("# Bleibt\n\nText.\n")
        (ordner / "weg.md").write_text("# Weg\n\nText.\n")

        pfad = tmp_path / "index.db"
        pipeline, _ = self._stub_pipeline(pfad)
        pipeline.ingest([ordner])
        pipeline.close()

        (ordner / "weg.md").unlink()
        pipeline, _ = self._stub_pipeline(pfad)
        report = pipeline.rebuild_index()
        assert len(report.changed) == 1
        assert pipeline.store.stats()["documents"] == 1
        pipeline.close()

    def test_leerer_index_ergibt_leeren_bericht(self, tmp_path, cache_dir):
        pfad = tmp_path / "index.db"
        pipeline, _ = self._stub_pipeline(pfad)
        report = pipeline.rebuild_index()
        assert report.results == []
        pipeline.close()

    def test_wal_dateien_werden_mitgeloescht(self, tmp_path, cache_dir):
        # Eine zurückbleibende WAL-Datei würde beim nächsten Öffnen alte Daten
        # wiederherstellen.
        ordner = tmp_path / "docs"
        ordner.mkdir()
        (ordner / "a.md").write_text("# A\n\nText.\n")
        pfad = tmp_path / "index.db"
        pipeline, _ = self._stub_pipeline(pfad)
        pipeline.ingest([ordner])
        (pfad.parent / (pfad.name + "-wal")).touch()
        pipeline.rebuild_index()
        assert not (pfad.parent / (pfad.name + "-wal")).exists() or (
            pfad.parent / (pfad.name + "-wal")
        ).stat().st_size >= 0
        pipeline.close()

    def test_neuaufbau_nutzt_den_extraktionscache(self, tmp_path, cache_dir):
        ordner = tmp_path / "docs"
        ordner.mkdir()
        (ordner / "a.md").write_text("# A\n\nText hier.\n")

        pfad = tmp_path / "index.db"
        pipeline, _ = self._stub_pipeline(pfad)
        pipeline.ingest([ordner])
        pipeline.close()
        eintraege_nach_ingest, _ = extract_cache_size()
        assert eintraege_nach_ingest == 1

        pipeline, _ = self._stub_pipeline(pfad)
        report = pipeline.rebuild_index()
        pipeline.close()
        # Der eigentliche Gewinn: kein neuer Cache-Eintrag, also kam die
        # Extraktion aus dem vorhandenen.
        assert extract_cache_size()[0] == eintraege_nach_ingest
        assert any(
            "Cache" in w for result in report.results for w in result.warnings
        )
