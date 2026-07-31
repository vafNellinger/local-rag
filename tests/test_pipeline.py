"""Tests für Kette und Einstellungen.

Ohne Modelle: geprüft wird die Verdrahtung. Der Schwerpunkt liegt darauf, was
beim Ändern einer Einstellung verworfen werden muss — ein geänderter Embedder
zwingt den Index neu zu öffnen, eine geänderte Temperatur darf nichts
entladen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.pipeline import (
    PipelineError,
    RagPipeline,
    Settings,
    _parse_context,
)
from rag.rerank import Reranker
from rag.store import IndexStore


class TestParseContext:
    def test_k_schreibweise(self):
        assert _parse_context("32k") == 32768
        assert _parse_context("8k") == 8192

    def test_ganzzahl_bleibt(self):
        assert _parse_context(4096) == 4096

    def test_zahl_als_text(self):
        assert _parse_context("2048") == 2048


class TestSettingsForPlatform:
    def test_vorgaben_kommen_aus_der_klasse(self):
        config = {
            "platform_class": {
                "testklasse": {
                    "embedder_profile": "klein",
                    "embedder_device": "cpu",
                    "reranker_enabled": False,
                    "generator_device": "cpu",
                    "context_length": "16k",
                }
            }
        }

        class FakePlatform:
            platform_class = "testklasse"

            def describe(self):
                return "Testmaschine"

        settings = Settings.for_platform(FakePlatform(), config=config)
        assert settings.embedder_profile == "klein"
        assert settings.reranker_enabled is False
        assert settings.generator_context_length == 16384
        # Ohne GPU keine Layer — sonst verspricht die Anzeige etwas Falsches.
        assert settings.generator_gpu_layers == 0

    def test_gpu_klasse_setzt_alle_layer(self):
        config = {
            "platform_class": {
                "gross": {"generator_device": "gpu", "context_length": "64k"}
            }
        }

        class FakePlatform:
            platform_class = "gross"

            def describe(self):
                return "Grosse Maschine"

        settings = Settings.for_platform(FakePlatform(), config=config)
        assert settings.generator_gpu_layers == -1

    def test_unbekannte_klasse_nutzt_vorgaben(self):
        class FakePlatform:
            platform_class = "gibtsnicht"

            def describe(self):
                return "?"

        settings = Settings.for_platform(FakePlatform(), config={})
        assert settings.embedder_profile == "default"


class TestSettingsPersistence:
    def test_speichern_und_laden(self, tmp_path):
        ziel = tmp_path / "settings.json"
        original = Settings(
            index_path=tmp_path / "i.db",
            top_k=9,
            temperature=0.55,
            min_rerank_score=0.02,
        )
        original.save(ziel)

        geladen = Settings.load(ziel, fallback=Settings())
        assert geladen.top_k == 9
        assert geladen.temperature == pytest.approx(0.55)
        assert geladen.min_rerank_score == pytest.approx(0.02)
        assert geladen.index_path == tmp_path / "i.db"

    def test_index_path_wird_wieder_ein_pfad(self, tmp_path):
        ziel = tmp_path / "s.json"
        Settings(index_path=tmp_path / "x.db").save(ziel)
        assert isinstance(Settings.load(ziel, fallback=Settings()).index_path, Path)

    def test_fehlende_datei_gibt_vorgaben(self, tmp_path):
        basis = Settings(top_k=7)
        assert Settings.load(tmp_path / "nichts.json", fallback=basis).top_k == 7

    def test_kaputte_datei_gibt_vorgaben(self, tmp_path):
        ziel = tmp_path / "s.json"
        ziel.write_text("{kein gueltiges json")
        assert Settings.load(ziel, fallback=Settings(top_k=3)).top_k == 3

    def test_unbekannte_felder_werden_ignoriert(self, tmp_path):
        # Eine Datei aus einer aelteren oder neueren Version darf nicht
        # dazu fuehren, dass gar nichts geladen wird.
        ziel = tmp_path / "s.json"
        ziel.write_text('{"top_k": 4, "gibtsnichtmehr": 99}')
        assert Settings.load(ziel, fallback=Settings()).top_k == 4

    def test_plattformfelder_werden_nicht_uebernommen(self, tmp_path):
        # Sie beschreiben die Maschine, nicht den Wunsch des Anwenders — und
        # die Maschine kann eine andere sein.
        ziel = tmp_path / "s.json"
        ziel.write_text('{"platform_class": "fremde-hardware", "top_k": 2}')
        geladen = Settings.load(ziel, fallback=Settings(platform_class="echte"))
        assert geladen.platform_class == "echte"
        assert geladen.top_k == 2

    def test_fehlendes_verzeichnis_wird_angelegt(self, tmp_path):
        ziel = tmp_path / "tief" / "drin" / "s.json"
        Settings().save(ziel)
        assert ziel.exists()


class TestApply:
    def test_unbekannte_einstellung_wird_abgelehnt(self):
        pipeline = RagPipeline(Settings())
        with pytest.raises(PipelineError, match="Unbekannte Einstellung"):
            pipeline.apply(gibtsnicht=1)

    def test_temperatur_entlaedt_nichts(self, tmp_path):
        pipeline = RagPipeline(Settings(index_path=tmp_path / "i.db"))
        pipeline._generator = object()
        pipeline.apply(temperature=0.9)
        assert pipeline._generator is not None

    def test_kontextwechsel_verwirft_den_generator(self, tmp_path):
        pipeline = RagPipeline(Settings(index_path=tmp_path / "i.db"))
        pipeline._generator = object()
        pipeline.apply(generator_context_length=4096)
        assert pipeline._generator is None

    def test_embedderwechsel_verwirft_store_und_embedder(self, tmp_path):
        settings = Settings(index_path=tmp_path / "i.db")
        pipeline = RagPipeline(settings)
        # Einen echten Store öffnen, damit close() etwas zu tun hat.
        pipeline._store = IndexStore(
            tmp_path / "i.db", embedder="a/b", dimensions=4
        ).open()
        pipeline._embedder = object()
        pipeline.apply(embedder_profile="alternative")
        assert pipeline._store is None
        assert pipeline._embedder is None

    def test_rerankerwechsel_verwirft_nur_den_reranker(self, tmp_path):
        pipeline = RagPipeline(
            Settings(index_path=tmp_path / "i.db", reranker_device="auto")
        )
        pipeline._reranker = object()
        pipeline._generator = object()
        pipeline.apply(reranker_device="cpu")
        assert pipeline._reranker is None
        # Der Generator hat mit dem Reranker nichts zu tun und bleibt geladen.
        assert pipeline._generator is not None

    def test_gleicher_wert_verwirft_nichts(self, tmp_path):
        # Das Speichern der Einstellungen schreibt alle Felder zurück, auch
        # unveraenderte. Dabei darf nichts entladen werden, sonst kostet jeder
        # Klick auf "Speichern" die Ladezeit aller Modelle.
        pipeline = RagPipeline(
            Settings(index_path=tmp_path / "i.db", reranker_device="cpu")
        )
        pipeline._reranker = object()
        pipeline._generator = object()
        pipeline._embedder = object()
        pipeline.apply(reranker_device="cpu", temperature=0.2)
        assert pipeline._reranker is not None
        assert pipeline._generator is not None
        assert pipeline._embedder is not None


class TestReranker:
    def test_abgeschalteter_reranker_ist_none(self, tmp_path):
        pipeline = RagPipeline(
            Settings(index_path=tmp_path / "i.db", reranker_enabled=False)
        )
        assert pipeline.reranker is None

    def test_modell_wird_nicht_beim_anlegen_geladen(self):
        assert not Reranker(device="cpu").is_loaded

    def test_leere_treffer_brauchen_kein_modell(self):
        reranker = Reranker(device="cpu")
        assert reranker.rerank("Frage", []) == []
        assert not reranker.is_loaded

    def test_einzelner_treffer_braucht_kein_modell(self):
        # Bei einem Kandidaten gibt es nichts zu ordnen — das Modell dafür zu
        # laden wären zwei Gigabyte für nichts.
        from rag.store import SearchHit

        einer = SearchHit(
            chunk_id=1,
            document_path="/x.md",
            ordinal=0,
            text="Text",
            heading_path=(),
            kind="prosa",
            distance=0.1,
        )
        reranker = Reranker(device="cpu")
        assert len(reranker.rerank("Frage", [einer])) == 1
        assert not reranker.is_loaded


class TestIndexStats:
    def test_ohne_index_leeres_bild(self, tmp_path):
        pipeline = RagPipeline(Settings(index_path=tmp_path / "gibtsnicht.db"))
        stats = pipeline.index_stats()
        assert stats["exists"] is False
        assert stats["documents"] == 0

    def test_mit_index_echte_zahlen(self, tmp_path):
        pfad = tmp_path / "i.db"
        with IndexStore(pfad, embedder="BAAI/bge-m3", dimensions=1024):
            pass
        pipeline = RagPipeline(Settings(index_path=pfad))
        stats = pipeline.index_stats()
        assert stats["exists"] is True
        assert stats["documents"] == 0
        pipeline.close()
