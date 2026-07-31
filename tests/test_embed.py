"""Tests für die Embedder-Konfiguration.

Ohne Modell-Download: geprüft wird das Lesen der Profile aus platforms.toml
und die Gerätewahl. Der Schwerpunkt liegt auf den Instruktionspräfixen — ihr
Fehlen verschlechtert die Retrieval-Qualität, ohne etwas fehlschlagen zu
lassen.
"""

from __future__ import annotations

import pytest

from rag.embed import (
    BATCH_SIZE_CPU,
    BATCH_SIZE_GPU,
    Embedder,
    EmbedderConfig,
    EmbeddingError,
    load_embedder_config,
    resolve_device,
)


class TestLoadConfig:
    def test_default_ist_bge_m3(self):
        config = load_embedder_config("default")
        assert config.model_id == "BAAI/bge-m3"
        assert config.dimensions == 1024

    def test_bge_m3_braucht_keine_praefixe(self):
        # Der Grund, warum die Felder ueberhaupt existieren: bei den
        # Alternativen sind sie gefuellt, hier nicht.
        config = load_embedder_config("default")
        assert not config.needs_prefix
        assert config.query_prefix == ""

    def test_e5_hat_die_vorgeschriebenen_praefixe(self):
        # E5 ist auf genau diese Praefixe trainiert. Fehlen sie, sinkt die
        # Qualitaet still — deshalb stehen sie in der Konfiguration.
        config = load_embedder_config("alternative")
        assert config.query_prefix == "query: "
        assert config.passage_prefix == "passage: "
        assert config.needs_prefix

    def test_qwen3_profil_ist_vorhanden(self):
        config = load_embedder_config("qwen3")
        assert config.model_id == "Qwen/Qwen3-Embedding-0.6B"
        assert config.query_prefix.startswith("Instruct:")

    def test_unbekanntes_profil_nennt_die_vorhandenen(self):
        with pytest.raises(EmbeddingError, match="default"):
            load_embedder_config("gibtsnicht")

    def test_dimensionen_sind_ganzzahlen(self):
        config = load_embedder_config("default")
        assert isinstance(config.dimensions, int)
        assert isinstance(config.max_seq_length, int)

    def test_unvollstaendiges_profil_wird_abgelehnt(self, tmp_path):
        kaputt = tmp_path / "platforms.toml"
        kaputt.write_text('[embedder.default]\nmodel_id = "a/b"\n')
        with pytest.raises(EmbeddingError, match="dimensions"):
            load_embedder_config("default", config_path=kaputt)

    def test_profil_ohne_embedder_tabelle(self, tmp_path):
        leer = tmp_path / "platforms.toml"
        leer.write_text("[roles]\nstatic = []\n")
        with pytest.raises(EmbeddingError, match="keine"):
            load_embedder_config("default", config_path=leer)


class TestResolveDevice:
    def test_cpu_wird_immer_uebernommen(self):
        assert resolve_device("cpu") == "cpu"

    def test_auto_liefert_ein_bekanntes_geraet(self):
        assert resolve_device("auto") in {"cpu", "cuda", "mps"}

    def test_gpu_aus_platforms_toml_wird_uebersetzt(self):
        # platforms.toml schreibt "gpu", Torch kennt nur cuda/mps.
        assert resolve_device("gpu") in {"cpu", "cuda", "mps"}

    def test_fehlende_gpu_faellt_auf_cpu_zurueck(self, caplog):
        # Auf dieser Maschine der Normalfall: CUDA-Build von Torch, AMD-iGPU
        # unsichtbar. Der Fallback darf nicht still passieren, sonst sucht man
        # die Ursache der Langsamkeit an der falschen Stelle.
        import rag.embed

        if rag.embed._best_gpu() is not None:
            pytest.skip("Maschine hat einen von Torch nutzbaren Beschleuniger")
        assert resolve_device("cuda") == "cpu"
        assert any("kein" in r.message.lower() for r in caplog.records)


class TestEmbedder:
    def test_modell_wird_nicht_beim_anlegen_geladen(self):
        # Der Punkt: 'rag ingest' ueber ein unveraendertes Verzeichnis darf
        # die Ladezeit von gut zwei Gigabyte nicht zahlen.
        embedder = Embedder(
            EmbedderConfig(model_id="stub/x", dimensions=8, max_seq_length=128),
            device="cpu",
        )
        assert not embedder.is_loaded

    def test_batchgroesse_haengt_am_aufgeloesten_geraet(self):
        # Am aufgeloesten, nicht am gewuenschten: wer "cuda" verlangt und keine
        # Karte hat, laeuft auf CPU und braucht dann auch den CPU-Batch.
        config = EmbedderConfig(model_id="stub/x", dimensions=8, max_seq_length=128)
        embedder = Embedder(config, device="cpu")
        assert embedder.device == "cpu"
        assert embedder.batch_size == BATCH_SIZE_CPU

        auto = Embedder(config, device="auto")
        erwartet = BATCH_SIZE_CPU if auto.device == "cpu" else BATCH_SIZE_GPU
        assert auto.batch_size == erwartet

    def test_gpu_batch_ist_groesser_als_cpu_batch(self):
        assert BATCH_SIZE_GPU > BATCH_SIZE_CPU

    def test_batchgroesse_ist_ueberschreibbar(self):
        config = EmbedderConfig(model_id="stub/x", dimensions=8, max_seq_length=128)
        assert Embedder(config, device="cpu", batch_size=4).batch_size == 4

    def test_leere_liste_ergibt_keine_vektoren(self):
        # Muss ohne Modell funktionieren — sonst laedt ein leerer Lauf das
        # Modell fuer nichts.
        embedder = Embedder(
            EmbedderConfig(model_id="stub/x", dimensions=8, max_seq_length=128),
            device="cpu",
        )
        assert embedder.embed_passages([]) == []
        assert not embedder.is_loaded

    def test_fehlendes_modell_meldet_klar(self):
        embedder = Embedder(
            EmbedderConfig(
                model_id="gibtsnicht/wirklichnicht-xyz",
                dimensions=8,
                max_seq_length=128,
            ),
            device="cpu",
        )
        with pytest.raises(EmbeddingError, match="konnte nicht geladen werden"):
            embedder.embed_passages(["Text"])


class TestEmbedderConfig:
    def test_needs_prefix_bei_nur_query(self):
        config = EmbedderConfig(
            model_id="a/b", dimensions=8, max_seq_length=128, query_prefix="q: "
        )
        assert config.needs_prefix

    def test_needs_prefix_bei_keinem(self):
        assert not EmbedderConfig(
            model_id="a/b", dimensions=8, max_seq_length=128
        ).needs_prefix
