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
    def test_explizite_angabe_wird_uebernommen(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"

    def test_auto_liefert_ein_bekanntes_geraet(self):
        assert resolve_device("auto") in {"cpu", "cuda", "mps"}


class TestEmbedder:
    def test_modell_wird_nicht_beim_anlegen_geladen(self):
        # Der Punkt: 'rag ingest' ueber ein unveraendertes Verzeichnis darf
        # die Ladezeit von gut zwei Gigabyte nicht zahlen.
        embedder = Embedder(
            EmbedderConfig(model_id="stub/x", dimensions=8, max_seq_length=128),
            device="cpu",
        )
        assert not embedder.is_loaded

    def test_batchgroesse_haengt_am_geraet(self):
        config = EmbedderConfig(model_id="stub/x", dimensions=8, max_seq_length=128)
        assert Embedder(config, device="cpu").batch_size == BATCH_SIZE_CPU
        assert Embedder(config, device="cuda").batch_size == BATCH_SIZE_GPU

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
