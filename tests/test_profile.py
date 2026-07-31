"""Tests für die plattformabhängige Embedder-Wahl.

Vorher stand in ``_resolve_static`` hart ``default``, und die übrigen Einträge
in ``[embedder.*]`` waren toter Text. Geprüft wird hier, dass die
Plattformklasse das Modell bestimmt und dass ein bestehender Index Vorrang hat
— sonst würde ein Hardwarewechsel den Index bei jedem Lauf invalidieren.
"""

from __future__ import annotations

import pytest

from rag.cli import _choose_embedder
from rag.detect import Platform
from rag.resolve import ResolutionError, _resolve_static
from rag.store import IndexStore, read_index_meta

CONFIG = {
    "embedder": {
        "default": {
            "model_id": "BAAI/bge-m3",
            "dimensions": 1024,
            "max_seq_length": 8192,
            "vram_estimate_mb": 1100,
        },
        "klein": {
            "model_id": "winzig/modell",
            "dimensions": 384,
            "max_seq_length": 512,
            "vram_estimate_mb": 200,
        },
    },
    "reranker": {"default": {"model_id": "BAAI/bge-reranker-v2-m3"}},
    "platform_class": {
        "cpu_only": {
            "embedder_device": "cpu",
            "embedder_profile": "klein",
            "reranker_enabled": False,
            "generator_device": "cpu",
        },
        "dgpu_large": {
            "embedder_device": "gpu",
            "embedder_profile": "default",
            "reranker_device": "gpu",
            "generator_device": "gpu",
        },
    },
}


class TestResolveStatic:
    def test_profil_bestimmt_das_modell(self):
        spec = _resolve_static("embedder", CONFIG, "cpu", "klein")
        assert spec.model_id == "winzig/modell"
        assert spec.dimensions == 384

    def test_default_bleibt_der_vorgabewert(self):
        assert _resolve_static("embedder", CONFIG, "cpu").model_id == "BAAI/bge-m3"

    def test_unbekanntes_profil_nennt_die_vorhandenen(self):
        with pytest.raises(ResolutionError, match="klein"):
            _resolve_static("embedder", CONFIG, "cpu", "gibtsnicht")

    def test_notiz_nennt_die_herkunft(self):
        # Bei einem falsch aufgeloesten Modell muss aus der Ausgabe hervorgehen,
        # welcher Config-Eintrag dahinter steckt.
        spec = _resolve_static("embedder", CONFIG, "cpu", "klein")
        assert "[embedder.klein]" in spec.notes[0]

    def test_vram_nur_bei_gpu_zugewiesen(self):
        assert _resolve_static("embedder", CONFIG, "cpu", "klein").vram_required_bytes is (
            None
        )
        assert _resolve_static("embedder", CONFIG, "gpu", "klein").vram_required_bytes


class TestResolvePipelineProfile:
    """Die Plattformklasse muss das Profil bis in den Plan durchreichen."""

    def test_klasse_waehlt_das_profil(self, monkeypatch):
        from rag import resolve

        # Den Generator stillstellen: hier geht es nur um die statischen Rollen,
        # und whichllm hat in einem Test nichts zu suchen.
        monkeypatch.setattr(
            resolve,
            "_resolve_generator",
            lambda *a, **k: resolve.ModelSpec(
                role="generator", model_id="stub/gen", device="cpu", source="stub"
            ),
        )
        platform = Platform(
            label="test",
            platform_class="cpu_only",
            gpu_name=None,
            gpu_count=0,
            usable_vram_bytes=0,
            shared_memory=False,
            ram_bytes=8 * 1024**3,
            cpu_cores=4,
            os_name="linux",
        )
        plan = resolve.resolve_pipeline(platform, config=CONFIG)
        # cpu_only zeigt in dieser Testkonfiguration auf "klein".
        assert plan.embedder.model_id == "winzig/modell"

    def test_andere_klasse_anderes_profil(self, monkeypatch):
        from rag import resolve

        monkeypatch.setattr(
            resolve,
            "_resolve_generator",
            lambda *a, **k: resolve.ModelSpec(
                role="generator", model_id="stub/gen", device="gpu", source="stub"
            ),
        )
        platform = Platform(
            label="test",
            platform_class="dgpu_large",
            gpu_name="RTX 4090",
            gpu_count=1,
            usable_vram_bytes=24 * 1024**3,
            shared_memory=False,
            ram_bytes=64 * 1024**3,
            cpu_cores=16,
            os_name="linux",
        )
        plan = resolve.resolve_pipeline(platform, config=CONFIG)
        assert plan.embedder.model_id == "BAAI/bge-m3"
        assert plan.embedder.device == "gpu"


class TestChooseEmbedder:
    """Die Rangfolge in der CLI: Angabe > Index > Plattform."""

    def test_explizite_angabe_gewinnt(self, tmp_path):
        choice = _choose_embedder(tmp_path / "kein-index.db", "qwen3", "cpu")
        assert choice.profile == "qwen3"
        assert choice.device == "cpu"
        # Keine Plattformerkennung noetig, wenn beides angegeben ist.
        assert "explizit" in choice.origin

    def test_index_gewinnt_gegen_plattform(self, tmp_path):
        # Der Punkt: ein Hardwarewechsel darf einen bestehenden Index nicht
        # bei jedem Lauf invalidieren.
        pfad = tmp_path / "index.db"
        with IndexStore(
            pfad, embedder="winzig/modell", dimensions=384, profile="klein"
        ):
            pass
        choice = _choose_embedder(pfad, None, "cpu")
        assert choice.profile == "klein"
        assert "Index" in choice.origin

    def test_angabe_gewinnt_gegen_index(self, tmp_path):
        pfad = tmp_path / "index.db"
        with IndexStore(
            pfad, embedder="winzig/modell", dimensions=384, profile="klein"
        ):
            pass
        assert _choose_embedder(pfad, "qwen3", "cpu").profile == "qwen3"

    def test_ohne_index_und_ohne_angabe_kommt_die_plattform(self, tmp_path):
        # Greift auf die echte platforms.toml zu; dort zeigt derzeit jede
        # Klasse auf "default". Geprueft wird, dass die Herkunft benannt wird.
        choice = _choose_embedder(tmp_path / "keiner.db", None, "cpu")
        assert choice.profile == "default"
        assert "Plattformklasse" in choice.origin


class TestIndexMeta:
    def test_profil_landet_im_index(self, tmp_path):
        pfad = tmp_path / "index.db"
        with IndexStore(
            pfad, embedder="winzig/modell", dimensions=384, profile="klein"
        ):
            pass
        assert read_index_meta(pfad)["embedder_profile"] == "klein"

    def test_meta_ohne_sqlite_vec_lesbar(self, tmp_path):
        # read_index_meta muss ohne die Erweiterung auskommen, sonst kann
        # 'rag search' das Profil nicht ermitteln, bevor es den Index oeffnet.
        pfad = tmp_path / "index.db"
        with IndexStore(pfad, embedder="a/b", dimensions=8, profile="default"):
            pass
        meta = read_index_meta(pfad)
        assert meta["embedder"] == "a/b"
        assert meta["dimensions"] == "8"

    def test_fehlende_datei_gibt_leeres_dict(self, tmp_path):
        assert read_index_meta(tmp_path / "nichts.db") == {}

    def test_fremde_datei_gibt_leeres_dict(self, tmp_path):
        fremd = tmp_path / "kein-index.db"
        fremd.write_text("das ist keine Datenbank")
        assert read_index_meta(fremd) == {}

    def test_altindex_ohne_profilfeld_wird_ergaenzt(self, tmp_path):
        # Ein Index aus einer Version vor dem Profilfeld darf nicht abgelehnt
        # werden — das Modell stimmt ja, es fehlt nur ein Metadatum.
        import sqlite3

        pfad = tmp_path / "index.db"
        with IndexStore(pfad, embedder="a/b", dimensions=8):
            pass
        raw = sqlite3.connect(pfad)
        raw.execute("DELETE FROM meta WHERE key = 'embedder_profile'")
        raw.commit()
        raw.close()

        with IndexStore(pfad, embedder="a/b", dimensions=8, profile="default"):
            pass
        assert read_index_meta(pfad)["embedder_profile"] == "default"
