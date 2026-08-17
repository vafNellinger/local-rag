"""Tests für die austauschbaren Inferenz-Engines (ONNX vs. sentence-transformers).

Zwei Ebenen: schnelle Unit-Tests für Engine-Wahl, Provider-Rangfolge und den
Fallback (ohne echtes Modell, über Mocks), und ein Treue-Integrationstest, der
das exportierte ONNX-Modell gegen sentence-transformers stellt und übersprungen
wird, wenn nichts exportiert ist (``python tools/export_onnx.py``).

Der Zweck ist Gleichheit: ONNX muss dieselben Vektoren bzw. Scores liefern wie
der PyTorch-Weg, sonst passt ein ONNX-gebauter Index nicht zu einer
sentence-transformers-Abfrage.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rag import embed as embed_mod
from rag import rerank as rerank_mod
from rag.embed import (
    EmbedderConfig,
    _load_engine,
    _onnx_model_dir,
    _onnx_providers,
    load_embedder_config,
)
from rag.rerank import RerankerConfig, _load_reranker_engine, load_reranker_config


class TestEngineKonfiguration:
    def test_default_engine_ist_sentence_transformers(self):
        assert load_embedder_config("default").engine == "sentence-transformers"
        assert load_reranker_config("default").engine == "sentence-transformers"

    def test_engine_wird_aus_der_konfiguration_gelesen(self, tmp_path):
        toml = tmp_path / "platforms.toml"
        toml.write_text(
            '[embedder.default]\nmodel_id = "a/b"\ndimensions = 8\n'
            'max_seq_length = 128\nengine = "onnx"\n'
        )
        assert load_embedder_config("default", config_path=toml).engine == "onnx"


class TestProviderWahl:
    def test_cpu_liefert_nur_cpu_provider(self):
        assert _onnx_providers("cpu") == ["CPUExecutionProvider"]

    def test_cpu_ist_immer_letzter_fallback(self):
        # Auch wenn eine GPU gewünscht ist, bleibt CPU als Auffanglösung hinten.
        assert _onnx_providers("cuda")[-1] == "CPUExecutionProvider"

    def test_modellverzeichnis_folgt_dem_basisnamen(self):
        pfad = _onnx_model_dir("BAAI/bge-m3")
        assert pfad.name == "bge-m3"
        assert pfad.parent.name == "onnx"


class _FakeEmbeddingEngine:
    def __init__(self, config, device):
        self.dimension = config.dimensions
        self.max_seq_length = config.max_seq_length

    def encode(self, texts, *, batch_size, progress):
        return [[0.0] * self.dimension for _ in texts]

    def count_tokens(self, text):
        return len(text.split())


class TestFallback:
    def test_onnx_faellt_auf_sentence_transformers_zurueck(
        self, tmp_path, monkeypatch, caplog
    ):
        # ONNX gewünscht, aber kein exportiertes Modell unter dem Cache-Pfad:
        # es muss auf sentence-transformers zurückfallen statt zu scheitern.
        monkeypatch.setattr(embed_mod, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(embed_mod, "_SentenceTransformersEngine", _FakeEmbeddingEngine)
        cfg = EmbedderConfig(
            model_id="BAAI/bge-m3",
            dimensions=8,
            max_seq_length=128,
            engine="onnx",
        )
        with caplog.at_level("WARNING"):
            engine = _load_engine(cfg, "cpu")
        assert isinstance(engine, _FakeEmbeddingEngine)
        assert any("ONNX-Modell fehlt" in r.message for r in caplog.records)

    def test_reranker_onnx_faellt_zurueck(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(embed_mod, "CACHE_DIR", tmp_path)

        class _FakeCE:
            def __init__(self, config, device):
                self.config = config

            def predict(self, pairs, batch_size):
                return [0.0 for _ in pairs]

        monkeypatch.setattr(rerank_mod, "_CrossEncoderEngine", _FakeCE)
        cfg = RerankerConfig(model_id="BAAI/bge-reranker-v2-m3", engine="onnx")
        with caplog.at_level("WARNING"):
            engine = _load_reranker_engine(cfg, "cpu")
        assert isinstance(engine, _FakeCE)
        assert any("ONNX-Reranker fehlt" in r.message for r in caplog.records)


def _onnx_vorhanden(basename: str) -> bool:
    return (_onnx_model_dir(basename) / "model.onnx").exists()


@pytest.mark.skipif(
    not _onnx_vorhanden("BAAI/bge-m3"),
    reason="ONNX-Modell nicht exportiert (python tools/export_onnx.py)",
)
class TestTreueIntegration:
    """Langsam: lädt echte Modelle. Belegt, dass ONNX == sentence-transformers."""

    def test_embedder_onnx_gleicht_sentence_transformers(self):
        import numpy as np

        cfg = load_embedder_config("default")
        onnx = embed_mod.Embedder(replace(cfg, engine="onnx"), device="cpu")
        st = embed_mod.Embedder(cfg, device="cpu")
        texts = ["Die Kündigungsfrist beträgt drei Monate.", "Ein zweiter Satz."]
        ov = onnx.embed_passages(texts)
        sv = st.embed_passages(texts)
        for o, s in zip(ov, sv):
            assert float(np.dot(o, s)) > 0.9999

    def test_reranker_onnx_gleicht_crossencoder(self):
        cfg = load_reranker_config("default")
        onnx = rerank_mod._OnnxRerankerEngine(cfg, "cpu", _onnx_model_dir(cfg.model_id))
        ce = rerank_mod._CrossEncoderEngine(cfg, "cpu")
        pairs = [
            ("Wie lang ist die Frist?", "Die Frist beträgt drei Monate."),
            ("Wie lang ist die Frist?", "Der Urlaub beträgt 30 Tage."),
        ]
        o = onnx.predict(pairs, 8)
        c = ce.predict(pairs, 8)
        assert max(abs(a - b) for a, b in zip(o, c)) < 1e-3
