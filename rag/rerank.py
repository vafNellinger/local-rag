"""Treffer neu ordnen, bevor sie in den Prompt gehen.

Die Vektorsuche vergleicht zwei Vektoren, die unabhängig voneinander
entstanden sind — Anfrage und Chunk haben sich beim Embedden nie gesehen. Ein
Cross-Encoder liest beide zusammen und kann deshalb beurteilen, was ein
Bi-Encoder nur schätzen kann: ob dieser Chunk *diese* Frage beantwortet.

Der Preis ist die Rechenzeit. Ein Bi-Encoder embeddet einmal pro Chunk und
vergleicht dann billig; ein Cross-Encoder rechnet pro Paar neu. Deshalb die
Arbeitsteilung: die Vektorsuche holt breit (30 Kandidaten), der Reranker
ordnet und schneidet auf das, was in den Prompt passt (5).

Abschaltbar, und das ist keine Bequemlichkeit: auf einer Maschine ohne GPU
verdoppelt Reranking die Query-Latenz. ``platforms.toml`` schaltet ihn in der
Klasse ``cpu_only`` deshalb aus.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

from rag.detect import load_config
from rag.embed import (
    _onnx_model_dir,
    _onnx_providers,
    _onnx_session_options,
    resolve_device,
)
from rag.hfload import load_offline_first
from rag.store import SearchHit

logger = logging.getLogger(__name__)

# Wie viele Kandidaten die Vektorsuche liefert, damit der Reranker etwas zu
# ordnen hat. Sechsfach überholt: unter etwa dem Vierfachen bringt Reranking
# kaum etwas, weil der richtige Chunk dann meist schon oben steht.
DEFAULT_CANDIDATES = 30

# Wie viele Chunks nach dem Reranking in den Prompt gehen.
DEFAULT_TOP_K = 5

# Batch-Größe für die Paar-Bewertung. Kleiner als beim Embedden, weil jedes
# Paar Anfrage *und* Chunk enthält und damit doppelt so lang ist.
BATCH_SIZE = 8


class RerankError(RuntimeError):
    """Das Reranking-Modell ist nicht benutzbar."""


@dataclass(frozen=True)
class RerankerConfig:
    """Was ein Reranking-Modell ausmacht, aus platforms.toml gelesen."""

    model_id: str
    max_seq_length: int = 8192
    vram_estimate_mb: int = 0
    # "sentence-transformers" (CrossEncoder, Standard) oder "onnx" (ONNX
    # Runtime). "onnx" fällt sauber auf den CrossEncoder zurück, wenn das
    # exportierte Modell fehlt.
    engine: str = "sentence-transformers"


def load_reranker_config(profile: str = "default") -> RerankerConfig:
    """Reranker-Konfiguration aus platforms.toml lesen."""
    table = load_config().get("reranker", {})
    if profile not in table:
        available = ", ".join(sorted(table)) or "(keine)"
        raise RerankError(
            f"Reranker-Profil '{profile}' fehlt in der Konfiguration. "
            f"Vorhanden: {available}"
        )
    entry = table[profile]
    if "model_id" not in entry:
        raise RerankError(f"Reranker-Profil '{profile}' hat kein Feld 'model_id'")
    return RerankerConfig(
        model_id=str(entry["model_id"]),
        max_seq_length=int(entry.get("max_seq_length", 8192)),
        vram_estimate_mb=int(entry.get("vram_estimate_mb", 0)),
        engine=str(entry.get("engine", "sentence-transformers")),
    )


class _CrossEncoderEngine:
    """Der PyTorch-Weg über sentence-transformers CrossEncoder — die Vorgabe."""

    def __init__(self, config: RerankerConfig, device: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RerankError(
                "sentence-transformers fehlt: uv pip install -e '.[ingest]'"
            ) from exc
        # Offline zuerst, wie beim Embedder (siehe rag/hfload.py).
        self._model = load_offline_first(
            lambda offline: CrossEncoder(
                config.model_id,
                device=device,
                max_length=config.max_seq_length,
                local_files_only=offline,
            ),
            was=f"Reranker {config.model_id}",
        )

    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[float]:
        return [float(s) for s in self._model.predict(list(pairs), batch_size=batch_size)]


class _OnnxRerankerEngine:
    """ONNX Runtime für den Cross-Encoder: derselbe Score, GPU-portabel.

    bge-reranker-v2-m3 ist ein Sequence-Classification-Modell mit einem Logit
    pro (Frage, Passage)-Paar; ``CrossEncoder.predict`` schickt diesen Logit bei
    ``num_labels=1`` durch eine Sigmoid-Funktion. Diese Engine bildet das nach.
    """

    def __init__(self, config: RerankerConfig, device: str, onnx_dir) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._session = ort.InferenceSession(
            str(onnx_dir / "model.onnx"),
            sess_options=_onnx_session_options(),
            providers=_onnx_providers(device),
        )
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
        self.max_seq_length = config.max_seq_length

    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[float]:
        import numpy as np

        pairs = list(pairs)
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            stapel = pairs[start : start + batch_size]
            kodiert = self._tokenizer(
                [q for q, _ in stapel],
                [p for _, p in stapel],
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            feeds = {k: v for k, v in kodiert.items() if k in self._input_names}
            logits = self._session.run(None, feeds)[0].reshape(-1)
            scores.extend((1.0 / (1.0 + np.exp(-logits))).tolist())  # Sigmoid
        return scores


def _load_reranker_engine(config: RerankerConfig, device: str):
    """Reranker-Engine nach Konfiguration wählen, mit CrossEncoder-Fallback."""
    if config.engine == "onnx":
        onnx_dir = _onnx_model_dir(config.model_id)
        if (onnx_dir / "model.onnx").exists():
            try:
                return _OnnxRerankerEngine(config, device, onnx_dir)
            except Exception as exc:  # noqa: BLE001 — Grund wird geloggt
                logger.warning(
                    "ONNX-Reranker für %s nicht ladbar (%s) — CrossEncoder",
                    config.model_id, exc,
                )
        else:
            logger.warning(
                "ONNX-Reranker fehlt unter %s — CrossEncoder. "
                "Export: python tools/export_onnx.py",
                onnx_dir,
            )
    try:
        return _CrossEncoderEngine(config, device)
    except RerankError:
        raise
    except Exception as exc:
        raise RerankError(
            f"Reranker '{config.model_id}' konnte nicht geladen werden: {exc}"
        ) from exc


class Reranker:
    """Cross-Encoder, der Anfrage und Chunk zusammen liest.

    Lädt das Modell beim ersten Gebrauch. Eine Query, die keinen Reranker
    braucht (weil er abgeschaltet ist), zahlt die Ladezeit nicht.
    """

    def __init__(
        self,
        config: RerankerConfig | None = None,
        *,
        profile: str = "default",
        device: str = "auto",
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.config = config or load_reranker_config(profile)
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._engine = None

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    @property
    def engine(self):
        if self._engine is None:
            logger.debug(
                "Lade Reranker %s (%s) auf %s",
                self.config.model_id,
                self.config.engine,
                self.device,
            )
            self._engine = _load_reranker_engine(self.config, self.device)
        return self._engine

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """Treffer nach Relevanz für die Anfrage ordnen und auf ``top_k`` kürzen.

        Bewertet wird gegen ``embed_context`` — denselben Text mit
        Überschriften-Kontext, mit dem auch indiziert wurde. Ohne den Pfad
        müsste der Cross-Encoder einen Chunk beurteilen, dessen Thema nur in
        der Überschrift steht.
        """
        if not hits:
            return []
        if len(hits) == 1:
            return list(hits)

        pairs = [(query, _context_text(hit)) for hit in hits]
        try:
            scores = self.engine.predict(pairs, self.batch_size)
        except Exception as exc:
            raise RerankError(f"Reranking fehlgeschlagen: {exc}") from exc

        scored = [
            replace(hit, rerank_score=float(score))
            for hit, score in zip(hits, scores)
        ]
        # Absteigend: hohe Cross-Encoder-Punktzahl heißt relevant. Das ist die
        # umgekehrte Richtung zur Vektordistanz, wo klein gut ist — eine
        # Verwechslung hier dreht das Ranking um und fällt kaum auf.
        scored.sort(key=lambda hit: hit.rerank_score, reverse=True)
        top = scored[:top_k]

        if min_score <= 0:
            return top

        # Gemessen an einem kleinen Index: bei einer Frage zur Kündigungsfrist
        # bekamen die beiden treffenden Chunks 0,038 und 0,031, drei
        # sachfremde 0,005 bis 0,000 — und landeten trotzdem im Prompt, weil
        # top_k sie auffüllte. Die Schwelle wirft sie heraus, statt das
        # Kontextfenster mit Rauschen zu füllen.
        #
        # Der erste Treffer bleibt immer: fällt auch er durch, ist die Frage
        # mit diesem Index nicht beantwortbar, und das soll die Antwort sagen
        # dürfen, statt hier still zu einem leeren Ergebnis zu werden.
        kept = [hit for hit in top if (hit.rerank_score or 0.0) >= min_score]
        if not kept:
            return top[:1]
        if len(kept) < len(top):
            logger.debug(
                "%d von %d Treffern unter der Relevanzschwelle %.3f verworfen",
                len(top) - len(kept),
                len(top),
                min_score,
            )
        return kept


def _context_text(hit: SearchHit) -> str:
    """Chunk mit Überschriften-Pfad, wie beim Indizieren."""
    if not hit.heading_path:
        return hit.text
    return f"{hit.heading}\n\n{hit.text}"
