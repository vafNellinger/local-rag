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
from rag.embed import resolve_device
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
    )


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
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RerankError(
                "sentence-transformers fehlt: uv pip install -e '.[ingest]'"
            ) from exc

        logger.debug("Lade Reranker %s auf %s", self.config.model_id, self.device)
        try:
            return CrossEncoder(
                self.config.model_id,
                device=self.device,
                max_length=self.config.max_seq_length,
            )
        except Exception as exc:
            raise RerankError(
                f"Reranker '{self.config.model_id}' konnte nicht geladen "
                f"werden: {exc}"
            ) from exc

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
            scores = self.model.predict(pairs, batch_size=self.batch_size)
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
