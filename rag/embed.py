"""Chunks → Vektoren, über sentence-transformers.

Das Modell steht in ``config/platforms.toml`` unter ``[embedder.*]``, nicht
hier. Der Code kennt nur die Rolle — dieselbe Trennung wie beim Generator.

Zwei Dinge, die man leicht übersieht und die beide still schiefgehen:

**Anfrage und Dokument sind nicht symmetrisch.** E5-Modelle erwarten
``query:`` und ``passage:`` als Präfix, bge-m3 erwartet nichts. Wer das
falsch macht, bekommt keine Fehlermeldung, nur schlechtere Treffer. Die
Präfixe stehen deshalb in der Konfiguration und werden hier angewandt.

**Der Tokenizer gehört dem Modell.** Beim Chunking muss dieselbe Zählung
gelten wie beim Embedden, sonst schneidet das Modell Chunks ab, die der
Chunker für passend hielt. ``token_counter()`` gibt den echten Zähler heraus,
sobald das Modell geladen ist.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.detect import load_config
from rag.hfload import load_offline_first

logger = logging.getLogger(__name__)

# Batch-Größen. Auf CPU ist der Speicher nicht das Problem, sondern dass zu
# große Batches die Latenz pro Fortschrittsschritt unnötig grob machen; auf
# GPU zahlt sich der größere Batch direkt in Durchsatz aus.
BATCH_SIZE_CPU = 16
BATCH_SIZE_GPU = 64

# Sicherheitsabstand zum Modell-Limit. Die Chunk-Größe ist in Token geschätzt
# oder gezählt, aber der Überschriften-Präfix und Sonderzeichen können sie
# nachträglich erhöhen. Wird das Limit trotzdem überschritten, schneidet
# sentence-transformers still ab — deshalb wird hier gewarnt.
TRUNCATION_HEADROOM = 0.95


class EmbeddingError(RuntimeError):
    """Das Embedding-Modell ist nicht benutzbar."""


@dataclass(frozen=True)
class EmbedderConfig:
    """Was ein Embedding-Modell ausmacht, aus platforms.toml gelesen."""

    model_id: str
    dimensions: int
    max_seq_length: int
    query_prefix: str = ""
    passage_prefix: str = ""
    vram_estimate_mb: int = 0

    @property
    def needs_prefix(self) -> bool:
        return bool(self.query_prefix or self.passage_prefix)


def load_embedder_config(
    profile: str = "default", *, config_path: Path | None = None
) -> EmbedderConfig:
    """Embedder-Konfiguration aus platforms.toml lesen."""
    config = load_config(config_path)
    table = config.get("embedder", {})
    if profile not in table:
        available = ", ".join(sorted(table)) or "(keine)"
        raise EmbeddingError(
            f"Embedder-Profil '{profile}' fehlt in der Konfiguration. "
            f"Vorhanden: {available}"
        )

    entry = table[profile]
    for required in ("model_id", "dimensions", "max_seq_length"):
        if required not in entry:
            raise EmbeddingError(
                f"Embedder-Profil '{profile}' hat kein Feld '{required}'"
            )

    return EmbedderConfig(
        model_id=entry["model_id"],
        dimensions=int(entry["dimensions"]),
        max_seq_length=int(entry["max_seq_length"]),
        query_prefix=entry.get("query_prefix", ""),
        passage_prefix=entry.get("passage_prefix", ""),
        vram_estimate_mb=int(entry.get("vram_estimate_mb", 0)),
    )


def _best_gpu() -> str | None:
    """Der von Torch nutzbare Beschleuniger, oder None."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None

    if torch.cuda.is_available():
        return "cuda"
    # MPS auf Apple-Silicon. Kein ROCm-Zweig: torch.cuda.is_available() ist
    # bei einem ROCm-Build ebenfalls True, die AMD-Karte kommt also über den
    # Zweig oben herein.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return None


def resolve_device(requested: str = "auto") -> str:
    """Gerätename für Torch bestimmen.

    Nimmt auch ``"gpu"`` an, weil ``platforms.toml`` in dieser Sprache
    schreibt — Torch kennt nur ``cuda`` und ``mps``. Ist die gewünschte GPU
    nicht da, wird auf CPU zurückgefallen *und gewarnt*: auf dieser Maschine
    ist genau das der Normalfall (CUDA-Build von Torch, AMD-iGPU unsichtbar),
    und ein stiller Fallback würde die Ursache verschleiern.
    """
    if requested in {"auto", "gpu"}:
        if gpu := _best_gpu():
            return gpu
        if requested == "gpu":
            logger.warning(
                "Gerät 'gpu' gewünscht, aber Torch sieht keinen Beschleuniger "
                "— Embedding läuft auf der CPU"
            )
        return "cpu"

    if requested in {"cuda", "mps"} and _best_gpu() is None:
        logger.warning(
            "Gerät '%s' gewünscht, aber Torch sieht keinen Beschleuniger "
            "— Embedding läuft auf der CPU",
            requested,
        )
        return "cpu"

    return requested


class Embedder:
    """Lädt das Modell beim ersten Gebrauch und embeddet Text.

    Der verzögerte Ladevorgang ist nicht Kosmetik: das Modell wiegt gut 2 GB
    und braucht zweistellige Sekunden. Ein ``rag ingest`` über ein
    Verzeichnis, in dem sich nichts geändert hat, soll das nicht zahlen.
    """

    def __init__(
        self,
        config: EmbedderConfig | None = None,
        *,
        profile: str = "default",
        device: str = "auto",
        batch_size: int | None = None,
    ) -> None:
        self.config = config or load_embedder_config(profile)
        self.device = resolve_device(device)
        self.batch_size = batch_size or (
            BATCH_SIZE_CPU if self.device == "cpu" else BATCH_SIZE_GPU
        )
        self._model = None
        self._counter = None

    # ─── Modell ──────────────────────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "sentence-transformers fehlt: uv pip install -e '.[ingest]'"
            ) from exc

        logger.debug(
            "Lade %s auf %s (Batch %d)",
            self.config.model_id,
            self.device,
            self.batch_size,
        )
        # Offline zuerst (siehe rag/hfload.py): ein zwischengespeichertes Modell
        # lädt ohne Netz, ohne HF-Roundtrip und ohne die Unauthenticated-Warnung.
        # Heruntergeladen wird nur, was fehlt.
        try:
            model = load_offline_first(
                lambda offline: SentenceTransformer(
                    self.config.model_id,
                    device=self.device,
                    local_files_only=offline,
                ),
                was=self.config.model_id,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Modell '{self.config.model_id}' konnte nicht geladen "
                f"werden: {exc}"
            ) from exc

        # Das Modell-Limit gewinnt gegen die Konfiguration: steht dort eine
        # größere Zahl als das Modell kann, würde still abgeschnitten.
        native = getattr(model, "max_seq_length", None)
        if native and native < self.config.max_seq_length:
            logger.warning(
                "%s kann %d Token, konfiguriert sind %d — es gilt %d",
                self.config.model_id,
                native,
                self.config.max_seq_length,
                native,
            )
        elif native:
            model.max_seq_length = self.config.max_seq_length

        # sentence-transformers 5.x hat die Methode umbenannt; der alte Name
        # lebt weiter, warnt aber. Beide Namen prüfen, damit weder eine ältere
        # noch eine neuere Installation eine FutureWarning auslöst.
        dimension_of = getattr(
            model, "get_embedding_dimension", None
        ) or model.get_sentence_embedding_dimension
        actual = dimension_of()
        if actual != self.config.dimensions:
            raise EmbeddingError(
                f"{self.config.model_id} liefert {actual} Dimensionen, "
                f"konfiguriert sind {self.config.dimensions} — "
                "platforms.toml korrigieren, sonst passt der Index nicht"
            )
        return model

    def token_counter(self):
        """Der Tokenizer des Modells als Zählfunktion für den Chunker.

        Damit zählt das Chunking mit demselben Vokabular, mit dem später
        embeddet wird. Der Zugriff lädt das Modell — wer nur schätzen will,
        nimmt ``chunk.estimate_tokens``.
        """
        if self._counter is None:
            tokenizer = self.model.tokenizer

            def count(text: str) -> int:
                return len(tokenizer.encode(text, add_special_tokens=False))

            self._counter = count
        return self._counter

    # ─── Embedden ────────────────────────────────────────────────────────────

    def _encode(
        self, texts: Sequence[str], *, prefix: str, progress: bool
    ) -> list[list[float]]:
        if not texts:
            return []

        prepared = [f"{prefix}{text}" for text in texts] if prefix else list(texts)

        # normalize_embeddings: der Index rechnet mit Cosine, und normalisierte
        # Vektoren machen die Distanz vergleichbar, ohne pro Abfrage die Norm
        # nachzuziehen.
        vectors = self.model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=progress,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_passages(
        self, texts: Sequence[str], *, progress: bool = False
    ) -> list[list[float]]:
        """Dokumentseite: Chunks, die in den Index gehen."""
        return self._encode(
            texts, prefix=self.config.passage_prefix, progress=progress
        )

    def embed_query(self, text: str) -> list[float]:
        """Anfrageseite. Getrennt, weil das Präfix ein anderes ist."""
        vectors = self._encode([text], prefix=self.config.query_prefix, progress=False)
        return vectors[0]

    def check_lengths(self, texts: Sequence[str]) -> list[int]:
        """Indizes der Texte, die das Modell abschneiden würde.

        Existiert, weil das Abschneiden sonst unsichtbar bleibt: das Modell
        liefert brav einen Vektor, nur eben für den halben Chunk.
        """
        count = self.token_counter()
        limit = int(min(self.config.max_seq_length, self.model.max_seq_length))
        threshold = int(limit * TRUNCATION_HEADROOM)
        return [i for i, text in enumerate(texts) if count(text) > threshold]
