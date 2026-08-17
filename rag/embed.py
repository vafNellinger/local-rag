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
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.detect import CACHE_DIR, load_config
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
    # Welche Inferenz-Engine die Vektoren rechnet: "sentence-transformers"
    # (PyTorch, Standard) oder "onnx" (ONNX Runtime, herstellerübergreifende
    # GPU). "onnx" fällt sauber auf sentence-transformers zurück, wenn das
    # exportierte Modell fehlt — der Index bleibt derselbe, die Vektoren sind
    # bitgleich (verifiziert), nur der Rechenweg unterscheidet sich.
    engine: str = "sentence-transformers"

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
        engine=entry.get("engine", "sentence-transformers"),
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


# ─── Inferenz-Engines ────────────────────────────────────────────────────────
#
# Dieselbe Rolle wie die Vektor-Backends in rag/vectors.py: der Embedder kennt
# nur das schmale Protokoll (encode / count_tokens / max_seq_length /
# dimension), die eigentliche Rechnung liegt austauschbar dahinter.
# sentence-transformers ist die Vorgabe; ONNX Runtime rechnet dieselben Vektoren
# (Treue verifiziert: Cosine-Abweichung < 1e-7), aber herstellerübergreifend auf
# GPU und ohne den PyTorch-Ballast zur Laufzeit.


def _onnx_model_dir(model_id: str) -> Path:
    """Wo ``tools/export_onnx.py`` das ONNX-Modell dieses Embedders ablegt."""
    return CACHE_DIR / "onnx" / model_id.split("/")[-1]


def _onnx_providers(device: str) -> list[str]:
    """ExecutionProvider in fester Rangfolge, herstellerübergreifend.

    Auf einem reinen CPU-onnxruntime steht nur der CPU-Provider bereit; die
    GPU-Provider kommen mit onnxruntime-gpu (CUDA) bzw. onnxruntime-directml
    (Windows) hinzu, ohne dass hier etwas zu ändern wäre.
    """
    import onnxruntime as ort

    verfuegbar = set(ort.get_available_providers())
    bevorzugt: list[str] = []
    if device in {"cuda", "gpu", "mps", "auto"}:
        bevorzugt = [
            p
            for p in (
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
            )
            if p in verfuegbar
        ]
    return [*bevorzugt, "CPUExecutionProvider"]


def _physical_cores() -> int:
    """Physische Kerne — onnxruntimes CPU-Kernel ist mit Hyperthreads langsamer.

    Gemessen auf 24 logischen Kernen: 150 statt 210 ms/Chunk, wenn intra_op auf
    die 12 physischen Kerne begrenzt wird. Auf Linux exakt aus ``/proc/cpuinfo``;
    sonst die logischen halbieren (Hyperthreading angenommen). Nie unter 1.
    """
    try:
        paare: set[tuple[str, str]] = set()
        physical = core = None
        with open("/proc/cpuinfo", encoding="ascii") as f:
            for zeile in f:
                if zeile.startswith("physical id"):
                    physical = zeile.split(":", 1)[1].strip()
                elif zeile.startswith("core id"):
                    core = zeile.split(":", 1)[1].strip()
                elif not zeile.strip() and physical is not None and core is not None:
                    paare.add((physical, core))
                    physical = core = None
        if paare:
            return len(paare)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _onnx_session_options():
    """SessionOptions mit voller Graph-Optimierung und passender Thread-Zahl."""
    import onnxruntime as ort

    optionen = ort.SessionOptions()
    optionen.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    optionen.intra_op_num_threads = _physical_cores()
    return optionen


class _SentenceTransformersEngine:
    """Der PyTorch-Weg über sentence-transformers — die Vorgabe."""

    def __init__(self, config: EmbedderConfig, device: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "sentence-transformers fehlt: uv pip install -e '.[ingest]'"
            ) from exc

        model = load_offline_first(
            lambda offline: SentenceTransformer(
                config.model_id, device=device, local_files_only=offline
            ),
            was=config.model_id,
        )
        # Das Modell-Limit gewinnt gegen die Konfiguration: steht dort eine
        # größere Zahl als das Modell kann, würde still abgeschnitten.
        native = getattr(model, "max_seq_length", None)
        if native and native < config.max_seq_length:
            logger.warning(
                "%s kann %d Token, konfiguriert sind %d — es gilt %d",
                config.model_id, native, config.max_seq_length, native,
            )
        elif native:
            model.max_seq_length = config.max_seq_length
        self._model = model
        self.max_seq_length = int(native or config.max_seq_length)
        # sentence-transformers 5.x hat die Methode umbenannt; beide Namen
        # prüfen, damit keine FutureWarning auslöst.
        dimension_of = getattr(
            model, "get_embedding_dimension", None
        ) or model.get_sentence_embedding_dimension
        self.dimension = int(dimension_of())

    def encode(
        self, texts: Sequence[str], *, batch_size: int, progress: bool
    ) -> list[list[float]]:
        vektoren = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=progress,
            convert_to_numpy=True,
        )
        return [vektor.tolist() for vektor in vektoren]

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))


class _OnnxEngine:
    """ONNX Runtime: dieselben Vektoren, herstellerübergreifende GPU.

    bge-m3 poolt über das CLS-Token und normalisiert danach (siehe
    ``1_Pooling/config.json``: ``pooling_mode_cls_token=true``, dann
    ``2_Normalize``). Beides macht diese Engine von Hand, weil der ONNX-Graph
    nur den Transformer bis ``last_hidden_state`` umfasst.
    """

    def __init__(self, config: EmbedderConfig, device: str, onnx_dir: Path) -> None:
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
        self.dimension = config.dimensions

    def encode(
        self, texts: Sequence[str], *, batch_size: int, progress: bool
    ) -> list[list[float]]:
        import numpy as np

        texts = list(texts)
        vektoren: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            stapel = texts[start : start + batch_size]
            kodiert = self._tokenizer(
                stapel,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
            )
            feeds = {k: v for k, v in kodiert.items() if k in self._input_names}
            last_hidden = self._session.run(None, feeds)[0]
            cls = last_hidden[:, 0].astype(np.float32)  # CLS-Pooling
            normiert = cls / np.linalg.norm(cls, axis=1, keepdims=True)  # L2
            vektoren.extend(normiert.tolist())
        return vektoren

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def _load_engine(config: EmbedderConfig, device: str):
    """Engine nach Konfiguration wählen, mit sauberem Fallback.

    ``engine="onnx"`` verlangt ein vorab exportiertes Modell (siehe
    ``tools/export_onnx.py``). Fehlt es oder scheitert ONNX Runtime, wird auf
    sentence-transformers zurückgefallen — sichtbar im Protokoll, nie still,
    denn beide liefern denselben Index.
    """
    if config.engine == "onnx":
        onnx_dir = _onnx_model_dir(config.model_id)
        if (onnx_dir / "model.onnx").exists():
            try:
                return _OnnxEngine(config, device, onnx_dir)
            except Exception as exc:  # noqa: BLE001 — Grund wird geloggt
                logger.warning(
                    "ONNX-Engine für %s nicht ladbar (%s) — sentence-transformers",
                    config.model_id, exc,
                )
        else:
            logger.warning(
                "ONNX-Modell fehlt unter %s — sentence-transformers. "
                "Export: python tools/export_onnx.py",
                onnx_dir,
            )
    try:
        return _SentenceTransformersEngine(config, device)
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(
            f"Modell '{config.model_id}' konnte nicht geladen werden: {exc}"
        ) from exc


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
        self._engine = None
        self._counter = None

    # ─── Engine ──────────────────────────────────────────────────────────────

    @property
    def engine(self):
        """Die Inferenz-Engine, beim ersten Gebrauch geladen.

        Verzögert, weil das Modell gut 2 GB wiegt: ein ``rag ingest`` über ein
        unverändertes Verzeichnis soll die Ladezeit nicht zahlen.
        """
        if self._engine is None:
            logger.debug(
                "Lade %s (%s) auf %s (Batch %d)",
                self.config.model_id,
                self.config.engine,
                self.device,
                self.batch_size,
            )
            engine = _load_engine(self.config, self.device)
            # Das Modell muss die konfigurierte Dimension liefern, sonst passt
            # der Index nicht zur Anfrage — beide Engines gleich behandelt.
            if engine.dimension != self.config.dimensions:
                raise EmbeddingError(
                    f"{self.config.model_id} liefert {engine.dimension} "
                    f"Dimensionen, konfiguriert sind {self.config.dimensions} "
                    "— platforms.toml korrigieren, sonst passt der Index nicht"
                )
            self._engine = engine
        return self._engine

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    def token_counter(self):
        """Der Tokenizer des Modells als Zählfunktion für den Chunker.

        Damit zählt das Chunking mit demselben Vokabular, mit dem später
        embeddet wird. Der Zugriff lädt das Modell — wer nur schätzen will,
        nimmt ``chunk.estimate_tokens``.
        """
        if self._counter is None:
            self._counter = self.engine.count_tokens
        return self._counter

    # ─── Embedden ────────────────────────────────────────────────────────────

    def _encode(
        self, texts: Sequence[str], *, prefix: str, progress: bool
    ) -> list[list[float]]:
        if not texts:
            return []

        prepared = [f"{prefix}{text}" for text in texts] if prefix else list(texts)

        # Die Engine normalisiert selbst: der Index rechnet mit Cosine, und
        # normalisierte Vektoren machen die Distanz vergleichbar, ohne pro
        # Abfrage die Norm nachzuziehen.
        return self.engine.encode(
            prepared, batch_size=self.batch_size, progress=progress
        )

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
        limit = int(min(self.config.max_seq_length, self.engine.max_seq_length))
        threshold = int(limit * TRUNCATION_HEADROOM)
        return [i for i, text in enumerate(texts) if count(text) > threshold]
