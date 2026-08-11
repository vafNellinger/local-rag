"""Die vollständige RAG-Kette als ein Objekt.

Existiert, weil CLI und Oberfläche dieselbe Kette brauchen und dieselben
Modelle nicht zweimal laden sollen: Index, Embedder, Reranker und Generator
hängen alle an derselben Plattformentscheidung, und alle vier sind teuer.

``RagPipeline`` hält diesen Zustand und lädt jeden Teil erst, wenn er
gebraucht wird. Eine Suche ohne Antwort lädt keinen Generator; eine Antwort
ohne Reranking lädt keinen Cross-Encoder. In der Oberfläche ist das der
Unterschied zwischen einem benutzbaren Programm und dreißig Sekunden
Startbildschirm.

Die Einstellungen kommen aus ``platforms.toml`` über die erkannte Plattform,
lassen sich aber einzeln überschreiben — genau das, was die Settings-Seite
der Oberfläche braucht.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from rag.chunk import TARGET_TOKENS
from rag.detect import Platform, WhichllmError, detect_local, load_config
from rag.embed import Embedder, EmbeddingError, load_embedder_config, resolve_device
from rag.generate import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    NO_CONTEXT_ANSWER,
    Answer,
    GenerationError,
    Generator,
    RouteDecision,
    Turn,
    build_prompt,
    build_route_prompt,
    build_sources,
    gguf_path,
    parse_route,
    supports_gpu_offload,
)
from rag.ingest import IngestReport, ingest_paths
from rag.rerank import (
    DEFAULT_CANDIDATES,
    DEFAULT_TOP_K,
    RerankError,
    Reranker,
    load_reranker_config,
)
from rag.resolve import (
    PipelinePlan,
    ResolutionError,
    resolve_pipeline,
    resolve_vector_backend,
)
from rag.store import (
    DEFAULT_INDEX_NAME,
    IndexStore,
    SearchHit,
    StoreError,
    read_index_documents,
    read_index_meta,
)
from rag.vectors import DEFAULT_BACKEND, VectorBackendError, clear_vectors

logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = Path.home() / ".cache" / "local-rag" / DEFAULT_INDEX_NAME

# Einstellungen liegen neben dem Index. Anders als der Index sind sie nicht
# neu erzeugbar — sie sind die einzige Datei hier, die echte Nutzereingabe
# enthält.
SETTINGS_PATH = Path.home() / ".config" / "local-rag" / "settings.json"

# Wie viele GPU-Layer llama.cpp auslagert, wenn der Plan "gpu" sagt. -1 heißt
# "alle". Auf CPU bleibt es bei 0.
ALL_GPU_LAYERS = -1


class PipelineError(RuntimeError):
    """Die Kette ist nicht benutzbar."""


@dataclass
class Settings:
    """Alle Regler an einem Ort.

    Die Vorgaben hier sind die neutralen; ``Settings.for_platform()`` füllt
    sie aus ``platforms.toml``. Was der Anwender in der Oberfläche verstellt,
    überschreibt beides.
    """

    index_path: Path = field(default_factory=lambda: DEFAULT_INDEX_PATH)

    embedder_profile: str = "default"
    embedder_device: str = "auto"

    # Wo die Vektoren liegen. Gehört zum Index, nicht zur Maschine: ein
    # bestehender Index bringt seine Wahl im Metadatum mit und übersteuert
    # diesen Wert, sonst ließe sich ein Index nach dem Umstellen nicht mehr
    # öffnen.
    vector_backend: str = DEFAULT_BACKEND
    vector_backend_options: dict = field(default_factory=dict)

    reranker_enabled: bool = True
    reranker_profile: str = "default"
    reranker_device: str = "auto"

    # Retrieval-Breite: die Vektorsuche holt ``candidates``, der Reranker
    # schneidet auf ``top_k``. Ohne Reranker wird direkt auf top_k gekürzt.
    candidates: int = DEFAULT_CANDIDATES
    top_k: int = DEFAULT_TOP_K

    # Mindest-Relevanz nach dem Reranking. 0 heißt: nicht filtern, top_k
    # allein entscheidet. Über 0 fallen sachfremde Chunks aus dem Prompt, die
    # top_k sonst auffüllt — gemessen lagen treffende Chunks bei 0,03 und
    # sachfremde bei 0,001. Die Skala hängt am Modell, deshalb ist die
    # Vorgabe aus und der Wert einstellbar; die Oberfläche zeigt die Punkte
    # an, damit man ihn an eigenen Dokumenten wählen kann.
    min_rerank_score: float = 0.0

    chunk_target_tokens: int = TARGET_TOKENS

    generator_enabled: bool = True
    generator_context_length: int = 8192
    generator_gpu_layers: int = 0
    generator_threads: int | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    # Nur zur Anzeige: woher die Werte kommen.
    platform_class: str | None = None
    platform_label: str | None = None

    @classmethod
    def for_platform(
        cls, platform: Platform | None = None, *, config: dict | None = None
    ) -> Settings:
        """Vorgaben aus der erkannten Plattform und ``platforms.toml``.

        Bewusst ohne ``resolve_pipeline()``: das löst den Generator über
        whichllm auf und ist für ein Settings-Objekt zu teuer. Der Generator
        wird erst aufgelöst, wenn er wirklich gebraucht wird.
        """
        detected = platform or detect_local()
        cfg = config or load_config()
        class_config = cfg.get("platform_class", {}).get(detected.platform_class, {})

        generator_device = str(class_config.get("generator_device", "cpu"))
        context = _parse_context(class_config.get("context_length", "8k"))
        vector_backend, vector_options = resolve_vector_backend(cfg, class_config)

        return cls(
            vector_backend=vector_backend,
            vector_backend_options=vector_options,
            embedder_profile=str(class_config.get("embedder_profile", "default")),
            embedder_device=str(class_config.get("embedder_device", "cpu")),
            reranker_enabled=bool(class_config.get("reranker_enabled", True)),
            reranker_profile=str(class_config.get("reranker_profile", "default")),
            reranker_device=str(class_config.get("reranker_device", "cpu")),
            generator_context_length=context,
            generator_gpu_layers=(
                ALL_GPU_LAYERS if generator_device == "gpu" else 0
            ),
            platform_class=detected.platform_class,
            platform_label=detected.describe(),
        )

    def as_dict(self) -> dict:
        data = asdict(self)
        data["index_path"] = str(self.index_path)
        return data

    def save(self, path: Path | None = None) -> Path:
        """Einstellungen als JSON ablegen.

        Ohne das wären in der Oberfläche verstellte Werte nach jedem Neustart
        weg. ``platform_class`` und ``platform_label`` werden mitgeschrieben,
        aber beim Laden verworfen — sie beschreiben die Maschine, nicht den
        Wunsch des Anwenders, und die Maschine kann eine andere sein.
        """
        target = path or SETTINGS_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(
        cls, path: Path | None = None, *, fallback: Settings | None = None
    ) -> Settings:
        """Gespeicherte Einstellungen über die Plattformvorgaben legen.

        Die Plattform liefert die Grundlage, die Datei überschreibt nur, was
        wirklich darin steht. So wirkt ein neues Feld mit seinem Vorgabewert,
        statt an einer alten Datei zu scheitern.
        """
        base = fallback if fallback is not None else cls.for_platform()
        source = path or SETTINGS_PATH
        if not source.exists():
            return base

        try:
            stored = json.loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Einstellungen nicht lesbar (%s), nutze Vorgaben", exc)
            return base

        known = set(asdict(base))
        # Die Plattformfelder beschreiben die erkannte Maschine — aus der
        # Datei übernommen wären sie eine Behauptung über fremde Hardware.
        ignored = {"platform_class", "platform_label"}
        changes = {
            key: value
            for key, value in stored.items()
            if key in known and key not in ignored
        }
        if "index_path" in changes:
            changes["index_path"] = Path(changes["index_path"])

        return replace(base, **changes)


def _parse_context(value: str | int) -> int:
    """``"32k"`` → 32768. Dieselbe Schreibweise wie in platforms.toml."""
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1024)
    return int(text)


@dataclass
class ModelStatus:
    """Was die Oberfläche über eine Rolle anzeigen muss."""

    role: str
    model_id: str
    device: str
    source: str
    loaded: bool
    available: bool = True
    # Bei GGUF-Modellen: liegt die Datei schon lokal?
    detail: str | None = None


class RagPipeline:
    """Index, Embedder, Reranker und Generator als eine Einheit."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._store: IndexStore | None = None
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None
        self._generator: Generator | None = None
        self._plan: PipelinePlan | None = None

    # ─── Lebenszyklus ────────────────────────────────────────────────────────

    @classmethod
    def from_platform(cls) -> RagPipeline:
        """Kette mit den Vorgaben der erkannten Plattform."""
        return cls(Settings.for_platform())

    def __enter__(self) -> RagPipeline:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
        # Der Generator wird ausdrücklich geschlossen: llama.cpp räumt im
        # Destruktor auf, und der läuft beim Interpreter-Ende zu spät.
        if self._generator is not None:
            self._generator.close()
            self._generator = None
        # Embedder und Reranker sind reine Python-Objekte, die der Garbage
        # Collector freigibt.
        self._embedder = None
        self._reranker = None

    def apply(self, **changes) -> RagPipeline:
        """Einstellungen ändern und betroffene Modelle verwerfen.

        Die Oberfläche ruft das beim Speichern der Settings. Ein geänderter
        Embedder muss den Index neu öffnen, weil Modell und Index
        zusammengehören; eine geänderte Temperatur darf nichts entladen.
        """
        unknown = set(changes) - set(self.settings.as_dict())
        if unknown:
            raise PipelineError(f"Unbekannte Einstellung: {', '.join(sorted(unknown))}")

        previous = self.settings
        self.settings = replace(self.settings, **changes)

        if (
            self.settings.embedder_profile != previous.embedder_profile
            or self.settings.embedder_device != previous.embedder_device
            or self.settings.index_path != previous.index_path
        ):
            if self._store is not None:
                self._store.close()
            self._store = None
            self._embedder = None

        if (
            self.settings.reranker_profile != previous.reranker_profile
            or self.settings.reranker_device != previous.reranker_device
        ):
            self._reranker = None

        if (
            self.settings.generator_context_length
            != previous.generator_context_length
            or self.settings.generator_gpu_layers != previous.generator_gpu_layers
            or self.settings.generator_threads != previous.generator_threads
        ):
            self._generator = None

        return self

    # ─── Bausteine, je einzeln beim ersten Bedarf ────────────────────────────

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            profile = self._effective_embedder_profile()
            config = load_embedder_config(profile)
            self._embedder = Embedder(
                config, device=self.settings.embedder_device
            )
        return self._embedder

    def profile_conflict(self) -> tuple[str, str] | None:
        """``(im Index, gewünscht)``, falls beide auseinanderlaufen.

        Existiert, weil der Index gegen die Einstellung gewinnt und das nicht
        stillschweigend passieren darf: wer in der Oberfläche ein anderes
        Modell wählt und „Gespeichert“ bestätigt bekommt, würde sonst weiter
        mit dem alten suchen, ohne es zu merken. Ein Wechsel verlangt einen
        Neuaufbau — ``rebuild_index()`` erledigt ihn.
        """
        stored = read_index_meta(self.settings.index_path).get("embedder_profile")
        if stored and stored != self.settings.embedder_profile:
            return stored, self.settings.embedder_profile
        return None

    def _effective_embedder_profile(self) -> str:
        """Das Profil des Index gewinnt gegen die Einstellung.

        Sonst würde ein Plattformwechsel den bestehenden Index bei der ersten
        Suche unbrauchbar machen — gesucht werden muss mit dem Modell, mit dem
        indiziert wurde. Der Konflikt wird gemeldet und nicht verschwiegen;
        ``profile_conflict()`` macht ihn für die Oberfläche greifbar.
        """
        if conflict := self.profile_conflict():
            stored, requested = conflict
            logger.warning(
                "Index wurde mit Profil '%s' gebaut, eingestellt ist '%s' — "
                "es gilt '%s'. Für den Wechsel muss der Index neu aufgebaut "
                "werden.",
                stored,
                requested,
                stored,
            )
            return stored
        return self.settings.embedder_profile

    def backend_conflict(self) -> tuple[str, str] | None:
        """``(im Index, gewünscht)`` für das Vektor-Backend, falls verschieden.

        Gegenstück zu ``profile_conflict()`` und aus demselben Grund da: wer in
        der Oberfläche LanceDB wählt, würde sonst weiter gegen die sqlite-vec-
        Vektoren des bestehenden Index suchen, ohne es zu merken.
        """
        stored = read_index_meta(self.settings.index_path).get("vector_backend")
        if stored and stored != self.settings.vector_backend:
            return stored, self.settings.vector_backend
        return None

    def _effective_vector_backend(self) -> str:
        """Das Backend des Index gewinnt gegen die Einstellung.

        Ohne diese Regel würde ein Umstellen den bestehenden Index nicht nur
        unbrauchbar machen, sondern auf einen leeren Vektorspeicher zeigen —
        die Suche liefe fehlerfrei und fände nichts.
        """
        if conflict := self.backend_conflict():
            stored, requested = conflict
            logger.warning(
                "Index hat Vektor-Backend '%s', eingestellt ist '%s' — es "
                "gilt '%s'. Für den Wechsel muss der Index neu aufgebaut "
                "werden.",
                stored,
                requested,
                stored,
            )
            return stored
        return self.settings.vector_backend

    def rebuild_index(
        self,
        *,
        progress=None,
        extra_paths: Iterable[str | Path] = (),
    ) -> IngestReport:
        """Index mit dem eingestellten Profil neu aufbauen.

        Liest die Dateiliste aus dem alten Index, löscht ihn und nimmt alles
        erneut auf. Die Extraktion kommt dabei aus dem Cache, sofern die
        Dateien unverändert sind — genau dafür liegt er außerhalb des Index.
        Dateien, die inzwischen verschwunden sind, fallen still heraus.
        """
        # Vor dem Löschen lesen: danach ist nicht mehr feststellbar, welches
        # Backend der alte Index benutzt hat, und sein Nebenverzeichnis würde
        # als Waise zurückbleiben.
        altes_meta = read_index_meta(self.settings.index_path)
        altes_backend = altes_meta.get("vector_backend", DEFAULT_BACKEND)
        # Für das Aufräumen einer Server-Sammlung: der Qdrant-Client verlangt
        # die Vektorgröße beim Anlegen, auch wenn wir nur löschen wollen.
        # Notfalls die eingestellte, sonst scheitert das Aufräumen an einem
        # fehlenden Metadatum.
        alte_dimensionen = int(
            altes_meta.get("dimensions") or load_embedder_config(
                self.settings.embedder_profile
            ).dimensions
        )

        known = [Path(p) for p in read_index_documents(self.settings.index_path)]
        vorhanden = [p for p in known if p.exists()]
        if len(vorhanden) < len(known):
            logger.info(
                "%d von %d Dateien aus dem alten Index sind verschwunden",
                len(known) - len(vorhanden),
                len(known),
            )

        paths = vorhanden + [Path(p) for p in extra_paths]

        # Store schließen, bevor die Datei verschwindet.
        if self._store is not None:
            self._store.close()
            self._store = None
        self._embedder = None

        for leftover in self.settings.index_path.parent.glob(
            self.settings.index_path.name + "*"
        ):
            # Auch -wal und -shm: eine zurückbleibende WAL-Datei würde beim
            # nächsten Öffnen alte Daten wiederherstellen.
            try:
                leftover.unlink()
            except OSError as exc:  # pragma: no cover
                logger.warning("%s nicht löschbar: %s", leftover, exc)

        # Die Vektoren fallen nicht unter das Glob oben: aus index.db wird
        # index.lance, nicht index.db-lance, und eine Qdrant-Sammlung auf einem
        # Server liegt gar nicht im Dateisystem. Beide Backends aufräumen — das
        # alte, damit nichts zurückbleibt, das neue, weil ein Rest von einem
        # früheren Lauf sonst mit dem frischen Index verschmolzen würde.
        for backend in {altes_backend, self.settings.vector_backend}:
            try:
                ergebnis = clear_vectors(
                    self.settings.index_path,
                    backend,
                    dimensions=alte_dimensionen,
                    options=self.settings.vector_backend_options,
                )
                logger.info("Vektoren (%s): %s", backend, ergebnis)
            except (VectorBackendError, OSError) as exc:
                # Kein Abbruch: der Neuaufbau soll auch gelingen, wenn das alte
                # Backend nicht mehr erreichbar ist. Der Rest ist dann eine
                # verwaiste Sammlung, die niemand mehr abfragt — sichtbar im
                # Protokoll statt stillschweigend.
                logger.warning(
                    "Vektoren von '%s' nicht aufräumbar: %s", backend, exc
                )

        logger.info(
            "Index wird mit Profil '%s' neu aufgebaut, %d Datei(en)",
            self.settings.embedder_profile,
            len(paths),
        )
        if not paths:
            return IngestReport()
        return self.ingest(paths, progress=progress)

    @property
    def store(self) -> IndexStore:
        if self._store is None:
            profile = self._effective_embedder_profile()
            config = load_embedder_config(profile)
            self._store = IndexStore(
                self.settings.index_path,
                embedder=config.model_id,
                dimensions=config.dimensions,
                profile=profile,
                vector_backend=self._effective_vector_backend(),
                backend_options=self.settings.vector_backend_options,
            ).open()
        return self._store

    @property
    def reranker(self) -> Reranker | None:
        if not self.settings.reranker_enabled:
            return None
        if self._reranker is None:
            self._reranker = Reranker(
                profile=self.settings.reranker_profile,
                device=self.settings.reranker_device,
            )
        return self._reranker

    @property
    def plan(self) -> PipelinePlan:
        """Der Modellplan, inklusive whichllm-Auflösung des Generators.

        Teuer beim ersten Aufruf (bis zu drei Minuten, danach 24 h gecacht),
        deshalb ausdrücklich nur hier und nicht in ``Settings``.
        """
        if self._plan is None:
            self._plan = resolve_pipeline(detect_local())
        return self._plan

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            spec = self.plan.generator
            if not spec.artifact_repo_id or not spec.artifact_filename:
                raise PipelineError(
                    f"whichllm nennt für {spec.model_id} keine GGUF-Datei — "
                    "Quantisierung muss manuell gewählt werden"
                )
            path = gguf_path(spec.artifact_repo_id, spec.artifact_filename)
            if path is None:
                raise PipelineError(
                    f"{spec.artifact_filename} liegt nicht lokal. "
                    "Mit 'rag pull' herunterladen (mehrere Gigabyte)."
                )
            self._generator = Generator(
                path,
                context_length=self.settings.generator_context_length,
                gpu_layers=self.settings.generator_gpu_layers,
                threads=self.settings.generator_threads,
            )
        return self._generator

    # ─── Anzeige ─────────────────────────────────────────────────────────────

    def model_status(self, *, resolve_generator: bool = False) -> list[ModelStatus]:
        """Welche Modelle die Kette benutzt und was davon geladen ist.

        ``resolve_generator=False`` lässt whichllm in Ruhe und meldet den
        Generator als "nicht aufgelöst" — die Oberfläche kann so sofort etwas
        anzeigen und den teuren Scan auf Knopfdruck nachholen.
        """
        rows: list[ModelStatus] = []

        embedder_profile = self._effective_embedder_profile()
        try:
            embedder_config = load_embedder_config(embedder_profile)
            rows.append(
                ModelStatus(
                    role="embedder",
                    model_id=embedder_config.model_id,
                    device=resolve_device(self.settings.embedder_device),
                    source=f"config:{embedder_profile}",
                    loaded=self._embedder is not None
                    and self._embedder.is_loaded,
                )
            )
        except EmbeddingError as exc:
            rows.append(
                ModelStatus(
                    role="embedder",
                    model_id="—",
                    device="—",
                    source="Fehler",
                    loaded=False,
                    available=False,
                    detail=str(exc),
                )
            )

        if self.settings.reranker_enabled:
            try:
                reranker_config = load_reranker_config(self.settings.reranker_profile)
                rows.append(
                    ModelStatus(
                        role="reranker",
                        model_id=reranker_config.model_id,
                        device=resolve_device(self.settings.reranker_device),
                        source=f"config:{self.settings.reranker_profile}",
                        loaded=self._reranker is not None
                        and self._reranker.is_loaded,
                    )
                )
            except RerankError as exc:
                rows.append(
                    ModelStatus(
                        role="reranker",
                        model_id="—",
                        device="—",
                        source="Fehler",
                        loaded=False,
                        available=False,
                        detail=str(exc),
                    )
                )
        else:
            rows.append(
                ModelStatus(
                    role="reranker",
                    model_id="— abgeschaltet",
                    device="—",
                    source=f"Klasse {self.settings.platform_class or '?'}",
                    loaded=False,
                    available=False,
                    detail="Reranking verdoppelt die Query-Latenz auf CPU",
                )
            )

        rows.append(self._generator_status(resolve_generator=resolve_generator))
        return rows

    def _generator_status(self, *, resolve_generator: bool) -> ModelStatus:
        if not self.settings.generator_enabled:
            return ModelStatus(
                role="generator",
                model_id="— abgeschaltet",
                device="—",
                source="Einstellung",
                loaded=False,
                available=False,
                detail="Suche liefert Treffer ohne formulierte Antwort",
            )

        if not resolve_generator and self._plan is None:
            return ModelStatus(
                role="generator",
                model_id="— nicht aufgelöst",
                device="—",
                source="whichllm",
                loaded=False,
                available=False,
                detail="Modell-Scan noch nicht gelaufen",
            )

        try:
            spec = self.plan.generator
        except (WhichllmError, ResolutionError) as exc:
            return ModelStatus(
                role="generator",
                model_id="—",
                device="—",
                source="Fehler",
                loaded=False,
                available=False,
                detail=str(exc),
            )

        local = None
        if spec.artifact_repo_id and spec.artifact_filename:
            local = gguf_path(spec.artifact_repo_id, spec.artifact_filename)

        # Das *tatsächliche* Gerät, nicht das gewünschte: die Plattformklasse
        # kann "gpu" verlangen, während dieser llama.cpp-Build nur ein
        # CPU-Backend hat. Eine Anzeige, die dann "gpu" behauptet, ist
        # schlimmer als keine — man sucht die Langsamkeit an der falschen
        # Stelle. Hier die stille Prüfung, weil die Anzeige bei jedem
        # Seitenaufruf läuft; gewarnt wird beim Laden.
        wants_gpu = bool(self.settings.generator_gpu_layers)
        can_gpu = supports_gpu_offload()

        if local:
            detail = f"{spec.quant_type}, {spec.artifact_filename}"
            if wants_gpu and not can_gpu:
                detail += (
                    " — GPU verlangt, aber dieser llama.cpp-Build hat nur ein "
                    "CPU-Backend"
                )
        else:
            detail = f"{spec.quant_type} — GGUF nicht lokal, 'rag pull' nötig"

        return ModelStatus(
            role="generator",
            model_id=spec.model_id,
            device="gpu" if (wants_gpu and can_gpu) else "cpu",
            source="whichllm",
            loaded=self._generator is not None and self._generator.is_loaded,
            available=local is not None,
            detail=detail,
        )

    def index_stats(self) -> dict:
        """Kennzahlen des Index, oder ein leeres Bild ohne Index.

        Ein vorhandener, aber nicht öffenbarer Index — falsche Schema-Version,
        fehlende Backend-Bibliothek — liefert hier ``error`` statt einer
        Ausnahme. Die Oberfläche ruft das beim Aufbau jeder Seite auf; eine
        Ausnahme wäre dort eine 500-Seite ohne Hinweis, was zu tun ist. Genau
        das ist beim Sprung auf Schema-Version 2 passiert.
        """
        leer = {
            "documents": 0,
            "chunks": 0,
            "vectors": 0,
            "tokens": 0,
            "path": str(self.settings.index_path),
            "exists": False,
        }
        if not self.settings.index_path.exists():
            return leer

        try:
            return {**self.store.stats(), "exists": True}
        except StoreError as exc:
            logger.warning("Index nicht lesbar: %s", exc)
            return {**leer, "exists": True, "error": str(exc)}

    # ─── Arbeit ──────────────────────────────────────────────────────────────

    def ingest(self, paths: Iterable[str | Path], **kwargs) -> IngestReport:
        """Dateien in den Index aufnehmen."""
        return ingest_paths(
            paths,
            store=self.store,
            embedder=self.embedder,
            target_tokens=self.settings.chunk_target_tokens,
            **kwargs,
        )

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[SearchHit]:
        """Suchen und, wenn eingeschaltet, neu ordnen.

        Ohne Reranker wird direkt auf ``top_k`` gekürzt — die breite Suche
        wäre sonst nur Aufwand ohne Nutzen, weil niemand die Kandidaten
        sortiert.
        """
        limit = top_k or self.settings.top_k
        reranker = self.reranker

        width = self.settings.candidates if reranker else limit
        hits = self.store.search(self.embedder.embed_query(question), limit=width)

        if not hits or reranker is None:
            return hits[:limit]
        return reranker.rerank(
            question,
            hits,
            top_k=limit,
            min_score=self.settings.min_rerank_score,
        )

    @staticmethod
    def reusable_hits(history: Sequence[Turn]) -> list[SearchHit]:
        """Die Quellen des ganzen Verlaufs, dedupliziert, jüngste zuerst.

        Grundlage für die Wiederverwendung: eine Folgefrage darf sich auf
        alles beziehen, was das Gespräch schon geholt hat, nicht nur auf die
        letzte Antwort. Dedupliziert über ``chunk_id`` — dieselbe Stelle taucht
        über mehrere Turns sonst mehrfach auf. Jüngste zuerst, damit bei
        knappem Budget das zuletzt Besprochene erhalten bleibt.
        """
        gesehen: set[int] = set()
        hits: list[SearchHit] = []
        for turn in reversed(list(history)):
            for source in turn.sources:
                if source.hit.chunk_id not in gesehen:
                    gesehen.add(source.hit.chunk_id)
                    hits.append(source.hit)
        return hits

    def plan_retrieval(
        self, question: str, history: Sequence[Turn]
    ) -> tuple[RouteDecision, list[SearchHit]]:
        """Entscheiden, ob neu gesucht wird — und wenn ja, womit.

        Ohne Verlauf gibt es nichts wiederzuverwenden und nichts aufzulösen:
        die Frage geht unverändert in die Suche, ohne Generatoraufruf. Das ist
        der Normalfall der ersten Frage und soll nichts kosten.

        Mit Verlauf entscheidet der Router in *einem* Aufruf: reichen die
        vorliegenden Quellen (``reuse``), oder braucht es eine neue Suche — dann
        mit der umgeschriebenen Query. Schlägt der Aufruf fehl, wird gesucht,
        mit der Originalfrage; ein kaputter Router darf die Antwort nicht
        verhindern.
        """
        reusable = self.reusable_hits(history)
        if not history:
            return RouteDecision(reuse=False, query=question), reusable

        zitate = [hit.citation for hit in reusable]
        try:
            text, _, _ = self.generator.complete(
                build_route_prompt(history, question, zitate),
                # Kurz: ein Marker oder eine Suchquery, kein Absatz. Und
                # deterministisch — das ist Entscheidung und Umformung, keine
                # Erzeugung.
                max_tokens=64,
                temperature=0.0,
            )
        except GenerationError as exc:
            logger.warning("Router fehlgeschlagen, suche mit Originalfrage: %s", exc)
            return RouteDecision(reuse=False, query=question), reusable

        decision = parse_route(text, question, allow_reuse=bool(reusable))
        if decision.reuse:
            logger.debug("Router: vorliegende Quellen reichen, keine neue Suche")
        elif decision.query != question:
            logger.debug("Router: neue Suche, umgeschrieben zu %r", decision.query)
        return decision, reusable

    def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        """Vollständige Antwort mit Quellen."""
        started = time.time()
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            return Answer(
                question=question,
                text=NO_CONTEXT_ANSWER,
                duration_seconds=time.time() - started,
            )

        sources, dropped = build_sources(
            hits, context_tokens=self.settings.generator_context_length
        )

        if not self.settings.generator_enabled:
            raise PipelineError(
                "Generator ist abgeschaltet — retrieve() liefert die Treffer"
            )

        text, prompt_tokens, completion_tokens = self.generator.complete(
            build_prompt(question, sources),
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
        )

        return Answer(
            question=question,
            text=text,
            sources=sources,
            dropped=dropped,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_seconds=time.time() - started,
        )

    def ask_stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        history: Sequence[Turn] = (),
    ) -> tuple[list, Iterator[str], bool]:
        """Wie ``ask()``, aber die Antwort kommt stückweise.

        Gibt (Quellen, Token-Strom, wiederverwendet) zurück. Die Quellen stehen
        sofort fest und können angezeigt werden, während die Antwort noch
        entsteht — bei 5 bis 18 Token pro Sekunde ist das der Unterschied
        zwischen einer Anzeige und einer Wartezeit.

        ``history`` macht daraus einen Mehrturn-Chat. Ein Router entscheidet
        vorab (``plan_retrieval``), ob die schon vorliegenden Quellen genügen:
        dann wird nicht gesucht, sondern sie werden wiederverwendet
        (``wiederverwendet=True``). Sonst wird mit der umgeschriebenen Query neu
        gesucht. Ohne ``history`` verhält sich alles wie eine Einzelfrage.
        """
        decision, reusable = self.plan_retrieval(question, history)

        if decision.reuse:
            # Keine Vektorsuche, kein Reranking: die Quellen liegen schon vor,
            # nur aufs Kontextbudget gekürzt und neu numeriert.
            sources, _ = build_sources(
                reusable, context_tokens=self.settings.generator_context_length
            )
        else:
            hits = self.retrieve(decision.query, top_k=top_k)
            if not hits:
                return [], iter([NO_CONTEXT_ANSWER]), False
            sources, _ = build_sources(
                hits, context_tokens=self.settings.generator_context_length
            )

        stream = self.generator.stream(
            build_prompt(question, sources, history=history),
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature,
        )
        return sources, stream, decision.reuse

