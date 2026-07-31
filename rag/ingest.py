"""Dateien → Index. Die Verkettung der vier Stufen.

``extract.convert()`` → ``chunk.chunk_markdown()`` → ``embed.Embedder`` →
``store.IndexStore``. Diese Datei hält nur die Reihenfolge und die
Fehlerbehandlung; die Arbeit liegt in den vier Modulen.

Zwei Regeln bestimmen das Verhalten:

**Ein Fehler in einer Datei beendet den Lauf nicht.** Über hundert Dateien ist
eine kaputte die Regel, nicht die Ausnahme. Sie wird gemeldet und der Lauf
geht weiter — ein Abbruch bei Datei 80 von 100 verschenkt die Extraktionszeit
der ersten 79.

**Unverändertes wird übersprungen.** Der Hash-Vergleich kostet Millisekunden,
die Extraktion Sekunden bis Minuten pro Datei. Ein zweiter Lauf über ein
gepflegtes Verzeichnis ist damit fast kostenlos.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunk import TARGET_TOKENS, chunk_markdown
from rag.embed import Embedder, EmbeddingError
from rag.extract import SUPPORTED_SUFFIXES, ExtractionError, convert
from rag.store import IndexStore, StoreError, file_sha256

logger = logging.getLogger(__name__)

# Status einer einzelnen Datei nach dem Lauf.
STATUS_NEW = "neu"
STATUS_UPDATED = "aktualisiert"
STATUS_SKIPPED = "unverändert"
STATUS_EMPTY = "leer"
STATUS_FAILED = "fehler"

# Fortschrittsmeldung: (Datei, Index, Gesamtzahl, Phase).
ProgressCallback = Callable[[Path, int, int, str], None]


@dataclass
class FileResult:
    """Was mit einer Datei passiert ist."""

    path: Path
    status: str
    chunk_count: int = 0
    duration_seconds: float = 0.0
    ocr_used: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.status in {STATUS_NEW, STATUS_UPDATED}


@dataclass
class IngestReport:
    """Ergebnis eines Ingest-Laufs."""

    results: list[FileResult] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def chunk_count(self) -> int:
        return sum(r.chunk_count for r in self.results)

    @property
    def changed(self) -> list[FileResult]:
        return [r for r in self.results if r.changed]

    @property
    def skipped(self) -> list[FileResult]:
        return [r for r in self.results if r.status == STATUS_SKIPPED]

    @property
    def failed(self) -> list[FileResult]:
        return [r for r in self.results if r.status == STATUS_FAILED]

    @property
    def empty(self) -> list[FileResult]:
        return [r for r in self.results if r.status == STATUS_EMPTY]

    @property
    def ocr_count(self) -> int:
        return len([r for r in self.results if r.ocr_used])


def collect_files(paths: Iterable[str | Path]) -> list[Path]:
    """Verzeichnisse aufklappen, Formate filtern, Duplikate entfernen.

    Nicht unterstützte Endungen fallen still heraus, wenn sie aus einem
    Verzeichnis kommen — ein Ordner mit Bildern und Tabellen soll nicht
    hundert Fehlermeldungen erzeugen. Direkt benannte Dateien bleiben drin,
    damit ihr Format-Fehler den Anwender erreicht.
    """
    found: list[Path] = []
    seen: set[Path] = set()

    for entry in paths:
        path = Path(entry).expanduser()
        if path.is_dir():
            candidates = [
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            ]
        else:
            candidates = [path]

        for candidate in candidates:
            # resolve() gegen doppelte Aufnahme derselben Datei über zwei Pfade.
            key = candidate.resolve() if candidate.exists() else candidate
            if key not in seen:
                seen.add(key)
                found.append(candidate)

    return found


def ingest_paths(
    paths: Iterable[str | Path],
    *,
    store: IndexStore,
    embedder: Embedder,
    target_tokens: int = TARGET_TOKENS,
    force: bool = False,
    prune: bool = False,
    ocr: bool | None = None,
    progress: ProgressCallback | None = None,
) -> IngestReport:
    """Dateien extrahieren, chunken, embedden und in den Index schreiben.

    ``force`` ignoriert den Hash-Vergleich und schreibt alles neu.
    ``prune`` entfernt Dokumente, deren Datei verschwunden ist.
    """
    files = collect_files(paths)
    report = IngestReport()
    started = time.time()

    for index, path in enumerate(files, start=1):
        report.results.append(
            _ingest_one(
                path,
                store=store,
                embedder=embedder,
                target_tokens=target_tokens,
                force=force,
                ocr=ocr,
                index=index,
                total=len(files),
                progress=progress,
            )
        )

    if prune:
        report.removed = store.forget_missing(files)

    report.duration_seconds = time.time() - started
    return report


def _ingest_one(
    path: Path,
    *,
    store: IndexStore,
    embedder: Embedder,
    target_tokens: int,
    force: bool,
    ocr: bool | None,
    index: int,
    total: int,
    progress: ProgressCallback | None,
) -> FileResult:
    """Eine Datei durch die Kette schicken. Fängt alle erwartbaren Fehler."""

    def announce(phase: str) -> None:
        if progress:
            progress(path, index, total, phase)

    if not path.exists():
        return FileResult(path, STATUS_FAILED, error="Datei nicht gefunden")

    started = time.time()

    try:
        digest = file_sha256(path)
    except OSError as exc:
        return FileResult(path, STATUS_FAILED, error=f"nicht lesbar: {exc}")

    existing = store.document_record(path)
    if not force and existing is not None and existing.sha256 == digest:
        announce(STATUS_SKIPPED)
        return FileResult(
            path,
            STATUS_SKIPPED,
            chunk_count=existing.chunk_count,
            duration_seconds=time.time() - started,
        )

    announce("extrahieren")
    try:
        document = convert(path, ocr=ocr)
    except ExtractionError as exc:
        return FileResult(path, STATUS_FAILED, error=str(exc))
    except Exception as exc:  # pragma: no cover
        # Docling und seine Abhängigkeiten werfen breit gestreut. Eine Datei
        # darf den Lauf nicht beenden, deshalb hier der weite Fang.
        logger.debug("Unerwarteter Fehler bei %s", path, exc_info=True)
        return FileResult(path, STATUS_FAILED, error=f"unerwartet: {exc}")

    warnings = list(document.warnings)

    announce("chunken")
    # Mit dem Tokenizer des Embedding-Modells zählen, nicht schätzen: sonst
    # schneidet das Modell Chunks ab, die der Chunker für passend hielt. Der
    # Zugriff lädt das Modell — das ist der Punkt, an dem die Ladezeit anfällt,
    # und sie fällt ohnehin an, weil gleich embeddet wird.
    chunks = chunk_markdown(
        document.markdown,
        target_tokens=target_tokens,
        counter=embedder.token_counter(),
    )

    if not chunks:
        announce(STATUS_EMPTY)
        return FileResult(
            path,
            STATUS_EMPTY,
            duration_seconds=time.time() - started,
            ocr_used=document.ocr_used,
            warnings=warnings + ["kein extrahierbarer Text"],
        )

    announce("embedden")
    try:
        vectors = embedder.embed_passages([c.embed_text for c in chunks])
    except EmbeddingError as exc:
        return FileResult(path, STATUS_FAILED, error=str(exc))

    # Abgeschnittene Chunks bleiben sonst unsichtbar: das Modell liefert brav
    # einen Vektor, nur eben für den halben Chunk.
    if overlong := embedder.check_lengths([c.embed_text for c in chunks]):
        warnings.append(
            f"{len(overlong)} Chunk(s) am Token-Limit des Embedders — "
            "möglicherweise abgeschnitten"
        )

    announce("schreiben")
    try:
        written = store.replace_document(
            path,
            sha256=digest,
            format=document.format,
            chunks=chunks,
            embeddings=vectors,
            page_count=document.page_count,
            char_count=document.char_count,
            ocr_used=document.ocr_used,
        )
    except StoreError as exc:
        return FileResult(path, STATUS_FAILED, error=str(exc))

    return FileResult(
        path,
        STATUS_UPDATED if existing else STATUS_NEW,
        chunk_count=written,
        duration_seconds=time.time() - started,
        ocr_used=document.ocr_used,
        warnings=warnings,
    )


def search_index(
    query: str,
    *,
    store: IndexStore,
    embedder: Embedder,
    limit: int = 10,
) -> Sequence:
    """Freitextsuche über den Index.

    Reine Vektorsuche, ohne Reranking und ohne Generierung — der Abschluss
    von Schritt 2 und die Grundlage, gegen die sich Schritt 3 messen lässt.
    """
    return store.search(embedder.embed_query(query), limit=limit)
