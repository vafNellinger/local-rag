"""Index: SQLite plus sqlite-vec.

Eine Datei, kein Server. Das ist für ein lokales RAG die richtige Größe —
ein Vektorstore-Dienst neben einem Ein-Nutzer-Werkzeug wäre Betriebsaufwand
ohne Gegenwert.

Zwei Eigenschaften tragen den Entwurf:

**Der Index kennt sein Embedding-Modell.** Vektoren aus zwei Modellen im
selben Raum sind Unsinn, und der Fehler ist still: die Suche liefert weiter
Ergebnisse, nur falsche. Modell und Dimension stehen deshalb in ``meta`` und
werden bei jedem Öffnen geprüft.

**Ingest ist idempotent über den Dateihash.** Ein zweiter Lauf über dasselbe
Verzeichnis kostet nur das Hashen. Geänderte Dateien werden komplett ersetzt,
nicht ergänzt — Chunk-Grenzen verschieben sich bei jeder Textänderung, ein
Abgleich einzelner Chunks wäre aufwendiger als das Neuschreiben.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag.chunk import Chunk

logger = logging.getLogger(__name__)

# Hochzählen, wenn sich das Schema ändert. Ein Index mit anderer Version wird
# abgelehnt statt migriert: solange das Projekt keine Nutzerdaten hat, ist
# Neuaufbau billiger als Migrationscode, der nie getestet wird.
SCHEMA_VERSION = 1

DEFAULT_INDEX_NAME = "index.db"

# Wie viele Bytes pro Lesevorgang beim Hashen. 1 MiB hält den Speicher flach,
# ohne bei großen PDFs in Syscall-Overhead zu laufen.
HASH_CHUNK_BYTES = 1024 * 1024


class StoreError(RuntimeError):
    """Der Index ist nicht benutzbar."""


@dataclass(frozen=True)
class SearchHit:
    """Ein Treffer aus der Vektorsuche."""

    chunk_id: int
    document_path: str
    ordinal: int
    text: str
    heading_path: tuple[str, ...]
    kind: str
    # Cosine-Distanz: 0 bei identischer Richtung, 1 bei Orthogonalität.
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine-Ähnlichkeit — für die Anzeige greifbarer als die Distanz."""
        return 1.0 - self.distance

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path)

    @property
    def citation(self) -> str:
        """Kurze Quellenangabe für den Prompt."""
        name = Path(self.document_path).name
        return f"{name} — {self.heading}" if self.heading_path else name


@dataclass(frozen=True)
class DocumentRecord:
    """Was der Index über eine bereits aufgenommene Datei weiß."""

    id: int
    path: str
    sha256: str
    chunk_count: int


def file_sha256(path: str | Path) -> str:
    """Hash über den Dateiinhalt, für die Änderungserkennung.

    Inhalt statt mtime: Kopieren und Auspacken setzen die mtime neu, ohne dass
    sich etwas geändert hat. Das Hashen kostet Millisekunden, ein unnötiger
    Reingest kostet Minuten.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        import sqlite_vec
    except ImportError as exc:  # pragma: no cover
        raise StoreError("sqlite-vec fehlt: uv pip install sqlite-vec") from exc

    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError) as exc:
        # Manche Distributions-Builds von Python liefern SQLite ohne
        # Extension-Unterstützung. Ohne sie gibt es keine Vektorsuche, und die
        # Meldung muss den Grund nennen statt nur "geht nicht".
        raise StoreError(
            "SQLite dieser Python-Installation kann keine Erweiterungen laden — "
            "sqlite-vec ist damit nicht nutzbar. Abhilfe: Python aus einem Build "
            "mit --enable-loadable-sqlite-extensions verwenden (z.B. über uv)."
        ) from exc

    # WAL: der Ingest schreibt lange, eine parallele Abfrage soll dabei lesen
    # können, ohne auf den Abschluss zu warten.
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


class IndexStore:
    """Der Index. Als Context Manager verwenden."""

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: str,
        dimensions: int,
    ) -> None:
        self.path = Path(path).expanduser()
        self.embedder = embedder
        self.dimensions = dimensions
        self._connection: sqlite3.Connection | None = None

    # ─── Lebenszyklus ────────────────────────────────────────────────────────

    def __enter__(self) -> IndexStore:
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> IndexStore:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = _connect(self.path)
        self._create_schema()
        self._check_compatibility()
        return self

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise StoreError("Index ist nicht geöffnet")
        return self._connection

    def _create_schema(self) -> None:
        db = self.connection
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY,
                path        TEXT NOT NULL UNIQUE,
                sha256      TEXT NOT NULL,
                format      TEXT NOT NULL,
                page_count  INTEGER NOT NULL DEFAULT 0,
                char_count  INTEGER NOT NULL DEFAULT 0,
                ocr_used    INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                ingested_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id           INTEGER PRIMARY KEY,
                document_id  INTEGER NOT NULL
                             REFERENCES documents(id) ON DELETE CASCADE,
                ordinal      INTEGER NOT NULL,
                text         TEXT NOT NULL,
                heading_path TEXT NOT NULL DEFAULT '[]',
                kind         TEXT NOT NULL DEFAULT 'prosa',
                token_count  INTEGER NOT NULL DEFAULT 0,
                UNIQUE (document_id, ordinal)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document_id);
            """
        )

        # Cosine, weil bge-m3 normalisierte Vektoren liefert und Cosine dort
        # die Metrik ist, gegen die trainiert wurde. Die vec0-Tabelle kennt
        # keine Fremdschlüssel, ihre Zeilen werden in _delete_document
        # zusammen mit den Chunks entfernt.
        db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id  INTEGER PRIMARY KEY,
                embedding float[{self.dimensions}] distance_metric=cosine
            )
            """
        )
        db.commit()

    def _check_compatibility(self) -> None:
        """Modell, Dimension und Schema-Version gegen den Bestand prüfen."""
        db = self.connection
        stored = {
            row["key"]: row["value"]
            for row in db.execute("SELECT key, value FROM meta")
        }

        expected = {
            "schema_version": str(SCHEMA_VERSION),
            "embedder": self.embedder,
            "dimensions": str(self.dimensions),
        }

        if not stored:
            db.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)", expected.items()
            )
            db.commit()
            return

        if stored.get("schema_version") != expected["schema_version"]:
            raise StoreError(
                f"Index hat Schema-Version {stored.get('schema_version')}, "
                f"erwartet {SCHEMA_VERSION} — Index neu aufbauen "
                f"(Datei löschen: {self.path})"
            )

        # Der stille Fehler, gegen den diese Prüfung existiert: Vektoren aus
        # zwei Modellen im selben Raum liefern weiter Treffer, nur falsche.
        if stored.get("embedder") != expected["embedder"]:
            raise StoreError(
                f"Index wurde mit '{stored.get('embedder')}' gebaut, "
                f"angefragt ist '{self.embedder}'. Vektoren verschiedener "
                f"Modelle sind nicht vergleichbar — Index neu aufbauen "
                f"(Datei löschen: {self.path})"
            )

        if stored.get("dimensions") != expected["dimensions"]:
            raise StoreError(
                f"Index hat {stored.get('dimensions')} Dimensionen, "
                f"angefragt sind {self.dimensions} — Index neu aufbauen"
            )

    # ─── Schreiben ───────────────────────────────────────────────────────────

    def document_record(self, path: str | Path) -> DocumentRecord | None:
        """Was der Index über diese Datei weiß, oder None."""
        row = self.connection.execute(
            "SELECT id, path, sha256, chunk_count FROM documents WHERE path = ?",
            (self._normalize(path),),
        ).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            id=row["id"],
            path=row["path"],
            sha256=row["sha256"],
            chunk_count=row["chunk_count"],
        )

    def is_current(self, path: str | Path, sha256: str) -> bool:
        """Ist die Datei unverändert im Index?"""
        record = self.document_record(path)
        return record is not None and record.sha256 == sha256

    def _delete_document(self, document_id: int) -> None:
        """Dokument samt Chunks und Vektoren entfernen.

        Die Vektoren zuerst und explizit: ``chunk_vectors`` ist eine
        vec0-Tabelle und hängt nicht am ON-DELETE-CASCADE der Chunks. Ohne
        diesen Schritt bleiben verwaiste Vektoren zurück, die bei der Suche
        auf nicht mehr existierende Chunk-IDs zeigen.
        """
        db = self.connection
        db.execute(
            "DELETE FROM chunk_vectors WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE document_id = ?)",
            (document_id,),
        )
        db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        db.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def replace_document(
        self,
        path: str | Path,
        *,
        sha256: str,
        format: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        page_count: int = 0,
        char_count: int = 0,
        ocr_used: bool = False,
    ) -> int:
        """Dokument mit seinen Chunks in den Index schreiben.

        Ersetzt einen vorhandenen Eintrag vollständig. Gibt die Zahl der
        geschriebenen Chunks zurück.
        """
        if len(chunks) != len(embeddings):
            raise StoreError(
                f"{len(chunks)} Chunks, aber {len(embeddings)} Vektoren — "
                "das darf nicht auseinanderlaufen"
            )

        import sqlite_vec

        normalized = self._normalize(path)
        db = self.connection

        # Eine Transaktion über Löschen und Neuschreiben: bricht der Ingest
        # mitten im Dokument ab, bleibt der alte Stand statt einer halben Datei.
        with db:
            if existing := self.document_record(normalized):
                self._delete_document(existing.id)

            cursor = db.execute(
                """
                INSERT INTO documents
                    (path, sha256, format, page_count, char_count,
                     ocr_used, chunk_count, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized,
                    sha256,
                    format,
                    page_count,
                    char_count,
                    int(ocr_used),
                    len(chunks),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            document_id = int(cursor.lastrowid)

            for chunk, vector in zip(chunks, embeddings):
                if len(vector) != self.dimensions:
                    raise StoreError(
                        f"Vektor hat {len(vector)} Dimensionen, "
                        f"der Index erwartet {self.dimensions}"
                    )
                chunk_cursor = db.execute(
                    """
                    INSERT INTO chunks
                        (document_id, ordinal, text, heading_path, kind, token_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        chunk.ordinal,
                        chunk.text,
                        json.dumps(list(chunk.heading_path), ensure_ascii=False),
                        chunk.kind,
                        chunk.token_count,
                    ),
                )
                db.execute(
                    "INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?, ?)",
                    (
                        int(chunk_cursor.lastrowid),
                        sqlite_vec.serialize_float32(list(vector)),
                    ),
                )

        return len(chunks)

    def forget_missing(self, present: Iterable[str | Path]) -> list[str]:
        """Dokumente entfernen, deren Datei nicht mehr existiert.

        Gibt die entfernten Pfade zurück. Ohne diesen Schritt zitiert die
        Antwort irgendwann Dateien, die es nicht mehr gibt.
        """
        keep = {self._normalize(p) for p in present}
        removed: list[str] = []
        with self.connection:
            for row in self.connection.execute(
                "SELECT id, path FROM documents"
            ).fetchall():
                if row["path"] in keep:
                    continue
                if Path(row["path"]).exists():
                    continue
                self._delete_document(row["id"])
                removed.append(row["path"])
        return removed

    # ─── Lesen ───────────────────────────────────────────────────────────────

    def search(self, vector: Sequence[float], *, limit: int = 10) -> list[SearchHit]:
        """Die ``limit`` nächsten Chunks zum Anfragevektor."""
        if len(vector) != self.dimensions:
            raise StoreError(
                f"Anfragevektor hat {len(vector)} Dimensionen, "
                f"der Index erwartet {self.dimensions}"
            )

        import sqlite_vec

        rows = self.connection.execute(
            """
            SELECT v.chunk_id, v.distance, c.ordinal, c.text, c.heading_path,
                   c.kind, d.path
            FROM chunk_vectors AS v
            JOIN chunks AS c ON c.id = v.chunk_id
            JOIN documents AS d ON d.id = c.document_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (sqlite_vec.serialize_float32(list(vector)), limit),
        ).fetchall()

        return [
            SearchHit(
                chunk_id=row["chunk_id"],
                document_path=row["path"],
                ordinal=row["ordinal"],
                text=row["text"],
                heading_path=tuple(json.loads(row["heading_path"])),
                kind=row["kind"],
                distance=float(row["distance"]),
            )
            for row in rows
        ]

    def stats(self) -> dict[str, int | str]:
        """Kennzahlen für die Anzeige."""
        db = self.connection
        documents = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vectors = db.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
        tokens = db.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM chunks"
        ).fetchone()[0]
        return {
            "documents": documents,
            "chunks": chunks,
            "vectors": vectors,
            "tokens": tokens,
            "embedder": self.embedder,
            "dimensions": self.dimensions,
            "path": str(self.path),
        }

    def documents(self) -> list[DocumentRecord]:
        return [
            DocumentRecord(
                id=row["id"],
                path=row["path"],
                sha256=row["sha256"],
                chunk_count=row["chunk_count"],
            )
            for row in self.connection.execute(
                "SELECT id, path, sha256, chunk_count FROM documents ORDER BY path"
            )
        ]

    @staticmethod
    def _normalize(path: str | Path) -> str:
        """Pfade vereinheitlichen, damit dieselbe Datei nicht zweimal landet.

        ``resolve()`` löst Symlinks und relative Angaben auf — ohne das wären
        ``./akte.pdf`` und ``/home/x/akte.pdf`` zwei Dokumente.
        """
        return str(Path(path).expanduser().resolve())
