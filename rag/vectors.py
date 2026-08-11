"""Vektorsuche: das austauschbare Stück des Index.

``store.IndexStore`` behält Buchhaltung und Chunk-Texte in SQLite; allein die
Nachbarschaftssuche liegt hinter dem Protokoll unten. Der Schnitt ist bewusst
so eng: die Prüfung "kennt dieser Index sein Embedding-Modell" und die
Idempotenz über den Dateihash sind die Stellen, an denen stille Fehler
entstehen, und die will man nicht pro Backend wiederholen.

Jedes Backend speichert deshalb nur ``chunk_id`` und Vektor. Text, Überschrift
und Dokumentpfad holt der Store per JOIN aus SQLite — auch bei Backends, die
Metadaten könnten. Zwei Quellen für dieselbe Wahrheit wären ein Sync-Problem
ohne Gegenwert.

**Alle Backends liefern Cosine-Distanz**, nicht Ähnlichkeit: 0 bei gleicher
Richtung, 1 bei Orthogonalität. Qdrant gibt von sich aus einen Score in die
andere Richtung; das wird hier umgerechnet, damit ``SearchHit.distance``
überall dasselbe bedeutet. Ein Backend, das diese Konvention verletzt, dreht
die Trefferreihenfolge um, ohne dass irgendetwas fehlschlägt.

**Externe Backends werden erst nach dem SQLite-Commit geschrieben.** Sie
können nicht an dessen Transaktion teilnehmen, also muss die Reihenfolge
entscheiden, welcher Zustand ein Absturz hinterlässt: nach dem Commit fehlen
höchstens Vektoren zu vorhandenen Chunks (Treffer bleiben aus), davor gäbe es
Vektoren zu Chunk-IDs, die inzwischen anderen Text tragen (Treffer werden
falsch). Fehlende Treffer sind der bessere Fehler.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Name des Backends, das ohne Angabe gilt. sqlite-vec bleibt die Vorgabe: es
# liegt in derselben Datei wie alles andere, braucht keine zweite Abhängigkeit
# und macht einen Index zu genau einer Datei, die man kopieren kann.
DEFAULT_BACKEND = "sqlite-vec"


class VectorBackendError(RuntimeError):
    """Das Vektor-Backend ist nicht benutzbar."""


@runtime_checkable
class VectorBackend(Protocol):
    """Was der Store von einem Vektorspeicher braucht — nicht mehr."""

    #: Wird im Index-Metadatum ``vector_backend`` festgehalten.
    name: str

    def open(self) -> None:
        """Verbindung herstellen und Sammlung anlegen, falls sie fehlt."""

    def close(self) -> None:
        """Ressourcen freigeben. Muss mehrfach aufrufbar sein."""

    def add(
        self, ids: Sequence[int], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Vektoren unter ihren Chunk-IDs ablegen."""

    def delete(self, ids: Sequence[int]) -> None:
        """Vektoren zu diesen Chunk-IDs entfernen."""

    def search(
        self, vector: Sequence[float], limit: int
    ) -> list[tuple[int, float]]:
        """``limit`` nächste Nachbarn als (chunk_id, Cosine-Distanz)."""

    def count(self) -> int:
        """Zahl gespeicherter Vektoren, für die Kennzahlen."""

    def drop(self) -> None:
        """Alles verwerfen. Für den Neuaufbau, nicht für den Normalbetrieb."""

    def sidecar_path(self) -> Path | None:
        """Eigenes Verzeichnis neben der Index-Datei, falls es eines gibt.

        ``None`` bei sqlite-vec, das in der Index-Datei selbst wohnt. Die CLI
        braucht das, um beim Löschen eines Index nicht die Vektoren
        zurückzulassen.
        """


# ─── sqlite-vec ──────────────────────────────────────────────────────────────


class SqliteVecBackend:
    """Vektoren in einer vec0-Tabelle derselben SQLite-Datei.

    Der Sonderfall unter den Backends: es teilt die Verbindung mit dem Store
    und liegt damit in dessen Transaktion. Ein Absturz mitten im Schreiben
    hinterlässt hier keinen Mischzustand — anders als bei allen anderen.
    """

    name = DEFAULT_BACKEND

    def __init__(
        self,
        connection: Callable[[], sqlite3.Connection],
        *,
        dimensions: int,
    ) -> None:
        # Nicht die Verbindung selbst, sondern ein Zugriff darauf: der Store
        # öffnet sie erst in open() und schließt sie wieder, das Backend wird
        # aber vorher gebaut.
        self._connection = connection
        self.dimensions = dimensions

    @property
    def db(self) -> sqlite3.Connection:
        return self._connection()

    def open(self) -> None:
        # Cosine, weil bge-m3 normalisierte Vektoren liefert und Cosine dort
        # die Metrik ist, gegen die trainiert wurde. Die vec0-Tabelle kennt
        # keine Fremdschlüssel, ihre Zeilen werden explizit gelöscht.
        self.db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id  INTEGER PRIMARY KEY,
                embedding float[{self.dimensions}] distance_metric=cosine
            )
            """
        )
        self.db.commit()

    def close(self) -> None:
        # Die Verbindung gehört dem Store, nicht diesem Backend.
        pass

    def add(
        self, ids: Sequence[int], vectors: Sequence[Sequence[float]]
    ) -> None:
        import sqlite_vec

        self.db.executemany(
            "INSERT INTO chunk_vectors(chunk_id, embedding) VALUES (?, ?)",
            [
                (int(i), sqlite_vec.serialize_float32(list(v)))
                for i, v in zip(ids, vectors)
            ],
        )
        self.db.commit()

    def delete(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        self.db.executemany(
            "DELETE FROM chunk_vectors WHERE chunk_id = ?",
            [(int(i),) for i in ids],
        )
        self.db.commit()

    def search(
        self, vector: Sequence[float], limit: int
    ) -> list[tuple[int, float]]:
        import sqlite_vec

        rows = self.db.execute(
            """
            SELECT chunk_id, distance FROM chunk_vectors
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (sqlite_vec.serialize_float32(list(vector)), limit),
        ).fetchall()
        return [(int(row[0]), float(row[1])) for row in rows]

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0])

    def drop(self) -> None:
        self.db.execute("DROP TABLE IF EXISTS chunk_vectors")
        self.db.commit()
        self.open()

    def sidecar_path(self) -> Path | None:
        return None


# ─── LanceDB ─────────────────────────────────────────────────────────────────


class LanceDBBackend:
    """Vektoren in einem Lance-Datensatz neben der Index-Datei.

    Wie sqlite-vec dateibasiert und serverlos, aber mit eigenem ANN-Index.
    Interessant für später: Lance kann Sparse- und Dense-Vektoren in einer
    Tabelle halten, womit sich bge-m3s Sparse-Ausgabe für Hybrid-Retrieval
    nutzen ließe, ohne ein zweites Modell zu laden.
    """

    name = "lancedb"
    _TABLE = "chunk_vectors"

    def __init__(self, path: Path, *, dimensions: int) -> None:
        self.path = path
        self.dimensions = dimensions
        self._db = None
        self._table = None

    def open(self) -> None:
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as exc:  # pragma: no cover
            raise VectorBackendError(
                "lancedb fehlt: uv pip install -e '.[lancedb]'"
            ) from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.path))

        # Schema explizit statt aus Beispieldaten geraten: eine leere Tabelle
        # mit falschem Vektortyp fällt erst beim ersten Suchlauf auf.
        schema = pa.schema(
            [
                pa.field("chunk_id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), self.dimensions)),
            ]
        )
        # exist_ok statt einer Existenzprüfung: table_names() ist veraltet und
        # list_tables() liefert eine paginierte Antwort, deren Behandlung hier
        # nichts einbringt — create_table gibt die bestehende Tabelle samt
        # Inhalt zurück.
        self._table = self._db.create_table(self._TABLE, schema=schema, exist_ok=True)

    @property
    def table(self):
        if self._table is None:
            raise VectorBackendError("LanceDB-Backend ist nicht geöffnet")
        return self._table

    def close(self) -> None:
        self._table = None
        self._db = None

    def add(
        self, ids: Sequence[int], vectors: Sequence[Sequence[float]]
    ) -> None:
        if not ids:
            return
        self.table.add(
            [
                {"chunk_id": int(i), "vector": [float(x) for x in v]}
                for i, v in zip(ids, vectors)
            ]
        )

    def delete(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        # Lance filtert per SQL-Ausdruck; IN-Liste statt Schleife, weil jedes
        # delete() eine neue Dateiversion schreibt.
        liste = ", ".join(str(int(i)) for i in ids)
        self.table.delete(f"chunk_id IN ({liste})")

    def search(
        self, vector: Sequence[float], limit: int
    ) -> list[tuple[int, float]]:
        # "_distance" ausdrücklich mitwählen: wählt man Spalten aus, ohne sie zu
        # nennen, hängt Lance sie noch automatisch an — aber unter einer
        # Deprecation-Warnung pro Query, und künftig gar nicht mehr. Da wir den
        # Wert unten brauchen, fordern wir ihn explizit an.
        treffer = (
            self.table.search([float(x) for x in vector])
            .metric("cosine")
            .limit(limit)
            .select(["chunk_id", "_distance"])
            .to_list()
        )
        # Lance liefert bei metric=cosine in "_distance" bereits 1 - Ähnlichkeit,
        # also dieselbe Konvention wie sqlite-vec.
        return [(int(t["chunk_id"]), float(t["_distance"])) for t in treffer]

    def count(self) -> int:
        return int(self.table.count_rows())

    def drop(self) -> None:
        if self._db is not None:
            # ignore_missing, weil drop() auch auf einem frischen Index
            # aufrufbar sein muss.
            self._db.drop_table(self._TABLE, ignore_missing=True)
        self._table = None
        self.open()

    def sidecar_path(self) -> Path | None:
        return self.path


# ─── ChromaDB ────────────────────────────────────────────────────────────────


class ChromaBackend:
    """Vektoren in einer persistenten Chroma-Sammlung neben der Index-Datei.

    Chroma bringt eigene Metadaten- und Dokumenthaltung mit, die hier
    ungenutzt bleibt — Texte stehen in SQLite. Das ist Absicht: Chroma ist
    hier ANN-Index und nichts weiter.
    """

    name = "chromadb"
    _COLLECTION = "chunk_vectors"

    def __init__(self, path: Path, *, dimensions: int) -> None:
        self.path = path
        self.dimensions = dimensions
        self._client = None
        self._collection = None

    def open(self) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise VectorBackendError(
                "chromadb fehlt: uv pip install -e '.[chromadb]'"
            ) from exc

        self.path.mkdir(parents=True, exist_ok=True)
        # anonymized_telemetry aus: ein lokales Werkzeug soll nicht ins Netz
        # funken, und Chroma tut das ohne diese Einstellung.
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._COLLECTION,
            # Ohne diese Angabe rechnet Chroma L2 statt Cosine.
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        if self._collection is None:
            raise VectorBackendError("Chroma-Backend ist nicht geöffnet")
        return self._collection

    def close(self) -> None:
        self._collection = None
        self._client = None

    def add(
        self, ids: Sequence[int], vectors: Sequence[Sequence[float]]
    ) -> None:
        if not ids:
            return
        # Chroma-IDs sind Strings; die Chunk-ID wird beim Lesen zurückgewandelt.
        self.collection.upsert(
            ids=[str(int(i)) for i in ids],
            embeddings=[[float(x) for x in v] for v in vectors],
        )

    def delete(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        self.collection.delete(ids=[str(int(i)) for i in ids])

    def search(
        self, vector: Sequence[float], limit: int
    ) -> list[tuple[int, float]]:
        ergebnis = self.collection.query(
            query_embeddings=[[float(x) for x in vector]],
            n_results=limit,
            include=["distances"],
        )
        ids = ergebnis.get("ids") or [[]]
        distances = ergebnis.get("distances") or [[]]
        if not ids[0]:
            return []
        return [
            (int(i), float(d)) for i, d in zip(ids[0], distances[0])
        ]

    def count(self) -> int:
        return int(self.collection.count())

    def drop(self) -> None:
        if self._client is not None:
            try:
                self._client.delete_collection(self._COLLECTION)
            except Exception:  # noqa: BLE001 — Sammlung war nicht da
                pass
        self._collection = None
        self.open()

    def sidecar_path(self) -> Path | None:
        return self.path


# ─── Qdrant ──────────────────────────────────────────────────────────────────


class QdrantBackend:
    """Vektoren in Qdrant — eingebettet als Verzeichnis oder über einen Server.

    Ohne ``url`` läuft der Client im lokalen Modus und schreibt in ein
    Verzeichnis neben der Index-Datei; damit bleibt die Zusage "kein Server"
    des Projekts erhalten. Mit ``url`` wird ein laufender Qdrant benutzt, was
    echte Filter und größere Bestände erlaubt, aber Betrieb kostet.

    Achtung, lokaler Modus: Qdrant sperrt das Verzeichnis exklusiv. Zwei
    Prozesse gleichzeitig — etwa CLI-Ingest und laufende Oberfläche — gehen
    hier nicht, bei sqlite-vec schon.
    """

    name = "qdrant"
    _COLLECTION = "chunk_vectors"

    def __init__(
        self, path: Path, *, dimensions: int, url: str | None = None
    ) -> None:
        self.path = path
        self.dimensions = dimensions
        self.url = url
        self._client = None

    def open(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:  # pragma: no cover
            raise VectorBackendError(
                "qdrant-client fehlt: uv pip install -e '.[qdrant]'"
            ) from exc

        if self.url:
            self._client = QdrantClient(url=self.url)
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.path))

        if not self._client.collection_exists(self._COLLECTION):
            self._client.create_collection(
                collection_name=self._COLLECTION,
                vectors_config=VectorParams(
                    size=self.dimensions, distance=Distance.COSINE
                ),
            )

    @property
    def client(self):
        if self._client is None:
            raise VectorBackendError("Qdrant-Backend ist nicht geöffnet")
        return self._client

    def close(self) -> None:
        if self._client is not None:
            # Im lokalen Modus gibt erst close() die Verzeichnissperre frei.
            self._client.close()
            self._client = None

    def add(
        self, ids: Sequence[int], vectors: Sequence[Sequence[float]]
    ) -> None:
        if not ids:
            return
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=self._COLLECTION,
            points=[
                PointStruct(id=int(i), vector=[float(x) for x in v])
                for i, v in zip(ids, vectors)
            ],
        )

    def delete(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        self.client.delete(
            collection_name=self._COLLECTION,
            points_selector=[int(i) for i in ids],
        )

    def search(
        self, vector: Sequence[float], limit: int
    ) -> list[tuple[int, float]]:
        treffer = self.client.query_points(
            collection_name=self._COLLECTION,
            query=[float(x) for x in vector],
            limit=limit,
            with_payload=False,
        ).points
        # Qdrant liefert bei COSINE die Ähnlichkeit, nicht die Distanz — hier
        # wird umgedreht, damit klein überall "näher" heißt.
        return [(int(t.id), 1.0 - float(t.score)) for t in treffer]

    def count(self) -> int:
        return int(self.client.count(self._COLLECTION, exact=True).count)

    def drop(self) -> None:
        self.client.delete_collection(self._COLLECTION)
        from qdrant_client.models import Distance, VectorParams

        self.client.create_collection(
            collection_name=self._COLLECTION,
            vectors_config=VectorParams(
                size=self.dimensions, distance=Distance.COSINE
            ),
        )

    def sidecar_path(self) -> Path | None:
        return None if self.url else self.path


# ─── Auflösung ───────────────────────────────────────────────────────────────

# Dateiendung des Nebenverzeichnisses pro Backend. Aus index.db wird
# index.lance/, index.chroma/ oder index.qdrant/ — so bleibt sichtbar, welche
# Verzeichnisse zu welchem Index gehören.
_SIDECAR_SUFFIX = {
    "lancedb": ".lance",
    "chromadb": ".chroma",
    "qdrant": ".qdrant",
}

BACKENDS = (DEFAULT_BACKEND, "lancedb", "chromadb", "qdrant")


def sidecar_for(index_path: Path, backend: str) -> Path | None:
    """Wo das Backend zu diesem Index seine Dateien ablegt."""
    suffix = _SIDECAR_SUFFIX.get(backend)
    return None if suffix is None else index_path.with_suffix(suffix)


def create_backend(
    name: str,
    *,
    index_path: Path,
    dimensions: int,
    connection: Callable[[], sqlite3.Connection],
    options: dict | None = None,
) -> VectorBackend:
    """Backend nach Namen bauen. Lädt keine Bibliothek — das tut ``open()``.

    Die Trennung ist wichtig für die Oberfläche: sie muss die Auswahl
    anzeigen können, ohne alle vier Bibliotheken zu importieren.
    """
    opts = options or {}
    if name == DEFAULT_BACKEND:
        return SqliteVecBackend(connection, dimensions=dimensions)

    sidecar = sidecar_for(index_path, name)
    if sidecar is None:
        bekannt = ", ".join(BACKENDS)
        raise VectorBackendError(
            f"Unbekanntes Vektor-Backend '{name}'. Bekannt: {bekannt}"
        )

    if name == "lancedb":
        return LanceDBBackend(sidecar, dimensions=dimensions)
    if name == "chromadb":
        return ChromaBackend(sidecar, dimensions=dimensions)
    if name == "qdrant":
        return QdrantBackend(
            sidecar, dimensions=dimensions, url=opts.get("url") or None
        )

    raise VectorBackendError(f"Unbekanntes Vektor-Backend '{name}'")


def remove_sidecar(index_path: Path, backend: str) -> Path | None:
    """Nebenverzeichnis eines Backends löschen, falls es eines hat.

    Wird beim Neuaufbau gebraucht: eine gelöschte ``index.db`` ohne das
    zugehörige ``index.lance/`` hinterlässt Vektoren zu Chunks, die es nicht
    mehr gibt.
    """
    pfad = sidecar_for(index_path, backend)
    if pfad is not None and pfad.exists():
        shutil.rmtree(pfad)
        return pfad
    return None


def clear_vectors(
    index_path: Path,
    backend: str,
    *,
    dimensions: int,
    options: dict | None = None,
) -> str:
    """Alle Vektoren eines Backends verwerfen, gleich wo sie liegen.

    Zwei Wege, weil es zwei Ablageorte gibt: ein Verzeichnis wird gelöscht,
    eine Sammlung auf einem Server muss der Client selbst verwerfen. Genau
    hier lag sonst ein stiller Rest — ein ``rmtree`` erreicht einen laufenden
    Qdrant nicht, dessen Vektoren hätten den Neuaufbau überlebt und wären mit
    den neuen vermischt worden.

    Gibt eine kurze Beschreibung des Geschehenen für das Protokoll zurück.
    """
    if backend == DEFAULT_BACKEND:
        # Liegt in der Index-Datei, die der Aufrufer ohnehin löscht.
        return "in der Index-Datei enthalten"

    # Das Backend selbst entscheidet, wo es liegt: ob Qdrant ein Verzeichnis
    # benutzt oder einen Server, steht in seinen Optionen und nicht im Namen.
    # Der Konstruktor lädt keine Bibliothek — ein LanceDB-Verzeichnis lässt
    # sich also auch löschen, wenn lancedb nicht installiert ist.
    speicher = create_backend(
        backend,
        index_path=index_path,
        dimensions=dimensions,
        connection=_kein_sqlite,
        options=options,
    )

    if (pfad := speicher.sidecar_path()) is not None:
        if not pfad.exists():
            return "kein Verzeichnis vorhanden"
        shutil.rmtree(pfad)
        return f"Verzeichnis entfernt: {pfad}"

    # Serverbasiert: nur der Client kommt an die Sammlung.
    speicher.open()
    try:
        speicher.drop()
    finally:
        speicher.close()
    return "Sammlung verworfen"


def _kein_sqlite() -> sqlite3.Connection:
    """Platzhalter für Backends, die keine SQLite-Verbindung brauchen.

    ``create_backend`` verlangt sie der einheitlichen Signatur wegen; nur
    sqlite-vec ruft sie je auf.
    """
    raise VectorBackendError(
        "Dieses Backend braucht keine SQLite-Verbindung — "
        "hier stimmt die Aufrufreihenfolge nicht."
    )
