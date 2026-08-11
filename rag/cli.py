"""CLI: Planung, Extraktion, Ingest und Suche."""

from __future__ import annotations

import json
import logging
import os
import re
import time

import typer
from rich.console import Console
from rich.table import Table

from dataclasses import dataclass, replace
from pathlib import Path

from rag.detect import (
    Platform,
    WhichllmError,
    detect_local,
    load_config,
    simulate,
)
from rag.embed import Embedder, EmbeddingError, load_embedder_config
from rag.evaluate import (
    DEFAULT_GOLD_PATH,
    EvaluationError,
    evaluate,
    load_gold,
)
from rag.extract import (
    ExtractionError,
    clear_extract_cache,
    convert,
    extract_cache_size,
    probe,
)
from rag.generate import NO_CONTEXT_ANSWER, GenerationError, gguf_path
from rag.ingest import IngestReport, ingest_paths, search_index
from rag.pipeline import SETTINGS_PATH, PipelineError, RagPipeline, Settings
from rag.rerank import RerankError
from rag.resolve import (
    PipelinePlan,
    ResolutionError,
    backend_options,
    resolve_pipeline,
    resolve_vector_backend,
)
from rag.store import (
    DEFAULT_INDEX_NAME,
    IndexStore,
    StoreError,
    read_index_documents,
    read_index_meta,
)
from rag.vectors import BACKENDS, sidecar_for

app = typer.Typer(
    add_completion=False,
    help="Lokales RAG mit plattformabhängiger Modellauswahl.",
)
console = Console()
logger = logging.getLogger(__name__)

_GIB = 1024**3

# Wo der Index liegt, wenn nichts anderes gesagt wird. Neben dem Cache, weil
# er wie der Cache aus den Quelldokumenten neu erzeugbar ist.
DEFAULT_INDEX_PATH = Path.home() / ".cache" / "local-rag" / DEFAULT_INDEX_NAME


@app.callback()
def main() -> None:
    """Ohne diesen Callback macht Typer den einzigen Befehl zum Root-Befehl,
    was ``rag plan`` bricht, sobald ``ingest`` und ``ask`` dazukommen."""


def _resolve_index(explicit: Path | None) -> Path:
    """Index-Pfad auflösen: Angabe, dann gespeicherte Einstellung, dann Vorgabe.

    ``--index`` muss ``None`` als Vorgabe haben, damit dieser Vorrang
    überhaupt feststellbar ist. Mit einem Vorgabewert wäre nicht zu
    unterscheiden, ob der Anwender ihn genannt hat — und der Vorgabewert würde
    stets gegen die gespeicherte Einstellung gewinnen. Genau so war der in der
    Oberfläche eingestellte Pfad nach jedem Neustart wirkungslos.
    """
    if explicit is not None:
        return explicit
    if SETTINGS_PATH.exists():
        try:
            stored = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if pfad := stored.get("index_path"):
                return Path(pfad).expanduser()
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Einstellungen nicht lesbar: %s", exc)
    return DEFAULT_INDEX_PATH


def _render(plan: PipelinePlan) -> None:
    console.print()
    console.print(f"[bold]{plan.platform.describe()}[/]")
    console.print(f"Plattformklasse: [cyan]{plan.platform_class}[/]")
    console.print()

    table = Table(show_lines=True)
    table.add_column("Rolle", style="bold")
    # overflow="fold": Modell-IDs sind lang und dürfen nicht abgeschnitten
    # werden — der abgeschnittene Name ist die eine Information, die man
    # aus dieser Tabelle wirklich braucht.
    table.add_column("Modell", overflow="fold")
    table.add_column("Quant")
    table.add_column("Gerät")
    table.add_column("VRAM", justify="right")
    table.add_column("Speed", justify="right")
    # fold statt abschneiden: "config:default" wird sonst zu "config:def…" und
    # das aktive Profil ist genau die Information, die hier zählt.
    table.add_column("Quelle", overflow="fold")

    for spec in plan.specs:
        vram = (
            f"{spec.vram_required_bytes / _GIB:.1f} GB"
            if spec.vram_required_bytes
            else "—"
        )
        speed = (
            f"~{spec.estimated_tok_per_sec:.0f} tok/s"
            if spec.estimated_tok_per_sec
            else "—"
        )
        device_style = "green" if spec.device == "gpu" else "yellow"
        table.add_row(
            spec.role,
            spec.model_id,
            spec.quant_type or "—",
            f"[{device_style}]{spec.device.upper()}[/]",
            vram,
            speed,
            spec.source,
        )

    console.print(table)

    if plan.generator.artifact_repo_id:
        console.print(
            f"  GGUF: [dim]{plan.generator.artifact_repo_id}[/] / "
            f"[dim]{plan.generator.artifact_filename}[/]"
        )
    if plan.generator.context_length:
        console.print(f"  Kontext: [dim]{plan.generator.context_length} Token[/]")

    ziel = plan.vector_backend_options.get("url") or "eingebettet"
    console.print(
        f"  Vektoren: [dim]{plan.vector_backend}[/]"
        + (f" [dim]({ziel})[/]" if plan.vector_backend == "qdrant" else "")
    )

    if plan.platform.has_gpu:
        console.print(
            f"  GPU-Belegung: [dim]{plan.total_gpu_bytes / _GIB:.1f} GB von "
            f"{plan.platform.usable_vram_gb:.1f} GB[/]"
        )

    for warning in plan.warnings:
        console.print(f"  [yellow]![/] {warning}")
    console.print()


@app.command()
def plan(
    gpu: str | None = typer.Option(
        None, "--gpu", help="Zielplattform simulieren, z.B. 'RTX 4090'"
    ),
    vram: float | None = typer.Option(
        None, "--vram", help="Nutzbares VRAM in GB überschreiben"
    ),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Ohne GPU planen"),
    label: str | None = typer.Option(None, "--label", help="Name der Zielplattform"),
    refresh: bool = typer.Option(
        False, "--refresh", help="whichllm-Cache ignorieren und neu abfragen"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Zeige, welche Modelle auf einer Plattform verwendet werden."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    use_cache = not refresh
    simulated = bool(gpu or vram is not None or cpu_only)

    try:
        platform: Platform = (
            simulate(
                gpu=gpu,
                vram_gb=vram,
                cpu_only=cpu_only,
                label=label,
                use_cache=use_cache,
            )
            if simulated
            else detect_local(use_cache=use_cache)
        )
        result = resolve_pipeline(platform, use_cache=use_cache)
    except (WhichllmError, ResolutionError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    _render(result)


@app.command()
def inspect(
    paths: list[Path] = typer.Argument(..., help="Dateien oder Verzeichnisse"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Prüfe, was aus Dokumenten extrahierbar ist und wo OCR nötig wäre."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    files: list[Path] = []
    for entry in paths:
        if entry.is_dir():
            files.extend(sorted(p for p in entry.rglob("*") if p.is_file()))
        else:
            files.append(entry)

    table = Table(show_lines=False)
    table.add_column("Datei", overflow="fold")
    table.add_column("Format")
    table.add_column("Seiten", justify="right")
    table.add_column("Zeichen", justify="right")
    table.add_column("Scan/OCR", justify="right")
    table.add_column("leer", justify="right")

    total_pages = 0
    total_ocr = 0
    total_sparse = 0
    failures: list[tuple[Path, str]] = []

    for file_path in files:
        try:
            doc = probe(file_path)
        except ExtractionError as exc:
            failures.append((file_path, str(exc)))
            continue

        ocr_pages = len(doc.pages_needing_ocr)
        sparse_pages = len(doc.sparse_pages)
        total_pages += len(doc.pages)
        total_ocr += ocr_pages
        total_sparse += sparse_pages

        if ocr_pages == 0:
            ocr_cell = "[green]nein[/]"
        elif ocr_pages == len(doc.pages):
            ocr_cell = f"[red]alle {ocr_pages}[/]"
        else:
            ocr_cell = f"[yellow]{ocr_pages} von {len(doc.pages)}[/]"

        table.add_row(
            file_path.name,
            doc.format,
            str(len(doc.pages)),
            f"{doc.char_count:,}".replace(",", "."),
            ocr_cell,
            f"[dim]{sparse_pages}[/]" if sparse_pages else "—",
        )
        for page in doc.pages:
            if page.status == "error":
                table.add_row(
                    "", "", f"[red]S.{page.number}[/]", f"[red]{page.note}[/]", "", ""
                )
        for warning in doc.warnings:
            table.add_row("", "", "", f"[yellow]{warning}[/]", "", "")

    console.print()
    console.print(table)

    if total_pages:
        share = total_ocr / total_pages
        console.print(
            f"\n  {total_ocr} von {total_pages} Seiten brauchen OCR "
            f"([bold]{share:.0%}[/])"
        )
        if total_sparse:
            console.print(
                f"  {total_sparse} Seite(n) ohne Text und ohne Bild — "
                "[dim]kein OCR nötig, dort ist nichts[/]"
            )

    for path, reason in failures:
        console.print(f"  [red]x[/] {path.name}: {reason}")
    console.print()


@app.command(name="convert")
def convert_cmd(
    path: Path = typer.Argument(..., help="Zu konvertierende Datei"),
    out: Path | None = typer.Option(None, "-o", "--out", help="Markdown hierhin"),
    ocr: bool | None = typer.Option(
        None, "--ocr/--no-ocr", help="OCR erzwingen oder unterdrücken"
    ),
    show: int = typer.Option(0, "--show", help="Erste N Zeichen ausgeben"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Extrahiere ein Dokument als strukturiertes Markdown."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        doc = convert(path, ocr=ocr)
    except ExtractionError as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    ocr_note = "[yellow]mit OCR[/]" if doc.ocr_used else "[dim]ohne OCR[/]"
    # Tausendertrennung nur auf der Zahl, nicht auf dem ganzen Satz — sonst
    # werden auch die Kommas zwischen den Feldern zu Punkten.
    chars = f"{doc.char_count:,}".replace(",", ".")
    console.print(
        f"\n[bold]{doc.path.name}[/] — {doc.format}, {doc.page_count} Seite(n), "
        f"{chars} Zeichen, {ocr_note}, {doc.duration_seconds:.1f}s"
    )
    for warning in doc.warnings:
        console.print(f"  [yellow]![/] {warning}")

    if out:
        out.write_text(doc.markdown, encoding="utf-8")
        console.print(f"  geschrieben nach [cyan]{out}[/]")
    if show:
        console.print()
        console.print(doc.markdown[:show])
    console.print()


@dataclass(frozen=True)
class EmbedderChoice:
    """Welcher Embedder auf welchem Gerät, und woher die Wahl kommt."""

    profile: str
    device: str
    origin: str


def _choose_embedder(
    index: Path, profile: str | None, device: str | None
) -> EmbedderChoice:
    """Embedder-Profil und Gerät bestimmen, in dieser Rangfolge.

    1. Was der Anwender ausdrücklich angibt.
    2. Was im Index steht. Ein bestehender Index gibt das Modell vor — sonst
       würde die Plattformerkennung ihn bei jedem Lauf invalidieren, obwohl
       sich nur die Hardware geändert hat.
    3. Die Plattformklasse aus ``platforms.toml``.

    Nur Schritt 3 erkennt die Plattform, und zwar über ``detect_local()``:
    das liest den Hardware-Block (``whichllm -n 1``, 24 h gecacht) und löst
    ausdrücklich *nicht* den Generator auf. ``resolve_pipeline()`` wäre hier
    falsch — es würde für eine Embedder-Frage das Modell-Ranking anstoßen.
    """
    if profile and device:
        return EmbedderChoice(profile, device, "explizit angegeben")

    stored = read_index_meta(index)
    chosen_profile = profile or stored.get("embedder_profile")
    origin = "aus dem Index" if chosen_profile and not profile else "explizit angegeben"

    if chosen_profile and device:
        return EmbedderChoice(chosen_profile, device, origin)

    # Für das Gerät (und ein noch unbekanntes Profil) die Plattform befragen.
    platform = detect_local()
    class_config = load_config().get("platform_class", {}).get(
        platform.platform_class, {}
    )

    if not chosen_profile:
        chosen_profile = str(class_config.get("embedder_profile", "default"))
        origin = f"aus Plattformklasse {platform.platform_class}"

    chosen_device = device or str(class_config.get("embedder_device", "cpu"))
    if not device:
        origin = f"{origin}, Gerät aus Klasse {platform.platform_class}"

    return EmbedderChoice(chosen_profile, chosen_device, origin)


def _choose_backend(index: Path, explicit: str | None) -> tuple[str, dict]:
    """Vektor-Backend bestimmen, in derselben Rangfolge wie beim Embedder.

    1. Was der Anwender per ``--store`` angibt.
    2. Was im Index steht — die Vektoren liegen dort und nirgends sonst.
    3. ``[vector_store]`` aus ``platforms.toml``.

    Die Optionen kommen immer aus der Konfiguration, auch bei einer Angabe per
    Flag: eine Qdrant-Adresse will niemand als Kommandozeilenparameter
    wiederholen.
    """
    config = load_config()
    name = explicit or read_index_meta(index).get("vector_backend")
    if not name:
        name, _ = resolve_vector_backend(config)
    if name not in BACKENDS:
        bekannt = ", ".join(BACKENDS)
        raise StoreError(f"Unbekanntes Vektor-Backend '{name}'. Bekannt: {bekannt}")
    return name, backend_options(config, name)


def _open_store(
    index: Path,
    profile: str,
    *,
    device: str = "auto",
    backend: str | None = None,
) -> tuple[IndexStore, Embedder]:
    """Index und Embedder passend zueinander öffnen.

    Beide aus derselben Konfiguration, damit Modell und Index nicht
    auseinanderlaufen können. Der Embedder wird zuerst gebaut: schlägt die
    Konfiguration fehl, bleibt kein offener Index zurück.
    """
    config = load_embedder_config(profile)
    embedder = Embedder(config, device=device)
    backend_name, options = _choose_backend(index, backend)
    store = IndexStore(
        index,
        embedder=config.model_id,
        dimensions=config.dimensions,
        profile=profile,
        vector_backend=backend_name,
        backend_options=options,
    ).open()
    return store, embedder


def _render_ingest(report: IngestReport) -> None:
    table = Table(show_lines=False)
    table.add_column("Datei", overflow="fold")
    table.add_column("Status")
    table.add_column("Chunks", justify="right")
    table.add_column("Dauer", justify="right")

    styles = {
        "neu": "green",
        "aktualisiert": "cyan",
        "unverändert": "dim",
        "leer": "yellow",
        "fehler": "red",
    }

    for result in report.results:
        style = styles.get(result.status, "")
        ocr = " [yellow]OCR[/]" if result.ocr_used else ""
        table.add_row(
            result.path.name,
            f"[{style}]{result.status}[/]{ocr}" if style else result.status,
            str(result.chunk_count) if result.chunk_count else "—",
            f"{result.duration_seconds:.1f}s" if result.duration_seconds else "—",
        )
        if result.error:
            table.add_row("", f"[red]{result.error}[/]", "", "")
        for warning in result.warnings:
            table.add_row("", f"[yellow]{warning}[/]", "", "")

    console.print()
    console.print(table)

    changed = len(report.changed)
    console.print(
        f"\n  {changed} Datei(en) verarbeitet, {len(report.skipped)} unverändert, "
        f"{report.chunk_count} Chunks im Index, {report.duration_seconds:.1f}s"
    )
    if report.ocr_count:
        console.print(f"  [yellow]{report.ocr_count} Datei(en) mit OCR[/]")
    if report.removed:
        console.print(
            f"  [dim]{len(report.removed)} verschwundene Datei(en) entfernt[/]"
        )
    if report.failed:
        console.print(f"  [red]{len(report.failed)} Datei(en) fehlgeschlagen[/]")
    console.print()


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="Dateien oder Verzeichnisse"),
    index: Path | None = typer.Option(
        None, "--index", help="Index-Datei; ohne Angabe aus den Einstellungen"
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Embedder-Profil erzwingen; ohne Angabe aus Index oder Plattform",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="cpu, gpu, cuda oder mps; ohne Angabe aus der Plattformklasse",
    ),
    store_backend: str | None = typer.Option(
        None,
        "--store",
        help=f"Vektor-Backend ({', '.join(BACKENDS)}); "
        "ohne Angabe aus Index oder platforms.toml",
    ),
    force: bool = typer.Option(
        False, "--force", help="Auch unveränderte Dateien neu einlesen"
    ),
    prune: bool = typer.Option(
        False, "--prune", help="Verschwundene Dateien aus dem Index entfernen"
    ),
    ocr: bool | None = typer.Option(
        None, "--ocr/--no-ocr", help="OCR erzwingen oder unterdrücken"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Dokumente extrahieren, chunken, embedden und indizieren."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    index = _resolve_index(index)

    try:
        choice = _choose_embedder(index, profile, device)
        store, embedder = _open_store(
            index, choice.profile, device=choice.device, backend=store_backend
        )
    except (StoreError, EmbeddingError, WhichllmError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        f"[dim]Index:[/] {index}\n"
        f"[dim]Embedder:[/] {embedder.config.model_id} auf {embedder.device} "
        f"[dim]({choice.profile}, {choice.origin})[/]\n"
        f"[dim]Vektoren:[/] {store.vector_backend}"
    )

    # Der erste Lauf lädt das Modell — ohne Hinweis sieht die Pause wie ein
    # Hänger aus.
    with console.status("[dim]verarbeite…[/]") as status:

        def on_progress(path: Path, position: int, total: int, phase: str) -> None:
            status.update(f"[dim]{position}/{total} {path.name} — {phase}[/]")

        try:
            report = ingest_paths(
                paths,
                store=store,
                embedder=embedder,
                force=force,
                prune=prune,
                ocr=ocr,
                progress=on_progress,
            )
        except (StoreError, EmbeddingError) as exc:
            console.print(f"[red]Fehler:[/] {exc}")
            raise typer.Exit(1) from exc
        finally:
            store.close()

    _render_ingest(report)

    if report.failed:
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Suchbegriff oder Frage"),
    index: Path | None = typer.Option(
        None, "--index", help="Index-Datei; ohne Angabe aus den Einstellungen"
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Embedder-Profil erzwingen; ohne Angabe aus dem Index"
    ),
    device: str | None = typer.Option(
        None, "--device", help="cpu, gpu, cuda oder mps"
    ),
    store_backend: str | None = typer.Option(
        None,
        "--store",
        help=f"Vektor-Backend ({', '.join(BACKENDS)}); "
        "ohne Angabe aus dem Index",
    ),
    limit: int = typer.Option(5, "-n", "--limit", help="Zahl der Treffer"),
    full: bool = typer.Option(False, "--full", help="Chunks vollständig zeigen"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Suche im Index. Reine Vektorsuche, ohne Reranking und Generierung."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    index = _resolve_index(index)
    if not index.exists():
        console.print(f"[red]Fehler:[/] kein Index unter {index} — erst 'rag ingest'")
        raise typer.Exit(1)

    # Das Profil kommt aus dem Index, nicht aus der Plattform: gesucht werden
    # muss mit dem Modell, mit dem indiziert wurde. Nebeneffekt — eine Suche
    # kostet keine Plattformerkennung, was bei 79 ms Query-Latenz auch nicht
    # tragbar wäre.
    resolved_profile = profile or read_index_meta(index).get(
        "embedder_profile", "default"
    )

    try:
        store, embedder = _open_store(
            index, resolved_profile, device=device or "auto", backend=store_backend
        )
    except (StoreError, EmbeddingError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    try:
        with console.status("[dim]suche…[/]"):
            hits = search_index(query, store=store, embedder=embedder, limit=limit)
    except (StoreError, EmbeddingError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc
    finally:
        store.close()

    if not hits:
        console.print("\n  [yellow]keine Treffer[/]\n")
        return

    console.print()
    for rank, hit in enumerate(hits, start=1):
        console.print(
            f"[bold]{rank}.[/] [cyan]{hit.citation}[/] "
            f"[dim]({hit.similarity:.3f})[/]"
        )
        text = hit.text if full else hit.text[:400].replace("\n", " ")
        suffix = "" if full or len(hit.text) <= 400 else " […]"
        console.print(f"   {text}{suffix}")
        console.print()


@app.command()
def status(
    index: Path | None = typer.Option(
        None, "--index", help="Index-Datei; ohne Angabe aus den Einstellungen"
    ),
    store_backend: str | None = typer.Option(
        None,
        "--store",
        help=f"Vektor-Backend ({', '.join(BACKENDS)}); "
        "ohne Angabe aus dem Index",
    ),
) -> None:
    """Zeige, was im Index liegt."""
    index = _resolve_index(index)
    if not index.exists():
        console.print(f"\n  [yellow]kein Index unter[/] {index}\n")
        return

    profile = read_index_meta(index).get("embedder_profile", "default")

    try:
        store, _ = _open_store(index, profile, backend=store_backend)
    except (StoreError, EmbeddingError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    try:
        stats = store.stats()
        documents = store.documents()
    finally:
        store.close()

    size_mb = index.stat().st_size / (1024 * 1024)
    console.print()
    console.print(f"[bold]{stats['path']}[/] — {size_mb:.1f} MB")
    # Tausendertrennung nur auf der Zahl: auf dem ganzen Satz würden auch die
    # Kommas zwischen den Feldern zu Punkten.
    tokens = f"{stats['tokens']:,}".replace(",", ".")
    console.print(
        f"  {stats['documents']} Dokumente, {stats['chunks']} Chunks, "
        f"{tokens} Token"
    )
    console.print(
        f"  Embedder: [cyan]{stats['embedder']}[/] "
        f"({stats['dimensions']} Dimensionen, Profil {stats['profile']})"
    )

    sidecar = sidecar_for(index, str(stats["vector_backend"]))
    ort = f", {sidecar.name}" if sidecar else ", in der Index-Datei"
    console.print(f"  Vektoren: [cyan]{stats['vector_backend']}[/] ({stats['vectors']}{ort})")

    # Jeder Chunk braucht genau einen Vektor. Weicht das ab, ist ein Ingest
    # abgebrochen, nachdem SQLite schon geschrieben hatte — die betroffenen
    # Chunks sind vorhanden, aber unauffindbar. Stillschweigend wäre das ein
    # unerklärlich schlechtes Retrieval.
    if stats["vectors"] != stats["chunks"]:
        differenz = int(stats["chunks"]) - int(stats["vectors"])
        console.print(
            f"  [yellow]![/] {abs(differenz)} Chunk(s) "
            f"{'ohne Vektor' if differenz > 0 else 'zu viele Vektoren'} — "
            f"'rag reindex' baut den Index sauber neu auf"
        )

    if documents:
        table = Table(show_lines=False)
        table.add_column("Dokument", overflow="fold")
        table.add_column("Chunks", justify="right")
        for record in documents:
            table.add_row(Path(record.path).name, str(record.chunk_count))
        console.print()
        console.print(table)
    console.print()


@app.command()
def pull(
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Lade die GGUF-Datei des vom Plan gewählten Generators herunter."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        result = resolve_pipeline(detect_local())
    except (WhichllmError, ResolutionError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    spec = result.generator
    if not spec.artifact_repo_id or not spec.artifact_filename:
        console.print(
            f"[red]Fehler:[/] whichllm nennt für {spec.model_id} keine "
            "GGUF-Datei — Quantisierung muss manuell gewählt werden"
        )
        raise typer.Exit(1)

    if existing := gguf_path(spec.artifact_repo_id, spec.artifact_filename):
        size_gb = existing.stat().st_size / (1024**3)
        console.print(
            f"\n  [green]schon da[/] {existing.name} ({size_gb:.1f} GB)\n"
            f"  [dim]{existing}[/]\n"
        )
        return

    console.print(
        f"\n  lade [cyan]{spec.artifact_filename}[/] "
        f"[dim]aus {spec.artifact_repo_id}[/]\n"
        f"  [yellow]mehrere Gigabyte, das dauert[/]\n"
    )
    try:
        path = gguf_path(
            spec.artifact_repo_id, spec.artifact_filename, download=True
        )
    except GenerationError as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    size_gb = path.stat().st_size / (1024**3)
    console.print(f"  [green]fertig[/] {size_gb:.1f} GB nach [dim]{path}[/]\n")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Die Frage"),
    index: Path | None = typer.Option(
        None, "--index", help="Index-Datei; ohne Angabe aus den Einstellungen"
    ),
    top_k: int = typer.Option(5, "-k", "--top-k", help="Quellen im Prompt"),
    context: int | None = typer.Option(
        None, "--context", help="Kontextfenster in Token"
    ),
    max_tokens: int = typer.Option(800, "--max-tokens", help="Länge der Antwort"),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="Reranking überspringen"
    ),
    show_sources: bool = typer.Option(
        True, "--sources/--no-sources", help="Quellen unter der Antwort listen"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Stelle eine Frage an den Index und lass sie beantworten."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    index = _resolve_index(index)
    if not index.exists():
        console.print(f"[red]Fehler:[/] kein Index unter {index} — erst 'rag ingest'")
        raise typer.Exit(1)

    try:
        settings = Settings.for_platform()
    except WhichllmError as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    settings = replace(
        settings,
        index_path=index,
        top_k=top_k,
        max_tokens=max_tokens,
        reranker_enabled=settings.reranker_enabled and not no_rerank,
        **({"generator_context_length": context} if context else {}),
    )

    pipeline = RagPipeline(settings)
    try:
        with console.status("[dim]suche…[/]"):
            hits = pipeline.retrieve(question)

        if not hits:
            console.print(f"\n  [yellow]{NO_CONTEXT_ANSWER}[/]\n")
            return

        console.print()
        console.print(f"[bold]{question}[/]")
        console.print()

        sources, stream = pipeline.ask_stream(question)
        # Streaming, weil die Generierung auf CPU gemessen bei 2 bis 8 Token
        # pro Sekunde liegt — ohne laufende Ausgabe sieht das wie ein Hänger aus.
        started = time.time()
        pieces: list[str] = []
        with console.status("[dim]denkt…[/]") as status:
            for piece in stream:
                if not pieces:
                    status.stop()
                pieces.append(piece)
                console.print(piece, end="")
        console.print()

        answer = "".join(pieces)
        duration = time.time() - started
    except (PipelineError, StoreError, EmbeddingError, RerankError, GenerationError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc
    finally:
        pipeline.close()

    if show_sources:
        console.print()
        cited = {
            int(n) for n in re.findall(r"\[(\d+)\]", answer) if n.isdigit()
        }
        for source in sources:
            mark = "[green]✓[/]" if source.number in cited else "[dim]·[/]"
            score = (
                f" [dim]{source.hit.rerank_score:+.3f}[/]"
                if source.hit.rerank_score is not None
                else f" [dim]{source.hit.similarity:.3f}[/]"
            )
            console.print(f"  {mark} [{source.number}] {source.citation}{score}")
        # Unzitierte Quellen sind ein Hinweis, nicht ein Fehler: viele davon
        # heißen, dass die Suche breit gestreut hat oder top_k zu hoch ist.
        if uncited := len(sources) - len(cited):
            console.print(
                f"  [dim]{uncited} Quelle(n) nicht zitiert — "
                f"top_k senken oder min_rerank_score setzen[/]"
            )

    console.print(f"\n  [dim]{duration:.1f}s[/]\n")


@app.command()
def reindex(
    index: Path | None = typer.Option(
        None, "--index", help="Index-Datei; ohne Angabe aus den Einstellungen"
    ),
    profile: str | None = typer.Option(
        None, "--profile", help="Embedder-Profil, mit dem neu gebaut wird"
    ),
    store_backend: str | None = typer.Option(
        None,
        "--store",
        help=f"Vektor-Backend, mit dem neu gebaut wird ({', '.join(BACKENDS)})",
    ),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Extraktions-Cache vorher leeren (erzwingt echte Neuextraktion)",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Baue den Index neu auf, etwa nach einem Wechsel des Embedders.

    Die Dateiliste kommt aus dem bestehenden Index. Die Extraktion wird aus
    dem Cache wiederverwendet, sofern die Dateien unverändert sind — bei einem
    Modellwechsel ändert sich nur das Embedding.

    Auch der Weg zum Wechsel des Vektor-Backends: ``--store lancedb`` baut die
    Vektoren im neuen Backend auf und räumt das alte weg. Die Extraktion
    entfällt dabei komplett, das Embedding nicht — Vektoren lassen sich nicht
    zwischen Backends kopieren, ohne sie zu kennen.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    index = _resolve_index(index)
    if not index.exists():
        console.print(f"[red]Fehler:[/] kein Index unter {index} — erst 'rag ingest'")
        raise typer.Exit(1)

    known = read_index_documents(index)
    if not known:
        console.print(f"[yellow]Der Index unter {index} führt keine Dokumente.[/]")
        raise typer.Exit(1)

    stored = read_index_meta(index)
    ziel = profile or stored.get("embedder_profile", "default")

    try:
        config = load_embedder_config(ziel)
    except EmbeddingError as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    altes_backend = stored.get("vector_backend", "?")
    try:
        # Ohne --store das Backend des Index behalten, nicht das aus der
        # Konfiguration: ein reiner Embedder-Wechsel soll den Speicherort der
        # Vektoren nicht mitverändern.
        neues_backend, _ = _choose_backend(index, store_backend)
    except StoreError as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print()
    console.print(
        f"  bisher: [cyan]{stored.get('embedder', '?')}[/] "
        f"[dim](Profil {stored.get('embedder_profile', '?')}, "
        f"Vektoren {altes_backend})[/]"
    )
    console.print(
        f"  neu:    [cyan]{config.model_id}[/] "
        f"[dim](Profil {ziel}, Vektoren {neues_backend})[/]"
    )
    console.print(f"  {len(known)} Dokument(e) werden neu aufgenommen")

    if clear_cache:
        entfernt = clear_extract_cache()
        console.print(f"  [dim]{entfernt} Cache-Eintrag/Einträge gelöscht[/]")
    else:
        anzahl, bytes_ = extract_cache_size()
        if anzahl:
            console.print(
                f"  [dim]Extraktions-Cache: {anzahl} Eintrag/Einträge, "
                f"{bytes_ / 1024 / 1024:.1f} MB — Extraktion wird "
                f"übersprungen, wo möglich[/]"
            )
    console.print()

    try:
        settings = replace(
            Settings.load(fallback=Settings.for_platform()),
            index_path=index,
            embedder_profile=ziel,
            vector_backend=neues_backend,
            vector_backend_options=backend_options(load_config(), neues_backend),
        )
    except WhichllmError as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    pipeline = RagPipeline(settings)
    try:
        with console.status("[dim]baut neu…[/]") as status:

            def on_progress(path: Path, position: int, total: int, phase: str) -> None:
                status.update(f"[dim]{position}/{total} {path.name} — {phase}[/]")

            report = pipeline.rebuild_index(progress=on_progress)
    except (PipelineError, StoreError, EmbeddingError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc
    finally:
        pipeline.close()

    _render_ingest(report)

    if report.failed:
        raise typer.Exit(1)


@app.command(name="eval")
def eval_cmd(
    index: Path | None = typer.Option(
        None, "--index", help="Index-Datei; ohne Angabe aus den Einstellungen"
    ),
    gold: Path = typer.Option(
        DEFAULT_GOLD_PATH, "--gold", help="Goldstandard als JSON"
    ),
    top_k: int = typer.Option(5, "-k", "--top-k", help="Zu bewertende Treffer"),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="Ohne Reranking messen, zum Vergleich"
    ),
    answers: bool = typer.Option(
        False, "--answers", help="Auch Antworten erzeugen und prüfen (langsam)"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Messe Retrieval-Güte gegen den Goldstandard."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    index = _resolve_index(index)
    if not index.exists():
        console.print(f"[red]Fehler:[/] kein Index unter {index} — erst 'rag ingest'")
        raise typer.Exit(1)

    try:
        questions = load_gold(gold)
        settings = replace(
            Settings.load(fallback=Settings.for_platform()),
            index_path=index,
            top_k=top_k,
        )
        if no_rerank:
            settings = replace(settings, reranker_enabled=False)
        # Ohne Schwelle messen: die Kurve unten braucht alle Treffer, sonst
        # wären die Zahlen bereits durch die Einstellung vorgefiltert.
        settings = replace(settings, min_rerank_score=0.0)
    except (EvaluationError, WhichllmError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    pipeline = RagPipeline(settings)
    try:
        with console.status("[dim]messe…[/]") as status:

            def on_progress(position: int, total: int, question) -> None:
                status.update(f"[dim]{position}/{total} {question.id}[/]")

            report = evaluate(pipeline, questions, progress=on_progress)

        _render_eval(report)

        if answers:
            _render_answer_checks(pipeline, questions, console)
    except (PipelineError, StoreError, EmbeddingError, RerankError, GenerationError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc
    finally:
        pipeline.close()


def _render_eval(report) -> None:
    metrik = "Rerank-Punkte" if report.reranked else "Vektorähnlichkeit"
    console.print()
    console.print(
        f"[bold]{len(report.results)} Fragen[/] — "
        f"{len(report.answerable)} beantwortbar, "
        f"{len(report.unanswerable)} ohne Antwort im Korpus. "
        f"Gemessen an {metrik}, {report.duration_seconds:.1f}s"
    )
    console.print()

    table = Table(show_lines=False)
    table.add_column("Maß")
    table.add_column("Wert", justify="right")
    for k in (1, 3, 5):
        if k <= report.top_k:
            table.add_row(f"Recall@{k}", f"{report.recall_at(k):.1%}")
    table.add_row("MRR", f"{report.mrr():.3f}")
    console.print(table)

    correct_low, correct_high = report.correct_score_range()
    noise_low, noise_high = report.noise_score_range()
    console.print()
    console.print("[bold]Punktwerte[/]")
    console.print(
        f"  richtige Treffer:  {correct_low:.4f} bis {correct_high:.4f}"
    )
    console.print(
        f"  Rauschen (Fragen ohne Antwort): {noise_low:.4f} bis {noise_high:.4f}"
    )
    if correct_low < noise_high:
        console.print(
            "  [yellow]![/] Die Bereiche überlappen — keine Schwelle trennt "
            "sauber zwischen richtig und Rauschen."
        )
    else:
        console.print(
            f"  [green]✓[/] Bereiche trennbar: eine Schwelle zwischen "
            f"{noise_high:.4f} und {correct_low:.4f} filtert Rauschen ohne Verlust."
        )

    console.print()
    console.print("[bold]Wirkung der Relevanzschwelle[/]")
    threshold_table = Table(show_lines=False)
    threshold_table.add_column("Schwelle", justify="right")
    threshold_table.add_column("Recall", justify="right")
    threshold_table.add_column("Präzision", justify="right")
    threshold_table.add_column("Rauschen bleibt", justify="right")
    threshold_table.add_column("verliert", overflow="fold")

    for row in report.thresholds():
        threshold_table.add_row(
            f"{row.threshold:.3f}",
            f"{row.recall:.1%}",
            f"{row.precision:.1%}",
            str(row.noise_kept),
            ", ".join(row.lost[:3]) + ("…" if len(row.lost) > 3 else "") or "—",
        )
    console.print(threshold_table)

    if report.misses:
        console.print()
        console.print("[bold]Nicht gefunden[/]")
        for result in report.misses:
            erwartet = ", ".join(result.question.dokumente) or "(keins)"
            gefunden = ", ".join(h.document for h in result.hits[:3]) or "(nichts)"
            console.print(f"  [red]×[/] {result.question.id}")
            console.print(f"    [dim]{result.question.frage}[/]")
            console.print(f"    erwartet: {erwartet}")
            console.print(f"    gefunden: {gefunden}")
    console.print()


def _render_answer_checks(pipeline, questions, out) -> None:
    from rag.evaluate import check_answers

    out.print("[bold]Antworten prüfen[/] [dim](rund 20 s je Frage)[/]")
    with out.status("[dim]generiere…[/]") as status:

        def on_progress(position: int, total: int, question) -> None:
            status.update(f"[dim]{position}/{total} {question.id}[/]")

        checks = check_answers(pipeline, questions, progress=on_progress)

    answerable = [c for c in checks if c.question.answerable and c.question.antwort_enthaelt]
    unanswerable = [c for c in checks if not c.question.answerable]

    korrekt = sum(c.answered_correctly for c in answerable)
    eingeraeumt = sum(c.admitted_ignorance for c in unanswerable)

    out.print()
    if answerable:
        out.print(
            f"  Erwartete Angabe in der Antwort: "
            f"[bold]{korrekt}/{len(answerable)}[/] ({korrekt / len(answerable):.0%})"
        )
    if unanswerable:
        out.print(
            f"  Nichtwissen korrekt eingeräumt: "
            f"[bold]{eingeraeumt}/{len(unanswerable)}[/] "
            f"({eingeraeumt / len(unanswerable):.0%})"
        )

    for check in answerable:
        if not check.answered_correctly:
            fehlt = ", ".join(check.question.antwort_enthaelt)
            out.print(f"  [yellow]![/] {check.question.id}: erwartet '{fehlt}'")
            out.print(f"    [dim]{check.text[:180]}[/]")
    for check in unanswerable:
        if not check.admitted_ignorance:
            out.print(
                f"  [red]×[/] {check.question.id}: hat geantwortet statt "
                "Nichtwissen einzuräumen"
            )
            out.print(f"    [dim]{check.text[:180]}[/]")
    out.print()


@app.command()
def gui(
    index: Path | None = typer.Option(
        None, "--index", help="Index-Datei; ohne Angabe aus den Einstellungen"
    ),
    port: int = typer.Option(8080, "--port", help="Port für die Oberfläche"),
    host: str = typer.Option("127.0.0.1", "--host", help="Adresse zum Binden"),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Browser nicht automatisch öffnen"
    ),
    native: bool = typer.Option(
        False,
        "--native",
        help="Als eigenständiges Fenster öffnen statt im Browser",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Bei Codeänderungen neu starten (für die Entwicklung)",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Starte die grafische Oberfläche."""
    try:
        from rag.ui import LOG_PATH, run
    except ImportError as exc:
        console.print(
            f"[red]Fehler:[/] NiceGUI fehlt: uv pip install -e '.[gui]' ({exc})"
        )
        raise typer.Exit(1) from exc

    console.print(f"[dim]Protokoll:[/] {LOG_PATH}")

    if reload:
        if native:
            # Der native Modus hält das Fenster im Hauptthread; uvicorns
            # Reload-Kindprozess kann es nicht bedienen. Reload ist ohnehin ein
            # Entwicklungswerkzeug — dort bleibt es beim Browser.
            console.print(
                "[yellow]--native und --reload zusammen nicht möglich — "
                "Reload läuft im Browser.[/]"
            )
        # Nicht hier weiterlaufen: uvicorns Reloader importiert im Kindprozess
        # das Hauptmodul erneut, und das ist bei einem Console-Script der
        # pip-Wrapper mit seinem Main-Guard — die Seiten würden dort nie
        # registriert. rag/guiapp.py ist genau dafür da; die Begründung steht
        # in seinem Modulkopf.
        raise typer.Exit(
            _run_with_reload(
                index=index, host=host, port=port, browser=not no_browser,
                verbose=verbose,
            )
        )

    run(
        index_path=index,
        host=host,
        port=port,
        open_browser=not no_browser,
        native=native,
        verbose=verbose,
    )


def _run_with_reload(
    *, index: Path | None, host: str, port: int, browser: bool, verbose: bool
) -> int:
    """``python -m rag.guiapp`` starten und dessen Rückgabewert liefern.

    Die Parameter gehen über die Umgebung, nicht über die Argumentliste: der
    Reload-Kindprozess entsteht per multiprocessing-spawn und erbt die Umgebung,
    aber nicht ``sys.argv``.
    """
    import subprocess
    import sys

    env = {
        **os.environ,
        "LOCAL_RAG_GUI_HOST": host,
        "LOCAL_RAG_GUI_PORT": str(port),
        "LOCAL_RAG_GUI_BROWSER": "1" if browser else "0",
        "LOCAL_RAG_GUI_VERBOSE": "1" if verbose else "0",
    }
    if index is not None:
        env["LOCAL_RAG_GUI_INDEX"] = str(index)

    console.print("[dim]Reload aktiv — Änderungen unter rag/ starten neu.[/]")
    try:
        return subprocess.call([sys.executable, "-m", "rag.guiapp"], env=env)
    except KeyboardInterrupt:  # pragma: no cover
        return 0


if __name__ == "__main__":
    app()
