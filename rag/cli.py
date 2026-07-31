"""CLI: Planung, Extraktion, Ingest und Suche."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from pathlib import Path

from rag.detect import Platform, WhichllmError, detect_local, simulate
from rag.embed import Embedder, EmbeddingError, load_embedder_config
from rag.extract import ExtractionError, convert, probe
from rag.ingest import IngestReport, ingest_paths, search_index
from rag.resolve import PipelinePlan, ResolutionError, resolve_pipeline
from rag.store import DEFAULT_INDEX_NAME, IndexStore, StoreError

app = typer.Typer(
    add_completion=False,
    help="Lokales RAG mit plattformabhängiger Modellauswahl.",
)
console = Console()

_GIB = 1024**3

# Wo der Index liegt, wenn nichts anderes gesagt wird. Neben dem Cache, weil
# er wie der Cache aus den Quelldokumenten neu erzeugbar ist.
DEFAULT_INDEX_PATH = Path.home() / ".cache" / "local-rag" / DEFAULT_INDEX_NAME


@app.callback()
def main() -> None:
    """Ohne diesen Callback macht Typer den einzigen Befehl zum Root-Befehl,
    was ``rag plan`` bricht, sobald ``ingest`` und ``ask`` dazukommen."""


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
    table.add_column("Quelle")

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


def _open_store(
    index: Path, profile: str, *, device: str = "auto"
) -> tuple[IndexStore, Embedder]:
    """Index und Embedder passend zueinander öffnen.

    Beide aus derselben Konfiguration, damit Modell und Index nicht
    auseinanderlaufen können. Der Embedder wird zuerst gebaut: schlägt die
    Konfiguration fehl, bleibt kein offener Index zurück.
    """
    config = load_embedder_config(profile)
    embedder = Embedder(config, device=device)
    store = IndexStore(
        index, embedder=config.model_id, dimensions=config.dimensions
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
    index: Path = typer.Option(DEFAULT_INDEX_PATH, "--index", help="Index-Datei"),
    profile: str = typer.Option(
        "default", "--profile", help="Embedder-Profil aus platforms.toml"
    ),
    device: str = typer.Option(
        "auto", "--device", help="auto, cpu, cuda oder mps"
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

    try:
        store, embedder = _open_store(index, profile, device=device)
    except (StoreError, EmbeddingError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        f"[dim]Index:[/] {index}\n"
        f"[dim]Embedder:[/] {embedder.config.model_id} auf {embedder.device}"
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
    index: Path = typer.Option(DEFAULT_INDEX_PATH, "--index", help="Index-Datei"),
    profile: str = typer.Option(
        "default", "--profile", help="Embedder-Profil aus platforms.toml"
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

    if not index.exists():
        console.print(f"[red]Fehler:[/] kein Index unter {index} — erst 'rag ingest'")
        raise typer.Exit(1)

    try:
        store, embedder = _open_store(index, profile)
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
    index: Path = typer.Option(DEFAULT_INDEX_PATH, "--index", help="Index-Datei"),
    profile: str = typer.Option(
        "default", "--profile", help="Embedder-Profil aus platforms.toml"
    ),
) -> None:
    """Zeige, was im Index liegt."""
    if not index.exists():
        console.print(f"\n  [yellow]kein Index unter[/] {index}\n")
        return

    try:
        store, _ = _open_store(index, profile)
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
        f"({stats['dimensions']} Dimensionen)"
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


if __name__ == "__main__":
    app()
