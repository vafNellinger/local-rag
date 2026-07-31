"""Grafische Oberfläche über NiceGUI.

Vier Seiten für die vier Dinge, die man mit diesem System tut: fragen,
Dokumente aufnehmen, sehen welche Modelle laufen, Regler verstellen.

Der ganze Aufwand hier dreht sich um ein einziges Problem: **alles Interessante
dauert lange.** Ein Ingest läuft Minuten, eine Antwort entsteht mit zwei bis
acht Token pro Sekunde, der Modell-Scan braucht bis zu drei Minuten, ein
GGUF-Download mehrere Gigabyte. Nichts davon darf den Browser blockieren, und
alles davon muss Fortschritt zeigen — sonst ist es von einem Absturz nicht zu
unterscheiden.

Deshalb: jede blockierende Arbeit läuft über ``asyncio.to_thread``, und die
lange laufenden Teile melden ihren Zustand zurück in die Oberfläche.

Ein Nutzer, ein Prozess, ein Modell im Speicher. Die Pipeline ist deshalb
modulglobal und durch einen Lock geschützt — llama.cpp verträgt keine zwei
gleichzeitigen Anfragen auf demselben Kontext.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import threading
from dataclasses import replace
from pathlib import Path

from nicegui import app as nicegui_app
from nicegui import ui

from rag.embed import EmbeddingError
from rag.extract import SUPPORTED_SUFFIXES
from rag.generate import GenerationError, gguf_path
from rag.ingest import IngestReport
from rag.pipeline import (
    SETTINGS_PATH,
    PipelineError,
    RagPipeline,
    Settings,
)
from rag.rerank import RerankError
from rag.resolve import ResolutionError
from rag.detect import WhichllmError
from rag.store import StoreError

logger = logging.getLogger(__name__)

# Wohin hochgeladene Dateien wandern. Der Index verweist auf Pfade, also
# müssen sie einen dauerhaften Ort haben — ein Temp-Verzeichnis wäre nach dem
# nächsten Neustart weg und der Index zeigte ins Leere.
UPLOAD_DIR = Path.home() / ".local" / "share" / "local-rag" / "dokumente"

ERRORS = (
    PipelineError,
    StoreError,
    EmbeddingError,
    RerankError,
    GenerationError,
    WhichllmError,
    ResolutionError,
)

# Ein Lock um alles, was ein Modell benutzt. llama.cpp ist auf demselben
# Kontext nicht wiedereintrittsfähig, und zwei parallele Ingests würden sich
# im Index gegenseitig überschreiben.
_work_lock = threading.Lock()


class State:
    """Der gemeinsame Zustand der Oberfläche."""

    def __init__(self, index_path: Path | None = None) -> None:
        base = Settings.for_platform()
        if index_path is not None:
            base = replace(base, index_path=index_path)
        self.settings = Settings.load(fallback=base)
        if index_path is not None:
            # Ein ausdrücklich übergebener Pfad gewinnt gegen die
            # gespeicherte Einstellung.
            self.settings = replace(self.settings, index_path=index_path)
        self.pipeline = RagPipeline(self.settings)
        self.busy: str | None = None

    def rebuild(self, **changes) -> None:
        self.settings = replace(self.settings, **changes)
        self.pipeline.apply(**changes)


state: State | None = None


def _state() -> State:
    if state is None:  # pragma: no cover
        raise RuntimeError("Oberfläche nicht initialisiert")
    return state


async def _in_thread(function, *args, **kwargs):
    """Blockierende Arbeit auslagern, serialisiert durch den Lock."""

    def guarded():
        with _work_lock:
            return function(*args, **kwargs)

    return await asyncio.to_thread(guarded)


def _notify_error(exc: Exception) -> None:
    logger.debug("Fehler in der Oberfläche", exc_info=True)
    ui.notify(str(exc), type="negative", multi_line=True, close_button="ok")


# ─── Seite: Fragen ───────────────────────────────────────────────────────────


@ui.page("/")
def page_ask() -> None:
    _layout("fragen")
    current = _state()

    with ui.column().classes("w-full max-w-4xl mx-auto gap-4 p-4"):
        ui.label("Frage an die Dokumente").classes("text-2xl font-bold")

        stats = current.pipeline.index_stats()
        if not stats.get("chunks"):
            with ui.card().classes("w-full bg-amber-50 dark:bg-amber-900"):
                ui.label("Der Index ist leer.").classes("font-medium")
                ui.label(
                    "Ohne aufgenommene Dokumente gibt es keine Quellen, aus "
                    "denen eine Antwort entstehen könnte."
                ).classes("text-sm")
                ui.button(
                    "Dokumente hinzufügen", on_click=lambda: ui.navigate.to("/dokumente")
                ).props("flat")

        question = (
            ui.textarea(placeholder="Wie lang ist die Kündigungsfrist?")
            .classes("w-full")
            .props("outlined autogrow")
        )

        with ui.row().classes("items-center gap-3"):
            send = ui.button("Fragen").props("unelevated")
            spinner = ui.spinner(size="sm")
            spinner.visible = False
            phase = ui.label("").classes("text-sm text-gray-500")

        answer_card = ui.card().classes("w-full")
        answer_card.visible = False
        with answer_card:
            answer_label = ui.markdown("").classes("w-full")
            meta_label = ui.label("").classes("text-xs text-gray-500 mt-2")

        sources_card = ui.card().classes("w-full")
        sources_card.visible = False
        with sources_card:
            ui.label("Quellen").classes("font-bold")
            sources_box = ui.column().classes("w-full gap-2")

    async def on_ask() -> None:
        text = (question.value or "").strip()
        if not text:
            ui.notify("Keine Frage eingegeben", type="warning")
            return

        send.disable()
        spinner.visible = True
        answer_card.visible = False
        sources_card.visible = False
        sources_box.clear()
        answer_label.content = ""

        try:
            phase.text = "sucht passende Stellen…"
            sources, stream = await _in_thread(current.pipeline.ask_stream, text)

            if not sources:
                phase.text = ""
                answer_card.visible = True
                answer_label.content = next(stream, "")
                return

            _render_sources(sources_box, sources)
            sources_card.visible = True

            phase.text = "formuliert die Antwort…"
            answer_card.visible = True

            # Token für Token über den Thread holen. Bei 2 bis 8 Token pro
            # Sekunde ist der Thread-Wechsel je Token kostenlos, und die
            # Antwort erscheint beim Entstehen statt am Ende.
            pieces: list[str] = []
            loop = asyncio.get_running_loop()
            started = loop.time()
            while True:
                piece = await asyncio.to_thread(next, stream, None)
                if piece is None:
                    break
                pieces.append(piece)
                answer_label.content = "".join(pieces)

            duration = loop.time() - started
            answer = "".join(pieces)
            cited = _cited_numbers(answer)
            _mark_cited(sources_box, sources, cited)

            rate = len(answer) / 3.2 / duration if duration else 0
            uncited = len(sources) - len(cited)
            note = f"{duration:.1f}s, etwa {rate:.1f} Token/s"
            if uncited:
                note += f" — {uncited} von {len(sources)} Quellen nicht zitiert"
            meta_label.text = note
            phase.text = ""
        except ERRORS as exc:
            phase.text = ""
            _notify_error(exc)
        finally:
            spinner.visible = False
            send.enable()

    send.on_click(on_ask)


def _render_sources(container, sources) -> None:
    container.clear()
    with container:
        for source in sources:
            with ui.row().classes("w-full items-start gap-2 no-wrap"):
                badge = ui.badge(str(source.number)).props("color=grey")
                badge._source_number = source.number  # für _mark_cited
                with ui.column().classes("gap-0 grow"):
                    ui.label(source.citation).classes("text-sm font-medium")
                    score = source.hit.rerank_score
                    detail = (
                        f"Rerank {score:+.3f}"
                        if score is not None
                        else f"Ähnlichkeit {source.hit.similarity:.3f}"
                    )
                    ui.label(detail).classes("text-xs text-gray-500")
                    with ui.expansion("Textstelle").classes("w-full"):
                        ui.markdown(source.hit.text).classes("text-sm")


def _mark_cited(container, sources, cited: set[int]) -> None:
    """Zitierte Quellen grün färben.

    Die Unterscheidung ist der schnellste Weg zu beurteilen, ob das Retrieval
    getroffen hat: viele graue Quellen heißen, dass breit gestreut wurde.
    """
    for element in container.descendants():
        number = getattr(element, "_source_number", None)
        if number is not None:
            element.props(f'color={"green" if number in cited else "grey"}')


def _cited_numbers(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", text) if n.isdigit()}


# ─── Seite: Dokumente ────────────────────────────────────────────────────────


@ui.page("/dokumente")
def page_documents() -> None:
    _layout("dokumente")
    current = _state()

    with ui.column().classes("w-full max-w-4xl mx-auto gap-4 p-4"):
        ui.label("Dokumente").classes("text-2xl font-bold")

        with ui.card().classes("w-full"):
            ui.label("Aus einem Pfad aufnehmen").classes("font-bold")
            ui.label(
                "Datei oder Verzeichnis auf diesem Rechner. Verzeichnisse "
                "werden rekursiv durchsucht; unbekannte Formate übersprungen."
            ).classes("text-sm text-gray-500")
            with ui.row().classes("w-full items-center gap-2"):
                path_input = (
                    ui.input(placeholder=str(Path.home() / "dokumente"))
                    .classes("grow")
                    .props("outlined dense")
                )
                add_path = ui.button("Aufnehmen").props("unelevated")

            with ui.row().classes("items-center gap-4"):
                force = ui.checkbox("Auch Unveränderte neu einlesen")
                prune = ui.checkbox("Verschwundene entfernen")
                ocr_mode = ui.select(
                    {None: "OCR automatisch", True: "OCR erzwingen", False: "kein OCR"},
                    value=None,
                ).props("dense outlined")

        with ui.card().classes("w-full"):
            ui.label("Dateien hochladen").classes("font-bold")
            ui.label(
                f"Landet in {UPLOAD_DIR} — der Index verweist auf Pfade, "
                "also brauchen die Dateien einen dauerhaften Ort."
            ).classes("text-sm text-gray-500")
            ui.upload(
                on_upload=lambda event: _handle_upload(event, upload_status),
                multiple=True,
                auto_upload=True,
            ).classes("w-full").props(
                f'accept="{",".join(sorted(SUPPORTED_SUFFIXES))}"'
            )
            upload_status = ui.label("").classes("text-sm")
            # Der Knopf gehört in diese Karte, nicht unter die Seite: er
            # bezieht sich auf das Verzeichnis, in das der Upload schreibt.
            ingest_uploads = ui.button("Hochgeladene aufnehmen").props("outline")

        progress_card = ui.card().classes("w-full")
        progress_card.visible = False
        with progress_card:
            progress_label = ui.label("").classes("font-medium")
            progress_bar = ui.linear_progress(value=0, show_value=False).classes(
                "w-full"
            )

        result_card = ui.card().classes("w-full")
        result_card.visible = False
        with result_card:
            ui.label("Ergebnis").classes("font-bold")
            result_box = ui.column().classes("w-full gap-1")

        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Im Index").classes("font-bold")
                ui.button(icon="refresh", on_click=lambda: refresh_documents()).props(
                    "flat dense"
                )
            documents_box = ui.column().classes("w-full gap-1")

    def refresh_documents() -> None:
        documents_box.clear()
        stats = current.pipeline.index_stats()
        with documents_box:
            if not stats.get("exists") or not stats.get("documents"):
                ui.label("Noch keine Dokumente aufgenommen.").classes(
                    "text-sm text-gray-500"
                )
                return
            tokens = f"{stats['tokens']:,}".replace(",", ".")
            ui.label(
                f"{stats['documents']} Dokumente, {stats['chunks']} Chunks, "
                f"{tokens} Token"
            ).classes("text-sm text-gray-500")
            for record in current.pipeline.store.documents():
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(Path(record.path).name).classes("text-sm")
                    ui.label(f"{record.chunk_count} Chunks").classes(
                        "text-xs text-gray-500"
                    )

    async def run_ingest(paths: list[Path]) -> None:
        add_path.disable()
        progress_card.visible = True
        result_card.visible = False
        progress_bar.value = 0
        progress_label.text = "startet…"

        loop = asyncio.get_running_loop()

        def on_progress(path: Path, position: int, total: int, phase: str) -> None:
            # Aus dem Arbeitsthread in den Event-Loop: NiceGUI-Elemente dürfen
            # nur dort angefasst werden.
            def update() -> None:
                progress_label.text = f"{position}/{total} — {path.name} ({phase})"
                progress_bar.value = position / total if total else 0

            loop.call_soon_threadsafe(update)

        try:
            report: IngestReport = await _in_thread(
                current.pipeline.ingest,
                paths,
                force=force.value,
                prune=prune.value,
                ocr=ocr_mode.value,
                progress=on_progress,
            )
            _render_ingest_result(result_box, report)
            result_card.visible = True
            refresh_documents()
        except ERRORS as exc:
            _notify_error(exc)
        finally:
            progress_card.visible = False
            add_path.enable()

    async def on_add_path() -> None:
        raw = (path_input.value or "").strip()
        if not raw:
            ui.notify("Kein Pfad angegeben", type="warning")
            return
        target = Path(raw).expanduser()
        if not target.exists():
            ui.notify(f"Nicht gefunden: {target}", type="negative")
            return
        await run_ingest([target])

    async def on_ingest_uploads() -> None:
        await run_ingest([UPLOAD_DIR])

    add_path.on_click(on_add_path)
    ingest_uploads.on_click(on_ingest_uploads)
    ui.timer(0.1, refresh_documents, once=True)


def _handle_upload(event, status_label) -> None:
    """Hochgeladene Datei an ihren dauerhaften Ort schreiben."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / Path(event.name).name
    try:
        with open(target, "wb") as handle:
            shutil.copyfileobj(event.content, handle)
    except OSError as exc:
        status_label.text = f"Fehler bei {event.name}: {exc}"
        return
    status_label.text = (
        f"{target.name} gespeichert — mit „Hochgeladene aufnehmen“ indizieren"
    )


def _render_ingest_result(container, report: IngestReport) -> None:
    container.clear()
    colors = {
        "neu": "text-green-600",
        "aktualisiert": "text-blue-600",
        "unverändert": "text-gray-500",
        "leer": "text-amber-600",
        "fehler": "text-red-600",
    }
    with container:
        ui.label(
            f"{len(report.changed)} verarbeitet, {len(report.skipped)} unverändert, "
            f"{report.chunk_count} Chunks, {report.duration_seconds:.1f}s"
        ).classes("text-sm font-medium")
        for result in report.results:
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(result.path.name).classes("text-sm grow")
                if result.ocr_used:
                    ui.badge("OCR").props("color=amber")
                ui.label(result.status).classes(
                    f"text-sm {colors.get(result.status, '')}"
                )
                if result.chunk_count:
                    ui.label(f"{result.chunk_count} Chunks").classes(
                        "text-xs text-gray-500"
                    )
            if result.error:
                ui.label(result.error).classes("text-xs text-red-600 ml-4")
            for warning in result.warnings:
                ui.label(warning).classes("text-xs text-amber-600 ml-4")
        if report.removed:
            ui.label(f"{len(report.removed)} verschwundene entfernt").classes(
                "text-xs text-gray-500"
            )


# ─── Seite: Modelle ──────────────────────────────────────────────────────────


@ui.page("/modelle")
def page_models() -> None:
    _layout("modelle")
    current = _state()

    with ui.column().classes("w-full max-w-4xl mx-auto gap-4 p-4"):
        ui.label("Modelle").classes("text-2xl font-bold")
        ui.label(
            f"Plattform: {current.settings.platform_label or 'unbekannt'} "
            f"→ Klasse {current.settings.platform_class or '?'}"
        ).classes("text-sm text-gray-500")

        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Welche Rolle welches Modell benutzt").classes("font-bold")
                with ui.row().classes("gap-2"):
                    scan_button = ui.button("Nach bestem Modell suchen").props(
                        "unelevated"
                    )
                    ui.button(icon="refresh", on_click=lambda: refresh(False)).props(
                        "flat dense"
                    )
            ui.label(
                "Der Generator wird von whichllm gegen die erkannte Hardware "
                "gerankt. Der erste Lauf dauert bis zu drei Minuten, danach "
                "gilt 24 Stunden der Cache."
            ).classes("text-sm text-gray-500")
            models_box = ui.column().classes("w-full gap-2 mt-2")

        download_card = ui.card().classes("w-full")
        download_card.visible = False
        with download_card:
            ui.label("Generator herunterladen").classes("font-bold")
            download_label = ui.label("").classes("text-sm")
            download_button = ui.button("Herunterladen").props("unelevated")
            download_spinner = ui.spinner(size="sm")
            download_spinner.visible = False

    def refresh(resolve_generator: bool) -> None:
        models_box.clear()
        rows = current.pipeline.model_status(resolve_generator=resolve_generator)
        with models_box:
            for row in rows:
                with ui.row().classes("w-full items-center gap-3 no-wrap"):
                    ui.label(row.role).classes("text-sm font-bold w-24")
                    with ui.column().classes("gap-0 grow"):
                        ui.label(row.model_id).classes("text-sm")
                        if row.detail:
                            ui.label(row.detail).classes("text-xs text-gray-500")
                    ui.badge(row.device).props(
                        f'color={"green" if row.device == "cuda" else "blue-grey"}'
                    )
                    ui.label(row.source).classes("text-xs text-gray-500 w-32")
                    if row.loaded:
                        ui.badge("geladen").props("color=green")
                    elif row.available:
                        ui.badge("bereit").props("color=grey")
                    else:
                        ui.badge("fehlt").props("color=orange")

        generator = next((r for r in rows if r.role == "generator"), None)
        if generator and not generator.available and "GGUF" in (generator.detail or ""):
            download_card.visible = True
            download_label.text = generator.detail or ""
        else:
            download_card.visible = False

    async def on_scan() -> None:
        scan_button.disable()
        models_box.clear()
        with models_box:
            ui.spinner(size="lg")
            ui.label(
                "whichllm rankt Modelle gegen die Hardware — das dauert."
            ).classes("text-sm text-gray-500")
        try:
            # Die Auflösung im Thread, damit die Oberfläche antwortet.
            await _in_thread(lambda: current.pipeline.plan)
            refresh(True)
            ui.notify("Modell-Scan abgeschlossen", type="positive")
        except ERRORS as exc:
            _notify_error(exc)
            refresh(False)
        finally:
            scan_button.enable()

    async def on_download() -> None:
        download_button.disable()
        download_spinner.visible = True
        try:
            spec = current.pipeline.plan.generator
            await _in_thread(
                gguf_path,
                spec.artifact_repo_id,
                spec.artifact_filename,
                download=True,
            )
            ui.notify("Modell heruntergeladen", type="positive")
            refresh(True)
        except ERRORS as exc:
            _notify_error(exc)
        finally:
            download_spinner.visible = False
            download_button.enable()

    scan_button.on_click(on_scan)
    download_button.on_click(on_download)
    ui.timer(0.1, lambda: refresh(False), once=True)


# ─── Seite: Einstellungen ────────────────────────────────────────────────────


@ui.page("/einstellungen")
def page_settings() -> None:
    _layout("einstellungen")
    current = _state()
    settings = current.settings

    with ui.column().classes("w-full max-w-4xl mx-auto gap-4 p-4"):
        ui.label("Einstellungen").classes("text-2xl font-bold")
        ui.label(
            f"Vorgaben aus Plattformklasse {settings.platform_class or '?'}; "
            f"gespeichert in {SETTINGS_PATH}"
        ).classes("text-sm text-gray-500")

        with ui.card().classes("w-full"):
            ui.label("Vektordatenbank").classes("font-bold")
            index_input = (
                ui.input("Index-Datei", value=str(settings.index_path))
                .classes("w-full")
                .props("outlined dense")
            )
            ui.label(
                "SQLite mit sqlite-vec, eine Datei. Ein Wechsel des "
                "Embedding-Modells macht einen bestehenden Index unbrauchbar "
                "— er wird dann abgelehnt, nicht still weiterbenutzt."
            ).classes("text-xs text-gray-500")

            stats = current.pipeline.index_stats()
            if stats.get("exists"):
                tokens = f"{stats['tokens']:,}".replace(",", ".")
                ui.label(
                    f"Aktuell: {stats['documents']} Dokumente, "
                    f"{stats['chunks']} Chunks, {tokens} Token, "
                    f"Embedder {stats.get('embedder')}"
                ).classes("text-xs text-gray-500")

            chunk_input = ui.number(
                "Chunk-Zielgröße in Token",
                value=settings.chunk_target_tokens,
                min=128,
                max=2048,
                step=64,
            ).props("outlined dense")
            ui.label(
                "Wirkt erst bei neu aufgenommenen Dokumenten. Größer heißt "
                "mehr Zusammenhang je Chunk, aber unschärfere Vektoren."
            ).classes("text-xs text-gray-500")

        with ui.card().classes("w-full"):
            ui.label("Embedder").classes("font-bold")
            embedder_profile = ui.select(
                _profiles("embedder"),
                value=settings.embedder_profile,
                label="Profil",
            ).props("outlined dense")
            embedder_device = ui.select(
                ["auto", "cpu", "gpu", "cuda", "mps"],
                value=settings.embedder_device,
                label="Gerät",
            ).props("outlined dense")
            ui.label(
                "Das Profil des bestehenden Index gewinnt gegen diese "
                "Einstellung — gesucht werden muss mit dem Modell, mit dem "
                "indiziert wurde."
            ).classes("text-xs text-gray-500")

        with ui.card().classes("w-full"):
            ui.label("Retrieval").classes("font-bold")
            rerank_switch = ui.switch(
                "Reranking", value=settings.reranker_enabled
            )
            ui.label(
                "Ein Cross-Encoder liest Frage und Textstelle zusammen und "
                "ordnet neu. Kostet Zeit: auf CPU verdoppelt das die "
                "Antwortlatenz."
            ).classes("text-xs text-gray-500")
            candidates_input = ui.number(
                "Kandidaten aus der Vektorsuche",
                value=settings.candidates,
                min=1,
                max=200,
            ).props("outlined dense")
            top_k_input = ui.number(
                "Quellen im Prompt", value=settings.top_k, min=1, max=50
            ).props("outlined dense")
            min_score_input = ui.number(
                "Mindest-Relevanz nach Reranking",
                value=settings.min_rerank_score,
                min=0,
                max=1,
                step=0.005,
                format="%.3f",
            ).props("outlined dense")
            ui.label(
                "0 heißt: nicht filtern. Über 0 fallen sachfremde Stellen aus "
                "dem Prompt, die die Quellenzahl sonst auffüllt. Gemessen "
                "lagen treffende Stellen bei etwa 0,03 und sachfremde bei "
                "0,001 — die Skala hängt am Modell, also an eigenen "
                "Dokumenten prüfen."
            ).classes("text-xs text-gray-500")

        with ui.card().classes("w-full"):
            ui.label("Generator").classes("font-bold")
            generator_switch = ui.switch(
                "Antworten erzeugen", value=settings.generator_enabled
            )
            ui.label(
                "Aus heißt: die Suche liefert Textstellen, aber keine "
                "formulierte Antwort. Spart das Laden mehrerer Gigabyte."
            ).classes("text-xs text-gray-500")
            context_input = ui.number(
                "Kontextfenster in Token",
                value=settings.generator_context_length,
                min=2048,
                max=131072,
                step=2048,
            ).props("outlined dense")
            max_tokens_input = ui.number(
                "Maximale Antwortlänge",
                value=settings.max_tokens,
                min=64,
                max=4096,
                step=64,
            ).props("outlined dense")
            temperature_input = ui.number(
                "Temperatur",
                value=settings.temperature,
                min=0,
                max=1,
                step=0.05,
                format="%.2f",
            ).props("outlined dense")
            ui.label(
                "Niedrig halten. Bei RAG steht die Antwort im Kontext — "
                "Kreativität ist hier ein Fehler, nicht eine Eigenschaft."
            ).classes("text-xs text-gray-500")
            threads_input = ui.number(
                "CPU-Threads (0 = llama.cpp entscheiden lassen)",
                value=settings.generator_threads or 0,
                min=0,
                max=128,
            ).props("outlined dense")
            ui.label(
                "Mehr ist nicht schneller: auf dieser Maschine waren alle 24 "
                "logischen Kerne siebenmal langsamer als die Vorgabe "
                "(1,2 gegen 8,3 Token/s). Im Zweifel bei 0 lassen."
            ).classes("text-xs text-amber-600")

        with ui.row().classes("gap-2"):
            save_button = ui.button("Speichern").props("unelevated")
            ui.button("Auf Plattformvorgaben zurücksetzen", on_click=lambda: reset()).props(
                "outline"
            )

    def reset() -> None:
        fresh = Settings.for_platform()
        fresh = replace(fresh, index_path=current.settings.index_path)
        current.settings = fresh
        current.pipeline = RagPipeline(fresh)
        fresh.save()
        ui.notify("Auf Plattformvorgaben zurückgesetzt", type="positive")
        ui.navigate.to("/einstellungen")

    def save() -> None:
        try:
            changes = {
                "index_path": Path(index_input.value).expanduser(),
                "chunk_target_tokens": int(chunk_input.value),
                "embedder_profile": embedder_profile.value,
                "embedder_device": embedder_device.value,
                "reranker_enabled": bool(rerank_switch.value),
                "candidates": int(candidates_input.value),
                "top_k": int(top_k_input.value),
                "min_rerank_score": float(min_score_input.value),
                "generator_enabled": bool(generator_switch.value),
                "generator_context_length": int(context_input.value),
                "max_tokens": int(max_tokens_input.value),
                "temperature": float(temperature_input.value),
                "generator_threads": int(threads_input.value) or None,
            }
            current.rebuild(**changes)
            path = current.settings.save()
            ui.notify(f"Gespeichert in {path}", type="positive")
        except (ERRORS, ValueError, TypeError) as exc:
            _notify_error(exc)

    save_button.on_click(save)


def _profiles(role: str) -> list[str]:
    """Verfügbare Profile einer Rolle aus platforms.toml."""
    from rag.detect import load_config

    return sorted(load_config().get(role, {})) or ["default"]


# ─── Rahmen ──────────────────────────────────────────────────────────────────


def _layout(active: str) -> None:
    """Kopfzeile mit Navigation, auf jeder Seite gleich."""
    seiten = [
        ("fragen", "Fragen", "/", "chat"),
        ("dokumente", "Dokumente", "/dokumente", "folder"),
        ("modelle", "Modelle", "/modelle", "memory"),
        ("einstellungen", "Einstellungen", "/einstellungen", "settings"),
    ]
    with ui.header().classes("items-center justify-between px-4"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("travel_explore", size="sm")
            ui.label("local-rag").classes("text-lg font-bold")
        with ui.row().classes("gap-1"):
            for key, title, target, icon in seiten:
                button = ui.button(
                    title, icon=icon, on_click=lambda t=target: ui.navigate.to(t)
                )
                # color=white für alle: ein flat-Button rendert sonst in der
                # Primärfarbe, und die ist genau die Farbe des Headers — drei
                # von vier Navigationspunkten waren dadurch unsichtbar. Der
                # aktive wird über Schriftschnitt hervorgehoben, nicht über
                # Farbe.
                button.props("flat color=white")
                button.classes(
                    "font-bold underline" if key == active else "opacity-70"
                )
        ui.switch("dunkel", on_change=lambda event: _toggle_dark(event.value)).props(
            "dense"
        )


def _dark_mode() -> ui.dark_mode:
    if not hasattr(_dark_mode, "_instance"):
        _dark_mode._instance = ui.dark_mode()
    return _dark_mode._instance


def _toggle_dark(enabled: bool) -> None:
    mode = _dark_mode()
    mode.enable() if enabled else mode.disable()


def run(
    *,
    index_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    """Oberfläche starten. Blockiert bis zum Beenden."""
    global state
    state = State(index_path)

    nicegui_app.on_shutdown(lambda: state.pipeline.close() if state else None)

    ui.run(
        host=host,
        port=port,
        title="local-rag",
        favicon="🔍",
        show=open_browser,
        reload=False,
        # Ein lokales Werkzeug für einen Nutzer: kein Grund, im Netz zu lauschen.
        dark=None,
    )
