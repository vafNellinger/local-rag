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
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

from nicegui import app as nicegui_app
from nicegui import ui

from rag.embed import EmbeddingError
from rag.extract import SUPPORTED_SUFFIXES
from rag.generate import GenerationError, Turn, gguf_path
from rag.ingest import IngestReport
from rag.pipeline import (
    SETTINGS_PATH,
    PipelineError,
    RagPipeline,
    Settings,
)
from rag.rerank import RerankError
from rag.resolve import ResolutionError, backend_options
from rag.detect import WhichllmError, load_config
from rag.paths import LOG_PATH, UPLOAD_DIR
from rag.store import StoreError, read_index_documents
from rag.vectors import BACKENDS

logger = logging.getLogger(__name__)

# Wohin hochgeladene Dateien wandern. Der Index verweist auf Pfade, also
# müssen sie einen dauerhaften Ort haben — ein Temp-Verzeichnis wäre nach dem
# nächsten Neustart weg und der Index zeigte ins Leere. UPLOAD_DIR (data) und
# LOG_PATH (cache) kommen plattformkorrekt aus rag.paths.

# Ab welcher Größe das Protokoll umgewälzt wird, und wie viele Generationen
# behalten werden. Klein gehalten: interessant ist der letzte Fehler, nicht
# die Geschichte.
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUPS = 2

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


@dataclass
class ChatMessage:
    """Ein abgeschlossener Wechsel im Chat, mit allem für die Anzeige.

    Reicher als ``generate.Turn``: die Oberfläche zeigt beim Zurückkommen auf
    die Seite auch die Quellen wieder, die der Prompt-Verlauf bewusst weglässt.
    Für die Pipeline wird daraus ein schlankes ``Turn`` (siehe ``as_turn``).
    """

    question: str
    answer: str
    sources: list = field(default_factory=list)
    cited: set = field(default_factory=set)

    def as_turn(self) -> Turn:
        # Die Quellen gehen mit: der Router prüft an ihnen, ob eine Folgefrage
        # ohne neue Suche auskommt. Der Prompt-Bau ignoriert sie weiterhin.
        return Turn(
            question=self.question, answer=self.answer, sources=self.sources
        )


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
        # Der Chatverlauf. Lebt im Prozess und überdauert damit den Wechsel
        # zwischen den Seiten der Oberfläche, aber keinen Neustart — genau die
        # Lebensdauer, die "nur diese Sitzung" meint. "Neuer Chat" leert ihn.
        self.history: list[ChatMessage] = []

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
    # exception statt debug: in der Oberfläche ist der Fehler sonst nur als
    # Kurztext im Browser sichtbar und nirgends nachlesbar. Genau das machte
    # einen fehlgeschlagenen Upload unauffindbar.
    logger.exception("Fehler in der Oberfläche: %s", exc)
    ui.notify(str(exc), type="negative", multi_line=True, close_button="ok")


# ─── Seite: Fragen ───────────────────────────────────────────────────────────


@ui.page("/")
def page_ask() -> None:
    _layout("fragen")
    current = _state()

    with ui.column().classes("w-full max-w-4xl mx-auto gap-4 p-4"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Chat mit den Dokumenten").classes("text-2xl font-bold")
            # Der einzige Weg, den Verlauf zu leeren — er lebt sonst bis zum
            # Prozessende. Ohne Verlauf ausgeblendet, sonst lädt der Knopf zum
            # Leeren eines leeren Chats ein.
            neu_button = ui.button("Neuer Chat", icon="add").props("flat")
            neu_button.visible = bool(current.history)

        stats = current.pipeline.index_stats()
        if fehler := stats.get("error"):
            # Ein vorhandener, aber unlesbarer Index ist nicht dasselbe wie ein
            # leerer: „Dokumente hinzufügen“ wäre hier der falsche Rat.
            with ui.card().classes("w-full bg-red-50 dark:bg-red-900"):
                ui.label("Der Index ist nicht lesbar.").classes("font-medium")
                ui.label(str(fehler)).classes("text-sm")
                ui.button(
                    "Zu den Einstellungen",
                    on_click=lambda: ui.navigate.to("/einstellungen"),
                ).props("flat")
        elif not stats.get("chunks"):
            with ui.card().classes("w-full bg-amber-50 dark:bg-amber-900"):
                ui.label("Der Index ist leer.").classes("font-medium")
                ui.label(
                    "Ohne aufgenommene Dokumente gibt es keine Quellen, aus "
                    "denen eine Antwort entstehen könnte."
                ).classes("text-sm")
                ui.button(
                    "Dokumente hinzufügen", on_click=lambda: ui.navigate.to("/dokumente")
                ).props("flat")

        # Der Gesprächsverlauf. Wächst nach unten; die Eingabe steht darunter.
        verlauf = ui.column().classes("w-full gap-4")

        question = (
            ui.textarea(placeholder="Wie lang ist die Kündigungsfrist?")
            .classes("w-full")
            .props("outlined autogrow")
        )
        ui.label("Enter fragt, Shift+Enter macht eine neue Zeile.").classes(
            "text-xs text-gray-500"
        )

        with ui.row().classes("items-center gap-3"):
            send = ui.button("Fragen").props("unelevated")
            spinner = ui.spinner(size="sm")
            spinner.visible = False
            phase = ui.label("").classes("text-sm text-gray-500")

    # Den bestehenden Verlauf zeichnen: nach einem Seitenwechsel (Fragen →
    # Dokumente → zurück) baut NiceGUI die Seite neu auf, der Zustand liegt aber
    # im State und ist noch da.
    with verlauf:
        for msg in current.history:
            _render_message(msg.question, msg.answer, msg.sources, msg.cited)

    async def scroll_ans_ende() -> None:
        # Nach dem Anhängen ans DOM, sonst ist die neue Höhe noch nicht bekannt.
        await ui.run_javascript(
            "window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})"
        )

    def neuer_chat() -> None:
        current.history.clear()
        verlauf.clear()
        neu_button.visible = False

    neu_button.on_click(neuer_chat)

    # Läuft gerade eine Anfrage? Der deaktivierte Knopf allein reicht als
    # Sperre nicht: die Enter-Taste unten geht an ihm vorbei. Zwei überlappende
    # Anfragen liefen auf denselben llama.cpp-Kontext, dessen Buchhaltung über
    # ``_scores`` und ``n_tokens`` das nicht verträgt — der Fehler kam als
    # "index 3115 is out of bounds for axis 0 with size 154" aus der
    # Bibliothek und war der Oberfläche nicht anzusehen.
    laeuft = {"wert": False}

    async def on_ask() -> None:
        if laeuft["wert"]:
            ui.notify("Läuft noch", type="warning")
            return

        text = (question.value or "").strip()
        if not text:
            ui.notify("Keine Frage eingegeben", type="warning")
            return

        laeuft["wert"] = True
        send.disable()
        spinner.visible = True
        question.value = ""
        neu_button.visible = True

        # Die Frage sofort als Blase zeigen, dann einen leeren Antwortblock, der
        # während des Streamings gefüllt wird.
        with verlauf:
            _user_bubble(text)
            answer_col = ui.column().classes("w-full gap-1")
            with answer_col:
                answer_label = ui.markdown("").classes("w-full")
                meta_label = ui.label("").classes("text-xs text-gray-500")
        await scroll_ans_ende()

        try:
            # Der Verlauf *vor* dieser Frage — die neue Frage steht noch nicht
            # drin, sonst bezöge der Router sie auf sich selbst.
            verlauf_turns = [m.as_turn() for m in current.history]
            # Bei Verlauf prüft der Router erst die vorhandenen Quellen; die
            # Phase soll sagen, was wirklich passiert.
            phase.text = (
                "prüft die bisherigen Quellen…"
                if verlauf_turns
                else "sucht passende Stellen…"
            )
            sources, stream, wiederverwendet = await _in_thread(
                current.pipeline.ask_stream, text, history=verlauf_turns
            )

            if not sources:
                phase.text = ""
                antwort = next(stream, "")
                answer_label.content = antwort
                # Auch die quellenlose Antwort gehört in den Verlauf: sonst
                # bezöge sich eine Folgefrage auf eine Lücke.
                current.history.append(ChatMessage(question=text, answer=antwort))
                return

            phase.text = "formuliert die Antwort…"

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

            with answer_col:
                _sources_expansion(sources, cited)

            rate = len(answer) / 3.2 / duration if duration else 0
            uncited = len(sources) - len(cited)
            note = f"{duration:.1f}s, etwa {rate:.1f} Token/s"
            if wiederverwendet:
                # Sichtbar machen, dass keine neue Suche lief — sonst wundert
                # man sich über dieselben Quellen und die fehlende Suchphase.
                note += " · aus den bisherigen Quellen, ohne neue Suche"
            if uncited:
                note += f" — {uncited} von {len(sources)} Quellen nicht zitiert"
            meta_label.text = note
            phase.text = ""

            current.history.append(
                ChatMessage(
                    question=text, answer=answer, sources=sources, cited=cited
                )
            )
        except ERRORS as exc:
            phase.text = ""
            _notify_error(exc)
        finally:
            laeuft["wert"] = False
            spinner.visible = False
            send.enable()
            await scroll_ans_ende()

    send.on_click(on_ask)

    # Enter fragt, Shift+Enter macht eine neue Zeile.
    #
    # Das entscheidet der Browser und nicht der Server: Vues Modifier ``.exact``
    # wäre der naheliegende Weg, steht aber nicht in NiceGUIs Modifier-Liste
    # (nur stop, prevent, self, ctrl, shift, alt, meta) und würde dort als
    # *Tastenname* landen. Ein reines ``keydown.enter.prevent`` unterdrückte
    # umgekehrt auch den erwünschten Umbruch bei Shift+Enter.
    #
    # Also ein js_handler, der selbst entscheidet: nur beim Enter ohne Shift
    # den Standardumbruch verhindern und an den Server melden. Bei Shift+Enter
    # passiert hier nichts, und die Textarea fügt ihre neue Zeile ganz normal
    # selbst ein — samt richtiger Cursorposition, die eine serverseitige Lösung
    # von Hand nachbilden müsste.
    question.on(
        "keydown",
        on_ask,
        js_handler=(
            "(e) => { if (e.key === 'Enter' && !e.shiftKey) "
            "{ e.preventDefault(); emit(); } }"
        ),
    )


def _user_bubble(text: str) -> None:
    """Die Frage als rechtsbündige Blase, wie in einem Chat.

    ``whitespace-pre-wrap``, damit eine mehrzeilige Frage (Shift+Enter) ihre
    Zeilenumbrüche behält statt zu einem Absatz zu verlaufen.
    """
    with ui.row().classes("w-full justify-end"):
        ui.label(text).classes(
            "bg-primary text-white rounded-2xl px-4 py-2 "
            "max-w-[85%] whitespace-pre-wrap"
        )


def _render_message(question: str, answer: str, sources, cited) -> None:
    """Einen abgeschlossenen Wechsel zeichnen — für die Wiederherstellung.

    Der Live-Pfad in ``on_ask`` baut dieselbe Struktur Stück für Stück auf;
    diese Funktion setzt sie in einem Zug, wenn der Verlauf aus dem State
    wiederhergestellt wird.
    """
    _user_bubble(question)
    with ui.column().classes("w-full gap-1"):
        ui.markdown(answer).classes("w-full")
        if sources:
            _sources_expansion(sources, cited)


def _sources_expansion(sources, cited) -> None:
    """Quellen als aufklappbaren Block. Läuft im Kontext eines Containers."""
    with ui.expansion(f"{len(sources)} Quellen").classes("w-full"):
        for source in sources:
            with ui.row().classes("w-full items-start gap-2 no-wrap"):
                farbe = "green" if source.number in cited else "grey"
                ui.badge(str(source.number)).props(f"color={farbe}")
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
            if fehler := stats.get("error"):
                ui.label(f"Index nicht lesbar: {fehler}").classes(
                    "text-sm text-red-600"
                )
                return
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


async def _handle_upload(event, status_label) -> None:
    """Hochgeladene Datei an ihren dauerhaften Ort schreiben.

    ``event.file`` ist ein ``FileUpload`` mit ``name`` und einem
    **asynchronen** ``save()`` — deshalb ist dieser Handler eine Coroutine.
    Eine frühere Fassung griff auf ``event.name`` und ``event.content`` zu,
    was einer älteren NiceGUI-Schnittstelle entsprach und mit
    ``AttributeError`` scheiterte.
    """
    upload = event.file
    # basename gegen Pfadanteile im Dateinamen: ein hochgeladenes
    # "../../.bashrc" darf nicht aus dem Zielverzeichnis herausführen.
    name = Path(upload.name).name
    if not name:
        status_label.text = "Datei ohne Namen — übersprungen"
        return

    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        # Das accept-Attribut im Browser ist nur ein Vorschlag; abgelehnt wird
        # hier, damit keine Datei im Zielverzeichnis landet, die der Ingest
        # anschließend als Fehler melden müsste.
        status_label.text = (
            f"{name}: Format {suffix or '(ohne Endung)'} wird nicht "
            f"unterstützt. Möglich: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
        logger.warning("Upload abgelehnt: %s", name)
        return

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / name
    try:
        await upload.save(target)
    except OSError as exc:
        logger.exception("Upload von %s fehlgeschlagen", name)
        status_label.text = f"Fehler bei {name}: {exc}"
        return

    size_kb = target.stat().st_size / 1024
    logger.info("Upload gespeichert: %s (%.1f KB)", target, size_kb)
    status_label.text = (
        f"{name} gespeichert ({size_kb:.0f} KB) — "
        "mit „Hochgeladene aufnehmen“ indizieren"
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
                "Dokumente, Texte und Metadaten liegen immer in dieser "
                "SQLite-Datei. Ein Wechsel des Embedding-Modells macht einen "
                "bestehenden Index unbrauchbar — er wird dann abgelehnt, nicht "
                "still weiterbenutzt."
            ).classes("text-xs text-gray-500")

            # Gehört in diese Karte und nicht in eine eigene: „Vektordatenbank“
            # und „Vektor-Backend“ nebeneinander lasen sich wie zwei Themen,
            # sind aber die Frage nach demselben Speicherort.
            vector_backend = ui.select(
                list(BACKENDS),
                value=settings.vector_backend,
                label="Vektor-Backend",
            ).props("outlined dense")
            ui.label(
                "Wo die Vektoren liegen. sqlite-vec hält sie in der Datei "
                "oben; die übrigen legen ein Verzeichnis daneben und brauchen "
                "eine eigene Installation. Bis zum Neuaufbau gilt das Backend "
                "des Index — die Vektoren liegen ja dort."
            ).classes("text-xs text-gray-500")

            stats = current.pipeline.index_stats()
            if fehler := stats.get("error"):
                # Die Seite, auf der man es beheben kann — hier muss der Grund
                # stehen, nicht nur „keine Kennzahlen“.
                ui.label(f"Index nicht lesbar: {fehler}").classes(
                    "text-xs text-red-600"
                )
            elif stats.get("exists"):
                tokens = f"{stats['tokens']:,}".replace(",", ".")
                ui.label(
                    f"Aktuell: {stats['documents']} Dokumente, "
                    f"{stats['chunks']} Chunks, {tokens} Token, "
                    f"Embedder {stats.get('embedder')}, "
                    f"Vektoren {stats.get('vector_backend')}"
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
                "Ein anderes Modell macht den bestehenden Index unbrauchbar: "
                "Vektoren verschiedener Modelle sind nicht vergleichbar. Bis "
                "zum Neuaufbau gilt weiter das Modell des Index."
            ).classes("text-xs text-gray-500")

            # Der Konflikt gehört hierhin und nicht ins Protokoll: wer das
            # Profil umstellt und "Gespeichert" liest, muss erfahren, dass
            # noch nichts davon wirkt.
            conflict_card = ui.card().classes("w-full bg-amber-50 dark:bg-amber-900")
            with conflict_card:
                conflict_label = ui.label("").classes("text-sm font-medium")
                rebuild_button = ui.button("Index jetzt neu aufbauen").props(
                    "unelevated color=amber-8"
                )
                rebuild_hint = ui.label("").classes("text-xs")
                rebuild_progress = ui.linear_progress(value=0, show_value=False)
                rebuild_progress.visible = False

            def refresh_conflict() -> None:
                # Beide Konflikte teilen sich diese Karte: in beiden Fällen ist
                # die Abhilfe derselbe Neuaufbau, und zwei Warnkarten
                # übereinander lesen sich wie zwei Probleme.
                conflict = current.pipeline.profile_conflict()
                backend = current.pipeline.backend_conflict()
                if not conflict and not backend:
                    conflict_card.visible = False
                    return

                anzahl = len(read_index_documents(current.settings.index_path))
                conflict_card.visible = True

                texte = []
                if conflict:
                    im_index, gewuenscht = conflict
                    texte.append(
                        f"Der Index wurde mit Profil „{im_index}“ gebaut, "
                        f"eingestellt ist „{gewuenscht}“. Es gilt weiter "
                        f"„{im_index}“."
                    )
                if backend:
                    im_index, gewuenscht = backend
                    texte.append(
                        f"Die Vektoren liegen in „{im_index}“, eingestellt ist "
                        f"„{gewuenscht}“. Es gilt weiter „{im_index}“."
                    )
                conflict_label.text = " ".join(texte)
                rebuild_hint.text = (
                    f"{anzahl} Dokument(e) werden neu eingelesen. Die "
                    "Extraktion kommt aus dem Cache, neu berechnet werden nur "
                    "die Vektoren."
                )

            refresh_conflict()

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

    async def on_rebuild() -> None:
        rebuild_button.disable()
        rebuild_progress.visible = True
        rebuild_progress.value = 0
        loop = asyncio.get_running_loop()

        def on_progress(path: Path, position: int, total: int, phase: str) -> None:
            def update() -> None:
                rebuild_hint.text = f"{position}/{total} — {path.name} ({phase})"
                rebuild_progress.value = position / total if total else 0

            loop.call_soon_threadsafe(update)

        try:
            report = await _in_thread(
                current.pipeline.rebuild_index, progress=on_progress
            )
            ui.notify(
                f"Index neu aufgebaut: {len(report.changed)} Dokument(e), "
                f"{report.chunk_count} Chunks, {report.duration_seconds:.0f}s",
                type="positive",
            )
            refresh_conflict()
        except ERRORS as exc:
            _notify_error(exc)
        finally:
            rebuild_progress.visible = False
            rebuild_button.enable()

    rebuild_button.on_click(on_rebuild)

    def save() -> None:
        try:
            changes = {
                "index_path": Path(index_input.value).expanduser(),
                "chunk_target_tokens": int(chunk_input.value),
                "embedder_profile": embedder_profile.value,
                "embedder_device": embedder_device.value,
                "vector_backend": vector_backend.value,
                "vector_backend_options": backend_options(
                    load_config(), str(vector_backend.value)
                ),
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
            refresh_conflict()
            if conflict := current.pipeline.profile_conflict():
                # Nicht bloß "Gespeichert" melden, wenn die Einstellung noch
                # nicht wirkt — das war der Fehler, den es hier zu vermeiden gilt.
                im_index, _ = conflict
                ui.notify(
                    f"Gespeichert, aber noch nicht wirksam: der Index läuft "
                    f"weiter mit „{im_index}“. Erst der Neuaufbau schaltet um.",
                    type="warning",
                    multi_line=True,
                    close_button="ok",
                )
            else:
                ui.notify(f"Gespeichert in {path}", type="positive")
        except (ERRORS, ValueError, TypeError) as exc:
            _notify_error(exc)

    save_button.on_click(save)


def _profiles(role: str) -> list[str]:
    """Verfügbare Profile einer Rolle aus platforms.toml."""
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


def setup_logging(*, verbose: bool = False, log_path: Path | None = None) -> Path:
    """Protokoll in eine Datei und auf die Konsole legen.

    Existiert, weil ein Fehler in der Oberfläche sonst spurlos verschwindet:
    im Browser steht eine Kurzmeldung, der Traceback landet nirgends. Ein
    fehlgeschlagener Upload war deshalb nicht nachvollziehbar, ohne den
    Serverprozess selbst zu beobachten.
    """
    from logging.handlers import RotatingFileHandler

    target = log_path or LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("rag")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Doppelte Handler vermeiden, falls run() im selben Prozess erneut läuft.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = RotatingFileHandler(
        target, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(console)

    # Unbehandelte Fehler in NiceGUIs Event-Behandlung landen in dessen
    # Logger, nicht in unserem — ohne diese Zeile stünde der Traceback eines
    # abgestürzten Handlers nicht in der Datei.
    nicegui_logger = logging.getLogger("nicegui")
    nicegui_logger.setLevel(logging.INFO)
    for handler in (file_handler, console):
        if handler not in nicegui_logger.handlers:
            nicegui_logger.addHandler(handler)

    return target


def _webview_verfuegbar() -> bool:
    """Ob ein natives Fenster möglich ist — sonst übernimmt der Browser.

    Geprüft wird, was sich vorab prüfen lässt: dass pywebview da ist und auf
    Linux auch seine WebKitGTK-Bindung. Genau dort liegt die Falle — pywebview
    kann installiert sein, während ``gi``/WebKit2 fehlen; dann scheitert das
    Fenster erst beim Öffnen. Der Linux-Zweig fängt das vorher ab. Auf Windows
    (Edge WebView2) und macOS (Cocoa) genügt der pywebview-Import; ob die
    Anzeige am Ende steht, zeigt sich erst zur Laufzeit auf einem Rechner mit
    Bildschirm.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("webview") is None:
        return False

    if sys.platform.startswith("linux"):
        for version in ("4.1", "4.0"):
            try:
                import gi

                gi.require_version("WebKit2", version)
                from gi.repository import WebKit2  # noqa: F401

                return True
            except Exception:
                continue
        return False

    return True


def run(
    *,
    index_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True,
    native: bool = False,
    verbose: bool = False,
    reload: bool = False,
) -> None:
    """Oberfläche starten. Blockiert bis zum Beenden.

    ``native`` öffnet ein eigenständiges Fenster (NiceGUIs native Modus über
    pywebview) statt eines Browser-Tabs. Fehlt pywebview, wird gewarnt und im
    Browser geöffnet — lieber ein Tab als gar keine Oberfläche.

    ``reload`` lässt uvicorn die Python-Dateien des Pakets überwachen und den
    Server bei jeder Änderung neu starten; die offenen Browser-Sitzungen
    verbinden sich von selbst wieder. Das geht **nur** über
    ``python -m rag.guiapp``, nicht über ``rag gui`` — der Grund steht dort.
    """
    global state
    log_file = setup_logging(verbose=verbose)

    if native and not _webview_verfuegbar():
        logger.warning(
            "Natives Fenster gewünscht, aber pywebview fehlt — Oberfläche "
            "öffnet im Browser. Nachrüsten: uv pip install -e '.[native]' "
            "(auf Linux zusätzlich die System-WebKitGTK)."
        )
        native = False

    logger.info(
        "Oberfläche startet auf http://%s:%d (%s)",
        host,
        port,
        "eigenständiges Fenster" if native else "Browser",
    )
    logger.info("Protokoll: %s", log_file)
    state = State(index_path)

    nicegui_app.on_shutdown(lambda: state.pipeline.close() if state else None)

    ui.run(
        host=host,
        port=port,
        title="local-rag",
        favicon="🔍",
        # Im nativen Modus kein zusätzlicher Browser-Tab — das Fenster ist die
        # Oberfläche.
        show=open_browser and not native,
        native=native,
        window_size=(1280, 900) if native else None,
        reload=reload,
        # Nur das Paket überwachen, nicht das Arbeitsverzeichnis: NiceGUIs
        # Vorgabe ist ".", und wer die Oberfläche aus einem Dokumentenordner
        # startet, würde damit nichts vom Code sehen — dafür jede neue Datei
        # im Ordner als Codeänderung.
        uvicorn_reload_dirs=str(Path(__file__).resolve().parent),
        # Ein lokales Werkzeug für einen Nutzer: kein Grund, im Netz zu lauschen.
        dark=None,
    )
