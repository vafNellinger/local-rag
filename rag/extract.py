"""Dokument → Text, formatabhängig.

Der interessante Teil sind PDFs. Ein PDF ist kein Format, sondern zwei:
digital erzeugte Seiten mit Text-Layer und gescannte Seiten, die nur ein Bild
enthalten. Beides kommt in derselben Datei vor — gescanntes Deckblatt vor
digitalem Rest ist der Normalfall.

Die Entscheidung fällt deshalb pro Seite, nicht pro Datei. Seiten ohne
brauchbaren Text-Layer werden als ``needs_ocr`` markiert und an das
konfigurierte OCR-Backend weitergereicht; ist keines verfügbar, bleiben sie
leer und tauchen im Bericht auf, statt still zu verschwinden.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Ab wie vielen extrahierten Zeichen eine PDF-Seite als "hat Text-Layer" gilt.
# Gescannte Seiten liefern typischerweise 0 Zeichen, manchmal ein paar
# Artefakte aus eingebetteten Wasserzeichen oder Seitenzahlen. 100 trennt das
# sauber, ohne echte, aber dünn belegte Seiten (Titelblatt, Kapiteltrenner)
# fälschlich als Scan zu werten.
TEXT_LAYER_MIN_CHARS = 100

# Zweites Kriterium: eine Seite mit viel Text, der zu großen Teilen aus
# Ersatzzeichen besteht, hat eine kaputte Zeichenkodierung (fehlendes
# ToUnicode-CMap). Das sieht wie Text aus, ist aber unbrauchbar — solche
# Seiten gehören ebenfalls ins OCR.
MAX_REPLACEMENT_RATIO = 0.15

# Drittes Kriterium, und das entscheidende: OCR lohnt nur, wenn es ein Bild
# gibt. Eine Seite mit wenig Text und ohne Bild ist dünn belegt (Schlussformel,
# Kapiteltrenner), nicht gescannt — OCR würde dort ein leeres Blatt abtasten.
#
# Die Mindestfläche trennt Seitenscans von Logos und Signaturbildchen. Ein
# Scan deckt die Seite ab und liegt bei mehreren Millionen Pixeln; ein Logo
# in der Kopfzeile bleibt weit darunter.
MIN_SCAN_IMAGE_PIXELS = 100_000

# Wie tief in verschachtelte Form-XObjects hinein nach Bildern gesucht wird.
# Zwei Ebenen decken die üblichen Verschachtelungen ab, ohne bei bösartig
# verketteten Dokumenten in eine Endlosschleife zu laufen.
MAX_XOBJECT_DEPTH = 2

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt", ".text"}

# Formate, die eine Konvertierung brauchen, bevor wir sie lesen können.
# Explizit benannt, damit die Fehlermeldung den Weg nennt statt nur "geht nicht".
CONVERSION_HINTS = {
    ".doc": "altes Word-Format — mit libreoffice --convert-to docx umwandeln",
    ".ppt": "altes PowerPoint-Format — mit libreoffice --convert-to pptx umwandeln",
    ".xls": "altes Excel-Format — mit libreoffice --convert-to xlsx umwandeln",
    ".pages": "Apple Pages — als PDF oder DOCX exportieren",
}


class ExtractionError(RuntimeError):
    """Die Datei konnte nicht gelesen werden."""


@dataclass
class Page:
    """Eine Seite oder ein Abschnitt eines Dokuments.

    ``status`` unterscheidet die Fälle, die sonst alle als "kein Text"
    zusammenfielen:

    - ``text``   — Text-Layer ist brauchbar
    - ``scan``   — kein brauchbarer Text, aber ein seitenfüllendes Bild: OCR hilft
    - ``sparse`` — kein Text und kein Bild: die Seite ist wirklich leer, OCR
      würde nichts finden
    - ``error``  — Seite nicht lesbar
    """

    number: int
    text: str
    status: str = "text"
    largest_image_pixels: int = 0
    # Gefüllt, sobald ein OCR-Backend die Seite bearbeitet hat.
    ocr_applied: bool = False
    note: str | None = None

    @property
    def needs_ocr(self) -> bool:
        return self.status == "scan" and not self.ocr_applied

    @property
    def char_count(self) -> int:
        return len(self.text.strip())


@dataclass
class Document:
    """Ergebnis der Extraktion."""

    path: Path
    format: str
    pages: list[Page] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def char_count(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def pages_needing_ocr(self) -> list[Page]:
        return [p for p in self.pages if p.needs_ocr]

    @property
    def sparse_pages(self) -> list[Page]:
        """Seiten ohne Text und ohne Bild — dort ist nichts zu holen."""
        return [p for p in self.pages if p.status == "sparse"]

    @property
    def ocr_ratio(self) -> float:
        """Anteil der Seiten, die ohne OCR nicht lesbar sind."""
        if not self.pages:
            return 0.0
        return len([p for p in self.pages if p.status == "scan"]) / len(self.pages)


def _replacement_ratio(text: str) -> float:
    """Anteil an Ersatz- und Steuerzeichen — Indikator für kaputte Kodierung."""
    stripped = text.strip()
    if not stripped:
        return 0.0
    bad = sum(
        1
        for ch in stripped
        if ch == "�" or (unicodedata.category(ch) == "Cc" and ch not in "\n\r\t")
    )
    return bad / len(stripped)


def _largest_image_pixels(page_obj, depth: int = 0) -> int:
    """Größtes Bild auf einer PDF-Seite, in Pixeln. 0 wenn keins.

    Sucht auch in Form-XObjects, weil Scanner-Software das Seitenbild gern in
    einen Form-Container verpackt statt direkt einzubetten.
    """
    if depth > MAX_XOBJECT_DEPTH:
        return 0

    try:
        resources = page_obj.get("/Resources")
        if resources is None:
            return 0
        xobjects = resources.get_object().get("/XObject")
        if xobjects is None:
            return 0
        xobjects = xobjects.get_object()
    except Exception as exc:
        logger.debug("XObject-Zugriff fehlgeschlagen: %s", exc)
        return 0

    largest = 0
    for key in xobjects:
        try:
            obj = xobjects[key].get_object()
            subtype = obj.get("/Subtype")
            if subtype == "/Image":
                pixels = int(obj.get("/Width") or 0) * int(obj.get("/Height") or 0)
                largest = max(largest, pixels)
            elif subtype == "/Form":
                largest = max(largest, _largest_image_pixels(obj, depth + 1))
        except Exception as exc:
            logger.debug("Bild %s nicht auswertbar: %s", key, exc)
            continue

    return largest


def _classify_page(text: str, image_pixels: int) -> tuple[str, str | None]:
    """Bestimme den Seitenstatus. Gibt (status, Grund).

    Die Bildprüfung ist ausschlaggebend: ohne seitenfüllendes Bild gibt es
    nichts zu erkennen, egal wie leer der Text-Layer ist.
    """
    stripped = text.strip()
    has_scan_image = image_pixels >= MIN_SCAN_IMAGE_PIXELS

    if len(stripped) < TEXT_LAYER_MIN_CHARS:
        if has_scan_image:
            return "scan", f"{len(stripped)} Zeichen, Bild mit {image_pixels:,} px"
        return "sparse", f"nur {len(stripped)} Zeichen, kein Bild — Seite ist leer"

    ratio = _replacement_ratio(stripped)
    if ratio > MAX_REPLACEMENT_RATIO:
        if has_scan_image:
            return "scan", f"{ratio:.0%} Ersatzzeichen, Bild vorhanden"
        # Kaputte Kodierung ohne Bild: OCR kann nicht helfen, aber der Text
        # ist unbrauchbar. Muss sichtbar bleiben statt still durchzurutschen.
        return "error", f"{ratio:.0%} Ersatzzeichen, kein Bild — Kodierung defekt"

    return "text", None


def _extract_pdf(path: Path) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("pypdf fehlt: uv pip install pypdf") from exc

    doc = Document(path=path, format="pdf")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ExtractionError(f"PDF nicht lesbar: {exc}") from exc

    if reader.is_encrypted:
        # Viele PDFs sind mit leerem Passwort verschlüsselt — das lässt sich
        # ohne Zutun öffnen. Erst wenn auch das scheitert, ist wirklich Schluss.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ExtractionError(f"PDF ist passwortgeschützt: {exc}") from exc
        doc.warnings.append("PDF war verschlüsselt, mit leerem Passwort geöffnet")

    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            # Eine kaputte Seite darf nicht das ganze Dokument verlieren.
            # Als "scan" markiert, weil OCR hier die einzige Chance ist.
            logger.debug("Seite %d nicht extrahierbar: %s", index, exc)
            doc.pages.append(
                Page(
                    number=index,
                    text="",
                    status="scan",
                    note=f"Text-Layer defekt ({exc}) — OCR als Rückfallebene",
                )
            )
            continue

        image_pixels = _largest_image_pixels(page)
        status, reason = _classify_page(text, image_pixels)
        doc.pages.append(
            Page(
                number=index,
                text=text,
                status=status,
                largest_image_pixels=image_pixels,
                note=reason,
            )
        )

    return doc


def _extract_docx(path: Path) -> Document:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("python-docx fehlt: uv pip install python-docx") from exc

    try:
        source = docx.Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"DOCX nicht lesbar: {exc}") from exc

    doc = Document(path=path, format="docx")

    blocks = [p.text for p in source.paragraphs if p.text.strip()]

    # Tabellen stehen in DOCX außerhalb des Absatzflusses und gingen sonst
    # verloren. Zeilenweise mit Tabs zusammengesetzt bleibt die Zuordnung
    # Spalte-zu-Wert für das Sprachmodell erkennbar.
    table_count = 0
    for table in source.tables:
        table_count += 1
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                blocks.append("\t".join(cells))

    if table_count:
        doc.warnings.append(
            f"{table_count} Tabelle(n) zeilenweise eingefügt — "
            "Layout geht verloren, Inhalt bleibt"
        )

    # DOCX hat keine verlässlichen Seitengrenzen (die entstehen erst beim
    # Rendern), deshalb ein Block.
    doc.pages.append(Page(number=1, text="\n".join(blocks)))
    return doc


def _extract_text(path: Path, fmt: str) -> Document:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback für Dateien aus Windows-Umgebungen.
        content = path.read_text(encoding="latin-1")
        doc = Document(path=path, format=fmt)
        doc.warnings.append("Datei war nicht UTF-8, als latin-1 gelesen")
        doc.pages.append(Page(number=1, text=content))
        return doc

    doc = Document(path=path, format=fmt)
    doc.pages.append(Page(number=1, text=content))
    return doc


def extract(path: str | Path) -> Document:
    """Lies ein Dokument. OCR wird hier nur markiert, nicht ausgeführt."""
    file_path = Path(path).expanduser()

    if not file_path.exists():
        raise ExtractionError(f"Datei nicht gefunden: {file_path}")
    if not file_path.is_file():
        raise ExtractionError(f"Kein Dateipfad: {file_path}")

    suffix = file_path.suffix.lower()

    if hint := CONVERSION_HINTS.get(suffix):
        raise ExtractionError(f"{suffix} wird nicht direkt unterstützt — {hint}")

    if suffix == ".pdf":
        return _extract_pdf(file_path)
    if suffix == ".docx":
        return _extract_docx(file_path)
    if suffix in {".md", ".markdown"}:
        return _extract_text(file_path, "markdown")
    if suffix in {".txt", ".text"}:
        return _extract_text(file_path, "text")

    raise ExtractionError(
        f"Format {suffix or '(ohne Endung)'} wird nicht unterstützt. "
        f"Möglich: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )
