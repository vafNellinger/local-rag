"""Markdown → Chunks, entlang der Dokumentstruktur.

Docling liefert Überschriften, Tabellen und Absätze als Markdown. Ein
Chunker, der stur alle N Zeichen schneidet, wirft genau diese Information
weg — deshalb arbeitet dieser hier auf Blöcken statt auf Zeichen.

Drei Entscheidungen tragen den Rest:

**Überschriften-Pfad als Präfix.** Ein Chunk aus der Mitte eines Dokuments ist
für sich genommen kontextlos: "Die Frist beträgt 14 Tage" ist ohne
"Kündigung > Fristen" nicht auffindbar und im Prompt nicht einordbar. Der
Pfad wird deshalb Teil des embeddeten Textes.

**Sektionsgrenzen sind harte Grenzen.** Overlap zwischen zwei Abschnitten
verbindet Themen, die nichts miteinander zu tun haben. Überlappt wird nur
innerhalb einer Sektion.

**Tabellen behalten ihren Kopf.** Wird eine große Tabelle geteilt, bekommt
jeder Teil die Kopfzeile erneut. Eine Tabellenzeile ohne Spaltennamen ist
bedeutungslos — "4.500 | 12 | ja" beantwortet keine Frage.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Zielgröße eines Chunks in Token. bge-m3 verkraftet 8192, aber das ist die
# falsche Obergrenze: je mehr Themen in einem Vektor landen, desto unschärfer
# wird er. 512 Token sind etwa ein bis zwei Absätze — groß genug für einen
# abgeschlossenen Gedanken, klein genug für einen scharfen Vektor.
TARGET_TOKENS = 512

# Überlappung zwischen aufeinanderfolgenden Chunks derselben Sektion. Fängt
# den Fall auf, dass eine Antwort genau auf der Schnittkante liegt.
OVERLAP_TOKENS = 64

# Darunter wird ein Rest nicht als eigener Chunk geführt, sondern an den
# Vorgänger gehängt. Ein Chunk aus zwölf Token ist im Index Rauschen: er
# matcht auf Allerweltsformulierungen und verdrängt echte Treffer.
MIN_CHUNK_TOKENS = 32

# Zeichen pro Token für die Schätzung ohne echten Tokenizer. Deutscher Text
# liegt beim XLM-RoBERTa-Vokabular von bge-m3 bei rund 3,2 bis 3,8 — die
# Komposita kosten mehr Subtokens als englischer Text. 3,2 schätzt die
# Tokenzahl also eher zu hoch, was die richtige Richtung ist: zu kleine
# Chunks sind harmlos, zu große werden vom Modell abgeschnitten.
CHARS_PER_TOKEN = 3.2

# Deutsche Abkürzungen, die auf einen Punkt enden, ohne einen Satz zu beenden.
# Ohne diese Liste zerschneidet die Satzsegmentierung "gemäß § 5 Abs. 2 Satz 1"
# mitten in der Fundstelle.
SENTENCE_ABBREVIATIONS = frozenset(
    {
        "z.b.", "d.h.", "u.a.", "u.ä.", "o.ä.", "bzw.", "ca.", "ggf.", "evtl.",
        "inkl.", "exkl.", "zzgl.", "abzgl.", "vgl.", "bspw.", "insb.", "sog.",
        "abs.", "art.", "nr.", "bd.", "kap.", "abb.", "tab.", "ziff.", "lit.",
        "buchst.", "az.", "bzgl.", "gem.",
        "dr.", "prof.", "hr.", "fr.", "dipl.", "ing.", "st.",
        "mio.", "mrd.", "tsd.", "mind.", "max.", "zzt.", "usw.", "etc.",
        "jan.", "feb.", "mär.", "apr.", "jun.", "jul.", "aug.", "sep.",
        "okt.", "nov.", "dez.",
    }
)

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")

# Satzende: Interpunktion, Leerraum, dann ein Großbuchstabe oder öffnendes
# Anführungszeichen. Die Abkürzungsprüfung passiert danach, nicht hier — als
# Lookbehind wäre die Liste unlesbar.
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?:])\s+(?=[A-ZÄÖÜ“„"(\d])')

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Tokenzahl ohne Tokenizer schätzen.

    Bewusst grob. Wer es genau braucht, übergibt den Tokenizer des
    Embedding-Modells als ``counter`` — beim Ingest ist der ohnehin geladen.
    """
    return max(1, int(len(text) / CHARS_PER_TOKEN))


@dataclass
class Chunk:
    """Ein Stück Text, das als ein Vektor in den Index geht."""

    ordinal: int
    text: str
    heading_path: tuple[str, ...] = ()
    token_count: int = 0
    # Woraus der Chunk entstand: "prosa", "tabelle", "code". Steuert später
    # das Reranking und macht im Debug sichtbar, ob Tabellen sinnvoll
    # geschnitten wurden.
    kind: str = "prosa"

    @property
    def heading(self) -> str:
        """Der Pfad als lesbare Zeile, z.B. 'Kündigung > Fristen'."""
        return " > ".join(self.heading_path)

    @property
    def embed_text(self) -> str:
        """Was tatsächlich embeddet wird — Text mit Überschriften-Kontext.

        Der Pfad steht vorn und nicht hinten: bei einem Modell mit
        Positions-Bias zählt der Anfang mehr, und der Kontext soll den Text
        einordnen, nicht nachträglich ergänzen.
        """
        if not self.heading_path:
            return self.text
        return f"{self.heading}\n\n{self.text}"


@dataclass
class Block:
    """Ein Markdown-Block: Absatz, Tabelle, Codeblock oder Überschrift."""

    kind: str  # "prosa" | "tabelle" | "code" | "heading"
    text: str
    level: int = 0  # nur bei kind == "heading"
    lines: list[str] = field(default_factory=list)  # nur bei Tabellen


def _parse_blocks(markdown: str) -> list[Block]:
    """Markdown in Blöcke zerlegen.

    Kein vollständiger Parser — nur die Konstrukte, die Docling erzeugt:
    ATX-Überschriften, Pipe-Tabellen, Codeblöcke, Absätze.
    """
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush_prosa() -> None:
        if text := "\n".join(buffer).strip():
            blocks.append(Block("prosa", text))
        buffer.clear()

    lines = markdown.splitlines()
    index = 0

    while index < len(lines):
        line = lines[index]

        if fence := _FENCE.match(line):
            flush_prosa()
            marker = fence.group(1)
            fenced = [line]
            index += 1
            while index < len(lines):
                fenced.append(lines[index])
                if lines[index].strip().startswith(marker):
                    index += 1
                    break
                index += 1
            blocks.append(Block("code", "\n".join(fenced)))
            continue

        if heading := _HEADING.match(line):
            flush_prosa()
            blocks.append(
                Block("heading", heading.group(2).strip(), level=len(heading.group(1)))
            )
            index += 1
            continue

        if _TABLE_ROW.match(line):
            flush_prosa()
            rows: list[str] = []
            while index < len(lines) and _TABLE_ROW.match(lines[index]):
                rows.append(lines[index].rstrip())
                index += 1
            blocks.append(Block("tabelle", "\n".join(rows), lines=rows))
            continue

        if not line.strip():
            flush_prosa()
            index += 1
            continue

        buffer.append(line)
        index += 1

    flush_prosa()
    return blocks


def split_sentences(text: str) -> list[str]:
    """Text in Sätze zerlegen, mit Rücksicht auf deutsche Abkürzungen.

    Erst grob an der Interpunktion trennen, dann die Fehltrennungen wieder
    zusammenfügen. Betroffen sind Abkürzungen ("Abs.", "z.B.") und
    Ordinalzahlen ("1. Januar", "§ 5 Nr. 3") — beide enden auf einen Punkt,
    der kein Satzende ist.
    """
    rough = _SENTENCE_SPLIT.split(text)
    if len(rough) < 2:
        return [text] if text.strip() else []

    merged: list[str] = []
    for fragment in rough:
        if merged and _ends_mid_sentence(merged[-1]):
            merged[-1] = f"{merged[-1]} {fragment}"
        else:
            merged.append(fragment)
    return [m for m in merged if m.strip()]


def _ends_mid_sentence(fragment: str) -> bool:
    """Endet das Fragment auf etwas, das keinen Satz beendet?"""
    last = fragment.rstrip().split()[-1].lower() if fragment.strip() else ""
    if not last:
        return False
    if last in SENTENCE_ABBREVIATIONS:
        return True
    # Aufzählungs- und Datumszahlen: "1.", "14.", "2026.".
    #
    # Bewusst konservativ: "Fällig am 1. Januar" bleibt zusammen, aber
    # "... gemäß Satz 1. Danach ..." bleibt es auch, obwohl dort ein Satz
    # endet. Ohne Semantik ist das nicht zu trennen, und die Richtung ist die
    # richtige — die Satzliste liefert nur Schnittkandidaten. Ein verpasster
    # Schnitt kostet Granularität, ein falscher zerreißt eine Fundstelle.
    if re.fullmatch(r"[\d.]+\.", last):
        return True
    # Einzelner Buchstabe mit Punkt — Initialen wie "J. Öztürk".
    if re.fullmatch(r"[a-zäöü]\.", last):
        return True
    return False


def _split_table(block: Block, limit: int, counter: TokenCounter) -> list[Block]:
    """Große Tabelle in Teile schneiden, Kopfzeile in jeden Teil.

    Kopf sind die ersten beiden Zeilen — Spaltennamen und Trennzeile. Ohne
    sie ist eine Datenzeile nicht interpretierbar, deshalb wiegt die
    Wiederholung den Platzverlust auf.
    """
    rows = block.lines
    if len(rows) <= 3:
        return [block]

    header, body = rows[:2], rows[2:]
    header_cost = counter("\n".join(header))
    if header_cost >= limit:
        logger.debug("Tabellenkopf allein überschreitet das Limit, splitte roh")
        return _split_lines(block, limit, counter)

    parts: list[Block] = []
    current: list[str] = []
    current_cost = 0

    for row in body:
        cost = counter(row)
        if current and header_cost + current_cost + cost > limit:
            parts.append(Block("tabelle", "\n".join(header + current), lines=header + current))
            current, current_cost = [], 0
        current.append(row)
        current_cost += cost

    if current:
        parts.append(Block("tabelle", "\n".join(header + current), lines=header + current))
    return parts


def _split_lines(block: Block, limit: int, counter: TokenCounter) -> list[Block]:
    """Notfall-Teilung an Zeilengrenzen, für Code und entartete Tabellen."""
    parts: list[Block] = []
    current: list[str] = []
    current_cost = 0

    for line in block.text.splitlines():
        cost = counter(line)
        if current and current_cost + cost > limit:
            parts.append(Block(block.kind, "\n".join(current)))
            current, current_cost = [], 0
        current.append(line)
        current_cost += cost

    if current:
        parts.append(Block(block.kind, "\n".join(current)))
    return parts


def _split_prosa(block: Block, limit: int, counter: TokenCounter) -> list[Block]:
    """Zu langen Absatz an Satzgrenzen teilen.

    Ein einzelner Satz, der das Limit überschreitet, wird nicht weiter
    zerlegt: die Teilung mitten im Satz kostet mehr Bedeutung, als der
    Überhang an Präzision bringt. Bei 512 Token Ziel und 8192 Token Kapazität
    ist der Überhang für das Modell unkritisch.
    """
    sentences = split_sentences(block.text)
    parts: list[Block] = []
    current: list[str] = []
    current_cost = 0

    for sentence in sentences:
        cost = counter(sentence)
        if current and current_cost + cost > limit:
            parts.append(Block("prosa", " ".join(current)))
            current, current_cost = [], 0
        current.append(sentence)
        current_cost += cost

    if current:
        parts.append(Block("prosa", " ".join(current)))
    return parts or [block]


def _overlap_tail(text: str, budget: int, counter: TokenCounter) -> str:
    """Die letzten ``budget`` Token eines Chunks, an Satzgrenzen ausgerichtet.

    Rückwärts über die Sätze, damit der Overlap mit einem Satzanfang beginnt
    und nicht mit einem halben Nebensatz.
    """
    if budget <= 0:
        return ""
    sentences = split_sentences(text)
    tail: list[str] = []
    cost = 0
    for sentence in reversed(sentences):
        sentence_cost = counter(sentence)
        if tail and cost + sentence_cost > budget:
            break
        tail.insert(0, sentence)
        cost += sentence_cost
    return " ".join(tail)


def chunk_markdown(
    markdown: str,
    *,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    counter: TokenCounter | None = None,
) -> list[Chunk]:
    """Markdown in Chunks zerlegen, entlang von Überschriften und Blöcken.

    ``counter`` ist der Tokenzähler; ohne Angabe wird über die Zeichenzahl
    geschätzt. Beim Ingest wird der Tokenizer des Embedding-Modells
    übergeben, weil er dort schon im Speicher liegt.
    """
    count = counter or estimate_tokens
    blocks = _parse_blocks(markdown)

    chunks: list[Chunk] = []
    # Überschriften-Stapel: Index 0 ist Ebene 1. Eine Überschrift der Ebene N
    # verdrängt alles ab Ebene N, damit ein neues Kapitel nicht die
    # Unterabschnitte des vorigen erbt.
    path: list[str] = []
    pending: list[Block] = []

    def flush_section() -> None:
        """Die gesammelten Blöcke einer Sektion zu Chunks packen."""
        if not pending:
            return
        heading_path = tuple(path)
        # Der Überschriften-Präfix belegt Platz im Chunk und muss vom Budget
        # abgehen, sonst überschreiten die Chunks das Ziel genau um seine Länge.
        prefix_cost = count(" > ".join(heading_path)) if heading_path else 0
        limit = max(MIN_CHUNK_TOKENS, target_tokens - prefix_cost)

        # Der Overlap belegt Platz im Folge-Chunk und muss vom Unit-Budget
        # abgehen — sonst ergibt Overlap plus volle Unit einen Chunk über dem
        # Ziel. Die Deckelung auf ein Viertel begrenzt außerdem die Redundanz
        # im Index: mehr als ein Viertel Wiederholung kostet Platz, ohne die
        # Trefferwahrscheinlichkeit an der Schnittkante weiter zu erhöhen.
        effective_overlap = min(overlap_tokens, limit // 4) if overlap_tokens else 0
        unit_limit = max(MIN_CHUNK_TOKENS, limit - effective_overlap)

        units: list[Block] = []
        for block in pending:
            if count(block.text) <= unit_limit:
                units.append(block)
            elif block.kind == "tabelle":
                units.extend(_split_table(block, unit_limit, count))
            elif block.kind == "code":
                units.extend(_split_lines(block, unit_limit, count))
            else:
                units.extend(_split_prosa(block, unit_limit, count))

        current: list[Block] = []
        current_cost = 0

        def emit() -> None:
            nonlocal current, current_cost
            if not current:
                return
            text = "\n\n".join(b.text for b in current).strip()
            if not text:
                current, current_cost = [], 0
                return
            # Tabellen und Prosa gemischt zählen als Prosa; reine
            # Tabellen-Chunks werden markiert, weil sie beim Reranking anders
            # zu behandeln sind.
            kinds = {b.kind for b in current}
            kind = kinds.pop() if len(kinds) == 1 else "prosa"
            chunks.append(
                Chunk(
                    ordinal=len(chunks),
                    text=text,
                    heading_path=heading_path,
                    token_count=count(text) + prefix_cost,
                    kind=kind,
                )
            )
            current, current_cost = [], 0

        for unit in units:
            cost = count(unit.text)
            if current and current_cost + cost > limit:
                previous_text = "\n\n".join(b.text for b in current)
                emit()
                # Overlap nur bei Prosa: eine wiederholte Tabellenhälfte
                # bringt nichts, ihr Kopf steht ohnehin in jedem Teil.
                if effective_overlap and unit.kind == "prosa":
                    if tail := _overlap_tail(previous_text, effective_overlap, count):
                        current = [Block("prosa", tail)]
                        current_cost = count(tail)
            current.append(unit)
            current_cost += cost

        # Reste unter der Mindestgröße an den Vorgänger derselben Sektion
        # hängen, statt einen Splitter-Chunk zu erzeugen.
        if current and chunks and chunks[-1].heading_path == heading_path:
            rest = "\n\n".join(b.text for b in current).strip()
            if rest and count(rest) < MIN_CHUNK_TOKENS:
                previous = chunks[-1]
                previous.text = f"{previous.text}\n\n{rest}"
                previous.token_count = count(previous.text) + prefix_cost
                current, current_cost = [], 0

        emit()
        pending.clear()

    for block in blocks:
        if block.kind == "heading":
            flush_section()
            # Ebene 3 landet auf Index 2; klaffende Sprünge (H1 → H3) werden
            # aufgefüllt, damit der Pfad nicht durch fehlende Ebenen verrutscht.
            del path[block.level - 1 :]
            while len(path) < block.level - 1:
                path.append("")
            path.append(block.text)
            continue
        pending.append(block)

    flush_section()

    # Leere Zwischenebenen erst hier entfernen: während des Aufbaus halten sie
    # die Indizes stabil, im Pfad wären sie nur Lücken ("Kapitel >  > Detail").
    for chunk in chunks:
        chunk.heading_path = tuple(h for h in chunk.heading_path if h)

    return chunks
