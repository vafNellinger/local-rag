"""Retrieval gegen einen Goldstandard messen.

Ohne bekannte richtige Antworten lässt sich Retrieval nur anschauen, nicht
beurteilen. Diese Datei rechnet drei Dinge aus:

**Recall@k** — steckt in den ersten k Treffern mindestens eine Quelle, die die
Frage belegen kann? Das ist die Zahl, die zählt: was nicht abgerufen wird, kann
der Generator nicht zitieren, egal wie gut er ist.

**MRR** — der reziproke Rang des ersten Treffers. Unterscheidet „auf Platz 1"
von „auf Platz 5", was Recall@5 gleich bewertet. Wichtig, weil der Platz über
das Kontextbudget entscheidet.

**Die Schwellwertkurve** — für jeden möglichen ``min_rerank_score``: wie viele
richtige Stellen fallen weg, und wie viel Rauschen bleibt draußen? Die
Einstellung war bisher eine Schätzung; hier wird sie eine Messung.

Die Fragen ohne Antwort im Korpus tragen diese Analyse. Sie liefern die
Gegenprobe: was die Suche für eine unbeantwortbare Frage ausspuckt, ist per
Definition Rauschen, und sein Punktwert ist die Obergrenze für eine sinnvolle
Schwelle.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rag.pipeline import RagPipeline
from rag.store import SearchHit

logger = logging.getLogger(__name__)

DEFAULT_GOLD_PATH = Path("testdaten/goldstandard.json")

# Schwellwerte, die geprüft werden. Feiner Raster im unteren Bereich, weil die
# Punktwerte des Cross-Encoders sich dort ballen.
THRESHOLD_GRID = (
    0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5
)

TYPE_ANSWERABLE = ("eindeutig", "mehrdeutig")
TYPE_UNANSWERABLE = "ohne_antwort"


class EvaluationError(RuntimeError):
    """Der Goldstandard ist nicht benutzbar."""


@dataclass
class Question:
    id: str
    frage: str
    typ: str
    dokumente: list[str] = field(default_factory=list)
    abschnitt: str | None = None
    antwort_enthaelt: list[str] = field(default_factory=list)
    notiz: str | None = None

    @property
    def answerable(self) -> bool:
        return self.typ in TYPE_ANSWERABLE


@dataclass
class ScoredHit:
    """Ein Treffer mit dem Urteil, ob er zur Frage passt."""

    document: str
    heading: str
    score: float
    correct: bool


@dataclass
class QuestionResult:
    question: Question
    hits: list[ScoredHit] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def rank(self) -> int | None:
        """1-basierter Rang des ersten passenden Treffers, oder None."""
        for index, hit in enumerate(self.hits, start=1):
            if hit.correct:
                return index
        return None

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    def hit_at(self, k: int) -> bool:
        return any(hit.correct for hit in self.hits[:k])


@dataclass
class ThresholdRow:
    """Was eine Schwelle bewirkt."""

    threshold: float
    # Anteil der beantwortbaren Fragen, die eine richtige Quelle behalten.
    recall: float
    # Anteil der behaltenen Treffer, die richtig sind.
    precision: float
    # Wie viele Treffer unbeantwortbarer Fragen die Schwelle passieren.
    noise_kept: int
    # Fragen, die durch die Schwelle ihre einzige richtige Quelle verlieren.
    lost: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    results: list[QuestionResult] = field(default_factory=list)
    top_k: int = 5
    reranked: bool = True
    duration_seconds: float = 0.0

    @property
    def answerable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.question.answerable]

    @property
    def unanswerable(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.question.answerable]

    def recall_at(self, k: int) -> float:
        relevant = self.answerable
        if not relevant:
            return 0.0
        return sum(r.hit_at(k) for r in relevant) / len(relevant)

    def mrr(self) -> float:
        relevant = self.answerable
        if not relevant:
            return 0.0
        return sum(1 / r.rank if r.rank else 0.0 for r in relevant) / len(relevant)

    @property
    def misses(self) -> list[QuestionResult]:
        """Fragen, für die keine passende Quelle gefunden wurde."""
        return [r for r in self.answerable if r.rank is None]

    def noise_score_range(self) -> tuple[float, float]:
        """Kleinster und größter Spitzenwert der unbeantwortbaren Fragen.

        Die Obergrenze ist der interessante Wert: eine Schwelle darunter lässt
        Rauschen durch, eine darüber schneidet es weg.
        """
        scores = [r.top_score for r in self.unanswerable if r.hits]
        return (min(scores), max(scores)) if scores else (0.0, 0.0)

    def correct_score_range(self) -> tuple[float, float]:
        """Kleinster und größter Punktwert richtiger Treffer.

        Die Untergrenze begrenzt die Schwelle nach oben: darüber verliert man
        richtige Stellen.
        """
        scores = [
            hit.score
            for result in self.answerable
            for hit in result.hits
            if hit.correct
        ]
        return (min(scores), max(scores)) if scores else (0.0, 0.0)

    def thresholds(
        self, grid: Sequence[float] = THRESHOLD_GRID
    ) -> list[ThresholdRow]:
        """Wirkung jeder Schwelle auf Recall, Präzision und Rauschen."""
        rows: list[ThresholdRow] = []
        answerable = self.answerable

        for threshold in grid:
            kept_total = 0
            kept_correct = 0
            still_answered = 0
            lost: list[str] = []

            for result in answerable:
                # Die erste Quelle bleibt immer — so verhält sich rerank() auch.
                kept = [h for h in result.hits if h.score >= threshold]
                if not kept and result.hits:
                    kept = result.hits[:1]
                kept_total += len(kept)
                kept_correct += sum(h.correct for h in kept)
                if any(h.correct for h in kept):
                    still_answered += 1
                elif result.rank is not None:
                    # Vorher auffindbar, jetzt nicht mehr — das ist der Preis.
                    lost.append(result.question.id)

            noise = 0
            for result in self.unanswerable:
                kept = [h for h in result.hits if h.score >= threshold]
                if not kept and result.hits:
                    kept = result.hits[:1]
                noise += len(kept)

            rows.append(
                ThresholdRow(
                    threshold=threshold,
                    recall=still_answered / len(answerable) if answerable else 0.0,
                    precision=kept_correct / kept_total if kept_total else 0.0,
                    noise_kept=noise,
                    lost=lost,
                )
            )
        return rows


def load_gold(path: Path | None = None) -> list[Question]:
    """Goldstandard aus JSON lesen."""
    source = path or DEFAULT_GOLD_PATH
    if not source.exists():
        raise EvaluationError(
            f"Goldstandard fehlt: {source} — "
            "Korpus mit 'python tools/testkorpus.py' erzeugen"
        )
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Goldstandard ist kein gültiges JSON: {exc}") from exc

    fragen = data.get("fragen")
    if not fragen:
        raise EvaluationError(f"Goldstandard enthält keine Fragen: {source}")

    questions: list[Question] = []
    for entry in fragen:
        for required in ("id", "frage", "typ"):
            if required not in entry:
                raise EvaluationError(f"Frage ohne Feld '{required}': {entry}")
        questions.append(
            Question(
                id=entry["id"],
                frage=entry["frage"],
                typ=entry["typ"],
                dokumente=list(entry.get("dokumente", [])),
                abschnitt=entry.get("abschnitt"),
                antwort_enthaelt=list(entry.get("antwort_enthaelt", [])),
                notiz=entry.get("notiz"),
            )
        )
    return questions


def _document_key(hit: SearchHit) -> str:
    """Dateiname ohne Endung — so steht er im Goldstandard."""
    return Path(hit.document_path).stem


def _score_of(hit: SearchHit) -> float:
    """Reranker-Punktwert, sonst die Vektorähnlichkeit.

    Ohne Reranking gibt es keine Cross-Encoder-Punkte; die Ähnlichkeit ist
    dann die einzige verfügbare Größe. Die Skalen sind nicht vergleichbar,
    deshalb steht in der Ausgabe, welche gemessen wurde.
    """
    return hit.rerank_score if hit.rerank_score is not None else hit.similarity


def evaluate(
    pipeline: RagPipeline,
    questions: Sequence[Question],
    *,
    top_k: int | None = None,
    progress: Callable[[int, int, Question], None] | None = None,
) -> EvalReport:
    """Jede Frage abrufen und gegen den Goldstandard bewerten.

    Ruft nur ``retrieve()``, nicht ``ask()`` — die Generierung kostet zwanzig
    Sekunden pro Frage und ändert am Retrieval nichts. Was der Generator aus
    den Quellen macht, ist eine eigene Messung.
    """
    limit = top_k or pipeline.settings.top_k
    report = EvalReport(
        top_k=limit, reranked=pipeline.settings.reranker_enabled
    )
    started = time.time()

    for index, question in enumerate(questions, start=1):
        if progress:
            progress(index, len(questions), question)

        question_started = time.time()
        hits = pipeline.retrieve(question.frage, top_k=limit)
        expected = set(question.dokumente)

        report.results.append(
            QuestionResult(
                question=question,
                hits=[
                    ScoredHit(
                        document=_document_key(hit),
                        heading=hit.heading,
                        score=_score_of(hit),
                        correct=_document_key(hit) in expected,
                    )
                    for hit in hits
                ],
                duration_seconds=time.time() - question_started,
            )
        )

    report.duration_seconds = time.time() - started
    return report


@dataclass
class AnswerCheck:
    """Ergebnis einer geprüften Antwort."""

    question: Question
    text: str
    # Rohe Zeichenkettenprüfung. Nicht direkt verwenden — ``answered_correctly``
    # ist das Urteil.
    contains_strings: bool
    cited_sources: int
    duration_seconds: float

    @property
    def admitted_ignorance(self) -> bool:
        """Hat die Antwort Nichtwissen eingeräumt?

        Grobe Erkennung über die Formulierungen, die der Systemprompt
        nahelegt. Für die unbeantwortbaren Fragen ist genau das die richtige
        Antwort, und sie muss messbar sein.
        """
        marker = (
            "nicht angegeben",
            "nicht enthalten",
            "keine angabe",
            "nicht hervor",
            "nicht genannt",
            "steht nicht",
            "liegen keine",
            "nicht ersichtlich",
            "keine information",
            "lässt sich nicht",
            "nicht beantwort",
        )
        lowered = self.text.lower()
        return any(m in lowered for m in marker)

    @property
    def answered_correctly(self) -> bool:
        """Enthält die Antwort die erwartete Angabe *als Antwort*?

        Die reine Zeichenkettenprüfung genügt nicht: gemessen kam
        „Die Frage … ist nicht in den bereitgestellten Quellen enthalten. [1]
        nennt Ruhezeiten von 13:00 bis 15:00 Uhr" — die erwartete Zeichenkette
        steht drin, beantwortet ist die Frage aber nicht. Wer Nichtwissen
        einräumt, hat nicht geantwortet, auch wenn er dabei zitiert.
        """
        return self.contains_strings and not self.admitted_ignorance


def check_answers(
    pipeline: RagPipeline,
    questions: Sequence[Question],
    *,
    progress: Callable[[int, int, Question], None] | None = None,
) -> list[AnswerCheck]:
    """Antworten erzeugen und gegen die erwarteten Angaben prüfen.

    Teuer: gemessen zwanzig Sekunden pro Frage auf CPU. Getrennt von
    ``evaluate()``, damit die Retrieval-Messung schnell bleibt.
    """
    checks: list[AnswerCheck] = []
    for index, question in enumerate(questions, start=1):
        if progress:
            progress(index, len(questions), question)

        started = time.time()
        answer = pipeline.ask(question.frage)
        text = answer.text

        expected = question.antwort_enthaelt
        checks.append(
            AnswerCheck(
                question=question,
                text=text,
                # Alle erwarteten Angaben müssen vorkommen; bei einer Frage
                # nach einem Betrag ist die halbe Zahl keine halbe Antwort.
                contains_strings=bool(expected)
                and all(needle in text for needle in expected),
                cited_sources=len(answer.cited_numbers),
                duration_seconds=time.time() - started,
            )
        )
    return checks
