"""Tests für die Retrieval-Messung.

Die Metriken müssen selbst geprüft sein, bevor man ihnen eine Aussage über das
System abnimmt. Gerechnet wird hier mit erfundenen Treffern, ohne Index und
ohne Modell.
"""

from __future__ import annotations

import json

import pytest

from rag.evaluate import (
    AnswerCheck,
    EvalReport,
    EvaluationError,
    Question,
    QuestionResult,
    ScoredHit,
    load_gold,
)


def frage(id: str, *docs: str, typ: str = "eindeutig") -> Question:
    return Question(id=id, frage=f"Frage {id}?", typ=typ, dokumente=list(docs))


def ergebnis(question: Question, *hits: tuple[str, float, bool]) -> QuestionResult:
    return QuestionResult(
        question=question,
        hits=[
            ScoredHit(document=d, heading="", score=s, correct=c) for d, s, c in hits
        ],
    )


class TestLoadGold:
    def test_echter_goldstandard_ist_lesbar(self):
        # Prüft die eingecheckte Datei mit — ein Tippfehler darin würde jede
        # Messung verfälschen.
        questions = load_gold()
        assert len(questions) > 30
        assert all(q.id and q.frage and q.typ for q in questions)

    def test_ids_sind_eindeutig(self):
        ids = [q.id for q in load_gold()]
        assert len(ids) == len(set(ids))

    def test_unbeantwortbare_haben_keine_dokumente(self):
        for question in load_gold():
            if question.typ == "ohne_antwort":
                assert question.dokumente == [], question.id

    def test_beantwortbare_haben_dokumente(self):
        for question in load_gold():
            if question.typ in ("eindeutig", "mehrdeutig"):
                assert question.dokumente, question.id

    def test_fehlende_datei(self, tmp_path):
        with pytest.raises(EvaluationError, match="fehlt"):
            load_gold(tmp_path / "gibtsnicht.json")

    def test_kaputtes_json(self, tmp_path):
        pfad = tmp_path / "gold.json"
        pfad.write_text("{kein json")
        with pytest.raises(EvaluationError, match="JSON"):
            load_gold(pfad)

    def test_ohne_fragen(self, tmp_path):
        pfad = tmp_path / "gold.json"
        pfad.write_text('{"fragen": []}')
        with pytest.raises(EvaluationError, match="keine Fragen"):
            load_gold(pfad)

    def test_frage_ohne_pflichtfeld(self, tmp_path):
        pfad = tmp_path / "gold.json"
        pfad.write_text(json.dumps({"fragen": [{"id": "x", "frage": "?"}]}))
        with pytest.raises(EvaluationError, match="typ"):
            load_gold(pfad)


class TestRank:
    def test_treffer_auf_platz_eins(self):
        result = ergebnis(frage("a", "doc"), ("doc", 0.5, True), ("x", 0.1, False))
        assert result.rank == 1

    def test_treffer_auf_platz_drei(self):
        result = ergebnis(
            frage("a", "doc"),
            ("x", 0.5, False),
            ("y", 0.4, False),
            ("doc", 0.3, True),
        )
        assert result.rank == 3

    def test_kein_treffer(self):
        assert ergebnis(frage("a", "doc"), ("x", 0.5, False)).rank is None

    def test_ohne_treffer_ueberhaupt(self):
        assert ergebnis(frage("a", "doc")).rank is None

    def test_hit_at_beachtet_die_grenze(self):
        result = ergebnis(
            frage("a", "doc"),
            ("x", 0.5, False),
            ("y", 0.4, False),
            ("doc", 0.3, True),
        )
        assert not result.hit_at(2)
        assert result.hit_at(3)


class TestMetrics:
    def test_recall_zaehlt_nur_beantwortbare(self):
        report = EvalReport(
            results=[
                ergebnis(frage("a", "doc"), ("doc", 0.5, True)),
                ergebnis(frage("b", "doc"), ("x", 0.5, False)),
                # Unbeantwortbare dürfen den Recall nicht verwässern.
                ergebnis(frage("c", typ="ohne_antwort"), ("x", 0.1, False)),
            ]
        )
        assert report.recall_at(5) == pytest.approx(0.5)

    def test_mrr_gewichtet_den_platz(self):
        report = EvalReport(
            results=[
                ergebnis(frage("a", "doc"), ("doc", 0.5, True)),
                ergebnis(
                    frage("b", "doc"), ("x", 0.5, False), ("doc", 0.4, True)
                ),
            ]
        )
        # (1/1 + 1/2) / 2
        assert report.mrr() == pytest.approx(0.75)

    def test_mrr_ohne_treffer_ist_null(self):
        report = EvalReport(results=[ergebnis(frage("a", "doc"), ("x", 0.5, False))])
        assert report.mrr() == 0.0

    def test_leerer_bericht_teilt_nicht_durch_null(self):
        report = EvalReport()
        assert report.recall_at(5) == 0.0
        assert report.mrr() == 0.0

    def test_misses_listet_die_fehlschlaege(self):
        report = EvalReport(
            results=[
                ergebnis(frage("gut", "doc"), ("doc", 0.5, True)),
                ergebnis(frage("schlecht", "doc"), ("x", 0.5, False)),
            ]
        )
        assert [r.question.id for r in report.misses] == ["schlecht"]


class TestScoreRanges:
    def test_bereiche_werden_getrennt_erfasst(self):
        report = EvalReport(
            results=[
                ergebnis(frage("a", "doc"), ("doc", 0.30, True), ("x", 0.02, False)),
                ergebnis(frage("b", "doc"), ("doc", 0.10, True)),
                ergebnis(frage("c", typ="ohne_antwort"), ("y", 0.03, False)),
                ergebnis(frage("d", typ="ohne_antwort"), ("z", 0.01, False)),
            ]
        )
        assert report.correct_score_range() == pytest.approx((0.10, 0.30))
        assert report.noise_score_range() == pytest.approx((0.01, 0.03))

    def test_leere_bereiche(self):
        assert EvalReport().correct_score_range() == (0.0, 0.0)
        assert EvalReport().noise_score_range() == (0.0, 0.0)


class TestThresholds:
    def test_schwelle_null_behaelt_alles(self):
        report = EvalReport(
            results=[ergebnis(frage("a", "doc"), ("doc", 0.3, True), ("x", 0.001, False))]
        )
        row = report.thresholds(grid=[0.0])[0]
        assert row.recall == 1.0
        assert row.precision == pytest.approx(0.5)

    def test_hohe_schwelle_verbessert_die_praezision(self):
        report = EvalReport(
            results=[ergebnis(frage("a", "doc"), ("doc", 0.3, True), ("x", 0.001, False))]
        )
        niedrig, hoch = report.thresholds(grid=[0.0, 0.1])
        assert hoch.precision > niedrig.precision
        assert hoch.recall == 1.0

    def test_zu_hohe_schwelle_kostet_recall(self):
        report = EvalReport(
            results=[
                ergebnis(frage("a", "doc"), ("x", 0.2, False), ("doc", 0.05, True))
            ]
        )
        row = report.thresholds(grid=[0.1])[0]
        # Der richtige Treffer faellt weg, der falsche bleibt.
        assert row.recall == 0.0
        assert "a" in row.lost

    def test_erste_quelle_bleibt_immer(self):
        # rerank() verhaelt sich so: fällt alles durch, bleibt der Erste.
        # Sonst wäre die Frage still unbeantwortbar statt mit dünner Quelle.
        report = EvalReport(
            results=[ergebnis(frage("a", "doc"), ("doc", 0.001, True))]
        )
        row = report.thresholds(grid=[0.5])[0]
        assert row.recall == 1.0

    def test_rauschen_wird_gezaehlt(self):
        report = EvalReport(
            results=[
                ergebnis(
                    frage("c", typ="ohne_antwort"),
                    ("x", 0.02, False),
                    ("y", 0.001, False),
                )
            ]
        )
        offen, streng = report.thresholds(grid=[0.0, 0.05])
        assert offen.noise_kept == 2
        # Bei 0.05 fällt alles durch, der erste bleibt als Rest.
        assert streng.noise_kept == 1


class TestAdmittedIgnorance:
    @pytest.mark.parametrize(
        "text",
        [
            "Die Angabe ist in den Quellen nicht angegeben.",
            "Dazu steht nichts in den bereitgestellten Unterlagen.",
            "Es liegen keine passenden Stellen vor.",
            "Aus den Quellen geht das nicht hervor.",
            "Die Frage lässt sich mit den Quellen nicht beantworten.",
        ],
    )
    def test_erkennt_eingeraeumtes_nichtwissen(self, text):
        check = AnswerCheck(
            question=frage("x", typ="ohne_antwort"),
            text=text,
            contains_strings=False,
            cited_sources=0,
            duration_seconds=1.0,
        )
        assert check.admitted_ignorance

    def test_erkennt_eine_echte_antwort(self):
        check = AnswerCheck(
            question=frage("x", typ="ohne_antwort"),
            text="Der Zuschuss beträgt 200 Euro monatlich [1].",
            contains_strings=False,
            cited_sources=1,
            duration_seconds=1.0,
        )
        assert not check.admitted_ignorance


class TestAnsweredCorrectly:
    """Die Zeichenkettenprüfung allein genügt nicht.

    Gemessen kam: "Die Frage ... ist nicht in den bereitgestellten Quellen
    enthalten. [1] nennt Ruhezeiten von 13:00 bis 15:00 Uhr" — die erwartete
    Zeichenkette steht drin, beantwortet ist die Frage aber nicht. Das hat die
    erste Fassung der Metrik als richtig gezählt.
    """

    def _check(self, text: str, treffer: bool = True) -> AnswerCheck:
        return AnswerCheck(
            question=frage("x", "doc"),
            text=text,
            contains_strings=treffer,
            cited_sources=1,
            duration_seconds=1.0,
        )

    def test_echte_antwort_zaehlt(self):
        assert self._check("Die Ruhezeit gilt von 13:00 bis 15:00 Uhr [1].").answered_correctly

    def test_nichtwissen_mit_zitat_zaehlt_nicht(self):
        check = self._check(
            "Die Frage ist in den Quellen nicht enthalten. [1] nennt 13:00 Uhr."
        )
        assert check.contains_strings
        assert check.admitted_ignorance
        assert not check.answered_correctly

    def test_ohne_treffer_kein_erfolg(self):
        assert not self._check("Eine ganz andere Auskunft.", treffer=False).answered_correctly
