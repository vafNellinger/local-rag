"""Tests für Prompt-Bau und Quellenverwaltung.

Ohne Modell: geprüft wird, was vor der Generierung passiert. Der Schwerpunkt
liegt auf dem Kontextbudget — wenn dort zu viel hineingeht, kostet das die
Antwort, nicht nur eine Quelle.
"""

from __future__ import annotations

import pytest

from rag.generate import (
    CONTEXT_BUDGET_SHARE,
    SYSTEM_PROMPT,
    Answer,
    GenerationError,
    Source,
    build_prompt,
    build_sources,
    estimate_tokens,
    resolve_gpu_layers,
    supports_gpu_offload,
)
from rag.store import SearchHit


def hit(text: str, *, heading: tuple[str, ...] = (), pfad: str = "/x/akte.pdf") -> SearchHit:
    return SearchHit(
        chunk_id=1,
        document_path=pfad,
        ordinal=0,
        text=text,
        heading_path=heading,
        kind="prosa",
        distance=0.2,
    )


class TestBuildSources:
    def test_numeriert_ab_eins(self):
        sources, dropped = build_sources(
            [hit("Eins"), hit("Zwei")], context_tokens=8192
        )
        assert [s.number for s in sources] == [1, 2]
        assert dropped == 0

    def test_reihenfolge_bleibt(self):
        # Der Reranker hat schon sortiert — die Nummerierung darf das nicht
        # durcheinanderbringen, sonst zitiert das Modell die falsche Stelle.
        sources, _ = build_sources(
            [hit("Wichtig"), hit("Weniger")], context_tokens=8192
        )
        assert sources[0].hit.text == "Wichtig"

    def test_budget_begrenzt_die_zahl(self):
        lang = "Wort " * 400
        sources, dropped = build_sources([hit(lang) for _ in range(20)], context_tokens=2048)
        assert len(sources) < 20
        assert dropped == 20 - len(sources)

    def test_erste_quelle_kommt_immer_mit(self):
        # Auch wenn sie allein das Budget sprengt: ohne Quelle gibt es keine
        # Antwort, und die beste ist die erste.
        riesig = "Wort " * 100_000
        sources, _ = build_sources([hit(riesig)], context_tokens=2048)
        assert len(sources) == 1

    def test_budget_nutzt_nur_einen_teil_des_fensters(self):
        # Der Rest bleibt für Systemprompt, Frage und Antwort.
        assert 0 < CONTEXT_BUDGET_SHARE < 1

    def test_leere_treffer(self):
        assert build_sources([], context_tokens=8192) == ([], 0)


class TestSourceRendering:
    def test_quelle_traegt_ihre_nummer(self):
        source = Source(number=3, hit=hit("Inhalt", heading=("Kapitel",)))
        assert source.render().startswith("[3] ")

    def test_citation_enthaelt_dateiname_und_pfad(self):
        source = Source(
            number=1,
            hit=hit("x", heading=("Kündigung", "Fristen"), pfad="/a/vertrag.pdf"),
        )
        assert source.citation == "vertrag.pdf — Kündigung > Fristen"

    def test_text_steht_im_render(self):
        source = Source(number=1, hit=hit("Die Frist beträgt 14 Tage."))
        assert "Die Frist beträgt 14 Tage." in source.render()


class TestBuildPrompt:
    def test_system_und_user_nachricht(self):
        messages = build_prompt("Wie lang?", [Source(1, hit("Sechs Monate."))])
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == SYSTEM_PROMPT

    def test_frage_steht_nach_den_quellen(self):
        # Bei langem Kontext gewichten Modelle das Ende stärker; die Frage
        # darf nicht in der Mitte der Quellen verschwinden.
        messages = build_prompt("MEINEFRAGE", [Source(1, hit("QUELLENTEXT"))])
        user = messages[1]["content"]
        assert user.index("QUELLENTEXT") < user.index("MEINEFRAGE")

    def test_ohne_quellen_ist_das_kein_rag(self):
        with pytest.raises(GenerationError, match="kein RAG"):
            build_prompt("Frage", [])

    def test_zitierpflicht_steht_im_systemprompt(self):
        assert "[2]" in SYSTEM_PROMPT

    def test_nichtwissen_ist_erlaubt(self):
        assert "Rate nicht" in SYSTEM_PROMPT

    def test_antwortsprache_ist_deutsch(self):
        assert "Deutsch" in SYSTEM_PROMPT


class TestAnswer:
    def test_zitierte_nummern_werden_erkannt(self):
        answer = Answer(question="?", text="Sechs Monate [1] und mehr [3].")
        assert answer.cited_numbers == {1, 3}

    def test_ohne_zitate_leere_menge(self):
        assert Answer(question="?", text="Keine Belege.").cited_numbers == set()

    def test_unzitierte_quellen_werden_gemeldet(self):
        answer = Answer(
            question="?",
            text="Nur die erste [1].",
            sources=[Source(1, hit("a")), Source(2, hit("b"))],
        )
        assert [s.number for s in answer.uncited_sources] == [2]

    def test_tokenrate(self):
        answer = Answer(
            question="?", text="x", completion_tokens=50, duration_seconds=10.0
        )
        assert answer.tokens_per_second == pytest.approx(5.0)

    def test_tokenrate_ohne_dauer(self):
        assert Answer(question="?", text="x").tokens_per_second == 0.0

    def test_klammern_ohne_zahl_stoeren_nicht(self):
        answer = Answer(question="?", text="Siehe [oben] und [2].")
        assert answer.cited_numbers == {2}


class TestGpuLayers:
    def test_null_bleibt_null(self):
        assert resolve_gpu_layers(0) == 0

    def test_wunsch_ohne_faehigkeit_faellt_auf_cpu(self, caplog):
        # Auf dieser Maschine der Normalfall: der pip-Build hat nur ein
        # CPU-Backend, n_gpu_layers wuerde sonst still ignoriert.
        if supports_gpu_offload():
            pytest.skip("Dieser llama.cpp-Build kann auslagern")
        assert resolve_gpu_layers(-1) == 0
        assert any("CPU" in r.message for r in caplog.records)

    def test_faehigkeit_ist_ein_bool(self):
        assert isinstance(supports_gpu_offload(), bool)


class TestEstimateTokens:
    def test_mindestens_eins(self):
        assert estimate_tokens("") == 1

    def test_waechst_mit_der_laenge(self):
        assert estimate_tokens("Wort " * 100) > estimate_tokens("Wort")
