"""Tests für den Sprachfilter.

Ohne Netzzugriff: ``fetch_languages`` wird gepatcht. Die Testfälle bilden
reale HF-Antworten ab, wie sie am 2026-07-29 über die Model-API gemessen
wurden.
"""

from __future__ import annotations

import pytest

from rag import hfmeta
from rag.hfmeta import language_verdict

# Gemessene Realdaten. Die drei None-Fälle sind der Grund für die
# "fehlende Tags nicht bestrafen"-Regel.
REAL_TAGS = {
    "Qwen/Qwen3-4B-Instruct-2507": None,
    "google/gemma-3-27b-it": None,
    "openai/gpt-oss-20b": None,
    "meta-llama/Meta-Llama-3-8B-Instruct": ["en"],
    "zai-org/GLM-4.7-Flash": ["en", "zh"],
    "microsoft/Phi-4-mini-instruct": ["multilingual", "de", "en", "fr"],
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": ["en", "de", "fr", "es"],
    "unsloth/Qwen3-8B-GGUF": ["en"],
}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(
        hfmeta, "fetch_languages", lambda mid, use_cache=True: REAL_TAGS.get(mid)
    )


class TestUntaggedModelsPassieren:
    """Die entscheidende Regel — sonst fliegen die besten Modelle raus."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "Qwen/Qwen3-4B-Instruct-2507",
            "google/gemma-3-27b-it",
            "openai/gpt-oss-20b",
        ],
    )
    def test_ohne_tag_durchgelassen(self, model_id):
        verdict = language_verdict(model_id, "de")
        assert verdict.ok
        assert "nicht bestraft" in verdict.reason

    def test_unbekanntes_modell_wird_durchgelassen(self):
        # Auch der Fehlerfall (Netz weg, 404) landet hier — fail-open.
        assert language_verdict("irgendwer/neues-modell", "de").ok


class TestGetaggteModelle:
    def test_englischzentriert_wird_abgelehnt(self):
        verdict = language_verdict("meta-llama/Meta-Llama-3-8B-Instruct", "de")
        assert not verdict.ok
        assert "'de' fehlt" in verdict.reason

    def test_andere_sprachkombination_wird_abgelehnt(self):
        verdict = language_verdict("zai-org/GLM-4.7-Flash", "de")
        assert not verdict.ok
        assert "zh" in verdict.reason

    def test_multilingual_marker_zaehlt(self):
        assert language_verdict("microsoft/Phi-4-mini-instruct", "de").ok

    def test_zielsprache_explizit_gelistet(self):
        assert language_verdict(
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "de"
        ).ok

    def test_andere_zielsprache_aendert_das_urteil(self):
        # Derselbe Filter muss für ein englisches RAG das Gegenteil liefern.
        assert language_verdict("meta-llama/Meta-Llama-3-8B-Instruct", "en").ok


class TestNamensmuster:
    def test_fremdsprachiges_modell_ohne_netzabfrage(self):
        verdict = language_verdict(
            "elyza/Llama-3-ELYZA-JP-8B", "de", exclude_name_patterns=("elyza",)
        )
        assert not verdict.ok
        assert "elyza" in verdict.reason

    def test_muster_ist_case_insensitive(self):
        verdict = language_verdict(
            "Some/Japanese-Model-7B", "de", exclude_name_patterns=("japanese",)
        )
        assert not verdict.ok

    def test_muster_schlaegt_tags(self):
        # Namensmuster greift vor der Tag-Auswertung, auch wenn die Tags
        # unauffällig wären.
        verdict = language_verdict(
            "microsoft/Phi-4-mini-instruct", "de", exclude_name_patterns=("phi-4",)
        )
        assert not verdict.ok


class TestRepackagerFreistellung:
    def test_falscher_tag_wird_ohne_freistellung_bestraft(self):
        # unsloth taggt das Qwen3-Repackage als 'en', obwohl das Original
        # multilingual ist.
        assert not language_verdict("unsloth/Qwen3-8B-GGUF", "de").ok

    def test_freistellung_hebt_den_tag_auf(self):
        verdict = language_verdict("unsloth/Qwen3-8B-GGUF", "de", ignore_tags=True)
        assert verdict.ok
        assert "freigestellt" in verdict.reason

    def test_freistellung_umgeht_das_namensmuster_nicht(self):
        # Reihenfolge: Namensmuster ist die härtere Aussage und gilt weiter.
        verdict = language_verdict(
            "some/japanese-repack",
            "de",
            exclude_name_patterns=("japanese",),
            ignore_tags=True,
        )
        assert not verdict.ok
