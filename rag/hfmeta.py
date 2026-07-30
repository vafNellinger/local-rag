"""Sprachmetadaten von HuggingFace, mit persistentem Cache.

whichllm liefert keine Sprachinformation — ``ModelInfo`` hat kein
entsprechendes Feld. Die HF-Model-API hat sie in ``cardData.language``, aber
lückenhaft: über die Hälfte der Kandidaten taggt gar nichts, darunter die
komplette Qwen3-Familie und Gemma, die beide ausgeprägt multilingual sind.

Daraus folgt die Auswertungsregel in ``language_verdict``: vorhandene Tags
werden ausgewertet, fehlende nicht bestraft.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_FILE = Path.home() / ".cache" / "local-rag" / "hf-language.json"

# Sprachtags eines Modells ändern sich praktisch nie — nur bei einem
# Card-Update. Eine Woche ist reichlich konservativ.
CACHE_TTL_SECONDS = 7 * 24 * 3600

HTTP_TIMEOUT_SECONDS = 15

# Ein Modell, das explizit "multilingual" angibt, zählt für jede Zielsprache.
MULTILINGUAL_MARKER = "multilingual"


@dataclass(frozen=True)
class LanguageVerdict:
    """Ergebnis der Sprachprüfung für ein Modell."""

    ok: bool
    reason: str
    # None = keine Tags vorhanden, [] = leere Liste, sonst die Tags.
    tags: list[str] | None = None


def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Sprach-Cache defekt, wird neu aufgebaut: %s", CACHE_FILE)
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        CACHE_FILE.write_text(json.dumps(cache))
    except OSError as exc:
        logger.warning("Sprach-Cache nicht schreibbar: %s", exc)


def fetch_languages(model_id: str, *, use_cache: bool = True) -> list[str] | None:
    """Hole ``cardData.language`` für ein Modell.

    Rückgabe ``None`` bedeutet "unbekannt" — entweder kein Tag gesetzt oder
    die Abfrage ist fehlgeschlagen. Beides wird bewusst gleich behandelt:
    fehlende Information darf ein Modell nicht disqualifizieren.
    """
    cache = _load_cache() if use_cache else {}
    entry = cache.get(model_id)
    if entry and (time.time() - entry.get("fetched_at", 0)) < CACHE_TTL_SECONDS:
        return entry.get("languages")

    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "local-rag"})
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            data = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        # Fail-open: ohne Netz soll die Auflösung weiterlaufen, nicht abbrechen.
        logger.debug("Sprachabfrage für %s fehlgeschlagen: %s", model_id, exc)
        return None

    languages = (data.get("cardData") or {}).get("language")
    if isinstance(languages, str):
        languages = [languages]
    if languages is not None and not isinstance(languages, list):
        languages = None

    cache[model_id] = {"languages": languages, "fetched_at": time.time()}
    _save_cache(cache)
    return languages


def language_verdict(
    model_id: str,
    target: str,
    *,
    exclude_name_patterns: tuple[str, ...] = (),
    ignore_tags: bool = False,
    use_cache: bool = True,
) -> LanguageVerdict:
    """Prüfe, ob ein Modell für die Zielsprache taugt.

    Drei Stufen, von billig nach teuer:

    1. Namensmuster für explizit fremdsprachige Modelle (kein Netzzugriff).
       Solche Modelle tragen ihre Sprache fast immer im Namen.
    2. Vorhandene HF-Sprachtags: fehlt die Zielsprache und ist das Modell
       nicht als multilingual markiert, wird es abgelehnt.
    3. Keine Tags: durchlassen. Ein harter Filter würde hier Qwen3 und Gemma
       verwerfen und damit die stärksten Kandidaten.

    ``ignore_tags`` überspringt Stufe 2 für Repos, deren Sprachtag
    nachweislich falsch ist — typischerweise Repackager, die den Tag des
    Originals nicht übernehmen.
    """
    lower = model_id.lower()

    for pattern in exclude_name_patterns:
        if pattern.lower() in lower:
            return LanguageVerdict(
                ok=False, reason=f"Namensmuster '{pattern}' (andere Zielsprache)"
            )

    if ignore_tags:
        return LanguageVerdict(ok=True, reason="Sprachtag freigestellt (Repackager)")

    languages = fetch_languages(model_id, use_cache=use_cache)

    if languages is None:
        return LanguageVerdict(
            ok=True, reason="keine Sprachangabe — nicht bestraft", tags=None
        )

    normalized = {str(entry).lower() for entry in languages}
    if MULTILINGUAL_MARKER in normalized or target.lower() in normalized:
        return LanguageVerdict(ok=True, reason="Zielsprache angegeben", tags=languages)

    listed = ", ".join(sorted(normalized)[:6])
    return LanguageVerdict(
        ok=False,
        reason=f"laut Model Card nur {listed} — '{target}' fehlt",
        tags=languages,
    )
