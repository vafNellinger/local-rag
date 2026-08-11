"""Modelle offline zuerst laden — und den nutzlosen HF-Hinweis dämpfen.

Zwei Dinge, die beim Laden zwischengespeicherter Modelle stören:

**Metadaten-Roundtrips.** sentence-transformers prüft beim Laden per Netz, ob
die zwischengespeicherten Dateien noch aktuell sind — ein paar kleine
HTTP-Anfragen pro Modell (ETag-Abgleich), kein Download der Gewichte.
``local_files_only=True`` schaltet das für Loader ab, die es beachten (der
CrossEncoder tut das vollständig). Fehlt das Modell im Cache, wirft der Loader,
und dann wird einmalig mit Netz nachgeladen — heruntergeladen wird also nur,
was fehlt.

**Der Hinweis „unauthenticated requests to the HF Hub".** Er kommt bei jeder
solchen Anfrage über den Logger ``huggingface_hub.utils._http`` und rät zu
einem ``HF_TOKEN`` für höhere Rate-Limits. Für ein lokales Ein-Nutzer-Werkzeug
an öffentlichen Modellen ist das nie umsetzbar und nie relevant — reines
Rauschen, das wie ein Download aussieht. Ein gezielter Log-Filter lässt genau
diese eine Meldung fallen und sonst nichts. bge-m3 macht seine Metadaten-Calls
auch unter ``local_files_only=True`` (eine sentence-transformers-Eigenheit);
der Filter greift dort, wo der Parameter nicht durchdringt.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _DropUnauthenticatedHint(logging.Filter):
    """Lässt genau den HF-Token-Hinweis fallen, alles andere durch."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "unauthenticated requests to the HF Hub" not in record.getMessage()


def _install_hint_filter() -> None:
    # Auf den konkreten Logger, nicht global: nur diese eine Quelle soll
    # gefiltert werden, echte Fehler des HF-Clients bleiben sichtbar. Idempotent
    # aufrufbar — die Modell-Loader importieren dieses Modul, der Filter steht
    # damit vor dem ersten Laden.
    hf_logger = logging.getLogger("huggingface_hub.utils._http")
    if not any(isinstance(f, _DropUnauthenticatedHint) for f in hf_logger.filters):
        hf_logger.addFilter(_DropUnauthenticatedHint())


_install_hint_filter()


def load_offline_first(build: Callable[[bool], T], *, was: str) -> T:
    """Modell laden: erst offline, bei Fehlschlag mit Netz.

    ``build(local_files_only)`` erzeugt das Modell und reicht das Flag an den
    Loader durch. Offline (``True``) heißt: was im Cache liegt, lädt ohne
    Download. Fehlt es, wirft der Loader; dann wird einmalig mit Netz geladen
    und dabei heruntergeladen. Scheitert auch das, schlägt dessen Ausnahme zum
    Aufrufer durch — sie ist die aussagekräftige.
    """
    try:
        return build(True)
    except Exception as offline_exc:  # noqa: BLE001 — Grund wird geloggt
        logger.info(
            "%s nicht offline ladbar (%s) — lade vom HF Hub (einmalig)",
            was,
            offline_exc,
        )
        return build(False)
