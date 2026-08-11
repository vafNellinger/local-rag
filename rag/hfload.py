"""Modelle offline zuerst laden — und den nutzlosen HF-Hinweis dämpfen.

Zwei Dinge, die beim Laden zwischengespeicherter Modelle stören:

**Metadaten-Roundtrips.** sentence-transformers und die transformers-Schicht
darunter prüfen beim Laden per Netz, ob die zwischengespeicherten Dateien noch
aktuell sind — ein paar ETag-Anfragen pro Modell, kein Download der Gewichte.
``local_files_only=True`` schaltet das nicht zuverlässig ab: der CrossEncoder
beachtet es, bge-m3 fragt trotzdem an. Ohne Netz scheitert der Offline-Ladeweg
dann an einem Timeout statt am fehlenden Modell, und im Protokoll sieht die
Abfrage aus wie ein Download. Der harte Schalter ist ``HF_HUB_OFFLINE`` samt
``TRANSFORMERS_OFFLINE`` — ``load_offline_first`` legt beide für den
Offline-Versuch um und nimmt sie für den Netz-Fallback wieder zurück, sodass
ein fehlendes Modell weiterhin einmalig geladen wird.

**Der Hinweis „unauthenticated requests to the HF Hub".** Er kommt beim
Nachladen über den Logger ``huggingface_hub.utils._http`` und rät zu einem
``HF_TOKEN`` für höhere Rate-Limits. Für ein lokales Ein-Nutzer-Werkzeug an
öffentlichen Modellen ist das nie umsetzbar und nie relevant — reines Rauschen,
das wie ein Download aussieht. Ein gezielter Log-Filter lässt genau diese eine
Meldung fallen und sonst nichts.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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


# Die Schalter, die die HF-Bibliotheken offline zwingen: der erste deckt
# huggingface_hub ab, der zweite die transformers-Schicht darunter, die
# sentence-transformers benutzt.
_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


@contextmanager
def _forced_offline() -> Iterator[None]:
    """huggingface_hub und transformers für die Dauer des Blocks offline zwingen.

    Die Umgebungsvariablen greifen für Module, die erst noch importiert werden;
    die schon importierten lesen ihr Offline-Flag nur einmal beim Import,
    deshalb werden ihre Konstanten zusätzlich direkt umgelegt — nur, wo das Modul
    bereits geladen ist, sonst genügt die Variable. Alles wird am Ende exakt
    zurückgesetzt, damit der Netz-Fallback wieder laden darf.

    Prozessweit, nicht threadlokal: os.environ und die Modulkonstanten sind
    global. Das Modell-Laden ist verzögert und läuft praktisch serialisiert,
    ein zweiter Thread mitten im selben Block ist also kein realer Fall.
    """
    zuvor_env = {name: os.environ.get(name) for name in _OFFLINE_ENV}
    for name in _OFFLINE_ENV:
        os.environ[name] = "1"

    # (Modul, Attribut, alter Wert) — nur für bereits importierte Bibliotheken.
    patches: list[tuple[object, str, object]] = []
    for modul, attribut in (
        ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
        ("transformers.utils.hub", "_is_offline_mode"),
    ):
        geladen = sys.modules.get(modul)
        if geladen is not None and hasattr(geladen, attribut):
            patches.append((geladen, attribut, getattr(geladen, attribut)))
            setattr(geladen, attribut, True)

    try:
        yield
    finally:
        for modul, attribut, alt in patches:
            setattr(modul, attribut, alt)
        for name, alt in zuvor_env.items():
            if alt is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = alt


def load_offline_first(build: Callable[[bool], T], *, was: str) -> T:
    """Modell laden: erst offline, bei Fehlschlag mit Netz.

    ``build(local_files_only)`` erzeugt das Modell und reicht das Flag an den
    Loader durch. Der Offline-Versuch läuft zusätzlich unter erzwungenem
    Offline-Modus (siehe ``_forced_offline``), damit auch die ETag-Anfragen
    entfallen, die ``local_files_only=True`` allein nicht abstellt: was im Cache
    liegt, lädt dann ganz ohne Netz. Fehlt es, wirft der Loader; dann wird
    einmalig mit Netz geladen und dabei heruntergeladen. Scheitert auch das,
    schlägt dessen Ausnahme zum Aufrufer durch — sie ist die aussagekräftige.
    """
    try:
        with _forced_offline():
            return build(True)
    except Exception as offline_exc:  # noqa: BLE001 — Grund wird geloggt
        logger.info(
            "%s nicht offline ladbar (%s) — lade vom HF Hub (einmalig)",
            was,
            offline_exc,
        )
        return build(False)
