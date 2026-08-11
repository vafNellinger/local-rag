"""Einstiegspunkt der Oberfläche für den Reload-Betrieb.

``python -m rag.guiapp`` statt ``rag gui``, und das ist kein Geschmack, sondern
Notwendigkeit. uvicorns Reloader startet einen Kindprozess, der das Hauptmodul
erneut importiert — unter dem Namen ``__mp_main__``. Beim Console-Script ``rag``
ist das Hauptmodul der von pip erzeugte Wrapper:

    from rag.cli import app
    if __name__ == "__main__":
        sys.exit(app())

Im Kind greift dieser Guard nicht, der Befehl läuft dort also nie, ``rag.ui``
wird nie importiert, und keine ``@ui.page`` ist registriert. NiceGUI bricht
folgerichtig ab mit „You must call ui.run() to start the server“.

Dieses Modul hat deshalb **absichtlich keinen Main-Guard**: sein Code läuft im
Kindprozess genauso wie im Eltern. Dort registriert der Import von ``rag.ui``
die Seiten, und ``ui.run()`` kehrt sofort zurück, weil NiceGUI den
Nicht-Hauptprozess erkennt.

Die Parameter kommen aus Umgebungsvariablen und nicht aus ``sys.argv``: der
Kindprozess wird über multiprocessing-spawn erzeugt und erbt die Umgebung, aber
nicht die Argumentliste. NiceGUI selbst löst Host und Port genauso.

``rag gui --reload`` setzt die Variablen und startet dieses Modul; von Hand
braucht man es nicht aufzurufen.
"""

from __future__ import annotations

import os
from pathlib import Path

from rag.ui import run

# Präfix aller hier gelesenen Variablen, damit sie im Prozessbaum
# zuordenbar bleiben.
_PREFIX = "LOCAL_RAG_GUI_"


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(_PREFIX + name)
    return default if raw is None else raw == "1"


_index = os.environ.get(_PREFIX + "INDEX")

run(
    index_path=Path(_index) if _index else None,
    host=os.environ.get(_PREFIX + "HOST", "127.0.0.1"),
    port=int(os.environ.get(_PREFIX + "PORT", "8080")),
    open_browser=_flag("BROWSER", True),
    verbose=_flag("VERBOSE"),
    reload=True,
)
