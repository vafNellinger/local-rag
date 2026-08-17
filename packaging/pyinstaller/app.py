"""Einstiegspunkt für das gebündelte Desktop-Programm.

Startet die Oberfläche als eigenständiges Fenster (NiceGUIs native Modus über
pywebview), ohne Browser. Der Reload-Pfad (``rag/guiapp.py``) ist ein
Entwicklungswerkzeug und im Bundle fehl am Platz — hier läuft alles in einem
Prozess.

Fehlt die System-Webview, fällt ``run`` selbst auf den Browser zurück (lieber
ein Tab als keine Oberfläche), deshalb bleibt ``open_browser`` an.
"""

from rag.ui import run

if __name__ == "__main__":
    run(native=True, open_browser=True)
