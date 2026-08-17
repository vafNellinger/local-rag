# Strang 3 — Browser-lose native GUI

**Ziel:** Die App startet in einem eigenen Fenster statt in einem Browser-Tab —
auf allen drei Plattformen.

## Ausgangslage

- `rag gui --native` existiert bereits (pywebview, Extra `native` in
  `pyproject.toml`). Es öffnet ein eigenständiges Fenster über die
  System-Webview (WebView2 auf Windows, WebKitGTK auf Linux, WKWebView auf Mac).
- Ausgeliefert wird bisher aber der **Browser-Modus** (`start.sh` öffnet die
  Oberfläche im Browser).
- Die Oberfläche `rag/ui.py` (~1.265 Zeilen) ist NiceGUI, also HTML/JS im
  Webview gerendert.

## Ansatz

**Zwei Stufen**, wobei Stufe 1 das erklärte Ziel bereits erfüllt:

### Stufe 1 — pywebview als Standard (empfohlen für die Auslieferung)

Der `--native`-Modus wird der **Standard** im ausgelieferten Launcher. Vorteile:
`ui.py` bleibt vollständig unverändert, es funktioniert schon, und es ist
cross-platform. „Ohne Browser" im Sinne von „eigenes Fenster, kein Tab, keine
URL-Leiste" ist damit erreicht. (Nutzer-Aussage: „Webview wäre nicht schlimm".)

Aufgaben:
1. Auslieferungs-Launcher/Entry-Point startet `--native` statt Browser.
2. Fenster-Feinschliff: App-Icon, Titel, sinnvolle Start-Fenstergröße,
   Schließen-/Minimieren-Verhalten, Single-Instance.
3. Sicherstellen, dass die Webview-Abhängigkeit im Bundle steckt bzw. als
   OS-Abhängigkeit deklariert ist (Debian: `webkit2gtk`; Windows: WebView2
   Runtime; macOS: systemeigen).

### Stufe 2 — echtes natives Toolkit (optional, später)

Für „echt nativ" (Nutzer-Präferenz, aber nicht zwingend): `ui.py` auf
**PySide6/Qt** neu aufbauen. Erheblicher Aufwand (die HTML-Oberfläche komplett
neu), daher nur, wenn Look-&-Feel später wirklich zählt. Bis dahin bleibt
Stufe 1.

## Risiken / offene Punkte

- **WebView2 (Windows)** muss vorhanden sein — auf aktuellen Windows-Systemen
  Standard, andernfalls im Installer mitliefern/anfordern.
- **WebKitGTK (Linux)** als `.deb`-Abhängigkeit (ist bereits in `build-deb.sh`
  deklariert).
- Der Sprung auf Qt (Stufe 2) ist ein eigenes Projekt — nicht mit Stufe 1
  vermengen.

## Verifikation (Erfolgskriterium)

Auf Windows, macOS und Debian: App startet als eigenes Fenster (kein Browser,
keine URL-Leiste), lässt sich normal bedienen, schließen und wieder öffnen.
