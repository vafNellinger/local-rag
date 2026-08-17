# Strang 5 — App-Reife / Produktisierung

**Ziel:** Aus dem funktionierenden Werkzeug eine ausgelieferte Anwendung machen,
die eine nicht-technische Person installieren, bedienen, warten und wieder
entfernen kann — auf allen drei Plattformen.

Dieser Strang liefert für sich keinen sichtbaren „Wow"-Effekt, ist aber die
Voraussetzung dafür, dass Strang 4 (Bundle) überhaupt ein *fertiges Produkt*
verpackt statt eines Entwickler-Werkzeugs. Er kann früh und unabhängig
beginnen.

## Ausgangslage

- **Datenpfade sind Linux-hartcodiert.** Index, Settings, Caches und Logs
  liegen unter `~/.cache/local-rag` bzw. `~/.config/local-rag`
  (`rag/detect.py:CACHE_DIR`, `rag/pipeline.py:SETTINGS_PATH`/`DEFAULT_INDEX_PATH`,
  `rag/hfmeta.py:CACHE_FILE`, `rag/extract.py:EXTRACT_CACHE_DIR`). Das ist der
  XDG-Stil — auf Windows und macOS der falsche Ort.
- **Konfiguration läuft über `platforms.toml` + CLI-Flags.** Für eine
  installierte App muss das Wichtigste (Modellwahl, GPU an/aus) aus der
  Oberfläche erreichbar sein.
- Ein GUI-Protokoll existiert bereits (`rag/ui.py:LOG_PATH`).

## Aufgaben

1. **Cross-Platform-Datenpfade** über `platformdirs`:
   - `user_data_dir` → Index, Modelle, extrahierter Cache
   - `user_config_dir` → `settings.json`
   - `user_log_dir` → Protokolle
   - Eine einzige Stelle (`rag/paths.py` o. ä.), die alle bisherigen
     hartcodierten `Path.home() / ".cache"/".config"` ersetzt. Bestehende
     Linux-Installationen ggf. migrieren (alten Pfad erkennen und verschieben).
2. **First-Run-Erlebnis:**
   - Hardware erkennen (vorhandene GPU/VRAM, RAM) und **sinnvolle Modell-Defaults
     vorschlagen** — kleines LLM/Quantisierung auf schwacher Hardware, größeres
     auf einer dGPU. Baut auf den Plattformklassen in `platforms.toml` auf.
   - Modelle mit **Fortschrittsanzeige** laden, **Wiederaufnahme** nach Abbruch,
     klare Größenangabe vorab.
3. **Einstellungen in der GUI:** Embedder-/LLM-Modell wählen, GPU-Nutzung
   umschalten, Reranking an/aus — ohne TOML/CLI. Schreibt in dieselbe
   `settings.json`.
4. **Fehler & Support für Nicht-Techniker:**
   - Verständliche Meldungen bei den typischen Fällen (keine GPU/Treiber, zu
     wenig Speicher, Download abgebrochen, Modell fehlt).
   - „Diagnose exportieren" — Logs + Umgebungsinfo in eine Datei, die man zur
     Fehlersuche weitergeben kann.
5. **Deinstallation:** sauberer Deinstaller je OS, der fragt, ob die
   heruntergeladenen Modelle (zweistellige GB) mitentfernt werden sollen.

## Risiken / offene Punkte

- **Pfad-Migration** bestehender Linux-Nutzer: nicht Daten verlieren, wenn der
  Speicherort wechselt.
- **Hardware-Erkennung** muss konservativ sein — lieber ein kleineres Modell
  vorschlagen, das sicher läuft, als eines, das den Speicher sprengt.

## Verifikation (Erfolgskriterium)

Eine nicht-technische Testperson kann auf jedem der drei Systeme: installieren,
beim ersten Start ein passendes Modell geführt herunterladen, Dokumente
indizieren, Fragen stellen, in den Einstellungen das Modell wechseln, bei einem
Fehler eine Diagnosedatei erzeugen und die App sauber deinstallieren — alles
ohne Terminal.
