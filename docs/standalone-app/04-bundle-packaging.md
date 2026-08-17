# Strang 4 — Cross-Platform-Bundle & Installer

**Ziel:** Ein per Wizard installierbares Paket je Plattform (Windows, macOS,
Debian), das nach der Installation sofort lauffähig ist — Python und alle
Bibliotheken sind enthalten, nur die Modelle kommen beim ersten Start.

## Ausgangslage

- Heutiges Modell ist **Bootstrap**: NSIS (Windows) und `.deb` (Debian) bündeln
  nur Quellcode + Launcher; beim ersten Start holt `uv` Python, alle
  Abhängigkeiten und Modelle. macOS fehlt.
- Nachteil für das Ziel: Der erste Start ist eine lange zweite Phase mit
  Internetzwang. Auf macOS zudem mit Notarisierung praktisch unvereinbar
  (Gatekeeper erwartet alle Binaries im signierten Paket).
- Entscheidung: Wechsel von Bootstrap zu **echtem Bundle** (PyInstaller oder
  Nuitka).

## Ansatz

Pro OS ein statisches Bundle (Python + Libs eingefroren), verpackt in einen
Wizard-Installer. **Modelle nicht bündeln** — sie kommen über einen
First-Run-Download-Flow (klein halten, Flexibilität beim Modellwechsel
erhalten, den wir bewusst unterstützen).

## Schritte

1. **Bundler wählen & Spec bauen.** PyInstaller (verbreiteter, mehr Rezepte für
   torch/onnxruntime) oder Nuitka (kompiliert, kleiner/schneller). Empfehlung:
   mit PyInstaller starten.
2. **Torch/docling/onnxruntime zähmen.** Bekannte PyInstaller-Fallstricke:
   fehlende Data-Files (docling-Modelldefinitionen, easyocr, tokenizers),
   lazy/hidden imports, native Libs (`onnxruntime`-Provider-DLLs,
   `llama_cpp`-Backend). Über `hiddenimports`/`datas`/`binaries` in der Spec
   lösen; iterativ auf jedem OS testen.
3. **First-Run-Modell-Download in der App.** Ein Schritt in der GUI/CLI: erkennt
   fehlende Modelle, lädt sie mit Fortschrittsanzeige (nutzt den vorhandenen
   `rag pull`-Weg und das Offline-first-Laden). Danach ist die App offline
   nutzbar.
4. **Pro Plattform verpacken:**
   - **Windows:** PyInstaller (onedir) → Inno Setup oder NSIS als Wizard;
     WebView2-Runtime prüfen/mitliefern. **Code-Signing nötig** (OV-/EV-
     Zertifikat) — sonst zeigt SmartScreen „unbekannter Herausgeber". Das
     Pendant zur Mac-Notarisierung, mit eigenem jährlichen Zertifikat.
   - **macOS:** PyInstaller → `.app` → `codesign` (Developer ID) → `notarytool`
     (Notarisierung) → `.dmg`. **Setzt einen Apple Developer Account voraus.**
   - **Debian:** gebündeltes venv in ein `.deb` (statt uv-Bootstrap) oder
     alternativ **AppImage** für ein distributionsunabhängiges Doppelklick-Paket.
5. **GPU-Backends einbetten** (aus Strang 1 & 2): ONNX-Runtime mit
   DirectML/CoreML/CUDA-Providern und Vulkan-/Metal-llama.cpp ins Bundle
   aufnehmen; Provider-/Backend-Wahl zur Laufzeit (herstellerübergreifend,
   daher ein Bundle pro OS statt Hardware-Matrix).
6. **CI-Pipeline (zwingend) — Forgejo Actions.** Das Team nutzt auf
   `git.parracidal.de` **Forgejo Actions** (`.forgejo/workflows/*.yml`,
   GitHub-Actions-kompatible Syntax) mit **wiederverwendbaren Workflows** aus dem
   zentralen Repo `niklas.nellinger/infrastructure` und Ablage in der
   Forgejo-**Container-Registry** (Login per `REGISTRY_TOKEN`-Secret). Belegt in
   `frostline-app`/`frostline-base`: Image bauen → pushen → „Bump"-Job
   aktualisiert den Tag in `infrastructure/stacks/.../docker-compose.yml` →
   Deploy über docker-compose/Traefik (GitOps).
   - **Passt nur teilweise auf local-rag:** Hier ist das Ziel **kein**
     Container-Deploy, sondern **native Desktop-Installer**. Statt eines
     Docker-Images entstehen `.exe`/`.dmg`/`.deb`, die als **Forgejo-Release**
     abgelegt werden. Übernehmbar sind die Forgejo-Actions-Mechanik (Workflows,
     Secrets, Registry für Zwischen-Artefakte) — nicht das GitOps-Deploy-Ziel.
   - **Kernproblem — Build-Runner:** Der vorhandene Runner ist ein **`arm64-pi`**
     (ARM64-Linux, Raspberry Pi). Der kann **keine** Windows-, macOS- oder
     x86-Linux-Bundles bauen. Für die drei Ziel-Plattformen braucht es entweder
     zusätzliche self-hosted Forgejo-Runner (Windows-VM, Mac-mini, x86-Linux)
     **oder** die Bundle-Builds werden auf GitHub Actions ausgelagert
     (öffentliche Win/Mac/Linux-Runner) und nur die fertigen Installer als
     Artefakte zurück nach `git.parracidal.de` (Forgejo-Release) gespiegelt.
   - Rauch-Test je Bundle (App startet headless, `rag status`); Vulkan-Wheels
     ggf. hier bauen (Strang 2).

## Auto-Update

Bei einem Bundle ist jede neue Version ein neuer, großer Installer. Damit
installierte Nutzer nicht auf altem Stand hängen bleiben:

- Mindestens eine **Update-Benachrichtigung** (App prüft eine Versions-Datei auf
  `git.parracidal.de`/Release-Feed und weist auf eine neue Version hin).
- Optional ein **Auto-Updater** (z. B. Sparkle auf macOS, WinSparkle/MSIX auf
  Windows, `.deb`-Repo auf Debian). Zweitrangig, aber früh mitdenken, damit die
  Installer-Struktur dazu passt.
- Die **Modelle** bleiben beim Update erhalten (liegen im user-data-Pfad, siehe
  Strang 5) — nur der App-Teil wird ersetzt.

## Lizenzen / Redistribution

Vor dem Bündeln prüfen, ob alles weiterverteilt werden darf:

- **Modelle:** bge-m3, bge-reranker, das gewählte LLM (GGUF) — jeweils Lizenz
  auf Redistribution/kommerzielle Nutzung prüfen. (Ein Argument mehr, Modelle
  **nicht** zu bündeln, sondern beim ersten Start vom jeweiligen Anbieter zu
  laden.)
- **Bibliotheken:** Lizenzhinweise der gebündelten Libs (PyTorch, docling,
  onnxruntime, llama.cpp, pywebview/Qt) im Installer beilegen.

## GPU-Verifikation auf echter Hardware

CI-Runner haben meist **keine** (schon gar nicht herstellerdiverse) GPU. Die
herstellerübergreifende GPU-Unterstützung (NVIDIA/AMD/Intel/Apple) lässt sich
dort nicht vollständig testen. Nötig ist eine kleine **manuelle Test-Matrix** auf
realen Geräten (mindestens je eine NVIDIA-, AMD-/Intel- und Apple-Maschine) vor
einem Release — der CI-Rauchtest deckt nur den CPU-Pfad ab.

## Risiken / offene Punkte

- **macOS-Notarisierung** braucht den Apple Developer Account (99 $/Jahr) — ohne
  ihn nur unsignierte Auslieferung mit Gatekeeper-Warnung. **Vor Mac-Arbeit
  klären.**
- **Windows-Code-Signing** braucht ein OV-/EV-Zertifikat (eigene jährliche
  Kosten) — sonst SmartScreen-Warnung.
- **Windows-/macOS-Build-Runner** auf der self-hosted Instanz (siehe Schritt 6).
- **Bundle-Größe:** ~1,5–2 GB je OS (torch bleibt wegen docling/easyocr).
  Modelle nicht eingerechnet.
- **PyInstaller + torch** ist erfahrungsgemäß die aufwendigste Einzelaufgabe —
  Zeitpuffer je Plattform einplanen.
- **CI-Laufzeit/-Kosten** für drei OS mit großen Bundles.

## Verifikation (Erfolgskriterium)

Auf je einem **frischen** Windows-, macOS- und Debian-Gerät (bzw. VM):
Installer ausführen → App startet als eigenes Fenster → First-Run lädt die
Modelle → ein Dokument-Set wird indiziert → eine Frage wird beantwortet. Kein
Terminal, keine manuelle Einrichtung, kein Compiler.
