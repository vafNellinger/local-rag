# Fahrplan: local-rag als eigenständige GPU-App

Zielbild und Umsetzungsplan, um local-rag als eigenständige, per Wizard
installierbare Desktop-Anwendung mit GPU-Nutzung auszuliefern — für Windows,
macOS und Debian, ohne Browser.

## Endziel

Eine Person lädt einen Installer, klickt sich durch einen Wizard, und startet
danach eine App in einem eigenen Fenster (kein Browser-Tab). Die App nutzt die
vorhandene GPU — herstellerübergreifend, ohne dass die Person etwas
konfiguriert. Dokumente werden indiziert und beantwortet, lokal, ohne dass
Daten das Gerät verlassen.

## Grundsatzentscheidung: Python bleibt

Kein Rewrite in einer anderen Sprache. Zwei belegte Gründe:

1. **docling ist unersetzlich.** Die zu indizierenden Dokumente sind komplex
   (Scans, Layouts, Tabellen). Die Extraktionsqualität ist das Fundament eines
   RAG — ist sie schlecht, hilft die beste GPU danach nichts. Für docling
   (`rag/extract.py`) gibt es außerhalb von Python keinen gleichwertigen Ersatz.
2. **Die Sprache ist nicht der Engpass.** Gemessen: bge-m3-Embedding = 100 %
   nativer torch/BLAS-Kernel (`torch._C._nn.linear`), 0 % messbares Python-Glue,
   ~77 ms/Chunk auf CPU. Ein Sprachwechsel ruft denselben Kernel auf und gewinnt
   nichts. Der Hebel ist die **GPU** (~5–15×), und die ist aus Python erreichbar.

Rust wäre nur für die Verpackung attraktiv (kleine Single-Binary) — das wiegt
den Verlust von docling nicht auf.

## Die fünf Stränge

| # | Strang | Wirkung | Abhängt von |
|---|--------|---------|-------------|
| 1 | [GPU-Embedding via ONNX](01-gpu-embedding-onnx.md) | schneller Ingest & Modellwechsel; herstellerübergreifende GPU (inkl. OCR/Extraktion) | — |
| 2 | [GPU-LLM (Vulkan/Metal)](02-gpu-llm.md) | schnelle Antworten auf jeder GPU | — |
| 3 | [Browser-lose native GUI](03-native-gui.md) | eigenes Fenster statt Browser | — |
| 5 | [App-Reife / Produktisierung](05-app-reife.md) | Cross-Platform-Datenpfade, First-Run, Einstellungen, Support | — |
| 4 | [Cross-Platform-Bundle & Installer](04-bundle-packaging.md) | Wizard-Installer Win/Mac/Debian | 1–3, 5 |

Stränge 1, 2, 3 und 5 sind **unabhängig** voneinander und liefern je für sich
Wert. Strang 4 verpackt das Ergebnis und kommt zuletzt. Die Nummern sind IDs,
nicht die Reihenfolge — siehe unten.

## Empfohlene Reihenfolge

1. **Strang 1 (GPU-Embedding)** zuerst — löst das eigentliche Problem („viele
   Dateien, Modellwechsel"), unabhängig, und der Gewinn ist messbar.
2. **Strang 2 (GPU-LLM)** parallel/danach — größtenteils Wheel-Beschaffung, da
   `n_gpu_layers` im Code schon vorgesehen ist.
3. **Strang 3 (native GUI)** — klein, `--native` existiert bereits.
4. **Strang 5 (App-Reife)** — parallel zu 1–3 beginnen; die Cross-Platform-
   Datenpfade müssen stehen, **bevor** gebündelt wird.
5. **Strang 4 (Bundle)** zuletzt — bündelt 1–3 + 5 zu Installern, braucht CI.

Jeder Strang wird nach dem Muster dieser Sitzung **verifiziert** (Benchmark bzw.
Frischinstallation), bevor der nächste beginnt.

## Übergreifende Risiken

- **Signierung kostet extern:** macOS-Notarisierung braucht einen Apple
  Developer Account (99 $/Jahr), Windows ein OV-/EV-Code-Signing-Zertifikat
  (eigene jährliche Kosten). Ohne sie warnen Gatekeeper bzw. SmartScreen —
  klären, bevor Strang 4 die jeweilige Plattform angeht.
- **Cross-Platform-Builds brauchen Windows-/macOS-Runner** — die self-hosted
  Instanz `git.parracidal.de` (Forgejo/Gitea) hat üblicherweise nur Linux-
  Runner; native Bundles brauchen eigene Win/Mac-Runner oder ausgelagerte
  Builds (Details in Strang 4).
- **Bundle-Größe:** PyTorch bleibt im Bundle (docling/easyocr brauchen es),
  selbst wenn Embedding/Rerank auf ONNX laufen. Rechne mit ~1,5–2 GB je OS,
  Modelle nicht eingerechnet (die kommen beim ersten Start).
- **TLS-Zertifikat von `git.parracidal.de` ist abgelaufen** (Stand 2026-08-17) —
  blockiert Remote-Push und CI-Zugriff, bis es erneuert ist.

## Erfolgskriterium

Auf einem frischen Windows-, macOS- und Debian-Gerät: Installer ausführen → App
startet als eigenes Fenster → erstes Dokument-Set wird auf der GPU indiziert →
Frage wird GPU-beschleunigt beantwortet. Ohne Terminal, ohne manuelle
Einrichtung.
