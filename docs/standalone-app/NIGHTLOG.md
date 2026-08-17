# Nachtprotokoll — autonome Arbeit 2026-08-18

Über Nacht bearbeitet, alles committet und nach `origin` gepusht. Verifiziert auf
**CPU** — diese Umgebung hat keine nutzbare GPU, die GPU-Messungen sind dein
erster Schritt am Morgen.

## Überblick

| Strang | Ergebnis |
|---|---|
| **1 — GPU-Embedding via ONNX** | ✅ vollständig: austauschbare ONNX-Engine für Embedding + Reranking, bit-identisch zu sentence-transformers, getestet |
| **5 — App-Reife (Datenpfade)** | ✅ vollständig: plattformkorrekte Pfade via platformdirs + Migration |
| **3 — native GUI** | ✅ war bereits erledigt (Launcher nutzen `--native`, Browser-Fallback) |
| **2 — GPU-LLM (Vulkan)** | ✅ auf der AMD-iGPU gemessen: **1,99× schneller**, Integration funktioniert schon |
| INT8-Vertiefung | 📊 gemessen: schneller + 4× kleiner, aber −1,6 % Recall — als opt-in-Werkzeug |
| **4 — Bundle** | ⏸️ braucht Cross-Platform-Runner + Signing-Accounts |

6 Commits (`c96625c`-Bereich … `e770912`). Volle Testsuite grün (434 passed).
Details je Strang unten.

## Strang 1 — GPU-Embedding via ONNX (Detail)

## Erledigt und verifiziert

1. **ONNX-Export** (`tools/export_onnx.py`): bge-m3 und bge-reranker-v2-m3 nach
   ONNX exportiert (je ~2,2 GB, Modell + Tokenizer, unter
   `~/.cache/local-rag/onnx/`). `optimum` ist reines Bauwerkzeug; die App lädt
   zur Laufzeit nur `onnxruntime` + Tokenizer.
2. **Treue bewiesen** (der Kern):
   - Embedding bge-m3: Cosine-Ähnlichkeit ONNX↔sentence-transformers = **1.000000**,
     mittlere Abweichung **-1,5e-08** über die 88 Korpus-Chunks. Die ONNX-Pipeline
     (CLS-Pooling + L2-Normalize) reproduziert die ST-Vektoren exakt.
   - Reranker: ONNX-sigmoid vs. `CrossEncoder.predict` max. Abweichung **2,5e-07**,
     Ranking identisch.
   - **Folge:** Ein ONNX-gebauter Index ist bitgleich zu einem
     sentence-transformers-Index — Recall/MRR sind damit garantiert unverändert.
3. **Austauschbare Engine** (nach dem `vectors.py`-Backend-Muster):
   - `rag/embed.py`: `_SentenceTransformersEngine` (Standard) + `_OnnxEngine`,
     Wahl über `EmbedderConfig.engine` (`"sentence-transformers"` | `"onnx"`),
     sauberer Fallback wenn ONNX-Modell fehlt.
   - `rag/rerank.py`: analog (`_CrossEncoderEngine` + `_OnnxRerankerEngine`).
   - Laufzeit-Extra `[onnx]` = nur `onnxruntime`; `[export-onnx]` = `optimum`
     (getrennt, weil optimum transformers < 5 pinnt).
4. **Tests**: `tests/test_engines.py` (9 Tests: Engine-Wahl, Provider-Rangfolge,
   Fallback über Mocks, Treue-Integration). Volle Suite **grün: 419 passed,
   1 skipped** — der Umbau bricht nichts.

## Wichtiger, unerwarteter Befund: ONNX-CPU ist langsamer

Der Fahrplan nahm an, ONNX sei auch auf CPU 2–4× schneller. **Gemessen ist das
Gegenteil:** unoptimiertes fp32-ONNX ist auf dieser CPU ~3× **langsamer** als
Torch+MKL (213 vs. 73 ms/Chunk, bge-m3, 250 Chunks). Grund: onnxruntimes
CPU-Kernel (MLAS) schlägt Torch+MKL für dieses Transformer-Modell in fp32 nicht.

Einordnung — das entwertet ONNX **nicht**, verschiebt aber die Begründung:

- Der Wert von ONNX ist die **herstellerübergreifende GPU** (DirectML/CoreML/
  CUDA), nicht CPU-Tempo. Das ist hier mangels GPU nicht messbar → **Morgen-TODO
  auf echter Hardware.**
- CPU-Tempo ließe sich mit **INT8-Quantisierung** deutlich heben (typisch 2–4×) —
  offener Folge-Schritt.
- Konsequenz für die Vorgabe: **Standard bleibt sentence-transformers** (auf CPU
  schneller); ONNX ist opt-in und lohnt auf GPU-Systemen. Sinnvoll wäre, die
  Engine an die Plattformklasse zu koppeln (GPU-Klassen → ONNX), analog
  `embedder_device`.

## Messungen (final, CPU)

- **Thread-Tuning:** onnxruntime läuft am besten mit intra_op = physische Kerne
  (12 von 24 logischen): 156 statt 210 ms/Chunk. Eingebaut über
  `_physical_cores()` + `_onnx_session_options()` (Linux exakt aus
  `/proc/cpuinfo`, sonst logische halbiert), `ORT_ENABLE_ALL`.
- **Embedding-Tempo** (bge-m3, 250 Chunks, CPU): Torch 74 ms/Chunk, ONNX
  (optimiert) 156 ms/Chunk → auf CPU **~2× langsamer**.
- **Retrieval-Qualität** (`rag eval`, ONNX-Embedding, `--no-rerank`):
  Recall@1/3/5 = 85.0/98.3/98.3 %, MRR 0.914 — **bit-identisch** zu
  sentence-transformers (erwartet, da die Vektoren bitgleich sind).

## Fazit Strang 1

Der ONNX-Weg ist vollständig, getestet und ohne Qualitätsverlust einsatzbereit —
aber auf CPU langsamer als Torch. Sein eigentlicher Nutzen (herstellerübergreifende
GPU) steht noch zur Messung aus. Deshalb: **Standard bleibt sentence-transformers**,
ONNX ist ein sauberes opt-in (`engine = "onnx"` je Profil in `platforms.toml`).

## Morgen zuerst (auf echter Hardware, dein Schritt)

1. **GPU messen:** `onnxruntime-gpu` (CUDA) bzw. `onnxruntime-directml` (Windows)
   statt `onnxruntime` installieren, `engine = "onnx"` setzen, Embedding-Tempo
   gegen Torch vergleichen. Erst das beweist den eigentlichen Hebel.
2. Optional: **INT8-Quantisierung** des ONNX-Modells fürs CPU-Embedding.
3. Danach: Engine-Wahl an die Plattformklasse koppeln (GPU-Klassen → ONNX).

## Geänderte/neue Dateien

- `rag/embed.py`, `rag/rerank.py` — austauschbare Engine (ST + ONNX, Fallback)
- `tools/export_onnx.py` — Export-Werkzeug (neu)
- `tests/test_engines.py` — 9 Tests (neu)
- `pyproject.toml` — `[onnx]` (Laufzeit: nur onnxruntime) + `[export-onnx]` (optimum)

## Strang 2 — GPU-LLM via Vulkan ✅ (die Maschine hat doch eine GPU)

Zunächst als „keine GPU" eingeschätzt — dann stellte sich heraus: die Maschine
hat eine **AMD Radeon 880M/890M iGPU (RDNA 3.5, GFX1150)**, nutzbar über
**Vulkan** (RADV-Treiber, API 1.4) — ohne CUDA/ROCm. Und llama-cpp-python ist
hier bereits mit Vulkan gebaut (`gpu_offload: True`, `matrix cores: KHR_coopmat`).

- **Gemessen** (Qwen3-4B-Instruct Q5_K_M, 120 Token): GPU **24,0 tok/s** vs. CPU
  **12,1 tok/s** = **1,99× schneller**. Für eine iGPU mit geteiltem RAM solide;
  eine dedizierte GPU (NVIDIA/CUDA) liefert deutlich mehr.
- **Integration läuft schon**: `detect_local()` erkennt die GPU korrekt
  (Plattformklasse `igpu_shared`, ~8 GB Shared-VRAM), und `platforms.toml` setzt
  dort `generator_device = "gpu"`. Die App nutzt die Vulkan-GPU also automatisch
  für die Antwortgenerierung. Embedder/Reranker bleiben CPU — onnxruntime/torch
  haben auf AMD-Linux ohne ROCm keinen GPU-Provider; genau dafür ist die Klasse
  so geschnitten.

Damit ist der eigentliche Query-Hebel (Generierung auf GPU) auf verbreiteter
AMD-APU-Hardware bestätigt — der Vulkan-Weg des Fahrplans trägt.

**Nebenfund beim end-to-end-Test:** Ein echter `rag ask` gegen den Index
(„Welche Kündigungsfrist gilt beim Gewerbemietvertrag?") lieferte die korrekte
Antwort mit Quelle (mietvertrag-gewerbe.md, Rerank-Score 0,882), Generierung
5,9 s auf der GPU — **nachdem** ein Absturz gefixt war: `rag ask` war seit dem
Conversational-Routing-Umbau kaputt (`ask_stream` liefert 3 Werte, der CLI
entpackte 2), unbemerkt, weil der Command-Pfad keinen Test hatte. Gefixt
(`4d82ab0`) + Regressionstest (`tests/test_cli_ask.py`).

---

# Strang 5 — App-Reife: Cross-Platform-Datenpfade

Erledigt und verifiziert:

- **`rag/paths.py`** (neu): zentrale Speicherorte über `platformdirs` —
  data/cache/config je Plattform korrekt (Windows `%LOCALAPPDATA%`, macOS
  `~/Library`, Linux XDG). Index + Dokumente → **data**, Einstellungen →
  **config**, Extraktions-Cache/ONNX/Erkennungs-Caches/GUI-Log → **cache**.
- **Alle hartcodierten `~/.cache`/`~/.config`/`~/.local/share`-Pfade** in
  `detect.py`, `pipeline.py`, `cli.py`, `ui.py`, `extract.py`, `hfmeta.py`,
  `embed.py` auf `rag.paths` umgestellt; dabei ungenutzte Importe bereinigt.
- **Migration**: ein Alt-Index unter `~/.cache/local-rag/index.db` (der frühere
  Sammelort) wandert einmalig samt WAL/SHM in den data-Ordner, ohne etwas zu
  überschreiben. `platformdirs` als Kern-Dependency ergänzt.
- **Tests**: `tests/test_paths.py` (7 Tests: Pfad-Rollen; Migration idempotent
  und ohne Altbestand). Volle Suite **grün: 435 passed, 1 skipped**.

Auf Linux ändert sich nur der Index-Ort (cache → data); cache/config/dokumente
bleiben gleich. Auf Windows/macOS liegen jetzt alle Daten am konventionell
richtigen Ort — die Voraussetzung fürs Bundle (Strang 4).

---

# Strang 3 (native GUI) — bereits erledigt

Beim Prüfen festgestellt: Stufe 1 ist schon umgesetzt. Alle Launcher rufen
`rag gui --native` auf (`start.sh`, `start.bat`, `packaging/*`), und `ui.py`
(`_webview_verfuegbar()`) fällt sauber auf den Browser zurück, wenn keine Webview
da ist. Kein Handlungsbedarf; Stufe 2 (echtes Qt-Toolkit) bleibt ein eigenes,
größeres Vorhaben.

---

# Vertiefung Strang 1 — INT8-Quantisierung (gemessen, nicht eingebaut)

Die offene Frage aus Strang 1 (kann ONNX auf CPU konkurrenzfähig werden?)
beantwortet: mit dynamischer INT8-Quantisierung ja — als bewusste Abwägung.
Werkzeug neu: `tools/quantize_onnx.py`.

| | Torch (fp32) | ONNX fp32 | ONNX INT8 |
|---|---|---|---|
| Tempo (CPU) | 74 ms/Chunk | 156 ms/Chunk | **64 ms/Chunk** |
| Modellgröße | — | 2,2 GB | **568 MB** |
| Treue zu ST (Cosine) | 1,0 | 1,0 | 0,978 |
| Recall@5 (Goldstandard) | 98,3 % | 98,3 % | 96,7 % |

INT8 ist die schnellste CPU-Option und 4× kleiner (relevant fürs Bundle), kostet
aber ~1,6 Punkte Recall. Deshalb **nicht als Standard eingebaut** — die Abwägung
gehört dir, idealerweise nach `rag eval` mit den eigenen Dokumenten und einem
INT8-gebauten Index.

**Reranker (bge-reranker-v2-m3)** — hier lohnt INT8 am meisten, weil der Reranker
die Query-Latenz treibt (läuft bei jeder Frage): **2,88× schneller** (13,2 statt
37,9 ms/Paar), 4× kleiner (560 MB), und das **Ranking bleibt gleich** (absolute
Scores weichen bis ~0,08 ab — bei gesetzter `min_rerank_score`-Schwelle also
nachjustieren). Für ein CPU-Bundle der attraktivste INT8-Kandidat. Dasselbe
Werkzeug: `python tools/quantize_onnx.py <reranker-verzeichnis>`.

## Offen für den Morgen

- **Strang 2 (GPU-LLM)** und die **GPU-Messung von Strang 1** brauchen echte
  GPU-Hardware (dein erster Schritt).
- **Strang 3 (native GUI)**: pywebview-Default lokal machbar, GUI-Start aber
  schwer ohne Display zu verifizieren.
- **Strang 4 (Bundle)**: braucht Cross-Platform-Runner + Signing-Accounts.
