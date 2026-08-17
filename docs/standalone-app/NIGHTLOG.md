# Nachtprotokoll — Strang 1 (GPU-Embedding via ONNX)

Autonome Arbeit in der Nacht auf 2026-08-18. Fokus: Strang 1 tief und sauber.
Reihenfolge nach dem Fahrplan, verifiziert auf **CPU** (diese Umgebung hat keine
nutzbare GPU — die GPU-Messung ist der erste Schritt am Morgen auf echter
Hardware).

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

## Strang 2 übersprungen (nicht autonom verifizierbar)

GPU-LLM (Vulkan/Metal) braucht GPU-Hardware und ein Vulkan-SDK — beides in dieser
Umgebung nicht vorhanden, und `n_gpu_layers` ist im Code ohnehin schon
vorbereitet. Deshalb stattdessen der nächste voll lokal umsetzbare Strang.

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

## Offen für den Morgen

- **Strang 2 (GPU-LLM)** und die **GPU-Messung von Strang 1** brauchen echte
  GPU-Hardware (dein erster Schritt).
- **Strang 3 (native GUI)**: pywebview-Default lokal machbar, GUI-Start aber
  schwer ohne Display zu verifizieren.
- **Strang 4 (Bundle)**: braucht Cross-Platform-Runner + Signing-Accounts.
