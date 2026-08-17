# Strang 1 — GPU-Embedding via ONNX Runtime

**Ziel:** Embedding (bge-m3) und Reranking (bge-reranker-v2-m3) auf der GPU
ausführen — herstellerübergreifend (NVIDIA/AMD/Intel/Apple) — für schnellen
Ingest, besonders beim Neu-Embedden nach einem Modellwechsel.

## Ausgangslage

- `rag/embed.py` (`Embedder`) und `rag/rerank.py` (`Reranker`) laden ihre
  Modelle über sentence-transformers/`CrossEncoder`, also PyTorch. GPU nur via
  CUDA-Torch — nicht herstellerübergreifend.
- Eine **Device-Abstraktion existiert schon**: `resolve_device()` in
  `embed.py`, und `platforms.toml` legt pro Plattformklasse (`cpu_only`,
  `igpu_shared`, `dgpu_small`, `dgpu_large`) je Rolle ein Gerät fest.
- Das Projekt kennt bereits das **austauschbare-Backend-Muster** (`rag/vectors.py`
  mit `VectorBackend`-Protokoll). Dasselbe Muster wenden wir auf die
  Inferenz-Engine an.

## Ansatz

Eine **Embedding-Engine hinter ein Protokoll** ziehen, analog zu `vectors.py`:

- `SentenceTransformersEngine` — der heutige Weg (Standard, bleibt als Fallback).
- `OnnxEngine` — lädt ein ONNX-Modell und rechnet über **ONNX Runtime** mit
  herstellerübergreifenden Execution-Providern:
  - `CUDAExecutionProvider` (NVIDIA),
  - `DmlExecutionProvider` (Windows/DirectML — deckt AMD & Intel mit ab),
  - `CoreMLExecutionProvider` (macOS),
  - `CPUExecutionProvider` (Fallback, auch hier oft 2–4× schneller als Torch).

Warum ONNX statt CUDA-Torch: **ein Modell, alle GPU-Hersteller** — dieselbe
Idee wie Vulkan beim LLM (Strang 2). Vermeidet eine CUDA/ROCm-Build-Matrix.

## Schritte

1. **Modelle nach ONNX exportieren.**
   - bge-m3 und bge-reranker-v2-m3 via `optimum-cli export onnx` (bzw.
     `optimum.onnxruntime`). Achtung bge-m3: nur die **Dense**-Ausgabe
     verwenden (die Sparse/ColBERT-Köpfe ignorieren) — das ist genau das, was
     `embed.py` heute nutzt.
   - Export als expliziten Schritt in `rag pull` einhängen (bzw. ein
     `tools/export_onnx.py`), damit reproduzierbar.
2. **Tokenizer entkoppeln.** Den HF-`tokenizers`-Tokenizer direkt laden (ohne
   das volle Modell), Ausgabe (input_ids/attention_mask) an die ONNX-Session
   geben. Präfix-Logik (`query_prefix`/`passage_prefix`) aus `EmbedderConfig`
   bleibt unverändert.
3. **`OnnxEngine` implementieren:** `encode(texts) -> vectors`, Mean-Pooling +
   L2-Normalisierung (wie sentence-transformers es tut), Batchgröße wie heute.
4. **Provider-Auswahl** analog `packaging/detect_gpu.py`: verfügbare Provider
   abfragen, in fester Rangfolge den ersten passenden wählen, sonst CPU.
5. **Engine-Wahl** an die bestehende Config hängen: neues Feld je Embedder-
   Profil (`engine = "onnx" | "sentence-transformers"`), Standard bleibt
   sentence-transformers, bis ONNX verifiziert ist.
6. **Fallback:** Fehlt das ONNX-Modell oder scheitert die Session, sauber auf
   `SentenceTransformersEngine` zurückfallen (Muster wie `load_offline_first`).
7. **Reranker analog** über dieselbe Engine-Abstraktion.

## Risiken / offene Punkte

- **Numerische Treue:** ONNX-Ausgabe muss den Torch-Vektoren entsprechen, sonst
  passt der Index nicht zur Query. → in der Verifikation Cosine-Abweichung
  Torch↔ONNX prüfen (Ziel: < 1e-3).
- **DirectML-Reife** bei großen Transformer-Modellen — auf Zielhardware testen.
- **bge-m3-Export** kann wegen der Multi-Head-Architektur fummelig sein; als
  Erstes isoliert verifizieren, bevor der Umbau von `embed.py` beginnt.
- **docling/easyocr bleiben Torch** — dieser Strang macht das Bundle nicht
  torch-frei; der Gewinn ist GPU-Portabilität und Tempo, nicht primär Größe.

## Erweiterung: GPU für Extraktion/OCR

Bei komplexen, **gescannten** Dokumenten ist nicht das Embedding der einzige
Ingest-Engpass, sondern auch **docling-Layout + easyocr** — beide laufen auf
Torch und heute auf der CPU (`ocr.device = "cpu"` in `platforms.toml`). OCR auf
CPU ist langsam; bei großen Scan-Mengen dominiert das den Ingest.

- **Kurzfristig:** docling/easyocr auf **CUDA** stellen, wo eine NVIDIA-GPU da
  ist (Torch kann das direkt; `ocr.device`/`embedder_device`-Logik erweitern).
- **Herstellerübergreifend** ist es schwerer als beim Embedding, weil docling
  kein ONNX-Austauschmodell mitbringt. Optionen prüfen: ONNX-Exporte der
  docling-Layoutmodelle bzw. ein ONNX-OCR (z. B. PaddleOCR-ONNX) als
  alternatives OCR-Backend hinter einer Abstraktion wie beim Embedder.
- **Priorität:** nach dem Embedding-Umbau. Erst messen, wie groß der
  OCR-Anteil am realen Ingest deiner Dokumente ist (Profiling wie in dieser
  Sitzung), dann entscheiden, wie weit die herstellerübergreifende OCR-GPU
  lohnt.

## Verifikation (Erfolgskriterium)

1. **Tempo:** Benchmark an den 88 Korpus-Chunks — ONNX-CPU vs. Torch-CPU vs.
   ONNX-GPU. Erwartung: ONNX-CPU 2–4× schneller, GPU 5–15×.
2. **Qualität:** `rag eval` gegen den Goldstandard muss **gleiche** Recall@k/MRR
   liefern wie heute (der Umbau darf die Retrieval-Güte nicht verändern).
3. **Treue:** mittlere Cosine-Abweichung Torch↔ONNX < 1e-3 über den Korpus.
