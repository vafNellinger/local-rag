# local-rag

Lokales RAG-System, dessen Modellauswahl von der Zielplattform abhängt.

Stand: **Schritt 1 von 5** — Plattformerkennung und Modellauflösung stehen.
Ingest und Query fehlen noch.

## Idee

Pipeline-Code kennt nur Rollen (`generator`, `embedder`, `reranker`), niemals
Modellnamen. Welches Modell eine Rolle auf einer Plattform bekommt, entscheidet
`rag/resolve.py`. Wechselt die Hardware, wechselt dort die Antwort — sonst
nirgends.

Den Generator löst [whichllm](https://github.com/Andyyyy64/whichllm) auf.
Embedder und Reranker stehen in `config/platforms.toml`, weil whichllm sie
strukturell nicht kennt: `models/hf.py` filtert hart auf
`pipeline_tag=text-generation`, Embedding-Modelle tragen aber
`feature-extraction`.

## Setup

```bash
uv venv && uv pip install -e .
pipx install whichllm     # falls noch nicht vorhanden
```

## Verwendung

```bash
rag plan                      # lokale Maschine
rag plan --gpu "RTX 4090"     # Zielrechner simulieren, ohne dort zu sein
rag plan --vram 16            # nutzbares VRAM überschreiben
rag plan --cpu-only
rag plan --refresh            # whichllm-Cache umgehen (dauert Minuten)

rag inspect ~/dokumente       # was ist extrahierbar, wo ist OCR nötig
rag convert datei.pdf -o out.md
rag convert scan.pdf --ocr    # OCR erzwingen statt automatisch entscheiden
```

Ergebnisse werden 24 h in `~/.cache/local-rag/` gecacht; ein
whichllm-Lauf dauert ein bis drei Minuten.

## Was die Auflösung über whichllm hinaus tut

whichllm rankt ein einzelnes Modell gegen die gesamte Hardware. Eine
RAG-Pipeline lädt mehrere Modelle, die sich denselben Speicher teilen, und hat
andere Qualitätsanforderungen als allgemeine Nutzung. `resolve.py` ergänzt
deshalb:

- **VRAM-Budget-Aufteilung.** GPU-resident geplante Nebenrollen werden vom
  Kartenspeicher abgezogen, der Rest via `--vram` als Budget an whichllm
  übergeben. Auf einer 24-GB-Karte plant der Generator so gegen 20.7 GB statt
  gegen 22.8 GB.
- **Kein Reasoning-Modell.** whichllm rankt Thinking-Varianten nach oben; bei
  RAG steht die Antwort im Kontext und die Thinking-Tokens kosten nur Latenz.
- **Harte Quantisierungsgrenze.** whichllms Penalty für Q3_K_M liegt bei 8 %.
  Für wörtliche Kontexttreue ist das zu mild, deshalb Q4_K_S als Untergrenze.
- **Natives Kontextfenster als Filter.** whichllm berechnet `context_fits`
  in `engine/compatibility.py`, nutzt es aber nicht im Ranking — ein Modell
  mit 8k Fenster kann bei `-c 32k` auf Platz 1 stehen, mit einer Warnung, die
  leicht untergeht.
- **Sprachfilter** (siehe unten) und **Base-Modell-Ausschluss**: `-pt` ist bei
  Google die Endung für `pretrained`, das Gegenstück zum instruction-tuned
  `-it`. Ein Base-Modell folgt keiner Anweisung und taugt nicht für RAG.

Die meisten davon sind Konfiguration in `platforms.toml`, Kontextfenster und
Sprache sind Code. Alle sind Kandidaten für einen späteren Upstream-Beitrag.

## Sprachfilter

Die Zieldokumente sind deutsch, also muss der Generator Deutsch können.
whichllm hat kein Sprachkonzept, und die HF-Metadaten sind lückenhaft.
Gemessen über 59 reale Kandidaten:

| | Anzahl |
|---|---|
| ohne jeden `language`-Tag | 32 |
| getaggt, ohne `de`/`multilingual` | 19 |
| getaggt, mit `de`/`multilingual` | 8 |

Unter den 32 untaggten sind Qwen3 und Gemma — beide ausgeprägt multilingual.
Ein harter Positiv-Filter würde also die stärksten Kandidaten verwerfen.
`rag/hfmeta.py` prüft deshalb dreistufig, von billig nach teuer:

1. **Namensmuster** (kein Netzzugriff): Modelle mit fremder Zielsprache tragen
   sie fast immer im Namen (`elyza`, `swallow`, `chatglm`, …).
2. **Vorhandene Tags**: fehlt die Zielsprache und ist das Modell nicht als
   `multilingual` markiert → raus. Sortiert real Llama-2/3, Mistral-7B-v0.1,
   Phi-3, Olmo, Falcon und Qwen2.5 aus.
3. **Keine Tags** → durchlassen, fehlende Metadaten disqualifizieren nicht.

Repackager übernehmen Sprachtags nicht immer korrekt (`unsloth/Qwen3-8B-GGUF`
ist als reines `en` getaggt); solche Repos lassen sich über `ignore_tags_for`
freistellen. Ergebnisse werden 7 Tage in `~/.cache/local-rag/` gecacht, die
Abfrage läuft fail-open — ohne Netz bricht die Auflösung nicht ab.

Zielsprache umstellen: `target` unter `[generator.language]` in
`config/platforms.toml`.

## Plattformklassen

| Klasse | Bedingung | Generator | Embedder | Reranker |
|---|---|---|---|---|
| `cpu_only` | kein nutzbares VRAM | CPU | CPU | aus |
| `igpu_shared` | geteilter Speicher oder ≤ 10 GB | GPU | CPU | CPU |
| `dgpu_small` | 10–20 GB dediziert | GPU | GPU | CPU |
| `dgpu_large` | > 20 GB dediziert | GPU | GPU | GPU |

Geteilter Speicher schlägt die VRAM-Zahl: bei einer APU konkurriert jedes
GPU-resident geladene Modell direkt mit dem System-RAM.

## Extraktion

Zwei Stufen, weil sie zehnfach unterschiedlich teuer sind:

`probe()` liest den Text-Layer mit pypdf und klassifiziert jede Seite.
`convert()` macht die eigentliche Arbeit über Docling — Layout, Tabellen,
Überschriften, bei Bedarf OCR (EasyOCR, `de`+`en`).

Gemessen auf dieser Maschine (CPU): **~0,8 s pro Seite ohne OCR, ~8,5 s mit
OCR.** Bei 1000 Seiten mit 10 % Scans sind das 29 Minuten selektiv gegen 2,4
Stunden pauschal — deshalb entscheidet `probe()` vorab pro Seite.

Ein PDF ist zwei Formate in einer Datei: digital erzeugte Seiten mit
Text-Layer und gescannte Seiten, die nur ein Bild enthalten. Gemischt ist der
Normalfall, also fällt die Entscheidung pro Seite. Vier Zustände:

| Status | Bedeutung |
|---|---|
| `text` | Text-Layer brauchbar |
| `scan` | kein Text, aber seitenfüllendes Bild → OCR hilft |
| `sparse` | kein Text **und** kein Bild → Seite ist leer, OCR bringt nichts |
| `error` | kaputte Kodierung ohne Bild → OCR kann auch nicht helfen |

Die Bildprüfung trennt `sparse` von `scan` und hat einen echten Fehlalarm
gefunden: eine Seite mit 60 Zeichen und ohne jedes Bild war als Scan markiert.
OCR hätte dort ein leeres Blatt abgetastet. Eine Mindestpixelfläche trennt
Seitenscans von Kopfzeilen-Logos.

OCR-Qualität am erzeugten deutschen Testscan verifiziert — Umlaute und ß
fehlerfrei („Änderungsantrag", „Größen", „gemäß", „Jürgen Öztürk").

Altformate (`.doc`, `.ppt`, `.xls`, `.pages`) scheitern mit dem
Konvertierungsbefehl in der Meldung statt mit einem pauschalen „nicht
unterstützt".

## Tests

```bash
uv pip install -e ".[dev]" && pytest -q
```

Getestet wird die Entscheidungslogik, ohne whichllm-Aufruf und ohne Netz.

## Nächste Schritte

2. **Ingest** (angefangen): Extraktion steht, fehlen Chunking → bge-m3 → sqlite-vec
3. Query: Retrieval → Rerank → Prompt → llama-cpp-python
4. Messen: Latenz pro Stufe, Ingest-Durchsatz
5. whichllm erweitern: Rollen, MTEB-Adapter, Budget-Allokation

Offen: Phasen-getrenntes VRAM-Budget (Ingest lädt keinen Generator, Query kein
OCR). Auf dieser Maschine ohne Wirkung, weil Torch die AMD-iGPU nicht sieht —
relevant erst mit einem NVIDIA-Zielrechner.
