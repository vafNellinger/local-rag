# local-rag

Lokales RAG-System, dessen Modellauswahl von der Zielplattform abhängt.

Stand: **Schritt 2 von 5** — Plattformerkennung, Modellauflösung und der
komplette Ingest stehen: Extraktion, Chunking, Embedding, Index, Vektorsuche.
Reranking und Generierung fehlen noch.

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

rag ingest ~/dokumente        # extrahieren, chunken, embedden, indizieren
rag ingest ~/dokumente --prune   # verschwundene Dateien aus dem Index werfen
rag ingest datei.pdf --force     # auch unverändert neu einlesen
rag status                    # was liegt im Index
rag search "Wie lang ist die Kündigungsfrist?"
rag search "Löschfristen" -n 10 --full
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

## Chunking

Docling liefert Überschriften, Tabellen und Absätze. Ein Chunker, der alle N
Zeichen schneidet, wirft genau das weg — `rag/chunk.py` arbeitet deshalb auf
Blöcken. Drei Entscheidungen tragen den Rest:

- **Überschriften-Pfad als Präfix.** „Die Frist beträgt 14 Tage" ist ohne
  „Kündigung > Fristen" nicht auffindbar und im Prompt nicht einordbar. Der
  Pfad wird Teil des embeddeten Textes, bleibt aber aus dem gespeicherten
  Chunk-Text heraus.
- **Sektionsgrenzen sind hart.** Overlap zwischen zwei Abschnitten verbindet
  Themen, die nichts miteinander zu tun haben. Überlappt wird nur innerhalb
  einer Sektion, und der Overlap geht vom Chunk-Budget ab statt dazu — sonst
  überschreiten die Chunks das Ziel genau um seine Länge.
- **Tabellen behalten ihren Kopf.** Wird eine große Tabelle geteilt, bekommt
  jeder Teil die Kopfzeile erneut. „4.500 | 12 | ja" beantwortet ohne
  Spaltennamen keine Frage.

Die Satzsegmentierung kennt deutsche Abkürzungen — ohne sie zerschneidet
`gemäß § 5 Abs. 2 Satz 1` mitten in der Fundstelle. Umgekehrt bleibt sie bei
Ordinalzahlen bewusst konservativ: „1. Januar" und „Satz 1. Danach" sind ohne
Semantik nicht zu unterscheiden, und ein verpasster Schnitt kostet nur
Granularität, ein falscher zerreißt eine Fundstelle.

Gezählt wird mit dem Tokenizer des Embedding-Modells, nicht geschätzt — sonst
schneidet das Modell Chunks ab, die der Chunker für passend hielt. Ohne
geladenes Modell greift eine Zeichenheuristik (3,2 Zeichen pro Token; deutsche
Komposita kosten mehr Subtokens als englischer Text).

Zielgröße 512 Token bei 64 Token Overlap. bge-m3 verkraftet 8192, aber das ist
die falsche Obergrenze: je mehr Themen in einem Vektor landen, desto unschärfer
wird er.

## Index

SQLite plus [sqlite-vec](https://github.com/asg017/sqlite-vec), eine Datei,
kein Server. Cosine-Distanz, weil bge-m3 normalisierte Vektoren liefert.

Zwei Eigenschaften, die beide gegen einen *stillen* Fehler existieren:

- **Der Index kennt sein Embedding-Modell.** Vektoren aus zwei Modellen im
  selben Raum liefern weiter Treffer, nur falsche. Modell, Dimension und
  Schema-Version stehen in `meta` und werden bei jedem Öffnen geprüft; ein
  Wechsel bricht mit Hinweis ab, statt schlechtere Ergebnisse zu liefern.
- **Vektoren werden explizit mitgelöscht.** Die `vec0`-Tabelle hängt nicht am
  `ON DELETE CASCADE` der Chunks. Ohne den eigenen Löschschritt bleiben
  verwaiste Vektoren zurück, die auf nicht mehr existierende Chunks zeigen.

Ingest ist idempotent über den SHA-256 des Dateiinhalts, nicht über die mtime:
Kopieren und Auspacken setzen die mtime neu, ohne dass sich etwas geändert hat.
Ein zweiter Lauf über ein gepflegtes Verzeichnis kostet nur das Hashen. Eine
kaputte Datei beendet den Lauf nicht — bei hundert Dateien ist eine kaputte die
Regel, und ein Abbruch bei Datei 80 verschenkt die Extraktionszeit der ersten
79.

### Welcher Embedder, auf welchem Gerät

Wie beim Generator entscheidet die Plattform, nicht der Code. `platforms.toml`
legt pro Plattformklasse `embedder_profile` und `embedder_device` fest; die
Profile stehen unter `[embedder.*]`. Die Rangfolge:

1. **Was angegeben ist.** `--profile` / `--device`.
2. **Was im Index steht.** Ein bestehender Index gibt das Modell vor, sonst
   würde ein Hardwarewechsel ihn bei jedem Lauf invalidieren. Das Profil liegt
   dafür in der `meta`-Tabelle.
3. **Die Plattformklasse.**

`rag search` fragt nur den Index — gesucht werden *muss* mit dem Modell, mit
dem indiziert wurde. Nebeneffekt: eine Suche kostet keine Plattformerkennung,
was bei 79 ms Query-Latenz auch nicht tragbar wäre. Nur `rag ingest` erkennt
die Plattform, und zwar über `detect_local()` (Hardware-Block, 24 h gecacht) —
nicht über `resolve_pipeline()`, das für eine Embedder-Frage das
Generator-Ranking anstoßen würde.

Beim Embedder ist die plattformabhängige Wahl **enger als beim Generator**:
das Modell muss zwischen Ingest und Query dasselbe sein, sonst passt der Index
nicht zur Anfrage. Ein großes Embedding-Modell konkurriert bei der Query also
direkt mit dem Generator um VRAM, und „beim Ingest groß, bei der Query klein"
ist keine Option. Damit bleibt als sinnvolle Staffelung nur ein *kleineres*
Modell auf schwacher Hardware, nicht ein größeres auf starker — deshalb zeigen
derzeit alle Klassen auf `default`, und die Zuordnung wartet auf eine Messung
an echten Dokumenten statt auf eine Leaderboard-Zahl.

Fällt ein gewünschtes Gerät weg, wird gewarnt statt still auf CPU
zurückgefallen: auf dieser Maschine ist genau das der Normalfall (CUDA-Build
von Torch, AMD-iGPU unsichtbar), und ein stiller Fallback lässt einen die
Ursache der Langsamkeit an der falschen Stelle suchen.

Gemessen auf dieser Maschine (12 Kerne, CPU, bge-m3):

| | |
|---|---|
| Modell laden | 7,5 s, einmalig pro Lauf |
| Embedding | 1,2 Chunks/s bei ~500 Token pro Chunk |
| Query-Embedding | 79 ms |
| zweiter Lauf, nichts geändert | 1,1 s für drei Dateien, Modell wird nicht geladen |

Das Modell wird erst beim ersten zu verarbeitenden Dokument geladen. Ein Lauf
über ein unverändertes Verzeichnis zahlt die 7,5 Sekunden deshalb nicht.

Hochgerechnet auf 1000 Seiten ohne Scans: ~13 Minuten Extraktion plus ~18
Minuten Embedding. Das Embedding ist damit die teurere Hälfte des Ingests —
auf einem Rechner mit sichtbarer GPU der erste Hebel.

## Tests

```bash
uv pip install -e ".[dev]" && pytest -q
```

Getestet wird die Entscheidungslogik: ohne whichllm-Aufruf, ohne Netz und ohne
Modell-Download. Der Embedder wird in den Ingest-Tests durch einen Stub
ersetzt, der aus dem Text deterministische Vektoren ableitet — geprüft wird die
Steuerung (Dateiauswahl, Idempotenz, Fehlerbehandlung), nicht die
Retrieval-Qualität.

## Nächste Schritte

3. Query: Reranking (bge-reranker-v2-m3) → Prompt → llama-cpp-python
4. Messen: Latenz pro Stufe, Ingest-Durchsatz, Retrieval-Qualität an echten
   deutschen Dokumenten
5. whichllm erweitern: Rollen, MTEB-Adapter, Budget-Allokation

Offen:

- **Seitenzahlen pro Chunk.** Docling kennt die Provenance jedes Elements,
  `export_to_markdown()` verliert sie. Für Quellenangaben („Seite 12") müsste
  die Extraktion strukturiert statt als Markdown durchgereicht werden.
- **Hybrid-Retrieval.** bge-m3 liefert neben dem Dense-Vektor auch
  Sparse-Gewichte; genutzt wird bisher nur der Dense-Teil. Bei Aktenzeichen und
  Eigennamen wäre die lexikalische Komponente der größere Hebel — es war das
  ausschlaggebende Argument für dieses Modell.
- **Empirischer Embedder-Vergleich.** `[embedder.qwen3]` steht als Profil
  bereit. Ein Wechsel invalidiert den Index (bewusst: der Store lehnt ihn ab).
- **Phasen-getrenntes VRAM-Budget** (Ingest lädt keinen Generator, Query kein
  OCR). Für den Embedder beim Ingest ist das umgesetzt — er nimmt die GPU,
  wenn Torch eine sieht. Auf dieser Maschine ohne Wirkung, weil Torch die
  AMD-iGPU nicht sieht; relevant erst mit einem NVIDIA-Zielrechner.
