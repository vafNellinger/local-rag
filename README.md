# local-rag

Lokales RAG-System, dessen Modellauswahl von der Zielplattform abhängt.

Stand: **Schritt 3 von 5** — die Kette ist vollständig: Extraktion, Chunking,
Embedding, Index, Vektorsuche, Reranking, Antwortgenerierung. Bedienbar über
CLI und grafische Oberfläche. Offen sind Messungen (Schritt 4) und der
Upstream-Beitrag an whichllm (Schritt 5).

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
uv venv && uv pip install -e ".[ingest,generate,gui]"
pipx install whichllm     # falls noch nicht vorhanden
```

`llama-cpp-python` hat keine Wheels und kompiliert beim Installieren; ein
C++-Compiler muss da sein, cmake zieht es sich selbst. Der so entstehende Build
kann **nur CPU** — für GPU-Betrieb neu bauen, etwa mit
`CMAKE_ARGS="-DGGML_VULKAN=ON" uv pip install --force-reinstall --no-cache-dir llama-cpp-python`.
Ob der vorhandene Build auslagern kann, sagt `rag plan` beim Generator und die
Modelle-Seite der Oberfläche; wo es nicht geht, steht dort `cpu` samt Grund
statt einer GPU-Zusage, die nicht eingelöst wird.

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

rag pull                      # GGUF des geplanten Generators holen
rag ask "Wie lang ist die Kündigungsfrist und was gilt bei Verzug?"
rag ask "Löschfristen?" -k 3 --no-rerank

rag gui                       # Oberfläche auf http://127.0.0.1:8080
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

## Reranking

Die Vektorsuche vergleicht zwei Vektoren, die unabhängig voneinander entstanden
sind — Anfrage und Chunk haben sich beim Embedden nie gesehen. Ein Cross-Encoder
liest beide zusammen und kann deshalb beurteilen, was ein Bi-Encoder nur
schätzen kann: ob dieser Chunk *diese* Frage beantwortet. Der Preis ist
Rechenzeit pro Paar statt einmal pro Chunk. Deshalb die Arbeitsteilung: die
Vektorsuche holt breit (30 Kandidaten), der Reranker ordnet und schneidet auf
das, was in den Prompt passt (5).

Abschaltbar, und das ist keine Bequemlichkeit: auf CPU verdoppelt Reranking die
Query-Latenz. `platforms.toml` schaltet es in der Klasse `cpu_only` aus.

**Die Relevanzschwelle ist ein echter Kompromiss.** Bei einer Frage zur
Kündigungsfrist bekamen die zwei treffenden Chunks 0,28 und 0,06, drei
sachfremde 0,005 bis 0,000 — und landeten trotzdem im Prompt, weil `top_k` sie
auffüllte. `min_rerank_score` wirft sie heraus. Umgekehrt gemessen: bei 0,01
fiel eine Stelle heraus, die die Antwort enthalten hätte, und das Modell sagte
korrekt „steht nicht in den Quellen" — die Information war im Index, nur nicht
im Prompt. Die Vorgabe ist deshalb 0 (nicht filtern), und die Oberfläche zeigt
die Punktzahlen an, damit der Wert an eigenen Dokumenten gewählt werden kann.

## Antwort

Der Prompt ist der eigentliche Inhalt von `rag/generate.py`, nicht das Laden.
Drei Festlegungen:

- **Quellen sind numeriert, und die Nummer gehört in den Text.** Ohne
  Zitierpflicht mischt das Modell Kontextwissen mit Vorwissen, und niemand kann
  hinterher trennen, was woher kam.
- **Nichtwissen ist eine erlaubte Antwort.** Ein RAG-System, das bei fehlendem
  Kontext rät, ist schlimmer als eines das schweigt — der plausiblen Erfindung
  glaubt man. Am Testkorpus verifiziert: auf eine Frage, deren Antwort nicht in
  den mitgegebenen Quellen stand, kam „ist in den bereitgestellten Quellen
  nicht angegeben" statt einer Erfindung.
- **Deutsch, ausdrücklich.** Qwen3 antwortet sonst gern englisch auf eine
  deutsche Frage, sobald der Kontext englische Fachbegriffe enthält.

Die Frage steht *nach* den Quellen: bei langem Kontext gewichten Modelle das
Ende stärker, und die Frage soll nicht in der Mitte der Quellen verschwinden.
Vom Kontextfenster gehen nur 60 % an Quellen — ein überlaufendes Fenster kostet
die Antwort, nicht bloß eine Quelle.

## Oberfläche

`rag gui` startet einen lokalen Server (NiceGUI, nur auf localhost) mit vier
Seiten: Fragen, Dokumente, Modelle, Einstellungen.

Der ganze Aufwand dort dreht sich um ein Problem: **alles Interessante dauert
lange.** Ein Ingest läuft Minuten, eine Antwort entsteht mit zwei bis acht
Token pro Sekunde, der Modell-Scan braucht bis zu drei Minuten, ein
GGUF-Download mehrere Gigabyte. Nichts davon darf den Browser blockieren, und
alles muss Fortschritt zeigen — sonst ist es von einem Absturz nicht zu
unterscheiden. Jede blockierende Arbeit läuft deshalb über `asyncio.to_thread`,
der Ingest meldet Datei und Phase, die Antwort erscheint Token für Token.

Modelle werden erst geladen, wenn sie gebraucht werden: eine Suche ohne Antwort
lädt keinen Generator, eine Antwort ohne Reranking keinen Cross-Encoder. In der
Oberfläche ist das der Unterschied zwischen einem benutzbaren Programm und
dreißig Sekunden Startbildschirm.

Einstellungen liegen in `~/.config/local-rag/settings.json` — die einzige Datei
des Projekts, die echte Nutzereingabe enthält und nicht neu erzeugbar ist. Beim
Laden legt sie sich über die Plattformvorgaben, statt sie zu ersetzen: ein neues
Feld wirkt mit seinem Vorgabewert, statt an einer alten Datei zu scheitern.

Ein Nutzer, ein Prozess, ein Modell im Speicher. Der Index ist gegen
Mehrfachzugriff abgesichert (`check_same_thread=False` plus ein Lock im Store),
weil der Event-Loop ihn für die Kennzahlen öffnet und ein Arbeitsthread darin
sucht und schreibt — ein Fehler, der in der CLI nie auftritt, weil dort alles
ein Thread ist.

## Tests

```bash
uv pip install -e ".[dev]" && pytest -q
```

Getestet wird die Entscheidungslogik: ohne whichllm-Aufruf, ohne Netz und ohne
Modell-Download. Der Embedder wird in den Ingest-Tests durch einen Stub
ersetzt, der aus dem Text deterministische Vektoren ableitet — geprüft wird die
Steuerung (Dateiauswahl, Idempotenz, Fehlerbehandlung), nicht die
Retrieval-Qualität.

## Gemessen

Alles auf dieser Maschine (24 logische Kerne, CPU; Torch sieht die AMD-iGPU
nicht, llama.cpp ist CPU-only gebaut).

| Stufe | Wert |
|---|---|
| Extraktion ohne OCR | ~0,8 s pro Seite |
| Extraktion mit OCR | ~8,5 s pro Seite |
| Embedding | 1,2 Chunks/s bei ~500 Token |
| Query-Embedding | 79 ms |
| Antwort, Ende zu Ende | 19–23 s bei 2,4–2,8 Token/s |
| Generator laden (4B Q5_K_M) | ~10 s |
| Ingest, nichts geändert | 1,1 s für drei Dateien, kein Modell geladen |

**Threads nicht hochstellen.** Alle 24 logischen Kerne an llama.cpp zu geben war
siebenmal langsamer als die Vorgabe: 1,2 gegen 8,3 Token/s. Ursache ist
SMT-Oversubscription. Die Einstellung steht deshalb auf 0 („llama.cpp
entscheiden lassen") und die Oberfläche warnt daneben.

Die Antwortlatenz ist damit der teuerste Teil der Kette und der einzige, der
den Anwender direkt warten lässt. Auf einem Rechner mit sichtbarer GPU liegt
hier der Hebel — whichllm schätzt für dieses Modell 17,6 Token/s auf der GPU
gegen die gemessenen 2,4 auf CPU.

## Nächste Schritte

4. Messen an einem echten Korpus statt an drei Testdokumenten: Retrieval-Güte,
   sinnvolle Relevanzschwelle, Ingest-Durchsatz über hunderte Seiten
5. whichllm erweitern: Rollen, MTEB-Adapter, Budget-Allokation

Offen:

- **GPU-Betrieb.** Weder Torch noch llama.cpp sehen hier eine GPU: Torch ist
  der CUDA-Build (die Radeon 890M bleibt unsichtbar, dafür bräuchte es ROCm),
  llama-cpp-python wurde ohne Beschleuniger-Flags kompiliert. Beide Stellen
  melden das jetzt statt still auf CPU zu laufen, aber gelöst ist es nicht.
- **Hybrid-Retrieval.** bge-m3 liefert neben dem Dense-Vektor auch
  Sparse-Gewichte; genutzt wird bisher nur der Dense-Teil. Bei Aktenzeichen und
  Eigennamen wäre die lexikalische Komponente der größere Hebel — es war das
  ausschlaggebende Argument für dieses Modell.

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
