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

Die ersten drei sind Konfiguration in `platforms.toml`, der vierte ist Code.
Alle vier sind Kandidaten für einen späteren Upstream-Beitrag.

## Plattformklassen

| Klasse | Bedingung | Generator | Embedder | Reranker |
|---|---|---|---|---|
| `cpu_only` | kein nutzbares VRAM | CPU | CPU | aus |
| `igpu_shared` | geteilter Speicher oder ≤ 10 GB | GPU | CPU | CPU |
| `dgpu_small` | 10–20 GB dediziert | GPU | GPU | CPU |
| `dgpu_large` | > 20 GB dediziert | GPU | GPU | GPU |

Geteilter Speicher schlägt die VRAM-Zahl: bei einer APU konkurriert jedes
GPU-resident geladene Modell direkt mit dem System-RAM.

## Tests

```bash
uv pip install -e ".[dev]" && pytest -q
```

Getestet wird die Entscheidungslogik ohne whichllm-Aufruf.

## Nächste Schritte

2. Ingest: Datei → Text → Chunks → bge-m3 → sqlite-vec
3. Query: Retrieval → Rerank → Prompt → llama-cpp-python
4. Messen: Latenz pro Stufe, Ingest-Durchsatz
5. whichllm erweitern: Rollen, MTEB-Adapter, Budget-Allokation
