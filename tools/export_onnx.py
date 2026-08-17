"""bge-m3 und den Reranker nach ONNX exportieren.

Einmaliges Bauwerkzeug, kein Laufzeit-Code: ``optimum`` lädt das HF-Modell und
schreibt ein ONNX-Graph-Modell samt Tokenizer. Die Anwendung selbst lädt das
Ergebnis später allein über ``onnxruntime`` + ``tokenizers`` — ohne optimum und
ohne den transformers-Downgrade, den optimum erzwingt (optimum pinnt
transformers < 5). Deshalb gehört dieses Skript in eine getrennte, idealerweise
isolierte Umgebung und nicht in die Laufzeit-Abhängigkeiten.

Aufruf::

    python tools/export_onnx.py [zielverzeichnis]

Standardziel ist ``~/.cache/local-rag/onnx``. bge-m3 wird als
Feature-Extraction exportiert (liefert ``last_hidden_state``); das CLS-Pooling
und die L2-Normalisierung macht die Engine, denn so ist bge-m3 konfiguriert
(``1_Pooling/config.json``: ``pooling_mode_cls_token=true``, danach
``2_Normalize``). Der Reranker ist ein Sequence-Classification-Modell und
liefert einen Relevanz-Logit pro (Frage, Passage)-Paar.
"""

from __future__ import annotations

import sys
from pathlib import Path

EMBEDDER = "BAAI/bge-m3"
RERANKER = "BAAI/bge-reranker-v2-m3"
DEFAULT_ZIEL = Path.home() / ".cache" / "local-rag" / "onnx"


def main() -> int:
    ziel = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_ZIEL
    ziel.mkdir(parents=True, exist_ok=True)

    from optimum.onnxruntime import (
        ORTModelForFeatureExtraction,
        ORTModelForSequenceClassification,
    )
    from transformers import AutoTokenizer

    print(f"Exportiere {EMBEDDER} (Feature-Extraction) …", flush=True)
    emb_dir = ziel / "bge-m3"
    ORTModelForFeatureExtraction.from_pretrained(EMBEDDER, export=True).save_pretrained(
        emb_dir
    )
    AutoTokenizer.from_pretrained(EMBEDDER).save_pretrained(emb_dir)
    print(f"  → {emb_dir}", flush=True)

    print(f"Exportiere {RERANKER} (Sequence-Classification) …", flush=True)
    rr_dir = ziel / "bge-reranker-v2-m3"
    ORTModelForSequenceClassification.from_pretrained(
        RERANKER, export=True
    ).save_pretrained(rr_dir)
    AutoTokenizer.from_pretrained(RERANKER).save_pretrained(rr_dir)
    print(f"  → {rr_dir}", flush=True)

    print("Fertig.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
