"""Ein exportiertes ONNX-Modell dynamisch nach INT8 quantisieren.

Funktioniert für beide Modelle aus ``tools/export_onnx.py`` — den Embedder
(bge-m3) und den Reranker (bge-reranker-v2-m3). Optionaler Bauschritt: INT8 ist
die schnellste CPU-Variante und deutlich kleiner — aber verlustbehaftet. Am
Reranker lohnt es besonders (er treibt die Query-Latenz): dort ~2,9× schneller
bei erhaltenem Ranking. Am Embedder gemessen am Testkorpus (dynamisch,
avx512_vnni):

- Tempo: ~64 ms/Chunk auf CPU (fp32-ONNX ~156, Torch ~74) — INT8 ist auf CPU die
  schnellste Option.
- Größe: ~568 MB statt ~2,2 GB (≈ 4× kleiner) — relevant fürs Bundle.
- Qualität: Cosine-Treue zu sentence-transformers fällt auf ~0,978 (statt 1,0),
  Retrieval-Recall@5 sinkt von 98,3 % auf ~96,7 % (−1,6 Punkte).

Deshalb kein Standard, sondern eine bewusste Abwägung: klein und schnell gegen
einen kleinen Retrieval-Verlust. Vor dem Produktiveinsatz mit den eigenen
Dokumenten gegenprüfen (``rag eval``), am besten mit einem INT8-gebauten Index.

Aufruf::

    python tools/quantize_onnx.py [onnx-verzeichnis]

Erwartet ein mit ``export_onnx.py`` erzeugtes Modellverzeichnis (Standard
``~/.cache/local-rag/onnx/bge-m3``) und schreibt ``bge-m3-int8`` daneben.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DEFAULT_SRC = Path.home() / ".cache" / "local-rag" / "onnx" / "bge-m3"


def main() -> int:
    src = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_SRC
    if not (src / "model.onnx").exists():
        print(f"Kein ONNX-Modell unter {src} — erst tools/export_onnx.py", flush=True)
        return 1
    dst = src.with_name(src.name + "-int8")
    dst.mkdir(parents=True, exist_ok=True)

    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    print(f"Quantisiere {src.name} dynamisch nach INT8 …", flush=True)
    quantizer = ORTQuantizer.from_pretrained(src)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=dst, quantization_config=qconfig)

    # Tokenizer und Konfiguration mitnehmen, damit das Verzeichnis eigenständig
    # ladbar ist.
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    ):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    print(f"  → {dst} (model_quantized.onnx)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
