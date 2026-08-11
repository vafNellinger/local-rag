#!/usr/bin/env python3
"""GPU erkennen und den passenden llama-cpp-python-Wheel-Index ausgeben.

Gibt genau eine Zeile auf stdout aus: die Basis-URL des Wheel-Index, aus dem
llama-cpp-python installiert werden soll. Eine NVIDIA-Karte (erkannt über
``nvidia-smi``) bekommt ein vorkompiliertes CUDA-Wheel passend zur CUDA-Version,
die der Treiber trägt; alles andere fällt auf CPU zurück.

Bewusste Grenzen:

- **Nur Standardbibliothek.** Das Skript läuft *vor* der eigentlichen
  Installation, mit dem nackten Python der frischen Umgebung — es darf nichts
  importieren, was noch nicht da ist.
- **Kein Auto-GPU für AMD/Intel.** Dafür gibt es keine fertigen Wheels; das
  braucht einen eigenen Vulkan-/ROCm-Build und lässt sich nicht zuverlässig
  automatisieren. Diese Karten laufen hier auf CPU.
- **Die vom Treiber gemeldete CUDA-Version ist die Obergrenze.** ``nvidia-smi``
  zeigt die höchste CUDA-Laufzeit, die der Treiber ausführen kann. Es wird das
  höchste verfügbare Wheel gewählt, das darunter passt — so kann es der Treiber
  auch wirklich starten.
"""

from __future__ import annotations

import re
import shutil
import subprocess

BASE = "https://abetlen.github.io/llama-cpp-python/whl"
CPU = f"{BASE}/cpu"

# Verfügbare CUDA-Indizes, absteigend. Gewählt wird der höchste, dessen Version
# die vom Treiber gemeldete nicht überschreitet.
CUDA = [
    ((12, 4), "cu124"),
    ((12, 3), "cu123"),
    ((12, 2), "cu122"),
    ((12, 1), "cu121"),
]


def _cuda_version() -> tuple[int, int] | None:
    """Vom NVIDIA-Treiber getragene CUDA-Version, oder None ohne NVIDIA."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15
        )
    except Exception:
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", proc.stdout)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def wheel_index() -> str:
    version = _cuda_version()
    if version is None:
        return CPU
    for min_version, tag in CUDA:
        if version >= min_version:
            return f"{BASE}/{tag}"
    # Treiber älter als das kleinste CUDA-Wheel — kein passender Build, CPU.
    return CPU


if __name__ == "__main__":
    print(wheel_index())
