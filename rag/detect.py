"""Hardware-Erkennung und Plattformklassifizierung.

whichllm liefert die erkannte Hardware und das Modell-Ranking. Wir nutzen es
als Hardware-Orakel, weil seine GPU-Erkennung (inkl. Bandbreite, geteiltem
APU-Speicher und Headroom-Reservierung) deutlich gründlicher ist als was
psutil hergibt. whichllm ist eine echte Abhängigkeit (nicht mehr extern per
pipx): die lokale Hardware-Erkennung läuft über den Direktimport von
``whichllm.hardware.detector`` — das funktioniert auch im gebündelten Programm
ohne CLI im PATH und ohne Netzwerk (siehe ``_detect_hardware_via_import``).

Der ``whichllm --json``-Subprozess bleibt Ausweichweg und der Weg für
simulierte Zielplattformen (``--gpu`` / ``--vram`` / ``--cpu-only``), die nur
das CLI beherrscht.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from rag.paths import CACHE_DIR

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "platforms.toml"

# whichllm braucht pro Lauf bis zu ~3 Minuten (HF-Abfragen, Benchmark-Merge).
# Für eine Hardware- und Ranking-Abfrage, die sich täglich kaum ändert, ist ein
# Cache Pflicht — sonst ist jeder CLI-Aufruf unbenutzbar langsam.
CACHE_TTL_SECONDS = 24 * 3600

WHICHLLM_TIMEOUT_SECONDS = 300

_GIB = 1024**3


class WhichllmError(RuntimeError):
    """whichllm ist nicht installiert, bricht ab oder liefert kein JSON."""


@dataclass(frozen=True)
class Platform:
    """Eine Zielplattform — lokal erkannt oder simuliert."""

    label: str
    platform_class: str
    gpu_name: str | None
    gpu_count: int
    usable_vram_bytes: int
    shared_memory: bool
    ram_bytes: int
    cpu_cores: int
    os_name: str
    # Argumente, mit denen whichllm für diese Plattform aufgerufen wird.
    # Bei der lokalen Plattform leer, bei simulierten z.B. ["--gpu", "RTX 4090"].
    whichllm_args: tuple[str, ...] = ()

    @property
    def has_gpu(self) -> bool:
        return self.usable_vram_bytes > 0

    @property
    def usable_vram_gb(self) -> float:
        return self.usable_vram_bytes / _GIB

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / _GIB

    def describe(self) -> str:
        if not self.has_gpu:
            return f"{self.label}: CPU only, {self.cpu_cores} Kerne, {self.ram_gb:.1f} GB RAM"
        shared = " (shared)" if self.shared_memory else ""
        multi = f" x{self.gpu_count}" if self.gpu_count > 1 else ""
        return (
            f"{self.label}: {self.gpu_name}{multi}, "
            f"{self.usable_vram_gb:.1f} GB VRAM{shared}, {self.ram_gb:.1f} GB RAM"
        )


def load_config(path: Path | None = None) -> dict:
    """Lade platforms.toml."""
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Konfiguration fehlt: {config_path}")
    with config_path.open("rb") as fh:
        return tomllib.load(fh)


def _cache_path(args: list[str]) -> Path:
    key = hashlib.sha256(" ".join(args).encode()).hexdigest()[:16]
    return CACHE_DIR / f"whichllm-{key}.json"


def run_whichllm(
    args: list[str],
    *,
    use_cache: bool = True,
    timeout: int = WHICHLLM_TIMEOUT_SECONDS,
) -> dict:
    """Rufe ``whichllm --json`` auf und gib das geparste Ergebnis zurück.

    Der CLI-Subprozess ist der stabile Weg für das Modell-Ranking und für
    simulierte Plattformen: die ``--json``-Ausgabe ändert sich zwischen
    0.5.x-Versionen kaum. Die lokale Hardware-Erkennung dagegen läuft über den
    Direktimport (siehe ``_detect_hardware_via_import``), damit sie auch im
    Bundle ohne CLI und ohne Netzwerk funktioniert.
    """
    full_args = ["whichllm", "--json", *args]
    cache_file = _cache_path(full_args)

    if use_cache and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            logger.debug("whichllm-Cache-Treffer (%.0fs alt): %s", age, cache_file.name)
            try:
                return json.loads(cache_file.read_text())
            except json.JSONDecodeError:
                logger.warning("Cache-Datei defekt, wird neu geholt: %s", cache_file)

    logger.info("whichllm wird aufgerufen: %s", " ".join(full_args))
    try:
        proc = subprocess.run(
            full_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise WhichllmError(
            "whichllm-CLI nicht im PATH gefunden (wird fürs Modell-Ranking "
            "gebraucht; die Hardware-Erkennung läuft ohne CLI)."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise WhichllmError(f"whichllm-Timeout nach {timeout}s") from exc

    # whichllm beendet sich bei Argumentfehlern mit Code 0 und schreibt die
    # Fehlermeldung nach stdout (siehe `whichllm hardware --json`), deshalb
    # reicht die Prüfung des Exit-Codes allein nicht — wir müssen das Parsen
    # als eigentlichen Erfolgstest behandeln.
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        detail = (proc.stdout or proc.stderr or "").strip()[:400]
        raise WhichllmError(
            f"whichllm lieferte kein JSON (Exit {proc.returncode}): {detail}"
        ) from exc

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def classify(
    usable_vram_bytes: int,
    *,
    shared_memory: bool,
) -> str:
    """Bestimme die Plattformklasse — Schlüssel in platforms.toml."""
    vram_gb = usable_vram_bytes / _GIB
    if vram_gb < 1.0:
        return "cpu_only"
    if shared_memory or vram_gb <= 10.0:
        # Geteilter Speicher ist der entscheidende Faktor, nicht die reine
        # Zahl: bei einer APU konkurriert jedes GPU-resident geladene Modell
        # direkt mit dem System-RAM-Bedarf des restlichen Systems.
        return "igpu_shared"
    if vram_gb <= 20.0:
        return "dgpu_small"
    return "dgpu_large"


def _platform_from_hardware(
    hardware: dict,
    *,
    label: str,
    whichllm_args: tuple[str, ...],
) -> Platform:
    gpus = hardware.get("gpus") or []

    # Bei Multi-GPU summieren wir das nutzbare VRAM. Das ist für die
    # Klassifizierung brauchbar, aber nicht für die Modellplatzierung: ob ein
    # Modell über zwei Karten gesplittet werden kann, entscheidet whichllm
    # selbst über multi_gpu_effective_vram_bytes.
    usable = sum(int(g.get("usable_vram_bytes") or 0) for g in gpus)
    shared = any(bool(g.get("shared_memory")) for g in gpus)

    return Platform(
        label=label,
        platform_class=classify(usable, shared_memory=shared),
        gpu_name=gpus[0].get("name") if gpus else None,
        gpu_count=len(gpus),
        usable_vram_bytes=usable,
        shared_memory=shared,
        ram_bytes=int(hardware.get("ram_bytes") or 0),
        cpu_cores=int(hardware.get("cpu_cores") or 0),
        os_name=str(hardware.get("os") or "unknown"),
        whichllm_args=whichllm_args,
    )


def _auto_vram_headroom(vram_bytes: int) -> int:
    """VRAM-Reserve wie whichllm sie ansetzt (nachgebildet für den Direktpfad).

    whichllms CLI zieht vom rohen VRAM einen Puffer ab und schreibt das
    Ergebnis nach ``usable_vram_bytes``. Der direkte API-Aufruf
    (``detect_hardware``) lässt das Feld dagegen leer — ohne diesen Abzug sähe
    ``classify`` die GPU mit 0 nutzbarem VRAM. Formel identisch zu
    ``whichllm.cli._auto_vram_headroom``.
    """
    if vram_bytes <= 0:
        return 0
    return int(max(512 * 1024**2, min(vram_bytes * 0.05, 2 * _GIB)))


def _detect_hardware_via_import() -> dict | None:
    """Hardware-Block direkt über die mitgelieferte whichllm-Python-API.

    Im gebündelten Programm gibt es kein externes ``whichllm``-CLI im PATH; die
    Erkennung läuft deshalb über den Direktimport von ``detect_hardware``. Das
    umgeht zugleich die Modell-Abfrage des CLI (HF-Netzwerk), die für die reine
    Hardware-Erkennung unnötig ist und offline scheitern würde. Gibt ``None``
    zurück, wenn whichllm nicht importierbar ist oder die Erkennung fehlschlägt
    — dann greift der CLI-Subprozess als Ausweichweg.
    """
    try:
        from dataclasses import asdict

        from whichllm.hardware.detector import detect_hardware
    except Exception as exc:  # whichllm nicht importierbar
        logger.debug("whichllm-Direktimport nicht verfügbar: %s", exc)
        return None
    try:
        hardware = asdict(detect_hardware())
    except Exception as exc:  # Erkennung selbst gescheitert
        logger.warning("whichllm-Hardware-Erkennung fehlgeschlagen: %s", exc)
        return None
    # usable_vram_bytes setzt sonst erst der CLI (Headroom); hier nachziehen,
    # damit classify() die GPU nicht mit 0 nutzbarem VRAM sieht.
    for gpu in hardware.get("gpus", []):
        if gpu.get("usable_vram_bytes") is None:
            vram = int(gpu.get("vram_bytes") or 0)
            gpu["usable_vram_bytes"] = max(0, vram - _auto_vram_headroom(vram))
    return hardware


def _probe(args: list[str], *, label: str, use_cache: bool) -> Platform:
    """Hole nur den Hardware-Block.

    Bevorzugt den Direktimport der whichllm-API (funktioniert auch im Bundle
    ohne externes CLI und ohne Netzwerk); fällt auf den ``whichllm --json``-
    Subprozess zurück, wenn das Paket nicht importierbar ist. ``-n 1`` hält den
    CLI-Lauf klein.
    """
    hardware = None
    # Nur die echte lokale Hardware kommt aus dem Direktimport; Simulations-
    # argumente (--gpu/--vram/--cpu-only) beherrscht ausschließlich das CLI.
    if not args:
        hardware = _detect_hardware_via_import()
    if hardware is None:
        data = run_whichllm([*args, "-n", "1"], use_cache=use_cache)
        hardware = data.get("hardware")
    if not hardware:
        raise WhichllmError("whichllm-JSON enthält keinen 'hardware'-Block")
    return _platform_from_hardware(
        hardware, label=label, whichllm_args=tuple(args)
    )


def detect_local(*, use_cache: bool = True) -> Platform:
    """Erkenne die Plattform, auf der wir gerade laufen."""
    return _probe([], label="local", use_cache=use_cache)


def simulate(
    *,
    gpu: str | None = None,
    vram_gb: float | None = None,
    cpu_only: bool = False,
    label: str | None = None,
    use_cache: bool = True,
) -> Platform:
    """Beschreibe eine Zielplattform, auf der wir nicht sitzen.

    Nutzt whichllms eingebaute Simulation. Damit lässt sich die Modellauswahl
    für einen anderen Rechner bestimmen, ohne dort etwas zu installieren --
    der eigentliche Zweck der plattformabhängigen Auflösung.
    """
    args: list[str] = []
    if cpu_only:
        args.append("--cpu-only")
    if gpu:
        args.extend(["--gpu", gpu])
    if vram_gb is not None:
        args.extend(["--vram", str(vram_gb)])

    if not args:
        raise ValueError(
            "simulate() braucht mindestens gpu, vram_gb oder cpu_only — "
            "für die lokale Maschine ist detect_local() zuständig"
        )

    auto_label = label or gpu or ("cpu-only" if cpu_only else f"vram-{vram_gb}gb")
    return _probe(args, label=auto_label, use_cache=use_cache)
