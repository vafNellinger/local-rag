"""Lokale Hardware-Erkennung über den whichllm-Direktimport.

Deckt den Pfad ab, der das gebündelte Programm ohne externes whichllm-CLI
lauffähig macht (siehe rag/detect.py, ``_detect_hardware_via_import``).
"""

import pytest

from rag import detect

_GIB = 1024**3


def test_auto_vram_headroom_matches_whichllm():
    # 8 GB → 5 % = 400 MB, unter der 512-MB-Untergrenze → 512 MB Reserve.
    assert detect._auto_vram_headroom(8 * _GIB) == 512 * 1024**2
    # Kein/negatives VRAM → keine Reserve.
    assert detect._auto_vram_headroom(0) == 0
    # Sehr groß → 2-GB-Deckel.
    assert detect._auto_vram_headroom(100 * _GIB) == 2 * _GIB


def test_direct_import_fills_usable_vram(monkeypatch):
    """detect_hardware() lässt usable_vram_bytes leer; wir ziehen Headroom ab."""
    pytest.importorskip("whichllm")
    from whichllm.hardware.types import GPUInfo, HardwareInfo

    fake = HardwareInfo(
        cpu_cores=8,
        ram_bytes=16 * _GIB,
        os="linux",
        gpus=[
            GPUInfo(
                name="Test", vendor="amd", vram_bytes=8 * _GIB, shared_memory=True
            )
        ],
    )
    monkeypatch.setattr(
        "whichllm.hardware.detector.detect_hardware", lambda: fake
    )
    hw = detect._detect_hardware_via_import()
    assert hw is not None
    gpu = hw["gpus"][0]
    # None → mit Headroom-Abzug gefüllt, nicht 0.
    assert gpu["usable_vram_bytes"] == 8 * _GIB - detect._auto_vram_headroom(8 * _GIB)


def test_detect_local_prefers_direct_import(monkeypatch):
    """detect_local() nutzt den Direktimport, nicht den CLI-Subprozess."""
    pytest.importorskip("whichllm")

    def boom(*args, **kwargs):
        raise AssertionError("run_whichllm (CLI) darf hier nicht laufen")

    monkeypatch.setattr(detect, "run_whichllm", boom)
    platform = detect.detect_local()
    assert platform.platform_class
    assert platform.os_name in {"linux", "darwin", "windows"}


def test_probe_falls_back_to_cli_without_import(monkeypatch):
    """Fehlt der Direktimport, greift der CLI-Weg (Fallback/Simulation)."""
    monkeypatch.setattr(detect, "_detect_hardware_via_import", lambda: None)
    fake_json = {
        "hardware": {
            "gpus": [],
            "ram_bytes": 8 * _GIB,
            "cpu_cores": 4,
            "os": "linux",
        }
    }
    monkeypatch.setattr(detect, "run_whichllm", lambda *a, **k: fake_json)
    platform = detect.detect_local()
    assert platform.platform_class == "cpu_only"
