# Strang 2 — GPU-LLM (Vulkan / Metal)

**Ziel:** Die Antwortgenerierung (llama.cpp) auf der GPU ausführen —
herstellerübergreifend — für spürbar schnellere Antworten, in einem einzigen
Bundle pro OS ohne CUDA/ROCm-Matrix.

## Ausgangslage

- `rag/generate.py` (`Generator`) lädt das Modell über `llama_cpp.Llama` und
  setzt bereits **`n_gpu_layers`** (`self.gpu_layers`) sowie `n_ctx`. Die
  GPU-Offload-Fähigkeit ist im Code also schon vorgesehen.
- `platforms.toml` legt pro Plattformklasse `generator_device` fest (`gpu` bei
  `igpu_shared`/`dgpu_*`).
- Die README kennt bereits einen **Vulkan-Build** von llama-cpp-python
  (`CMAKE_ARGS="-DGGML_VULKAN=ON"`), heute nur als manueller Bauweg.
- Der bestehende Bootstrap wählt CPU- vs. CUDA-Wheels über
  `packaging/detect_gpu.py`.

## Ansatz

Statt einer CPU/CUDA-Wheel-Matrix **herstellerübergreifende GPU-Backends**
bündeln:

- **Windows/Linux:** llama-cpp-python im **Vulkan**-Build. Vulkan läuft auf
  NVIDIA + AMD + Intel mit demselben Binary. GGML fällt automatisch auf CPU
  zurück, wenn keine Vulkan-GPU vorhanden ist.
- **macOS:** **Metal**-Build (Standard bei llama.cpp auf Apple Silicon).

So deckt ein Binary pro OS alle GPUs ab und passt ins statische Bundle
(Strang 4).

## Schritte

1. **Vulkan-/Metal-Wheels beschaffen.** Prüfen, ob vorkompilierte
   Vulkan-Wheels verfügbar sind; falls nicht, in der CI (Strang 4) selbst bauen
   (`-DGGML_VULKAN=ON` bzw. Metal-Default) und als Artefakt ablegen.
2. **`n_gpu_layers`-Heuristik.** Automatisch nach erkanntem VRAM/Modellgröße
   setzen (so viele Layer wie möglich, Rest CPU). VRAM-Erkennung
   herstellerübergreifend (Vulkan-Device-Info / Metal), nicht nur `nvidia-smi`.
3. **CPU-Fallback absichern.** Startet das GPU-Backend nicht (keine
   Vulkan-Runtime, kein Treiber), sauber auf CPU zurückfallen und das im
   Protokoll sichtbar machen (nicht still).
4. **Laufzeit-Prüfung** wie im heutigen Launcher
   (`llama_supports_gpu_offload()`), um zu bestätigen, dass Offload aktiv ist.
5. **Modellwahl bleibt** unverändert (GGUF via `gguf_path`); nur das Backend
   wechselt.

## Risiken / offene Punkte

- **Vulkan-Wheel-Verfügbarkeit:** ggf. Eigenbau in der CI nötig (Shader-Toolkit
  `glslc`/`spirv-headers`, siehe README). Erhöht den CI-Aufwand.
- **VRAM-Erkennung cross-platform** ist uneinheitlich — konservativ schätzen und
  bei Fehlschlag lieber weniger Layer offloaden als abstürzen.
- **iGPU mit geteiltem Speicher** (`igpu_shared`): Offload kann langsamer sein
  als CPU — Heuristik muss das berücksichtigen (ggf. nur Teil-Offload).

## Verifikation (Erfolgskriterium)

1. **Tempo:** Tokens/Sekunde CPU vs. GPU auf derselben Frage; deutlicher
   Zuwachs auf einer dGPU.
2. **Robustheit:** auf einem Gerät ohne GPU startet die App weiterhin (CPU-
   Fallback greift, Protokoll weist es aus).
3. **Korrektheit:** identische Antwortqualität wie im CPU-Betrieb (gleicher
   GGUF, nur anderes Backend).
