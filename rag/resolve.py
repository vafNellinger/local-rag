"""Rollenauflösung: (Rolle, Plattform) -> konkretes Modell.

Das ist die Naht des Systems. Ingest und Query kennen nur Rollen
("embedder", "generator"), niemals Modellnamen. Wechselt die Plattform,
wechselt hier die Antwort — nicht im Pipeline-Code.

Der Generator wird dynamisch über whichllm aufgelöst, Embedder und Reranker
über Tabellen in platforms.toml, weil whichllm sie strukturell nicht kennt
(models/hf.py filtert hart auf ``pipeline_tag=text-generation``). Sobald das
Tool um Rollen erweitert ist, wird ``_resolve_static`` durch einen
whichllm-Aufruf ersetzt und der Rest des Systems merkt nichts davon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rag.detect import Platform, load_config, run_whichllm

logger = logging.getLogger(__name__)

_GIB = 1024**3
_MIB = 1024**2

# Qualitätsverlust je Quantisierung, als Bruchteil. Übernommen aus whichllms
# data/quantization.py (QUANT_QUALITY_PENALTY), weil das Tool in einem eigenen
# pipx-venv steckt und nicht importierbar ist. Bewusst nur die Formate, die
# in der Praxis auftauchen — Unbekanntes wird konservativ abgelehnt.
#
# Achtung bei Updates von whichllm: diese Werte können dort nachjustiert
# werden. Sie steuern hier nur die Untergrenze, nicht das Ranking selbst.
QUANT_QUALITY_PENALTY: dict[str, float] = {
    "F32": 0.0,
    "F16": 0.0,
    "BF16": 0.0,
    "Q8_0": 0.01,
    "Q6_K": 0.02,
    "Q5_K_M": 0.03,
    "Q5_K_S": 0.035,
    "Q5_0": 0.035,
    "Q4_K_M": 0.05,
    "NVFP4": 0.05,
    "IQ4_XS": 0.05,
    "Q4_K_S": 0.055,
    "IQ4_NL": 0.055,
    "Q4_0": 0.06,
    "MXFP4": 0.06,
    "Q3_K_L": 0.075,
    "Q3_K_M": 0.08,
    "Q3_K_S": 0.12,
    "IQ3_XS": 0.16,
    "IQ3_M": 0.16,
    "IQ3_S": 0.17,
    "IQ3_XXS": 0.18,
    "Q2_K": 0.25,
}

# Wie viele Kandidaten wir von whichllm holen, bevor wir nachfiltern. Muss
# deutlich über 1 liegen: die RAG-Nachfilterung (kein Thinking-Modell, harte
# Quant-Untergrenze, natives Kontextfenster) verwirft regelmäßig die vorderen
# Plätze.
CANDIDATE_POOL = 30


class ResolutionError(RuntimeError):
    """Für eine Rolle konnte auf dieser Plattform kein Modell bestimmt werden."""


@dataclass(frozen=True)
class ModelSpec:
    """Ein aufgelöstes Modell für eine Rolle auf einer Plattform."""

    role: str
    model_id: str
    device: str  # "gpu" | "cpu"
    source: str  # "whichllm" | "config"

    # Nur für GGUF-Generatoren belegt.
    artifact_repo_id: str | None = None
    artifact_filename: str | None = None
    quant_type: str | None = None
    context_length: int | None = None
    vram_required_bytes: int | None = None
    estimated_tok_per_sec: float | None = None
    quality_score: float | None = None

    # Nur für Embedder belegt.
    dimensions: int | None = None
    max_seq_length: int | None = None

    notes: tuple[str, ...] = ()

    def describe(self) -> str:
        parts = [f"{self.role}: {self.model_id}"]
        if self.quant_type:
            parts.append(self.quant_type)
        parts.append(f"auf {self.device.upper()}")
        if self.vram_required_bytes:
            parts.append(f"{self.vram_required_bytes / _GIB:.1f} GB")
        if self.estimated_tok_per_sec:
            parts.append(f"~{self.estimated_tok_per_sec:.0f} tok/s")
        return " | ".join(parts)


@dataclass
class PipelinePlan:
    """Vollständige Modellzuweisung für eine Plattform, inkl. VRAM-Aufteilung."""

    platform: Platform
    platform_class: str
    generator: ModelSpec
    embedder: ModelSpec
    reranker: ModelSpec | None
    gpu_reserved_bytes: int
    generator_budget_bytes: int
    warnings: list[str] = field(default_factory=list)

    @property
    def specs(self) -> list[ModelSpec]:
        out = [self.generator, self.embedder]
        if self.reranker:
            out.append(self.reranker)
        return out

    @property
    def total_gpu_bytes(self) -> int:
        """Summe des VRAM-Bedarfs aller GPU-resident geplanten Rollen."""
        total = 0
        for spec in self.specs:
            if spec.device != "gpu":
                continue
            if spec.vram_required_bytes:
                total += spec.vram_required_bytes
        return total


def parse_context_length(value: str | int) -> int:
    """'32k' -> 32768, '4096' -> 4096."""
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.endswith("k"):
        return int(float(text[:-1]) * 1024)
    return int(text)


def _quant_acceptable(quant_type: str | None, min_tier: str) -> bool:
    """Prüfe, ob eine Quantisierung die konfigurierte Untergrenze hält.

    Die Untergrenze wird als Format angegeben ("Q4_K_S") und über die
    Penalty-Tabelle in einen Schwellwert übersetzt — so muss die Konfiguration
    keine Zahlen kennen, und die Ordnung bleibt die von whichllm.
    """
    if not quant_type:
        return False
    threshold = QUANT_QUALITY_PENALTY.get(min_tier.upper())
    if threshold is None:
        raise ResolutionError(
            f"min_quant_tier '{min_tier}' ist unbekannt. "
            f"Erlaubt: {', '.join(sorted(QUANT_QUALITY_PENALTY))}"
        )
    penalty = QUANT_QUALITY_PENALTY.get(quant_type.upper())
    if penalty is None:
        logger.debug("Unbekannte Quantisierung '%s' wird abgelehnt", quant_type)
        return False
    return penalty <= threshold


def _context_truncation_warning(candidate: dict) -> str | None:
    """Finde whichllms Warnung über ein zu kleines natives Kontextfenster.

    whichllm berechnet ``context_fits`` in engine/compatibility.py, verwendet
    es aber nicht im Ranking — es landet nur als Warntext. Ein Modell mit 8k
    nativem Fenster kann deshalb bei ``-c 32k`` auf Platz 1 stehen. Für RAG ist
    das Kontextbudget die zentrale Größe, also filtern wir hier selbst.
    """
    for warning in candidate.get("warnings") or []:
        if "max context" in warning.lower():
            return warning
    return None


def _generator_candidates(
    platform: Platform,
    gen_config: dict,
    context_length: int,
    budget_bytes: int | None,
    *,
    use_cache: bool,
) -> list[dict]:
    """Hole die whichllm-Rangliste für die Generator-Rolle."""
    args = [
        *platform.whichllm_args,
        "--profile",
        str(gen_config.get("profile", "general")),
        "-c",
        str(context_length),
        "-n",
        str(CANDIDATE_POOL),
    ]
    if evidence := gen_config.get("evidence"):
        args.extend(["--evidence", str(evidence)])
    if speed := gen_config.get("speed"):
        args.extend(["--speed", str(speed)])

    # Das VRAM-Budget des Generators ist nicht das der Karte, sondern was nach
    # den GPU-resident geplanten Nebenrollen übrig bleibt. whichllms
    # --vram-Override ist genau der Hebel, um dieses geteilte Budget
    # abzubilden — das Tool selbst kennt nur "ein Modell, ganze Karte".
    if budget_bytes is not None:
        args.extend(["--vram", f"{budget_bytes / _GIB:.2f}"])

    data = run_whichllm(args, use_cache=use_cache)
    models = data.get("models") or []
    if not models:
        raise ResolutionError(
            f"whichllm fand keine lauffähigen Modelle für {platform.label} "
            f"bei {context_length} Token Kontext"
        )
    return models


def _resolve_generator(
    platform: Platform,
    config: dict,
    class_config: dict,
    budget_bytes: int | None,
    *,
    use_cache: bool,
) -> ModelSpec:
    # Klassenspezifische Overrides gewinnen über die globalen Kriterien. Nötig,
    # weil dieselben Schwellwerte nicht für alle Plattformen taugen: auf
    # CPU-only sind Speed-Filter und Parameter-Untergrenze zusammen
    # unerfüllbar.
    gen_config = {**config.get("generator", {}), **class_config.get("generator", {})}
    context_length = parse_context_length(class_config.get("context_length", 4096))
    device = str(class_config.get("generator_device", "gpu"))

    candidates = _generator_candidates(
        platform, gen_config, context_length, budget_bytes, use_cache=use_cache
    )

    exclude = [p.lower() for p in gen_config.get("exclude_name_patterns", [])]
    min_params_b = float(gen_config.get("min_params_b", 0.0))
    min_quant = str(gen_config.get("min_quant_tier", "Q4_K_S"))

    rejected: list[str] = []
    fallback: tuple[dict, str] | None = None

    for candidate in candidates:
        model_id = str(candidate.get("model_id", ""))
        lower = model_id.lower()

        if not candidate.get("can_run"):
            rejected.append(f"{model_id}: läuft nicht auf dieser Plattform")
            continue

        if hit := next((p for p in exclude if p in lower), None):
            rejected.append(f"{model_id}: Namensmuster '{hit}' ausgeschlossen")
            continue

        params_b = float(candidate.get("parameter_count") or 0) / 1e9
        if params_b < min_params_b:
            rejected.append(f"{model_id}: {params_b:.1f}B < {min_params_b}B Minimum")
            continue

        quant = candidate.get("quant_type")
        if not _quant_acceptable(quant, min_quant):
            rejected.append(f"{model_id}: Quantisierung {quant} unter {min_quant}")
            continue

        if warning := _context_truncation_warning(candidate):
            rejected.append(f"{model_id}: {warning}")
            continue

        fit_type = str(candidate.get("fit_type", ""))
        if device == "gpu" and fit_type != "full_gpu":
            # Nicht sofort verwerfen: wenn nichts vollständig auf die GPU
            # passt, ist ein teilweiser Offload immer noch besser als ein
            # Abbruch. Wir merken uns den ersten und nehmen ihn nur, falls
            # kein sauberer Treffer folgt.
            if fallback is None:
                fallback = (candidate, f"nur {fit_type}, kein full_gpu")
            rejected.append(f"{model_id}: {fit_type} statt full_gpu")
            continue

        return _spec_from_candidate(candidate, device, context_length)

    if fallback is not None:
        candidate, reason = fallback
        logger.warning("Generator nur als Fallback auflösbar: %s", reason)
        return _spec_from_candidate(
            candidate, device, context_length, notes=(f"Fallback: {reason}",)
        )

    detail = "\n  ".join(rejected[:10]) or "keine Kandidaten"
    raise ResolutionError(
        f"Kein Generator erfüllt die RAG-Kriterien auf {platform.label} "
        f"bei {context_length} Token.\nVerworfen:\n  {detail}"
    )


def _spec_from_candidate(
    candidate: dict,
    device: str,
    context_length: int,
    notes: tuple[str, ...] = (),
) -> ModelSpec:
    return ModelSpec(
        role="generator",
        model_id=str(candidate.get("model_id", "")),
        device=device,
        source="whichllm",
        artifact_repo_id=candidate.get("artifact_repo_id"),
        artifact_filename=candidate.get("artifact_filename"),
        quant_type=candidate.get("quant_type"),
        context_length=context_length,
        vram_required_bytes=candidate.get("vram_required_bytes"),
        estimated_tok_per_sec=candidate.get("estimated_tok_per_sec"),
        quality_score=candidate.get("quality_score"),
        notes=notes,
    )


def _resolve_static(role: str, config: dict, device: str) -> ModelSpec:
    """Löse Embedder oder Reranker über die Tabelle in platforms.toml auf."""
    section = config.get(role, {})
    entry = section.get("default")
    if not entry:
        raise ResolutionError(f"platforms.toml hat keinen Eintrag [{role}.default]")

    vram_bytes = None
    if device == "gpu" and (mb := entry.get("vram_estimate_mb")):
        vram_bytes = int(mb) * _MIB

    return ModelSpec(
        role=role,
        model_id=str(entry["model_id"]),
        device=device,
        source="config",
        vram_required_bytes=vram_bytes,
        dimensions=entry.get("dimensions"),
        max_seq_length=entry.get("max_seq_length"),
        notes=("aus platforms.toml; whichllm kennt diese Rolle nicht",),
    )


def resolve_pipeline(
    platform: Platform,
    *,
    config: dict | None = None,
    use_cache: bool = True,
) -> PipelinePlan:
    """Bestimme alle Modelle für eine Plattform.

    Die Reihenfolge ist wichtig: erst werden die Nebenrollen platziert, dann
    das verbleibende VRAM-Budget berechnet, dann der Generator aufgelöst. Nur
    so bekommt whichllm das Budget zu sehen, das dem Generator tatsächlich
    bleibt — sonst plant es mit der ganzen Karte und die Pipeline passt in
    Summe nicht in den Speicher.
    """
    cfg = config or load_config()
    classes = cfg.get("platform_class", {})
    class_config = classes.get(platform.platform_class)
    if not class_config:
        raise ResolutionError(
            f"platforms.toml kennt die Klasse '{platform.platform_class}' nicht. "
            f"Vorhanden: {', '.join(sorted(classes))}"
        )

    warnings: list[str] = []

    embedder = _resolve_static(
        "embedder", cfg, str(class_config.get("embedder_device", "cpu"))
    )

    reranker: ModelSpec | None = None
    if class_config.get("reranker_enabled", True):
        reranker = _resolve_static(
            "reranker", cfg, str(class_config.get("reranker_device", "cpu"))
        )
    else:
        warnings.append(
            f"Reranker in Klasse '{platform.platform_class}' deaktiviert — "
            "Retrieval-Präzision sinkt, Query-Latenz bleibt niedriger"
        )

    reserved = sum(
        spec.vram_required_bytes or 0
        for spec in (embedder, reranker)
        if spec is not None and spec.device == "gpu"
    )

    budget: int | None = None
    if platform.has_gpu and class_config.get("generator_device") == "gpu":
        budget = platform.usable_vram_bytes - reserved
        if budget <= 0:
            raise ResolutionError(
                f"Nebenrollen belegen {reserved / _GIB:.1f} GB von "
                f"{platform.usable_vram_gb:.1f} GB — für den Generator bleibt nichts. "
                f"Embedder/Reranker in platforms.toml auf 'cpu' setzen."
            )
        if reserved:
            warnings.append(
                f"Generator-Budget auf {budget / _GIB:.1f} GB begrenzt "
                f"({reserved / _GIB:.1f} GB für Nebenrollen reserviert)"
            )

    generator = _resolve_generator(
        platform, cfg, class_config, budget, use_cache=use_cache
    )
    warnings.extend(generator.notes)

    # Ohne konkretes GGUF-Artefakt lässt sich das Modell nicht laden: whichllm
    # rechnet für offizielle Repos ohne GGUF-Upload mit synthetischen
    # Schätzwerten und nennt dann keine Datei. Muss vor Schritt 2 auffallen,
    # nicht erst beim Download.
    if not generator.artifact_repo_id or not generator.artifact_filename:
        warnings.append(
            f"whichllm nennt für {generator.model_id} keine GGUF-Datei "
            "(synthetische Schätzung für ein offizielles Repo) — "
            "Quantisierung muss manuell gewählt werden"
        )

    return PipelinePlan(
        platform=platform,
        platform_class=platform.platform_class,
        generator=generator,
        embedder=embedder,
        reranker=reranker,
        gpu_reserved_bytes=reserved,
        generator_budget_bytes=budget or 0,
        warnings=warnings,
    )
