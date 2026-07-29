"""CLI. In Schritt 1 nur die Plattform- und Modellauflösung."""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.table import Table

from rag.detect import Platform, WhichllmError, detect_local, simulate
from rag.resolve import PipelinePlan, ResolutionError, resolve_pipeline

app = typer.Typer(
    add_completion=False,
    help="Lokales RAG mit plattformabhängiger Modellauswahl.",
)
console = Console()

_GIB = 1024**3


@app.callback()
def main() -> None:
    """Ohne diesen Callback macht Typer den einzigen Befehl zum Root-Befehl,
    was ``rag plan`` bricht, sobald ``ingest`` und ``ask`` dazukommen."""


def _render(plan: PipelinePlan) -> None:
    console.print()
    console.print(f"[bold]{plan.platform.describe()}[/]")
    console.print(f"Plattformklasse: [cyan]{plan.platform_class}[/]")
    console.print()

    table = Table(show_lines=True)
    table.add_column("Rolle", style="bold")
    # overflow="fold": Modell-IDs sind lang und dürfen nicht abgeschnitten
    # werden — der abgeschnittene Name ist die eine Information, die man
    # aus dieser Tabelle wirklich braucht.
    table.add_column("Modell", overflow="fold")
    table.add_column("Quant")
    table.add_column("Gerät")
    table.add_column("VRAM", justify="right")
    table.add_column("Speed", justify="right")
    table.add_column("Quelle")

    for spec in plan.specs:
        vram = (
            f"{spec.vram_required_bytes / _GIB:.1f} GB"
            if spec.vram_required_bytes
            else "—"
        )
        speed = (
            f"~{spec.estimated_tok_per_sec:.0f} tok/s"
            if spec.estimated_tok_per_sec
            else "—"
        )
        device_style = "green" if spec.device == "gpu" else "yellow"
        table.add_row(
            spec.role,
            spec.model_id,
            spec.quant_type or "—",
            f"[{device_style}]{spec.device.upper()}[/]",
            vram,
            speed,
            spec.source,
        )

    console.print(table)

    if plan.generator.artifact_repo_id:
        console.print(
            f"  GGUF: [dim]{plan.generator.artifact_repo_id}[/] / "
            f"[dim]{plan.generator.artifact_filename}[/]"
        )
    if plan.generator.context_length:
        console.print(f"  Kontext: [dim]{plan.generator.context_length} Token[/]")

    if plan.platform.has_gpu:
        console.print(
            f"  GPU-Belegung: [dim]{plan.total_gpu_bytes / _GIB:.1f} GB von "
            f"{plan.platform.usable_vram_gb:.1f} GB[/]"
        )

    for warning in plan.warnings:
        console.print(f"  [yellow]![/] {warning}")
    console.print()


@app.command()
def plan(
    gpu: str | None = typer.Option(
        None, "--gpu", help="Zielplattform simulieren, z.B. 'RTX 4090'"
    ),
    vram: float | None = typer.Option(
        None, "--vram", help="Nutzbares VRAM in GB überschreiben"
    ),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Ohne GPU planen"),
    label: str | None = typer.Option(None, "--label", help="Name der Zielplattform"),
    refresh: bool = typer.Option(
        False, "--refresh", help="whichllm-Cache ignorieren und neu abfragen"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Debug-Ausgabe"),
) -> None:
    """Zeige, welche Modelle auf einer Plattform verwendet werden."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    use_cache = not refresh
    simulated = bool(gpu or vram is not None or cpu_only)

    try:
        platform: Platform = (
            simulate(
                gpu=gpu,
                vram_gb=vram,
                cpu_only=cpu_only,
                label=label,
                use_cache=use_cache,
            )
            if simulated
            else detect_local(use_cache=use_cache)
        )
        result = resolve_pipeline(platform, use_cache=use_cache)
    except (WhichllmError, ResolutionError) as exc:
        console.print(f"[red]Fehler:[/] {exc}")
        raise typer.Exit(1) from exc

    _render(result)


if __name__ == "__main__":
    app()
