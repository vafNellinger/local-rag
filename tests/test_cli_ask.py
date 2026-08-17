"""Regressionstest für den ``rag ask``-Kommandopfad.

Hintergrund: ``ask_stream`` liefert seit dem Mehrturn-Umbau drei Werte
(Quellen, Token-Strom, wiederverwendet), der CLI-Befehl entpackte aber zwei —
``rag ask`` stürzte mit ``ValueError`` ab, bevor überhaupt etwas generiert
wurde. Der Pfad hatte keinen Test, deshalb blieb es unbemerkt. Dieser Test
sichert die Verdrahtung ab, ohne echte Modelle zu laden.
"""

from __future__ import annotations

from typer.testing import CliRunner

from rag import cli


class _FakePipeline:
    """Ersetzt RagPipeline: liefert Treffer und einen dreiwertigen Stream."""

    def __init__(self, settings):
        self.settings = settings

    def retrieve(self, question, **kwargs):
        return [object()]  # nicht leer, damit der Befehl bis ask_stream läuft

    def ask_stream(self, question, **kwargs):
        # Genau die Signatur, an der es krachte: drei Rückgabewerte.
        return [], iter(["Sechs ", "Monate."]), False

    def close(self):
        pass


def test_ask_entpackt_ask_stream_ohne_absturz(monkeypatch, tmp_path):
    index = tmp_path / "index.db"
    index.write_text("x")

    # Plattform-/Modellwahl umgehen (kein whichllm im Test) und die Pipeline
    # durch das Double ersetzen.
    monkeypatch.setattr(cli.Settings, "for_platform", staticmethod(cli.Settings))
    monkeypatch.setattr(cli, "RagPipeline", _FakePipeline)

    result = CliRunner().invoke(
        cli.app, ["ask", "Wie lang ist die Frist?", "--index", str(index)]
    )

    assert result.exit_code == 0, result.output
    assert "Sechs Monate." in result.output
