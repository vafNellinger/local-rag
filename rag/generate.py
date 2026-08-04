"""Kontext plus Frage → Antwort, über llama.cpp.

Das Modell kommt aus ``resolve_pipeline()`` — whichllm nennt Repo und
GGUF-Datei, die hier geladen wird. Der Code kennt weiterhin keinen
Modellnamen.

Der eigentliche Inhalt dieses Moduls ist der Prompt, nicht das Laden. Drei
Festlegungen:

**Quellen sind numeriert und die Nummer steht im Text.** Ohne
Zitierpflicht mischt das Modell Kontextwissen mit Vorwissen, und niemand kann
hinterher trennen, was woher kam. Mit Nummern ist jede Aussage prüfbar.

**Nichtwissen ist eine erlaubte Antwort.** Ein RAG-System, das bei fehlendem
Kontext zu raten anfängt, ist schlimmer als eines, das schweigt — der
plausible Erfindung glaubt man.

**Deutsch, weil die Dokumente deutsch sind.** Qwen3 antwortet sonst gern
englisch auf eine deutsche Frage, wenn der Kontext englische Fachbegriffe
enthält.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rag.store import SearchHit

logger = logging.getLogger(__name__)

# Anteil des Kontextfensters, der für Quellen verwendet wird. Der Rest bleibt
# für Systemprompt, Frage und Antwort. 60 % ist reichlich konservativ, aber ein
# überlaufendes Fenster kostet die Antwort, nicht nur eine Quelle.
CONTEXT_BUDGET_SHARE = 0.6

# Wie viele Token die Antwort höchstens lang wird.
DEFAULT_MAX_TOKENS = 800

# Niedrig, aber nicht null: bei RAG steht die Antwort im Kontext, Kreativität
# ist hier ein Fehler und keine Eigenschaft. Ganz auf 0 neigen manche Modelle
# zu Wiederholungsschleifen.
DEFAULT_TEMPERATURE = 0.2

# Zeichen pro Token für die Kontextabschätzung. Wie in chunk.py konservativ
# gewählt, damit eher zu wenige Quellen mitgehen als das Fenster überläuft.
CHARS_PER_TOKEN = 3.2

SYSTEM_PROMPT = """Du bist ein präziser Assistent für Dokumentenrecherche.

Regeln:
1. Antworte ausschließlich auf Grundlage der bereitgestellten Quellen.
2. Belege jede Aussage mit der Quellennummer in eckigen Klammern, z.B. [2].
3. Steht die Antwort nicht in den Quellen, sage genau das. Rate nicht und
   ergänze nichts aus eigenem Wissen.
4. Widersprechen sich Quellen, benenne den Widerspruch statt ihn aufzulösen.
5. Antworte auf Deutsch, sachlich und ohne Einleitungsfloskeln."""

NO_CONTEXT_ANSWER = (
    "Zu dieser Frage liegen keine passenden Stellen im Index. "
    "Entweder sind die betreffenden Dokumente nicht aufgenommen, "
    "oder die Frage trifft ihren Wortlaut nicht."
)


class GenerationError(RuntimeError):
    """Das Generator-Modell ist nicht benutzbar."""


def supports_gpu_offload() -> bool:
    """Kann dieser llama.cpp-Build überhaupt auf die GPU auslagern?

    Ein pip-Build ohne CUDA-, ROCm- oder Vulkan-Flags kann es nicht, und die
    Bibliothek sagt das nur, wenn man fragt: ``n_gpu_layers`` wird sonst
    stillschweigend ignoriert und alles läuft auf der CPU. Da die
    Plattformklasse in ``platforms.toml`` "gpu" verlangen kann, muss die
    Abweichung sichtbar werden.
    """
    try:
        import llama_cpp.llama_cpp as backend
    except ImportError:  # pragma: no cover
        return False
    checker = getattr(backend, "llama_supports_gpu_offload", None)
    return bool(checker()) if checker else False


def resolve_gpu_layers(requested: int) -> int:
    """Gewünschte GPU-Layer gegen die Fähigkeiten des Builds prüfen."""
    if requested and not supports_gpu_offload():
        logger.warning(
            "%s GPU-Layer gewünscht, aber dieser llama.cpp-Build kann nicht "
            "auslagern (nur CPU-Backend kompiliert) — Generierung läuft auf "
            "der CPU. Für GPU-Betrieb llama-cpp-python mit CMAKE_ARGS "
            "neu bauen, z.B. -DGGML_VULKAN=ON.",
            "Alle" if requested < 0 else requested,
        )
        return 0
    return requested


def gguf_path(
    repo_id: str, filename: str, *, download: bool = False
) -> Path | None:
    """Lokalen Pfad zur GGUF-Datei bestimmen.

    Ohne ``download`` wird nur der HF-Cache befragt und ``None``
    zurückgegeben, wenn die Datei fehlt. Das trennt "ist da" von "hole es" —
    die Oberfläche muss anzeigen können, ob ein Modell bereitsteht, ohne
    mehrere Gigabyte anzustoßen.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise GenerationError("huggingface-hub fehlt") from exc

    try:
        return Path(
            hf_hub_download(repo_id, filename, local_files_only=not download)
        )
    except Exception as exc:
        if download:
            raise GenerationError(
                f"Download von {repo_id}/{filename} fehlgeschlagen: {exc}"
            ) from exc
        logger.debug("%s/%s nicht im Cache: %s", repo_id, filename, exc)
        return None


@dataclass
class Source:
    """Eine numerierte Quelle im Prompt."""

    number: int
    hit: SearchHit

    @property
    def citation(self) -> str:
        return self.hit.citation

    def render(self) -> str:
        """Wie die Quelle im Prompt steht."""
        return f"[{self.number}] {self.hit.citation}\n{self.hit.text}"


@dataclass
class Answer:
    """Ergebnis einer Anfrage."""

    question: str
    text: str
    sources: list[Source] = field(default_factory=list)
    # Quellen, die wegen des Kontextbudgets nicht mitgingen. Sichtbar, damit
    # eine unvollständige Antwort erklärbar bleibt.
    dropped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        if not self.duration_seconds or not self.completion_tokens:
            return 0.0
        return self.completion_tokens / self.duration_seconds

    @property
    def cited_numbers(self) -> set[int]:
        """Quellennummern, die in der Antwort tatsächlich vorkommen."""
        import re

        return {
            int(match)
            for match in re.findall(r"\[(\d+)\]", self.text)
            if match.isdigit()
        }

    @property
    def uncited_sources(self) -> list[Source]:
        """Mitgegebene Quellen, die die Antwort nicht zitiert.

        Nützlich zur Beurteilung: viele unzitierte Quellen heißen, dass die
        Vektorsuche breit gestreut hat oder der Reranker daneben lag.
        """
        cited = self.cited_numbers
        return [s for s in self.sources if s.number not in cited]


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def build_sources(
    hits: Sequence[SearchHit], *, context_tokens: int
) -> tuple[list[Source], int]:
    """Treffer zu numerierten Quellen machen, bis das Budget erschöpft ist.

    Gibt (Quellen, Zahl der weggelassenen) zurück. Die Reihenfolge bleibt: was
    der Reranker oben hat, kommt zuerst und wird zuletzt weggelassen.
    """
    budget = int(context_tokens * CONTEXT_BUDGET_SHARE)
    sources: list[Source] = []
    used = 0

    for index, hit in enumerate(hits, start=1):
        candidate = Source(number=len(sources) + 1, hit=hit)
        cost = estimate_tokens(candidate.render())
        if sources and used + cost > budget:
            # Nicht abbrechen wäre falsch: die Treffer sind nach Relevanz
            # geordnet, also ist der erste, der nicht passt, auch der erste,
            # den man am ehesten entbehren kann — und alle danach ebenso.
            dropped = len(hits) - len(sources)
            logger.debug(
                "Kontextbudget erschöpft: %d von %d Quellen mitgegeben",
                len(sources),
                len(hits),
            )
            return sources, dropped
        sources.append(candidate)
        used += cost

    return sources, 0


def build_prompt(question: str, sources: Sequence[Source]) -> list[dict[str, str]]:
    """Chat-Nachrichten für llama.cpp bauen.

    Die Frage steht *nach* den Quellen. Bei langem Kontext gewichten Modelle
    das Ende stärker — die Frage soll das Letzte sein, was das Modell liest,
    nicht in der Mitte der Quellen verschwinden.
    """
    if not sources:
        raise GenerationError("Prompt ohne Quellen — das ist kein RAG")

    quellen = "\n\n".join(source.render() for source in sources)
    user = (
        f"Quellen:\n\n{quellen}\n\n"
        f"---\n\n"
        f"Frage: {question}\n\n"
        f"Antworte auf Deutsch und belege mit Quellennummern."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class Generator:
    """llama.cpp-Modell, geladen beim ersten Gebrauch.

    Der verzögerte Ladevorgang zählt hier mehr als beim Embedder: das GGUF
    wiegt mehrere Gigabyte, und eine reine Suche ohne Antwort soll das nicht
    zahlen.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        context_length: int = 8192,
        gpu_layers: int = 0,
        threads: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.context_length = context_length
        # Roh übernehmen und erst beim Laden auflösen: die Prüfung warnt, und
        # eine Warnung über die Generierung darf nicht schon dann erscheinen,
        # wenn das Objekt angelegt wird — sie tauchte sonst mitten im Ingest
        # auf, wo gar nichts generiert wird.
        self.requested_gpu_layers = gpu_layers
        self.threads = threads
        self.seed = seed
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def close(self) -> None:
        """Modell freigeben.

        Ausdrücklich und nicht dem Destruktor überlassen: ``Llama.__del__``
        läuft beim Interpreter-Ende erst, wenn die Modul-Globals schon
        abgebaut sind, und wirft dann ``TypeError: 'NoneType' object is not
        callable`` aus dem Aufräumcode. Wer vorher schließt, sieht das nicht.
        """
        if self._model is not None:
            try:
                self._model.close()
            except Exception as exc:  # pragma: no cover
                logger.debug("Aufräumen des Modells schlug fehl: %s", exc)
            self._model = None

    def _load(self):
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise GenerationError(
                "llama-cpp-python fehlt: uv pip install -e '.[generate]'"
            ) from exc

        if not self.model_path.exists():
            raise GenerationError(
                f"GGUF-Datei nicht gefunden: {self.model_path} — "
                "'rag pull' lädt das vom Plan genannte Modell herunter"
            )

        self.gpu_layers = resolve_gpu_layers(self.requested_gpu_layers)
        logger.debug(
            "Lade %s (Kontext %d, GPU-Layer %d)",
            self.model_path.name,
            self.context_length,
            self.gpu_layers,
        )
        kwargs: dict = {
            "model_path": str(self.model_path),
            "n_ctx": self.context_length,
            "n_gpu_layers": self.gpu_layers,
            "verbose": False,
        }
        if self.threads:
            kwargs["n_threads"] = self.threads
        if self.seed is not None:
            kwargs["seed"] = self.seed

        try:
            return Llama(**kwargs)
        except Exception as exc:
            raise GenerationError(
                f"Modell konnte nicht geladen werden: {exc}"
            ) from exc

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> tuple[str, int, int]:
        """Antwort in einem Stück. Gibt (Text, Prompt-Token, Antwort-Token)."""
        try:
            result = self.model.create_chat_completion(
                messages=list(messages),
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise GenerationError(f"Generierung fehlgeschlagen: {exc}") from exc

        text = result["choices"][0]["message"]["content"] or ""
        usage = result.get("usage", {})
        return (
            text.strip(),
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Iterator[str]:
        """Antwort Stück für Stück.

        Für die Oberfläche: bei 5 bis 18 Token pro Sekunde sind 800 Token
        knapp eine Minute. Ohne Streaming sieht das aus wie ein Hänger.
        """
        try:
            chunks = self.model.create_chat_completion(
                messages=list(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for chunk in chunks:
                delta = chunk["choices"][0].get("delta", {})
                if piece := delta.get("content"):
                    yield piece
        except Exception as exc:
            raise GenerationError(f"Generierung fehlgeschlagen: {exc}") from exc
