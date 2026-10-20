"""The Golden Bucket: past analyses, and the metric definitions taken from them.

Retrieving similar questions helps the model phrase a query. What actually stops
wrong numbers is the metric registry, which is why that is injected on every
analysis turn while the examples are only the top few.

The reason it matters here, measured on this dataset: revenue including cancelled
and returned orders is $10,814,271; excluding them it is $8,148,817. One unstated
choice moves the figure by 32.7%, and nothing in the schema tells a model which is
meant. The registry decides once, and the report quotes the definition it used.

Embeddings are precomputed into golden_bucket/index.npz by `analyst_assistant
index`, so a question costs one embedding call rather than one per trio. If the
index is missing the retriever falls back to word overlap, which is worse but keeps
the agent working offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:  # pragma: no cover
    from ..llm import Gemini

BUCKET_DIR = Path("golden_bucket")
INDEX_FILE = "index.npz"
METRICS_FILE = "metrics.yaml"

#: Trios below this score are not retrieved. Curation sets the score; see the
#: lifecycle in docs/01-architecture.md section 6.
MIN_QUALITY = 0.6

_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Trio:
    """One past analysis: the question, the SQL, and what the analyst wrote."""

    id: str
    question: str
    sql: str
    report: str
    metrics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    quality: float = 1.0

    def as_example(self) -> str:
        return f"Question: {self.question}\nSQL:\n{self.sql.strip()}"


class MetricRegistry:
    """Canonical definitions, so revenue means one thing across every answer."""

    def __init__(self, definitions: dict[str, dict[str, Any]]) -> None:
        self.definitions = definitions

    @classmethod
    def load(cls, directory: Path = BUCKET_DIR) -> MetricRegistry:
        path = directory / METRICS_FILE
        if not path.exists():
            return cls({})
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(data.get("metrics") or {})

    def prompt_block(self, names: list[str] | None = None) -> str:
        """The definitions to put in front of the model.

        Everything by default. The corpus is small enough that filtering to the
        retrieved trios' metrics would save few tokens and risks omitting the one
        definition the question actually needed.
        """
        wanted = names or list(self.definitions)
        lines = []
        for name in wanted:
            spec = self.definitions.get(name)
            if not spec:
                continue
            lines.append(f"- {name}: {spec['sql']}")
            if spec.get("filter"):
                lines.append(f"    always filtered by: {spec['filter']}")
            if spec.get("note"):
                lines.append(f"    note: {spec['note']}")
        return "\n".join(lines)

    def footnote(self, names: list[str]) -> str:
        """One line the report can quote, so the reader sees the definition used."""
        parts = [
            self.definitions[n]["plain"]
            for n in names
            if n in self.definitions and self.definitions[n].get("plain")
        ]
        return "; ".join(parts)


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class GoldenBucket:
    """Loads the trios and retrieves the ones closest to a question."""

    def __init__(self, directory: str | Path = BUCKET_DIR) -> None:
        self.directory = Path(directory)
        self.registry = MetricRegistry.load(self.directory)
        self.trios = self._load_trios()
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._load_index()

    def _load_trios(self) -> list[Trio]:
        folder = self.directory / "trios"
        if not folder.exists():
            return []
        out = []
        for path in sorted(folder.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue  # a malformed trio should not take the agent down
            meta = data.get("metadata", {})
            out.append(
                Trio(
                    id=data.get("id", path.stem),
                    question=data["question"],
                    sql=data["sql"],
                    report=data.get("report", ""),
                    metrics=meta.get("metrics", []),
                    tags=meta.get("tags", []),
                    quality=float(meta.get("quality_score", 1.0)),
                )
            )
        return out

    def _load_index(self) -> None:
        path = self.directory / INDEX_FILE
        if not path.exists():
            return
        try:
            data = np.load(path, allow_pickle=False)
            self._ids = [str(i) for i in data["ids"]]
            self._vectors = data["vectors"]
        except (OSError, KeyError, ValueError):
            self._ids, self._vectors = [], None

    @property
    def has_index(self) -> bool:
        return self._vectors is not None and len(self._ids) == len(self._vectors)

    def build_index(self, llm: Gemini) -> int:
        """Embed every trio question and save the vectors. Run after editing trios."""
        usable = [t for t in self.trios if t.quality >= MIN_QUALITY]
        if not usable:
            return 0
        vectors = llm.embed([t.question for t in usable])
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.directory / INDEX_FILE,
            ids=np.array([t.id for t in usable]),
            vectors=np.asarray(vectors, dtype=np.float32),
        )
        self._load_index()
        return len(usable)

    def retrieve(self, question: str, *, llm: Gemini | None = None, k: int = 3) -> list[Trio]:
        """The k closest trios.

        Cosine over precomputed vectors when the index is present, word overlap
        when it is not. Brute force is fine: at this corpus size it is exact and
        instant, and a vector database would be infrastructure for its own sake.
        """
        candidates = [t for t in self.trios if t.quality >= MIN_QUALITY]
        if not candidates:
            return []

        if llm is not None and self.has_index:
            try:
                return self._retrieve_dense(question, llm, k, candidates)
            except Exception:
                pass  # embeddings unavailable: fall through to the lexical path
        return self._retrieve_lexical(question, k, candidates)

    def _retrieve_dense(
        self, question: str, llm: Gemini, k: int, candidates: list[Trio]
    ) -> list[Trio]:
        query = np.asarray(llm.embed([question])[0], dtype=np.float32)
        assert self._vectors is not None
        matrix = self._vectors
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
        norms[norms == 0] = 1e-9
        scores = (matrix @ query) / norms

        by_id = {t.id: t for t in candidates}
        ranked = sorted(zip(self._ids, scores, strict=False), key=lambda p: -p[1])
        return [by_id[i] for i, _ in ranked if i in by_id][:k]

    def _retrieve_lexical(self, question: str, k: int, candidates: list[Trio]) -> list[Trio]:
        asked = _tokens(question)
        scored = []
        for trio in candidates:
            words = _tokens(trio.question) | _tokens(" ".join(trio.tags))
            overlap = len(asked & words) / max(1, len(asked | words))
            scored.append((overlap, trio))
        scored.sort(key=lambda p: -p[0])
        return [t for score, t in scored if score > 0][:k]

    def context(self, question: str, *, llm: Gemini | None = None, k: int = 3) -> str:
        """The block injected into the SQL prompt.

        Definitions first and always, examples second and only the nearest few.
        The definitions are the part that keeps answers consistent.
        """
        blocks = []
        definitions = self.registry.prompt_block()
        if definitions:
            blocks.append(
                "Use these metric definitions exactly. They are how the company's "
                "analysts measure these things:\n" + definitions
            )
        examples = self.retrieve(question, llm=llm, k=k)
        if examples:
            blocks.append(
                "Queries analysts wrote for similar questions:\n\n"
                + "\n\n".join(e.as_example() for e in examples)
            )
        return "\n\n".join(blocks)

    def question_catalog(self, limit: int = 8) -> list[str]:
        """Questions already answered, used to suggest what can be asked."""
        return [t.question for t in self.trios[:limit]]
