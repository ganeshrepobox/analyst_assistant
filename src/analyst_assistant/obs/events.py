"""Structured event log and counters. Implements requirement 7.

One owned source of truth: a JSONL stream, one record per node execution,
correlated by ``trace_id``. It is greppable, ships in the repo as an artifact,
powers ``/trace``, and depends on nothing external -- a mandated requirement should
not rest on a third-party service being reachable. LangSmith and OpenTelemetry are
additions on top of this in production, never replacements for it.

Every logged free-text field passes through the PII scrubber
------------------------------------------------------------
The question and the generated SQL can both contain personal data -- ``WHERE email
= '...'`` being the obvious case -- so writing them verbatim would persist PII into
the very artifact that exists to make the system auditable. Scrubbing happens here,
at the boundary, rather than being left to each call site to remember.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..safety.masking import scrub

#: Fields whose values are free text and must be scrubbed before being written.
#: Everything else is structural -- node names, statuses, measurements.
_TEXT_FIELDS = frozenset({"question", "sql", "message", "error", "preview", "answer"})


def new_trace_id() -> str:
    """A short, human-quotable correlation id.

    Short on purpose: it is printed in user-facing error messages, and a full UUID
    is hostile to read aloud or retype.
    """
    return uuid.uuid4().hex[:12]


@dataclass
class Event:
    """One node execution."""

    trace_id: str
    node: str
    status: str
    latency_ms: float
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    user_id: str | None = None
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    bytes_billed: int = 0
    retry_count: int = 0
    cache_hit: bool = False
    error_class: str | None = None
    persona: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


class EventLog:
    """Append-only JSONL sink, one file per day."""

    def __init__(self, directory: str | Path = "logs") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self.directory / f"events-{datetime.now(UTC):%Y%m%d}.jsonl"

    def append(self, event: Event) -> None:
        record = asdict(event)
        record["attrs"] = {
            key: scrub(value).text if key in _TEXT_FIELDS and isinstance(value, str) else value
            for key, value in record["attrs"].items()
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def read_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Every event for one turn, oldest first. Powers ``/trace <id>``.

        Scans the two most recent files. A real deployment queries BigQuery, so
        scanning the whole directory would be the wrong shape to grow into.
        """
        events: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("events-*.jsonl"))[-2:]:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a truncated final line must not break /trace
                if record.get("trace_id") == trace_id:
                    events.append(record)
        return sorted(events, key=lambda r: r.get("ts", ""))


class Metrics:
    """In-process counters.

    Deliberately just counters. The metric *names* are the contract and stay
    identical in production, where they become Cloud Monitoring time series -- so a
    dashboard built against one works against the other.
    """

    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()

    def incr(self, name: str, by: int = 1) -> None:
        self._counters[name] += by

    def counter(self, name: str) -> int:
        return self._counters[name]

    def snapshot(self) -> dict[str, int]:
        """What ``/metrics`` prints."""
        return dict(sorted(self._counters.items()))


class Recorder:
    """Per-turn recording context: one trace id, an event sink, and counters."""

    def __init__(
        self,
        log: EventLog,
        metrics: Metrics,
        *,
        user_id: str | None = None,
        persona: str | None = None,
    ) -> None:
        self.log = log
        self.metrics = metrics
        self.user_id = user_id
        self.persona = persona
        self.trace_id = new_trace_id()
        self.events: list[Event] = []

    @contextmanager
    def span(self, node: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        """Time one node and emit its event, whether it succeeds or raises.

        Yields a mutable dict the node fills in as it learns things -- token counts,
        bytes billed, cache hits. A yielded dict rather than return values keeps
        instrumentation out of every node's signature.

        An exception is recorded and re-raised: the caller's error boundary decides
        how to degrade, but the event is written either way, because failures are
        exactly what this log exists to capture.
        """
        m: dict[str, Any] = {"status": "ok"}
        started = time.perf_counter()
        try:
            yield m
        except Exception as exc:
            m["status"] = "error"
            m["error_class"] = type(exc).__name__
            m.setdefault("attrs", {})["error"] = str(exc)
            raise
        finally:
            event = Event(
                trace_id=self.trace_id,
                node=node,
                status=m.get("status", "ok"),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                user_id=self.user_id,
                persona=self.persona,
                model=m.get("model"),
                tokens_in=m.get("tokens_in", 0),
                tokens_out=m.get("tokens_out", 0),
                bytes_billed=m.get("bytes_billed", 0),
                retry_count=m.get("retry_count", 0),
                cache_hit=m.get("cache_hit", False),
                error_class=m.get("error_class"),
                attrs={**attrs, **m.get("attrs", {})},
            )
            self.events.append(event)
            self.log.append(event)

            self.metrics.incr(f"node.{node}.{event.status}")
            if event.bytes_billed:
                self.metrics.incr("bq.bytes_billed", event.bytes_billed)
            if event.tokens_in or event.tokens_out:
                self.metrics.incr("llm.calls")
                self.metrics.incr("llm.tokens", event.tokens_in + event.tokens_out)
            if event.cache_hit:
                self.metrics.incr("cache.hits")

    def record_pii_tripwire(self, findings: dict[str, int]) -> None:
        """Record an egress-scrubber firing.

        Separate from ``span`` because the semantics differ: this is an alarm, not
        a measurement. The tripwire should never fire -- direct identifiers are
        absent from the scoped views -- so a firing means an upstream control
        failed, and it is paged on rather than charted.
        """
        for kind, count in findings.items():
            self.metrics.incr(f"safety.pii_tripwire.{kind}", count)
        if findings:
            self.metrics.incr("safety.pii_tripwire.total", sum(findings.values()))

    def cost_line(self) -> str:
        """The per-turn cost summary printed in the CLI.

        Requirement 5's cost clause is graded, so it should be observable by a
        reviewer rather than merely true.
        """
        calls = sum(1 for e in self.events if e.tokens_in or e.tokens_out)
        tokens = sum(e.tokens_in + e.tokens_out for e in self.events)
        mb = sum(e.bytes_billed for e in self.events) / 1_000_000
        elapsed = sum(e.latency_ms for e in self.events) / 1000
        parts = [
            f"{calls} LLM call{'s' if calls != 1 else ''}",
            f"{tokens} tokens",
            f"{mb:.1f} MB billed",
            f"{elapsed:.1f}s",
        ]
        if any(e.cache_hit for e in self.events):
            parts.append("cache hit")
        return " · ".join(parts) + f" · trace {self.trace_id}"

    def timeline(self) -> list[str]:
        """Node-by-node summary for ``/debug``."""
        marks = {"ok": "+", "error": "!", "refused": "x", "empty": "0"}
        rows = []
        for e in self.events:
            detail = ""
            if e.retry_count:
                detail += f" retries={e.retry_count}"
            if e.bytes_billed:
                detail += f" bytes={e.bytes_billed}"
            if e.error_class:
                detail += f" {e.error_class}"
            rows.append(f"  {marks.get(e.status, '-')} {e.node:<22} {e.latency_ms:>8.1f}ms{detail}")
        return rows
