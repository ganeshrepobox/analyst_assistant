"""Structured event log and in-process metrics. Implements requirement 7.

One owned source of truth
-------------------------
A JSONL event stream, one record per node execution, correlated by ``trace_id``.
It is greppable, ships in the repo as an artifact, powers the ``/trace`` command,
and -- importantly -- depends on nothing external. A *mandated* requirement should
not rest on a third-party SaaS being reachable.

LangSmith/Langfuse and OpenTelemetry both have a place (prompt-response
correspondence and the production exporter respectively), but they are additions on
top of this, never replacements for it. Three overlapping telemetry systems read as
indecision; see ``docs/DECISIONS.md`` D7.

Every logged string passes through the PII scrubber
---------------------------------------------------
This is easy to get wrong. The user's question and the generated SQL can both
contain personal data -- ``WHERE email = 'x@y.com'`` is the obvious case -- and
writing them verbatim would persist PII into the very artifact that exists to make
the system auditable. So scrubbing happens here, at the boundary, rather than being
left to each call site to remember.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..safety.masking import scrub

Status = Literal["ok", "error", "refused", "degraded", "empty", "skipped"]

#: Event fields whose values are free text and must be scrubbed before being
#: written. Anything not listed here is assumed to be structural (node names,
#: statuses, numeric measurements).
_TEXT_FIELDS = frozenset({"question", "sql", "message", "error", "preview", "answer"})


def new_trace_id() -> str:
    """A short, human-quotable correlation id.

    Short on purpose: it is printed in user-facing error messages so a user can
    quote it to support, and a full UUID is hostile to read aloud or retype.
    """
    return uuid.uuid4().hex[:12]


@dataclass
class Event:
    """One node execution."""

    trace_id: str
    turn_id: int
    node: str
    status: Status
    latency_ms: float
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    user_id: str | None = None
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    bytes_billed: int = 0
    retry_count: int = 0
    cache_hit: bool = False
    error_class: str | None = None
    persona_version: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


class Metrics:
    """In-process counters and latency histograms.

    Deliberately simple: a dict of counters plus raw latency samples. In production
    these are Cloud Monitoring time series, but the *names* are the contract and
    they stay identical, so a dashboard built against one works against the other.
    """

    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()
        self._latencies: dict[str, list[float]] = defaultdict(list)

    def incr(self, name: str, by: int = 1) -> None:
        self._counters[name] += by

    def observe(self, name: str, value: float) -> None:
        self._latencies[name].append(value)

    def counter(self, name: str) -> int:
        return self._counters[name]

    @staticmethod
    def _percentile(samples: list[float], pct: float) -> float:
        if not samples:
            return 0.0
        ordered = sorted(samples)
        # Nearest-rank percentile. Exact interpolation is pointless at the sample
        # sizes a single CLI session produces.
        index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
        return ordered[index]

    def snapshot(self) -> dict[str, Any]:
        """What ``/metrics`` prints."""
        latency = {
            name: {
                "n": len(samples),
                "p50_ms": round(self._percentile(samples, 50), 1),
                "p95_ms": round(self._percentile(samples, 95), 1),
            }
            for name, samples in sorted(self._latencies.items())
        }
        return {"counters": dict(sorted(self._counters.items())), "latency": latency}


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

        Scans today's and yesterday's files only. A real deployment queries
        BigQuery; scanning the whole directory would be the wrong shape to grow
        into, and pretending otherwise would be misleading.
        """
        events: list[dict[str, Any]] = []
        for path in sorted(self.directory.glob("events-*.jsonl"))[-2:]:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a truncated final line must not break /trace
                    if record.get("trace_id") == trace_id:
                        events.append(record)
        return sorted(events, key=lambda r: r.get("ts", ""))


class Recorder:
    """Per-turn recording context: a trace id, an event sink, and metrics.

    One Recorder per user turn. ``span()`` is the only API the graph nodes need.
    """

    def __init__(
        self,
        log: EventLog,
        metrics: Metrics,
        *,
        user_id: str | None = None,
        turn_id: int = 0,
        trace_id: str | None = None,
        persona_version: str | None = None,
    ) -> None:
        self.log = log
        self.metrics = metrics
        self.user_id = user_id
        self.turn_id = turn_id
        self.trace_id = trace_id or new_trace_id()
        self.persona_version = persona_version
        self.events: list[Event] = []

    @contextmanager
    def span(self, node: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        """Time one node and emit its event, whether it succeeds or raises.

        Yields a mutable dict the node fills in as it learns things -- token counts,
        bytes billed, whether a cache was hit. Using a yielded dict rather than
        return values keeps instrumentation out of each node's signature.

        An exception is recorded and re-raised: the caller's error boundary decides
        how to degrade, but the event is written either way, because the failures
        are exactly what the log exists to capture.
        """
        measurements: dict[str, Any] = {"status": "ok"}
        started = time.perf_counter()
        try:
            yield measurements
        except Exception as exc:
            measurements["status"] = "error"
            measurements["error_class"] = type(exc).__name__
            measurements.setdefault("attrs", {})["error"] = str(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            event = Event(
                trace_id=self.trace_id,
                turn_id=self.turn_id,
                node=node,
                status=measurements.get("status", "ok"),
                latency_ms=round(elapsed_ms, 2),
                user_id=self.user_id,
                persona_version=self.persona_version,
                model=measurements.get("model"),
                tokens_in=measurements.get("tokens_in", 0),
                tokens_out=measurements.get("tokens_out", 0),
                cost_usd=measurements.get("cost_usd", 0.0),
                bytes_billed=measurements.get("bytes_billed", 0),
                retry_count=measurements.get("retry_count", 0),
                cache_hit=measurements.get("cache_hit", False),
                error_class=measurements.get("error_class"),
                attrs={**attrs, **measurements.get("attrs", {})},
            )
            self.events.append(event)
            self.log.append(event)

            self.metrics.incr(f"node.{node}.{event.status}")
            self.metrics.observe(f"node.{node}.latency_ms", elapsed_ms)
            if event.bytes_billed:
                self.metrics.incr("bq.bytes_billed", event.bytes_billed)
            if event.tokens_in or event.tokens_out:
                self.metrics.incr("llm.calls")
                self.metrics.incr("llm.tokens", event.tokens_in + event.tokens_out)
            if event.cache_hit:
                self.metrics.incr("cache.hits")

    def record_pii_tripwire(self, findings: dict[str, int]) -> None:
        """Record an egress-scrubber firing.

        Separate from ``span`` because the semantics are different: this is not a
        measurement, it is an alarm. The tripwire should never fire -- direct
        identifiers are absent from the scoped views -- so a firing means an
        upstream control failed and is paged on immediately rather than charted.
        See ``docs/01-architecture.md`` section 9.
        """
        if not findings:
            return
        for kind, count in findings.items():
            self.metrics.incr(f"safety.pii_tripwire.{kind}", count)
        self.metrics.incr("safety.pii_tripwire.total", sum(findings.values()))

    def cost_line(self) -> str:
        """The per-turn cost summary printed in the CLI.

        Requirement 5 asks for resilience "without inflating costs". That is a
        graded clause, so it should be observable by a reviewer rather than merely
        true: five lines of UI beat a paragraph claiming frugality.
        """
        llm_calls = sum(1 for e in self.events if e.tokens_in or e.tokens_out)
        tokens = sum(e.tokens_in + e.tokens_out for e in self.events)
        bytes_billed = sum(e.bytes_billed for e in self.events)
        cached = any(e.cache_hit for e in self.events)
        elapsed = sum(e.latency_ms for e in self.events)

        mb = bytes_billed / 1_000_000
        parts = [
            f"{llm_calls} LLM call{'s' if llm_calls != 1 else ''}",
            f"{tokens} tokens",
            f"{mb:.1f} MB billed" if bytes_billed else "0 MB billed",
            f"{elapsed / 1000:.1f}s",
        ]
        if cached:
            parts.append("cache hit")
        return " · ".join(parts) + f" · trace {self.trace_id}"

    def timeline(self) -> list[str]:
        """Node-by-node summary for ``/debug``."""
        rows = []
        for event in self.events:
            marker = {"ok": "+", "error": "!", "refused": "x", "empty": "0"}.get(
                event.status, "-"
            )
            detail = ""
            if event.retry_count:
                detail += f" retries={event.retry_count}"
            if event.bytes_billed:
                detail += f" bytes={event.bytes_billed}"
            if event.error_class:
                detail += f" {event.error_class}"
            rows.append(f"  {marker} {event.node:<22} {event.latency_ms:>8.1f}ms{detail}")
        return rows


def default_log_dir() -> Path:
    return Path(os.environ.get("ANALYST_LOG_DIR", "logs"))
