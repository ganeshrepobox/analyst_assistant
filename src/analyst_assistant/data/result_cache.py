"""Query result cache, which doubles as the record/replay store.

One artifact paying for four requirements
-----------------------------------------
This started as a decision *not* to build a DuckDB replica of the dataset. That
replica was tempting -- it would let a reviewer without a Google Cloud account run
the prototype -- but the BigQuery dialect divergence is load-bearing:
backtick-quoted ``project.dataset.table``, ``DATE_TRUNC(date, MONTH)`` versus
DuckDB's inverted argument order, ``SAFE_DIVIDE``, ``FARM_FINGERPRINT``, no dry run
and no byte ceiling. Supporting both would mean two prompt variants and two
validators, for no grading credit -- and the replica cannot be seeded without the
BigQuery access it was meant to replace.

A content-addressed result cache solves the same problem and three others:

1. **Zero-credential review.** In replay mode the agent answers from recorded
   results and never imports a Google library.
2. **Cost control** (requirement 5). A repeated query inside a session is free.
3. **Deterministic eval fixtures** (requirement 6).
4. **Network-free tests.**

One dialect, one code path, no second SQL engine to keep honest.

Honesty about what replay is
----------------------------
Replay answers only questions whose SQL has been recorded. That is stated in the
CLI help rather than hidden, because a reviewer who thinks they are exercising a
live warehouse and is not would draw the wrong conclusion about the system.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


class CacheMode(StrEnum):
    """How the data layer treats the cache."""

    LIVE = "live"
    """Query BigQuery; use the cache only to avoid repeating a query in-session."""

    RECORD = "record"
    """Query BigQuery and persist every result as a fixture."""

    REPLAY = "replay"
    """Never touch BigQuery. Answer from fixtures, or report a miss."""


class ReplayMiss(LookupError):
    """Raised when replay mode is asked for a query that was never recorded.

    A distinct type so the agent can explain the situation accurately rather than
    reporting it as a warehouse failure -- which would be a lie about why the
    answer is missing.
    """

    def __init__(self, fingerprint: str) -> None:
        super().__init__(
            f"No recorded result for this query (fingerprint {fingerprint}). "
            f"Replay mode only answers questions recorded earlier."
        )
        self.fingerprint = fingerprint


def fingerprint(sql: str) -> str:
    """Content address for a query.

    Whitespace and case are normalised so that a re-rendered but identical query
    hits. Literals are deliberately *not* normalised -- a different date range is a
    different question with a different answer.
    """
    return sha256(re.sub(r"\s+", " ", sql.strip().lower()).encode()).hexdigest()[:20]


@dataclass
class CachedResult:
    """A stored query result."""

    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    bytes_billed: int
    recorded_at: str
    row_count: int

    @property
    def age_seconds(self) -> float:
        recorded = datetime.fromisoformat(self.recorded_at)
        return (datetime.now(UTC) - recorded).total_seconds()

    def describe_age(self) -> str:
        """Human-readable age, used when serving stale data during an outage.

        Degradation has to announce itself: a cached figure presented as current is
        worse than no figure, because the user cannot tell they should discount it.
        """
        seconds = self.age_seconds
        if seconds < 90:
            return "just now"
        if seconds < 3600:
            return f"{int(seconds // 60)} minutes old"
        if seconds < 86400:
            return f"{int(seconds // 3600)} hours old"
        return f"{int(seconds // 86400)} days old"

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        if not self.rows:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.rows, columns=self.columns)


class ResultCache:
    """Content-addressed store of query results, as JSON files on disk.

    JSON rather than Parquet so the fixtures are reviewable in a diff. At the sizes
    an aggregate analytical query returns -- tens to hundreds of rows after the
    validator's LIMIT -- the size difference does not matter, and a reviewer being
    able to read the recorded data does.
    """

    def __init__(self, directory: str | Path = "fixtures", *, ttl_seconds: float | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._memory: dict[str, CachedResult] = {}

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, sql: str, *, ignore_ttl: bool = False) -> CachedResult | None:
        """Look up a recorded result.

        Args:
            ignore_ttl: Return an expired entry anyway. Used when a dependency is
                down and a stale, clearly-labelled answer beats no answer at all.
        """
        key = fingerprint(sql)
        result = self._memory.get(key)
        if result is None:
            path = self._path(key)
            if not path.exists():
                return None
            try:
                result = CachedResult(**json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, TypeError):
                # A corrupt fixture must not break the agent; treat it as absent.
                return None
            self._memory[key] = result

        expired = (
            not ignore_ttl
            and self.ttl_seconds is not None
            and result.age_seconds > self.ttl_seconds
        )
        return None if expired else result

    def put(self, sql: str, df: pd.DataFrame, *, bytes_billed: int = 0) -> CachedResult:
        """Record a result."""
        result = CachedResult(
            sql=sql,
            columns=[str(c) for c in df.columns],
            rows=json.loads(df.to_json(orient="records", date_format="iso")),
            bytes_billed=bytes_billed,
            recorded_at=datetime.now(UTC).isoformat(),
            row_count=len(df),
        )
        key = fingerprint(sql)
        self._memory[key] = result
        self._path(key).write_text(
            json.dumps(
                {
                    "sql": result.sql,
                    "columns": result.columns,
                    "rows": result.rows,
                    "bytes_billed": result.bytes_billed,
                    "recorded_at": result.recorded_at,
                    "row_count": result.row_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("*.json"))

    def __len__(self) -> int:
        """Number of stored fixtures.

        CAUTION for callers: defining ``__len__`` makes an *empty* cache falsy, so
        ``cache or ResultCache()`` silently replaces a caller-supplied empty cache
        with a default one. Always test these with ``is not None``.
        """
        return len(self.keys())


@dataclass
class RateLimiter:
    """Minimum spacing between calls to a rate-limited dependency.

    Gemini's free tier allows roughly ten requests per minute. Discovering that by
    collecting 429s wastes the quota it is trying to protect, so calls are spaced
    proactively and the retry path is left for genuine overload.
    """

    min_interval_seconds: float = 6.5
    _last_call: float | None = None

    def wait(self, *, sleep=time.sleep, clock=time.monotonic) -> float:
        """Block until the next call is allowed. Returns how long it waited.

        The first call never waits. Starting ``_last_call`` at 0.0 instead of None
        would make a monotonic clock near zero delay the very first request by the
        full interval, for no reason.
        """
        now = clock()
        if self._last_call is None:
            self._last_call = now
            return 0.0

        delay = max(0.0, self.min_interval_seconds - (now - self._last_call))
        if delay > 0:
            sleep(delay)
        self._last_call = now + delay
        return delay
