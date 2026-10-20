"""BigQuery access: dry-run cost control, result cache, and record/replay.

Wraps the ``BigQueryRunner`` supplied with the assignment and adds what a
production path needs around it.

Cost is checked before the money is spent
-----------------------------------------
BigQuery bills on bytes scanned, so a check *after* execution is a receipt, not a
control. Every query is dry-run first (free), the estimate is tested against the
remaining budget, and only then is it executed with ``maximum_bytes_billed`` set as
a second, server-side guarantee.

Why a result cache instead of an offline database replica
---------------------------------------------------------
A DuckDB copy of the dataset would let a reviewer run the prototype without Google
Cloud, but the BigQuery dialect divergence is load-bearing -- backtick-quoted
``project.dataset.table``, ``DATE_TRUNC(date, MONTH)`` argument order,
``SAFE_DIVIDE``, no dry run -- so it would mean two dialects and two validators. It
also cannot be seeded without the access it was meant to replace.

A content-addressed cache solves the same problem and three more: it gives a
credential-free review (replay mode), the requirement-5 cost cache, deterministic
test fixtures, and network-free tests. One dialect, one code path.

Google libraries are imported lazily, so replay mode works on a machine with no
credentials and no ``google-cloud-bigquery`` installed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..obs.events import Recorder
from ..resilience.breaker import CircuitBreaker, CircuitOpenError, call_with_resilience
from ..resilience.budget import Budget
from ..safety.schema import DATASET

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

#: Server-side ceiling, applied in addition to the in-process budget, so a bug in
#: our own accounting still cannot produce an unbounded bill.
DEFAULT_MAX_BYTES_BILLED = 2_000_000_000


class CacheMode(StrEnum):
    LIVE = "live"
    """Query BigQuery; use the cache only to avoid repeating a query in-session."""

    RECORD = "record"
    """Query BigQuery and persist each result as a fixture."""

    REPLAY = "replay"
    """Never touch BigQuery. Answer from fixtures, or report a miss."""


class ReplayMiss(LookupError):
    """Replay mode was asked for a query that was never recorded.

    A distinct type so this is never reported as a warehouse failure, which would
    misstate why the answer is absent.
    """

    def __init__(self, key: str) -> None:
        super().__init__(
            f"No recorded result for this query ({key}). Replay mode only answers "
            f"questions recorded earlier."
        )
        self.key = key


class BudgetExceeded(RuntimeError):
    """The query would scan more than the turn's remaining byte budget."""

    def __init__(self, estimated_bytes: int, remaining_bytes: int) -> None:
        super().__init__(
            f"query would scan ~{estimated_bytes / 1e6:.0f} MB but only "
            f"{remaining_bytes / 1e6:.0f} MB remain for this question"
        )
        self.estimated_bytes = estimated_bytes
        self.remaining_bytes = remaining_bytes


def fingerprint(sql: str) -> str:
    """Content address for a query.

    Whitespace and case are normalised so a re-rendered identical query still hits.
    Literals are deliberately not normalised -- a different date range is a
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

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def age_description(self) -> str:
        """How old this is, in words.

        Used when a cached figure is served during an outage. Degradation has to
        announce itself: a stale number presented as current is worse than none,
        because the user cannot tell they should discount it.
        """
        seconds = (datetime.now(UTC) - datetime.fromisoformat(self.recorded_at)).total_seconds()
        if seconds < 90:
            return "just now"
        if seconds < 3600:
            return f"{int(seconds // 60)} minutes old"
        return f"{int(seconds // 3600)} hours old"

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        if not self.rows:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(self.rows, columns=self.columns)


class ResultCache:
    """Content-addressed store of query results, as JSON files on disk.

    JSON rather than Parquet so fixtures are readable in a diff. After the
    validator's LIMIT, results are tens to hundreds of rows, so the size difference
    does not matter and reviewability does.

    Note there is deliberately no ``__len__``: defining it would make an *empty*
    cache falsy, and ``cache or ResultCache()`` would then silently discard a
    caller's cache. That bug wrote test fixtures into the repo before it was caught.
    """

    def __init__(self, directory: str | Path = "fixtures", *, ttl_seconds: float | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, sql: str, *, ignore_ttl: bool = False) -> CachedResult | None:
        """Look up a recorded result.

        Args:
            ignore_ttl: Return an expired entry anyway -- used when a dependency is
                down and a clearly-labelled stale answer beats no answer.
        """
        path = self._path(fingerprint(sql))
        if not path.exists():
            return None
        try:
            result = CachedResult(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            return None  # a corrupt fixture must not break the agent

        if not ignore_ttl and self.ttl_seconds is not None:
            age = (
                datetime.now(UTC) - datetime.fromisoformat(result.recorded_at)
            ).total_seconds()
            if age > self.ttl_seconds:
                return None
        return result

    def put(self, sql: str, df: pd.DataFrame, *, bytes_billed: int = 0) -> CachedResult:
        result = CachedResult(
            sql=sql,
            columns=[str(c) for c in df.columns],
            rows=json.loads(df.to_json(orient="records", date_format="iso")),
            bytes_billed=bytes_billed,
            recorded_at=datetime.now(UTC).isoformat(),
        )
        self._path(fingerprint(sql)).write_text(
            json.dumps(result.__dict__, indent=2), encoding="utf-8"
        )
        return result

    def count(self) -> int:
        return len(list(self.directory.glob("*.json")))


@dataclass
class RateLimiter:
    """Minimum spacing between calls to a rate-limited dependency.

    The Gemini free tier allows roughly ten requests per minute. Discovering that
    by collecting 429s wastes the quota it is meant to protect, so calls are spaced
    proactively and the retry path is left for genuine overload.
    """

    min_interval_seconds: float = 6.5
    _last_call: float | None = field(default=None, repr=False)

    def wait(self, *, sleep=time.sleep, clock=time.monotonic) -> float:
        """Block until the next call is allowed. Returns the delay applied.

        The first call never waits: starting from 0.0 would stall it by a full
        interval on a monotonic clock that begins near zero.
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


@dataclass
class QueryResult:
    """Outcome of running one query."""

    df: pd.DataFrame
    bytes_billed: int
    cache_hit: bool = False
    replayed: bool = False
    stale: CachedResult | None = None
    """Set when a cached result was served because the warehouse was unavailable.

    The caller must surface the age; silently serving stale data is the failure this
    field exists to prevent.
    """

    @property
    def row_count(self) -> int:
        return len(self.df)

    @property
    def is_empty(self) -> bool:
        """Empty is a distinct outcome, not an error.

        It usually means a well-formed question whose filter excluded everything --
        a date range outside the data, say -- and the right response is to diagnose
        why rather than report a failure or invent a zero.
        """
        return self.df.empty


class BigQueryGateway:
    """The only path to BigQuery.

    A single entry point by design: the dry run, the byte ceiling, the cache, the
    breaker and the replay switch must apply to *every* query, and the reliable way
    to ensure that is to leave nowhere else to go.
    """

    def __init__(
        self,
        *,
        project_id: str | None = None,
        mode: CacheMode = CacheMode.LIVE,
        cache: ResultCache | None = None,
        breaker: CircuitBreaker | None = None,
        max_bytes_billed: int = DEFAULT_MAX_BYTES_BILLED,
        dataset: str = DATASET,
    ) -> None:
        self.project_id = project_id
        self.mode = mode
        self.cache = cache if cache is not None else ResultCache()
        self.breaker = breaker if breaker is not None else CircuitBreaker("BigQuery")
        self.max_bytes_billed = max_bytes_billed
        self.dataset = dataset
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        """The BigQuery client, created on first use and never in replay mode."""
        if self.mode is CacheMode.REPLAY:
            raise RuntimeError("replay mode must not construct a BigQuery client")
        if self._client is None:
            from google.cloud import bigquery  # lazy; see module docstring

            self._client = bigquery.Client(project=self.project_id)
        return self._client

    def dry_run(self, sql: str) -> int:
        """Estimate bytes scanned without running the query. Free."""
        if self.mode is CacheMode.REPLAY:
            return 0
        from google.cloud import bigquery

        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = call_with_resilience(
            lambda: self.client.query(sql, job_config=config), breaker=self.breaker
        )
        return int(job.total_bytes_processed or 0)

    def execute(
        self,
        sql: str,
        *,
        budget: Budget | None = None,
        recorder: Recorder | None = None,
        allow_stale_on_failure: bool = True,
    ) -> QueryResult:
        """Run a query, or explain why it was not run.

        The order is the point of this method: cache, replay check, dry run, budget
        check, execute.

        Raises:
            ReplayMiss: replay mode with no recording for this query.
            BudgetExceeded: the dry-run estimate exceeds the remaining budget.
            CircuitOpenError: BigQuery is presumed down and no cached result exists.
        """
        # An exact repeat within the turn is the cheapest possible answer.
        cached = self.cache.get(sql)
        if cached is not None:
            if recorder is not None:
                with recorder.span("execute_bq") as m:
                    m["cache_hit"] = True
                    m["attrs"] = {"sql": sql, "row_count": cached.row_count}
            return QueryResult(
                df=cached.to_dataframe(),
                bytes_billed=0,
                cache_hit=True,
                replayed=self.mode is CacheMode.REPLAY,
            )

        if self.mode is CacheMode.REPLAY:
            raise ReplayMiss(fingerprint(sql))

        # The dry run must sit inside this try: an open breaker raises *there*,
        # before execution is attempted, so a try wrapping only the execute would
        # skip the stale fallback during exactly the outage it exists to handle.
        # BudgetExceeded is not a CircuitOpenError, so it still propagates.
        try:
            estimated = self.dry_run(sql)
            if budget is not None and not budget.can_run_query(estimated):
                raise BudgetExceeded(estimated, budget.remaining_bytes)
            result = self._execute_live(sql, estimated=estimated, recorder=recorder)
        except CircuitOpenError:
            if allow_stale_on_failure:
                stale = self.cache.get(sql, ignore_ttl=True)
                if stale is not None:
                    return QueryResult(
                        df=stale.to_dataframe(),
                        bytes_billed=0,
                        cache_hit=True,
                        stale=stale,
                    )
            raise

        if budget is not None:
            budget.charge_query(result.bytes_billed)
        return result

    def _execute_live(
        self, sql: str, *, estimated: int, recorder: Recorder | None
    ) -> QueryResult:
        from google.cloud import bigquery

        config = bigquery.QueryJobConfig(maximum_bytes_billed=self.max_bytes_billed)

        def run() -> tuple[pd.DataFrame, int]:
            job = self.client.query(sql, job_config=config)
            # db-dtypes is required here: thelook has DATE/TIMESTAMP/NUMERIC columns
            # and to_dataframe() raises without it. Pinned in requirements.txt.
            return job.result().to_dataframe(), int(job.total_bytes_billed or 0)

        if recorder is None:
            df, billed = call_with_resilience(run, breaker=self.breaker)
        else:
            with recorder.span("execute_bq") as m:
                df, billed = call_with_resilience(run, breaker=self.breaker)
                m["bytes_billed"] = billed
                m["status"] = "empty" if df.empty else "ok"
                m["attrs"] = {"sql": sql, "row_count": len(df), "estimated_bytes": estimated}

        if self.mode is CacheMode.RECORD:
            self.cache.put(sql, df, bytes_billed=billed)
        return QueryResult(df=df, bytes_billed=billed)

    def table_schema(self, table: str) -> list[dict[str, Any]]:
        """Live schema for one table, used by ``doctor``.

        Confirms the hard-coded classification in :mod:`..safety.schema` still
        matches reality. Silent drift could mean a new, unclassified PII column.
        """
        if self.mode is CacheMode.REPLAY:
            return []
        meta = call_with_resilience(
            lambda: self.client.get_table(f"{self.dataset}.{table}"), breaker=self.breaker
        )
        return [
            {"name": f.name, "type": f.field_type, "mode": f.mode} for f in meta.schema
        ]
