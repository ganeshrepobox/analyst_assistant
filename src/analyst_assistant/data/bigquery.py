"""BigQuery access: dry-run cost check, result cache, record/replay.

One entry point, so the byte ceiling and the cache apply to every query.

Cost is checked before the query runs. BigQuery bills on bytes scanned, so
checking afterwards is only a receipt. The dry run itself is free.

Replay mode answers from recorded results and never imports a Google library, so
the prototype runs without a GCP account. It is also where the test fixtures come
from.
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

# Server-side ceiling, applied in addition to the in-process budget, so a bug in
# our own accounting still cannot produce an unbounded bill.
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

    Its own type, so it is never reported as a warehouse failure.
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

    Whitespace and case are normalised; literals are not, since a different date
    range is a different question.
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
        """How old this is, in words. Shown when a cached figure is served."""
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
    """Query results as JSON files on disk, addressed by query fingerprint.

    JSON rather than Parquet so fixtures are readable in a diff. Note there is no
    __len__: it would make an empty cache falsy, and `cache or ResultCache()` then
    silently discards a caller's cache. That bug wrote fixtures into the repo once.
    """

    def __init__(self, directory: str | Path = "fixtures", *, ttl_seconds: float | None = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, sql: str, *, ignore_ttl: bool = False) -> CachedResult | None:
        """Look up a recorded result.

        ignore_ttl returns an expired entry anyway, used when a dependency is down and a
        labelled stale answer beats nothing.
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
    """Minimum spacing between calls. The Gemini free tier allows about 10/minute."""

    min_interval_seconds: float = 2.5
    _last_call: float | None = field(default=None, repr=False)

    def wait(self, *, sleep=time.sleep, clock=time.monotonic) -> float:
        """Block until the next call is allowed. Returns the delay applied.

        The first call never waits.
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
        """Empty is an outcome, not an error, usually a filter that excluded everything."""
        return self.df.empty


class BigQueryGateway:
    """The only path to BigQuery, so the cost checks apply to every query."""

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
        """BigQuery client, created on first use and never in replay mode."""
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

        Order matters: cache, replay check, dry run, budget check, execute. Raises
        ReplayMiss, BudgetExceeded or CircuitOpenError.
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
        """Live schema for one table, used by doctor to spot drift from safety.schema.

        Drift could mean a new column nobody has classified.
        """
        if self.mode is CacheMode.REPLAY:
            return []
        meta = call_with_resilience(
            lambda: self.client.get_table(f"{self.dataset}.{table}"), breaker=self.breaker
        )
        return [
            {"name": f.name, "type": f.field_type, "mode": f.mode} for f in meta.schema
        ]
