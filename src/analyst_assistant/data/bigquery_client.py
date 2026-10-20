"""BigQuery access: dry run, byte ceiling, cache, breaker, replay.

Wraps the ``BigQueryRunner`` supplied with the assignment and adds what a
production path needs around it: a cost estimate before execution, a hard byte
ceiling, resilience, and the record/replay modes.

Cost is checked before the money is spent
-----------------------------------------
BigQuery bills on bytes scanned, so a cost check *after* execution is a receipt,
not a control. Every query is therefore dry-run first, the estimate is tested
against the remaining budget, and only then is it executed with
``maximum_bytes_billed`` set as a second, server-side guarantee. The dry run is
free.

Google libraries are imported lazily
------------------------------------
Deliberate: replay mode must work on a machine with no credentials and, if it comes
to it, no ``google-cloud-bigquery`` installed. Importing at module scope would make
the whole agent unimportable without cloud access, which would defeat the point of
having an offline mode at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..obs.events import Recorder
from ..resilience.breaker import BreakerRegistry, CircuitOpenError, call_with_resilience
from ..resilience.budget import Budget
from ..safety.schema import DATASET
from .result_cache import CachedResult, CacheMode, ReplayMiss, ResultCache

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

#: Server-side ceiling, applied in addition to the in-process budget. Belt and
#: braces: a bug in our accounting cannot produce an unbounded bill.
DEFAULT_MAX_BYTES_BILLED = 2_000_000_000


class BudgetExceeded(RuntimeError):
    """A query would scan more than the turn's remaining byte budget.

    Carries the estimate so the agent can tell the user how much too large the
    question was, which is more useful than a bare refusal.
    """

    def __init__(self, estimated_bytes: int, remaining_bytes: int) -> None:
        super().__init__(
            f"query would scan ~{estimated_bytes / 1e6:.0f} MB but only "
            f"{remaining_bytes / 1e6:.0f} MB remain for this question"
        )
        self.estimated_bytes = estimated_bytes
        self.remaining_bytes = remaining_bytes


@dataclass
class QueryResult:
    """Outcome of running one query."""

    df: pd.DataFrame
    bytes_billed: int
    cache_hit: bool = False
    replayed: bool = False
    stale: CachedResult | None = None
    """Set when a cached result was served because the warehouse was unavailable.

    The caller must surface the age; silently serving stale data is the failure
    mode this field exists to prevent.
    """

    @property
    def row_count(self) -> int:
        return len(self.df)

    @property
    def is_empty(self) -> bool:
        """Empty is a distinct outcome, not an error.

        It usually means the question was well-formed but the filter excluded
        everything -- a date range outside the data, say -- and the right response
        is to diagnose why rather than report a failure or invent a zero.
        """
        return self.df.empty


class BigQueryGateway:
    """The only path to BigQuery.

    Single entry point by design: the dry run, the byte ceiling, the cache, the
    breaker and the replay switch all have to apply to *every* query, and the
    reliable way to guarantee that is to leave nowhere else to go.
    """

    def __init__(
        self,
        *,
        project_id: str | None = None,
        mode: CacheMode = CacheMode.LIVE,
        cache: ResultCache | None = None,
        breakers: BreakerRegistry | None = None,
        max_bytes_billed: int = DEFAULT_MAX_BYTES_BILLED,
        dataset: str = DATASET,
    ) -> None:
        self.project_id = project_id
        self.mode = mode
        # `is not None`, NOT `or`: ResultCache defines __len__, so an EMPTY cache is
        # falsy and `cache or ResultCache()` would silently discard the caller's
        # cache and fall back to the default directory. That bug wrote test
        # fixtures into the repo before it was caught.
        self.cache = cache if cache is not None else ResultCache()
        self.breakers = breakers if breakers is not None else BreakerRegistry()
        self.max_bytes_billed = max_bytes_billed
        self.dataset = dataset
        self._client: Any | None = None

    # -- client -------------------------------------------------------------

    @property
    def client(self) -> Any:
        """The BigQuery client, created on first use.

        Never constructed in replay mode, so no credentials are required there.
        """
        if self.mode is CacheMode.REPLAY:
            raise RuntimeError("replay mode must not construct a BigQuery client")
        if self._client is None:
            from google.cloud import bigquery  # imported lazily, see module docstring

            self._client = bigquery.Client(project=self.project_id)
        return self._client

    # -- cost estimation ----------------------------------------------------

    def dry_run(self, sql: str) -> int:
        """Estimate bytes scanned without running the query. Free.

        Returns 0 in replay mode, where nothing will be billed.
        """
        if self.mode is CacheMode.REPLAY:
            return 0

        from google.cloud import bigquery

        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        breaker = self.breakers.get("bigquery")
        job = call_with_resilience(
            lambda: self.client.query(sql, job_config=config),
            breaker=breaker,
        )
        return int(job.total_bytes_processed or 0)

    # -- execution ----------------------------------------------------------

    def execute(
        self,
        sql: str,
        *,
        budget: Budget | None = None,
        recorder: Recorder | None = None,
        allow_stale_on_failure: bool = True,
    ) -> QueryResult:
        """Run a query, or explain why it was not run.

        Order of operations matters and is the whole point of this method:
        cache, then replay check, then dry run, then budget check, then execute.

        Raises:
            ReplayMiss: In replay mode with no recording for this query.
            BudgetExceeded: If the dry-run estimate exceeds the remaining budget.
            CircuitOpenError: If BigQuery is presumed down and no cached result is
                available to fall back on.
        """
        # 1. An exact repeat inside the turn is free. This is the cheapest possible
        #    answer and so is checked before anything else.
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
            from .result_cache import fingerprint

            raise ReplayMiss(fingerprint(sql))

        # 2 and 3. Estimate before spending, then execute with a server-side
        # ceiling as a second guarantee.
        #
        # The dry run must sit INSIDE this try: an open breaker raises on the dry
        # run, before execution is ever attempted, so a try that wrapped only the
        # execute would skip the stale fallback during exactly the outage it exists
        # to handle. BudgetExceeded is not a CircuitOpenError, so it still
        # propagates rather than being answered from cache.
        try:
            estimated = self.dry_run(sql)
            if budget is not None and not budget.can_run_query(estimated):
                raise BudgetExceeded(estimated, budget.remaining_bytes)
            result = self._execute_live(sql, estimated=estimated, recorder=recorder)
        except CircuitOpenError:
            # The warehouse is presumed down. A clearly-labelled stale answer beats
            # no answer -- but only if the caller surfaces the label, which is why
            # the age travels with the result.
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
        breaker = self.breakers.get("bigquery")

        def run() -> tuple[pd.DataFrame, int]:
            job = self.client.query(sql, job_config=config)
            # db-dtypes is required here: thelook has DATE/TIMESTAMP/NUMERIC columns
            # and to_dataframe() raises without it. Pinned in requirements.txt.
            frame = job.result().to_dataframe()
            return frame, int(job.total_bytes_billed or 0)

        if recorder is None:
            df, billed = call_with_resilience(run, breaker=breaker)
        else:
            with recorder.span("execute_bq") as m:
                df, billed = call_with_resilience(run, breaker=breaker)
                m["bytes_billed"] = billed
                m["status"] = "empty" if df.empty else "ok"
                m["attrs"] = {
                    "sql": sql,
                    "row_count": len(df),
                    "estimated_bytes": estimated,
                }

        if self.mode is CacheMode.RECORD:
            self.cache.put(sql, df, bytes_billed=billed)

        return QueryResult(df=df, bytes_billed=billed)

    # -- schema -------------------------------------------------------------

    def table_schema(self, table: str) -> list[dict[str, Any]]:
        """Live schema for one table.

        Used by ``doctor`` to confirm the hard-coded schema in
        :mod:`analyst_assistant.safety.schema` still matches reality. A silent
        schema drift would mean the scoped views omit a column that now exists --
        which could be a new PII column nobody classified.
        """
        if self.mode is CacheMode.REPLAY:
            return []
        breaker = self.breakers.get("bigquery")
        meta = call_with_resilience(
            lambda: self.client.get_table(f"{self.dataset}.{table}"), breaker=breaker
        )
        return [
            {
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,
                "description": field.description or "",
            }
            for field in meta.schema
        ]
