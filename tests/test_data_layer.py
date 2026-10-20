"""Tests for the result cache and the BigQuery gateway.

The gateway is exercised with a fake client, so these run without credentials and
without a network. That is the same property that lets a reviewer run the prototype
in replay mode.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analyst_assistant.data.bigquery_client import (
    BigQueryGateway,
    BudgetExceeded,
    QueryResult,
)
from analyst_assistant.data.result_cache import (
    CacheMode,
    RateLimiter,
    ReplayMiss,
    ResultCache,
    fingerprint,
)
from analyst_assistant.resilience.breaker import BreakerRegistry
from analyst_assistant.resilience.budget import Budget

SQL = "SELECT state, SUM(sale_price) AS revenue FROM s_order_items GROUP BY state"


@pytest.fixture()
def cache(tmp_path):
    return ResultCache(tmp_path / "fixtures")


def _frame():
    return pd.DataFrame({"state": ["CA", "TX"], "revenue": [1200.5, 980.0]})


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


def test_cosmetically_different_queries_share_a_fingerprint():
    assert fingerprint("SELECT 1\n FROM t") == fingerprint("select 1 from t")


def test_different_literals_are_different_questions():
    """A changed date range is a different question with a different answer."""
    a = fingerprint("SELECT 1 FROM t WHERE d > '2024-01-01'")
    b = fingerprint("SELECT 1 FROM t WHERE d > '2025-01-01'")
    assert a != b


# --------------------------------------------------------------------------
# Cache round-trip
# --------------------------------------------------------------------------


def test_result_round_trips_through_disk(cache):
    cache.put(SQL, _frame(), bytes_billed=5000)

    reloaded = ResultCache(cache.directory)  # a fresh process
    got = reloaded.get(SQL)
    assert got is not None
    assert got.row_count == 2
    assert got.bytes_billed == 5000
    pd.testing.assert_frame_equal(got.to_dataframe(), _frame())


def test_empty_result_is_cacheable(cache):
    """Empty is a real answer and must survive the round trip."""
    empty = pd.DataFrame(columns=["state", "revenue"])
    cache.put(SQL, empty)
    got = cache.get(SQL)
    assert got is not None
    assert got.row_count == 0
    assert list(got.to_dataframe().columns) == ["state", "revenue"]


def test_unknown_query_misses(cache):
    assert cache.get("SELECT 999 FROM s_orders") is None


def test_a_corrupt_fixture_is_treated_as_absent(cache):
    """A bad file must not break the agent."""
    cache.put(SQL, _frame())
    key = fingerprint(SQL)
    (cache.directory / f"{key}.json").write_text("{ not json", encoding="utf-8")
    ResultCache(cache.directory)  # fresh instance, so no memo
    assert ResultCache(cache.directory).get(SQL) is None


def test_ttl_expiry_can_be_overridden_for_stale_serving(tmp_path):
    """During an outage a labelled stale answer beats no answer."""
    cache = ResultCache(tmp_path / "fixtures", ttl_seconds=0)
    cache.put(SQL, _frame())

    assert cache.get(SQL) is None, "expired by TTL"
    assert cache.get(SQL, ignore_ttl=True) is not None


def test_age_is_described_in_human_terms(cache):
    cache.put(SQL, _frame())
    got = cache.get(SQL)
    assert got is not None
    assert got.describe_age() == "just now"


# --------------------------------------------------------------------------
# Replay mode
# --------------------------------------------------------------------------


class ExplodingClient:
    """Any use of this is a bug: replay mode must never reach BigQuery."""

    def __getattr__(self, name):
        raise AssertionError(f"replay mode touched the BigQuery client ({name})")


def test_replay_answers_from_fixtures_without_a_client(cache):
    cache.put(SQL, _frame(), bytes_billed=5000)
    gateway = BigQueryGateway(mode=CacheMode.REPLAY, cache=cache)
    gateway._client = ExplodingClient()

    result = gateway.execute(SQL)
    assert result.row_count == 2
    assert result.replayed
    assert result.bytes_billed == 0, "replay must never report a charge"


def test_replay_constructing_a_client_is_an_error(cache):
    """Guards the property that makes a credential-free run possible."""
    gateway = BigQueryGateway(mode=CacheMode.REPLAY, cache=cache)
    with pytest.raises(RuntimeError, match="replay mode"):
        _ = gateway.client


def test_replay_miss_is_distinguishable_from_a_warehouse_failure(cache):
    """Reporting a miss as an outage would be a lie about why the answer is absent."""
    gateway = BigQueryGateway(mode=CacheMode.REPLAY, cache=cache)
    with pytest.raises(ReplayMiss) as excinfo:
        gateway.execute("SELECT 1 FROM s_orders")
    assert "recorded earlier" in str(excinfo.value)


# --------------------------------------------------------------------------
# Cost control
# --------------------------------------------------------------------------


class FakeJob:
    def __init__(self, df, *, billed=0, processed=0):
        self._df = df
        self.total_bytes_billed = billed
        self.total_bytes_processed = processed

    def result(self):
        return self

    def to_dataframe(self):
        return self._df


class FakeClient:
    """Records how it was called, so ordering can be asserted."""

    def __init__(self, df, *, processed=1000, billed=1000):
        self.df = df
        self.processed = processed
        self.billed = billed
        self.calls: list[str] = []

    def query(self, sql, job_config=None):
        is_dry = bool(getattr(job_config, "dry_run", False))
        self.calls.append("dry_run" if is_dry else "execute")
        if is_dry:
            return FakeJob(self.df, processed=self.processed)
        return FakeJob(self.df, billed=self.billed)


def _gateway_with(client, cache, **kwargs):
    gateway = BigQueryGateway(mode=CacheMode.LIVE, cache=cache, **kwargs)
    gateway._client = client
    return gateway


def test_every_query_is_dry_run_before_execution(cache):
    """A cost check after execution is a receipt, not a control."""
    client = FakeClient(_frame())
    gateway = _gateway_with(client, cache)

    gateway.execute(SQL, budget=Budget())
    assert client.calls == ["dry_run", "execute"]


def test_a_query_over_budget_is_refused_before_it_runs(cache):
    client = FakeClient(_frame(), processed=5_000_000)
    gateway = _gateway_with(client, cache)
    budget = Budget(max_bytes_billed=1_000_000)

    with pytest.raises(BudgetExceeded) as excinfo:
        gateway.execute(SQL, budget=budget)

    assert client.calls == ["dry_run"], "it must not have executed"
    assert excinfo.value.estimated_bytes == 5_000_000


def test_budget_exceeded_reports_the_scale_of_the_overrun(cache):
    """More useful to a user than a bare refusal."""
    client = FakeClient(_frame(), processed=5_000_000)
    gateway = _gateway_with(client, cache)
    with pytest.raises(BudgetExceeded) as excinfo:
        gateway.execute(SQL, budget=Budget(max_bytes_billed=1_000_000))
    assert "MB" in str(excinfo.value)


def test_executed_bytes_are_charged_to_the_budget(cache):
    client = FakeClient(_frame(), billed=250_000)
    gateway = _gateway_with(client, cache)
    budget = Budget()

    gateway.execute(SQL, budget=budget)
    assert budget.bytes_billed == 250_000


def test_a_repeated_query_is_free_and_skips_the_warehouse(cache):
    client = FakeClient(_frame(), billed=250_000)
    gateway = _gateway_with(client, cache)
    cache.put(SQL, _frame(), bytes_billed=250_000)

    result = gateway.execute(SQL, budget=Budget())
    assert result.cache_hit
    assert result.bytes_billed == 0
    assert client.calls == [], "a cache hit must not even dry-run"


# --------------------------------------------------------------------------
# Record mode
# --------------------------------------------------------------------------


def test_record_mode_persists_results_for_later_replay(cache):
    client = FakeClient(_frame(), billed=4242)
    gateway = BigQueryGateway(mode=CacheMode.RECORD, cache=cache)
    gateway._client = client

    gateway.execute(SQL, budget=Budget())

    stored = ResultCache(cache.directory).get(SQL)
    assert stored is not None
    assert stored.bytes_billed == 4242


def test_live_mode_does_not_write_fixtures(cache):
    """Fixtures are an explicit act, so the repo does not fill up by accident."""
    client = FakeClient(_frame())
    gateway = _gateway_with(client, cache)
    gateway.execute(SQL, budget=Budget())
    assert len(ResultCache(cache.directory)) == 0


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------


class DownClient:
    def __init__(self):
        self.calls: list[str] = []

    def query(self, sql, job_config=None):
        self.calls.append("attempt")
        raise ServiceUnavailable("bigquery is down")


class ServiceUnavailable(Exception):
    pass


def test_an_outage_serves_a_stale_result_and_flags_its_age(cache):
    """Degradation must announce itself; the age travels with the result."""
    cache.put(SQL, _frame(), bytes_billed=1000)
    breakers = BreakerRegistry(failure_threshold=1)
    breakers.get("bigquery").record_failure()  # breaker already open

    gateway = BigQueryGateway(
        mode=CacheMode.LIVE,
        cache=ResultCache(cache.directory, ttl_seconds=0),
        breakers=breakers,
    )
    gateway._client = DownClient()

    result = gateway.execute(SQL, budget=Budget())
    assert result.stale is not None
    assert result.cache_hit
    assert result.stale.describe_age()


def test_an_outage_with_no_cached_result_raises_rather_than_inventing_one(cache):
    from analyst_assistant.resilience.breaker import CircuitOpenError

    breakers = BreakerRegistry(failure_threshold=1)
    breakers.get("bigquery").record_failure()
    gateway = BigQueryGateway(mode=CacheMode.LIVE, cache=cache, breakers=breakers)
    gateway._client = DownClient()

    with pytest.raises(CircuitOpenError):
        gateway.execute(SQL, budget=Budget())


def test_stale_serving_can_be_disabled(cache):
    """Some clients would rather refuse than risk a stale number."""
    from analyst_assistant.resilience.breaker import CircuitOpenError

    cache.put(SQL, _frame())
    breakers = BreakerRegistry(failure_threshold=1)
    breakers.get("bigquery").record_failure()
    gateway = BigQueryGateway(
        mode=CacheMode.LIVE, cache=ResultCache(cache.directory, ttl_seconds=0), breakers=breakers
    )
    gateway._client = DownClient()

    with pytest.raises(CircuitOpenError):
        gateway.execute(SQL, budget=Budget(), allow_stale_on_failure=False)


# --------------------------------------------------------------------------
# Empty results
# --------------------------------------------------------------------------


def test_empty_is_not_an_error(cache):
    """It usually means a well-formed question whose filter excluded everything."""
    client = FakeClient(pd.DataFrame(columns=["state", "revenue"]))
    gateway = _gateway_with(client, cache)

    result = gateway.execute(SQL, budget=Budget())
    assert result.is_empty
    assert result.row_count == 0


def test_query_result_reports_row_count():
    assert QueryResult(df=_frame(), bytes_billed=0).row_count == 2


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_rate_limiter_does_not_delay_the_first_call():
    """A monotonic clock near zero must not stall the very first request."""
    limiter = RateLimiter(min_interval_seconds=6.0)
    assert limiter.wait(sleep=lambda _: None, clock=lambda: 0.0) == 0.0


def test_rate_limiter_spaces_calls_proactively():
    """Learning the limit by collecting 429s wastes the quota it protects."""
    clock_value = {"t": 0.0}
    limiter = RateLimiter(min_interval_seconds=6.0)
    clock = lambda: clock_value["t"]  # noqa: E731

    limiter.wait(sleep=lambda _: None, clock=clock)
    clock_value["t"] = 2.0
    waited = limiter.wait(sleep=lambda _: None, clock=clock)

    assert waited == pytest.approx(4.0)


def test_rate_limiter_does_not_wait_when_enough_time_passed():
    clock_value = {"t": 0.0}
    limiter = RateLimiter(min_interval_seconds=6.0)
    clock = lambda: clock_value["t"]  # noqa: E731

    limiter.wait(sleep=lambda _: None, clock=clock)
    clock_value["t"] = 100.0
    assert limiter.wait(sleep=lambda _: None, clock=clock) == 0.0
