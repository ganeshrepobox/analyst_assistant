"""The BigQuery gateway: cost control, caching, replay and degradation.

Exercised with a fake client, so these run with no credentials and no network,
the same property that lets a reviewer run the prototype in replay mode.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analyst_assistant.data.bigquery import (
    BigQueryGateway,
    BudgetExceeded,
    CacheMode,
    RateLimiter,
    ReplayMiss,
    ResultCache,
    fingerprint,
)
from analyst_assistant.resilience.breaker import CircuitBreaker, CircuitOpenError
from analyst_assistant.resilience.budget import Budget

SQL = "SELECT state, SUM(sale_price) AS revenue FROM s_order_items GROUP BY state"


@pytest.fixture()
def cache(tmp_path):
    return ResultCache(tmp_path / "fixtures")


def _frame():
    return pd.DataFrame({"state": ["CA", "TX"], "revenue": [1200.5, 980.0]})


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
        self.df, self.processed, self.billed = df, processed, billed
        self.calls: list[str] = []

    def query(self, sql, job_config=None):
        dry = bool(getattr(job_config, "dry_run", False))
        self.calls.append("dry_run" if dry else "execute")
        return (
            FakeJob(self.df, processed=self.processed)
            if dry
            else FakeJob(self.df, billed=self.billed)
        )


class DownClient:
    def query(self, sql, job_config=None):
        raise ServiceUnavailable("bigquery is down")


class ServiceUnavailable(Exception):
    pass


def _live(client, cache, **kwargs):
    gateway = BigQueryGateway(mode=CacheMode.LIVE, cache=cache, **kwargs)
    gateway._client = client
    return gateway


# Fingerprinting and the cache


def test_cosmetically_different_queries_share_a_fingerprint():
    assert fingerprint("SELECT 1\n FROM t") == fingerprint("select 1 from t")


def test_different_literals_are_different_questions():
    """A changed date range is a different question with a different answer."""
    assert fingerprint("SELECT 1 WHERE d > '2024-01-01'") != fingerprint(
        "SELECT 1 WHERE d > '2025-01-01'"
    )


def test_result_round_trips_through_disk(cache):
    cache.put(SQL, _frame(), bytes_billed=5000)

    got = ResultCache(cache.directory).get(SQL)  # a fresh process
    assert got is not None
    assert got.row_count == 2
    assert got.bytes_billed == 5000
    pd.testing.assert_frame_equal(got.to_dataframe(), _frame())


def test_empty_result_is_cacheable(cache):
    """Empty is a real answer and must survive the round trip."""
    cache.put(SQL, pd.DataFrame(columns=["state", "revenue"]))
    got = cache.get(SQL)
    assert got is not None and got.row_count == 0
    assert list(got.to_dataframe().columns) == ["state", "revenue"]


def test_a_corrupt_fixture_is_treated_as_absent(cache):
    """A bad file must not break the agent."""
    cache.put(SQL, _frame())
    (cache.directory / f"{fingerprint(SQL)}.json").write_text("{ not json", encoding="utf-8")
    assert cache.get(SQL) is None


def test_expired_entries_can_still_be_read_for_stale_serving(tmp_path):
    """During an outage a labelled stale answer beats no answer."""
    cache = ResultCache(tmp_path / "fx", ttl_seconds=0)
    cache.put(SQL, _frame())

    assert cache.get(SQL) is None, "expired by TTL"
    assert cache.get(SQL, ignore_ttl=True) is not None


# Replay mode


class ExplodingClient:
    """Any use of this is a bug: replay must never reach BigQuery."""

    def __getattr__(self, name):
        raise AssertionError(f"replay mode touched the client ({name})")


def test_replay_answers_from_fixtures_without_a_client(cache):
    cache.put(SQL, _frame(), bytes_billed=5000)
    gateway = BigQueryGateway(mode=CacheMode.REPLAY, cache=cache)
    gateway._client = ExplodingClient()

    result = gateway.execute(SQL)
    assert result.row_count == 2
    assert result.replayed
    assert result.bytes_billed == 0, "replay must never report a charge"


def test_replay_refuses_to_construct_a_client(cache):
    """Guards the property that makes a credential-free run possible."""
    gateway = BigQueryGateway(mode=CacheMode.REPLAY, cache=cache)
    with pytest.raises(RuntimeError, match="replay mode"):
        _ = gateway.client


def test_replay_miss_is_distinguishable_from_an_outage(cache):
    """Reporting a miss as an outage would misstate why the answer is absent."""
    gateway = BigQueryGateway(mode=CacheMode.REPLAY, cache=cache)
    with pytest.raises(ReplayMiss, match="recorded earlier"):
        gateway.execute("SELECT 1 FROM s_orders")


def test_record_mode_persists_results_for_later_replay(cache):
    client = FakeClient(_frame(), billed=4242)
    gateway = BigQueryGateway(mode=CacheMode.RECORD, cache=cache)
    gateway._client = client

    gateway.execute(SQL, budget=Budget())

    stored = ResultCache(cache.directory).get(SQL)
    assert stored is not None and stored.bytes_billed == 4242


def test_recorded_fixtures_are_masked(cache):
    """Fixtures are committed, so raw customer ids must not reach them.

    The aggregates recorded so far contain none, but a top-customers query would.
    """
    frame = pd.DataFrame({"user_id": [4211, 5002], "revenue": [900.0, 800.0]})
    gateway = BigQueryGateway(mode=CacheMode.RECORD, cache=cache)
    gateway._client = FakeClient(frame)

    gateway.execute("SELECT user_id, revenue FROM s_order_items", budget=Budget())

    stored = ResultCache(cache.directory).get("SELECT user_id, revenue FROM s_order_items")
    assert stored is not None
    values = [row["user_id"] for row in stored.rows]
    assert all(str(v).startswith("CUST-") for v in values), values
    assert 4211 not in values


def test_live_mode_does_not_write_fixtures(cache):
    """Fixtures are an explicit act, so the repo does not fill up by accident."""
    _live(FakeClient(_frame()), cache).execute(SQL, budget=Budget())
    assert ResultCache(cache.directory).count() == 0


def test_an_explicit_cache_is_honoured(cache):
    """Regression: `cache or ResultCache()` discarded an EMPTY caller cache.

    ResultCache deliberately has no __len__ now, but the gateway also tests with
    `is not None`. The old bug wrote fixtures into the repo.
    """
    gateway = BigQueryGateway(mode=CacheMode.LIVE, cache=cache)
    assert gateway.cache is cache


# Cost control


def test_every_query_is_dry_run_before_execution(cache):
    """A cost check after execution is a receipt, not a control."""
    client = FakeClient(_frame())
    _live(client, cache).execute(SQL, budget=Budget())
    assert client.calls == ["dry_run", "execute"]


def test_a_query_over_budget_is_refused_before_it_runs(cache):
    client = FakeClient(_frame(), processed=5_000_000)
    gateway = _live(client, cache)

    with pytest.raises(BudgetExceeded, match="MB"):
        gateway.execute(SQL, budget=Budget(max_bytes_billed=1_000_000))

    assert client.calls == ["dry_run"], "it must not have executed"


def test_executed_bytes_are_charged_to_the_budget(cache):
    budget = Budget()
    _live(FakeClient(_frame(), billed=250_000), cache).execute(SQL, budget=budget)
    assert budget.bytes_billed == 250_000


def test_a_repeated_query_is_free_and_skips_the_warehouse(cache):
    cache.put(SQL, _frame(), bytes_billed=250_000)
    client = FakeClient(_frame())

    result = _live(client, cache).execute(SQL, budget=Budget())

    assert result.cache_hit and result.bytes_billed == 0
    assert client.calls == [], "a cache hit must not even dry-run"


# Degradation


def _open_breaker():
    breaker = CircuitBreaker("BigQuery", failure_threshold=1)
    breaker.record_failure()
    return breaker


def test_an_outage_serves_a_stale_result_and_flags_its_age(tmp_path):
    """Degradation must announce itself, so the age travels with the result.

    The breaker raises on the dry run, which is why the stale fallback has to wrap
    the whole interaction rather than just the execute call.
    """
    cache = ResultCache(tmp_path / "fx", ttl_seconds=0)
    cache.put(SQL, _frame(), bytes_billed=1000)

    gateway = BigQueryGateway(mode=CacheMode.LIVE, cache=cache, breaker=_open_breaker())
    gateway._client = DownClient()

    result = gateway.execute(SQL, budget=Budget())
    assert result.stale is not None
    assert result.stale.age_description()


def test_an_outage_with_no_cached_result_raises_rather_than_inventing_one(cache):
    gateway = BigQueryGateway(mode=CacheMode.LIVE, cache=cache, breaker=_open_breaker())
    gateway._client = DownClient()

    with pytest.raises(CircuitOpenError):
        gateway.execute(SQL, budget=Budget())


def test_stale_serving_can_be_disabled(tmp_path):
    """Some clients would rather refuse than risk a stale number."""
    cache = ResultCache(tmp_path / "fx", ttl_seconds=0)
    cache.put(SQL, _frame())
    gateway = BigQueryGateway(mode=CacheMode.LIVE, cache=cache, breaker=_open_breaker())
    gateway._client = DownClient()

    with pytest.raises(CircuitOpenError):
        gateway.execute(SQL, budget=Budget(), allow_stale_on_failure=False)


def test_empty_is_not_an_error(cache):
    """Usually a well-formed question whose filter excluded everything."""
    client = FakeClient(pd.DataFrame(columns=["state", "revenue"]))
    result = _live(client, cache).execute(SQL, budget=Budget())
    assert result.is_empty and result.row_count == 0


# Rate limiting


def test_rate_limiter_does_not_delay_the_first_call():
    """A monotonic clock near zero must not stall the very first request."""
    assert RateLimiter(min_interval_seconds=6.0).wait(sleep=lambda _: None, clock=lambda: 0.0) == 0


def test_rate_limiter_spaces_calls_proactively():
    """Learning the limit by collecting 429s wastes the quota it protects."""
    now = {"t": 0.0}
    limiter = RateLimiter(min_interval_seconds=6.0)

    limiter.wait(sleep=lambda _: None, clock=lambda: now["t"])
    now["t"] = 2.0
    waited = limiter.wait(sleep=lambda _: None, clock=lambda: now["t"])

    assert waited == pytest.approx(4.0)
