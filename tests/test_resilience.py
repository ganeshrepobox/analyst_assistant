"""Tests for budgets, failure classification and circuit breakers.

Requirement 5 asks the agent to self-correct "without inflating costs", and those
clauses pull against each other -- every retry is another call. These tests pin the
boundary between the two.
"""

from __future__ import annotations

import contextlib

from analyst_assistant.resilience.breaker import (
    BreakerRegistry,
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
    call_with_resilience,
    describe_failure,
    is_transient,
    retry_after,
)
from analyst_assistant.resilience.budget import (
    Budget,
    BudgetBreach,
    RepairVerdict,
    fingerprint_sql,
)

# --------------------------------------------------------------------------
# Fakes that mimic SDK exception shapes without importing any SDK
# --------------------------------------------------------------------------


class FakeHttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeRateLimit(Exception):
    def __init__(self, retry_after_seconds: float) -> None:
        super().__init__("rate limited")
        self.status_code = 429
        self.retry_after = retry_after_seconds


class ServiceUnavailable(Exception):
    """Named to match google-api-core, but defined locally."""


class FakeResponse:
    def __init__(self, headers):
        self.headers = headers
        self.status_code = 429


class FakeHeaderRateLimit(Exception):
    def __init__(self, seconds):
        super().__init__("rate limited")
        self.response = FakeResponse({"Retry-After": str(seconds)})


# --------------------------------------------------------------------------
# Budget: LLM and query ceilings
# --------------------------------------------------------------------------


def test_llm_calls_are_capped():
    budget = Budget(max_llm_calls=2)
    assert budget.can_call_llm()
    budget.charge_llm(tokens_in=10, tokens_out=5)
    assert budget.can_call_llm()
    budget.charge_llm(tokens_in=10, tokens_out=5)
    assert not budget.can_call_llm()
    assert budget.breach() is BudgetBreach.LLM_CALLS


def test_tokens_are_capped_independently_of_call_count():
    """One huge call can breach the budget before the call ceiling is near."""
    budget = Budget(max_llm_calls=10, max_tokens=1000)
    budget.charge_llm(tokens_in=900, tokens_out=200)
    assert not budget.can_call_llm()
    assert budget.breach() is BudgetBreach.TOKENS


def test_query_is_checked_against_a_dry_run_estimate_before_running():
    """Checked before execution, the only point at which the answer still matters."""
    budget = Budget(max_bytes_billed=1000)
    assert budget.can_run_query(900)
    assert not budget.can_run_query(1001)


def test_bytes_accumulate_across_queries_in_a_turn():
    budget = Budget(max_bytes_billed=1000)
    budget.charge_query(600)
    assert budget.can_run_query(400)
    assert not budget.can_run_query(401)
    assert budget.remaining_bytes == 400


def test_negative_or_absent_estimates_are_treated_as_zero():
    budget = Budget(max_bytes_billed=1000)
    budget.charge_query(-5)
    assert budget.bytes_billed == 0
    assert budget.can_run_query(-1)


# --------------------------------------------------------------------------
# Budget: the repair loop
# --------------------------------------------------------------------------


def test_first_attempt_does_not_consume_the_repair_allowance():
    budget = Budget(max_repairs=2)
    assert budget.register_attempt("SELECT 1 FROM s_orders") is RepairVerdict.ALLOWED
    assert budget.repairs == 0


def test_repairs_are_capped():
    budget = Budget(max_repairs=2)
    assert budget.register_attempt("SELECT 1 FROM s_orders") is RepairVerdict.ALLOWED
    assert budget.register_attempt("SELECT 2 FROM s_orders") is RepairVerdict.ALLOWED
    assert budget.register_attempt("SELECT 3 FROM s_orders") is RepairVerdict.ALLOWED
    assert budget.register_attempt("SELECT 4 FROM s_orders") is RepairVerdict.EXHAUSTED
    assert budget.repairs == 2


def test_a_repair_that_repeats_a_failed_query_is_rejected_unrun():
    """The real waste case: a 'repair' equivalent to what just failed.

    Rejecting it costs nothing, where executing it costs an attempt and quota to
    learn the same thing twice.
    """
    budget = Budget(max_repairs=2)
    budget.register_attempt("SELECT revenue FROM s_orders")
    assert (
        budget.register_attempt("SELECT revenue FROM s_orders") is RepairVerdict.DUPLICATE
    )
    assert budget.repairs == 0, "a rejected duplicate must not consume an attempt"


def test_cosmetic_rewrites_count_as_duplicates():
    budget = Budget()
    budget.register_attempt("SELECT  revenue\n  FROM s_orders")
    assert budget.register_attempt("select revenue from s_orders") is RepairVerdict.DUPLICATE


def test_a_genuinely_different_query_is_allowed():
    budget = Budget()
    budget.register_attempt("SELECT revenue FROM s_orders")
    verdict = budget.register_attempt("SELECT SUM(sale_price) FROM s_order_items")
    assert verdict is RepairVerdict.ALLOWED


def test_fingerprint_is_stable_and_discriminating():
    assert fingerprint_sql("SELECT 1") == fingerprint_sql("select   1 ")
    assert fingerprint_sql("SELECT 1") != fingerprint_sql("SELECT 2")


# --------------------------------------------------------------------------
# Budget: plan steps and reporting
# --------------------------------------------------------------------------


def test_plan_steps_are_capped():
    budget = Budget(max_plan_steps=2)
    budget.charge_step()
    assert budget.can_take_step()
    budget.charge_step()
    assert not budget.can_take_step()


def test_within_budget_there_is_no_breach_and_no_message():
    budget = Budget()
    assert budget.breach() is None
    assert budget.explain_breach() is None


def test_breach_is_explained_in_terms_a_user_can_act_on():
    """Exceeding a budget is a degraded answer, not a stack trace."""
    budget = Budget(max_repairs=0)
    budget.register_attempt("SELECT 1")
    budget.register_attempt("SELECT 2")
    message = budget.explain_breach()
    assert message is not None
    assert "two attempts" in message or "stopped" in message


def test_snapshot_reports_consumption():
    budget = Budget()
    budget.charge_llm(tokens_in=100, tokens_out=50)
    budget.charge_query(2048)
    assert budget.snapshot() == {
        "llm_calls": 1,
        "tokens": 150,
        "bytes_billed": 2048,
        "repairs": 0,
        "plan_steps": 0,
    }


# --------------------------------------------------------------------------
# Transient classification
# --------------------------------------------------------------------------


def test_server_side_statuses_are_transient():
    for status in (408, 429, 500, 502, 503, 504):
        assert is_transient(FakeHttpError(status)), status


def test_client_side_statuses_are_permanent():
    """Retrying a 400 spends quota to learn nothing."""
    for status in (400, 401, 403, 404, 409, 422):
        assert not is_transient(FakeHttpError(status)), status


def test_well_known_exception_names_are_transient():
    assert is_transient(ServiceUnavailable("down"))
    assert is_transient(TimeoutError("slow"))
    assert is_transient(ConnectionError("reset"))


def test_unknown_exceptions_are_treated_as_permanent():
    """The conservative direction: retrying what we do not understand is a cost risk."""
    assert not is_transient(ValueError("who knows"))


def test_sql_errors_are_not_retried_blindly():
    """A syntax error needs the repair loop, which rewrites, not a retry."""
    assert not is_transient(FakeHttpError(400))


def test_retry_after_is_read_from_an_attribute():
    assert retry_after(FakeRateLimit(12.5)) == 12.5


def test_retry_after_is_read_from_a_header():
    assert retry_after(FakeHeaderRateLimit(7)) == 7.0


def test_a_retry_after_hint_alone_marks_a_fault_transient():
    class Hinted(Exception):
        retry_after = 3

    assert is_transient(Hinted())


# --------------------------------------------------------------------------
# Retry behaviour
# --------------------------------------------------------------------------


def test_transient_failure_is_retried_until_it_succeeds():
    breaker = CircuitBreaker("bq")
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ServiceUnavailable("try later")
        return "ok"

    assert call_with_resilience(flaky, breaker=breaker, sleep=lambda _: None) == "ok"
    assert attempts["n"] == 3


def test_permanent_failure_is_not_retried():
    breaker = CircuitBreaker("bq")
    attempts = {"n": 0}

    def broken():
        attempts["n"] += 1
        raise FakeHttpError(400)

    with contextlib.suppress(FakeHttpError):
        call_with_resilience(broken, breaker=breaker, sleep=lambda _: None)
    assert attempts["n"] == 1, "a permanent fault must fail on the first attempt"


def test_server_supplied_retry_after_is_honoured_over_our_backoff():
    """A server that said when to come back knows better than our curve."""
    breaker = CircuitBreaker("gemini")
    slept: list[float] = []
    attempts = {"n": 0}

    def rate_limited():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FakeRateLimit(4.0)
        return "ok"

    call_with_resilience(rate_limited, breaker=breaker, sleep=slept.append)
    assert slept == [4.0]


def test_retry_delays_are_jittered():
    """Without jitter, clients that failed together retry together."""
    breaker = CircuitBreaker("bq", failure_threshold=99)
    delays: list[float] = []

    def always_down():
        raise ServiceUnavailable("down")

    for _ in range(8):
        with contextlib.suppress(ServiceUnavailable):
            call_with_resilience(
                always_down, breaker=breaker, max_attempts=2, sleep=delays.append
            )

    assert len(set(delays)) > 1, f"delays look deterministic: {delays}"


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------


def test_breaker_opens_after_the_failure_threshold():
    breaker = CircuitBreaker("bq", failure_threshold=3)
    assert breaker.state is BreakerState.CLOSED
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allow()


def test_an_open_breaker_fails_fast_without_calling_the_dependency():
    """One outage must not become a slow outage for every user."""
    breaker = CircuitBreaker("bq", failure_threshold=1)
    breaker.record_failure()
    called = {"n": 0}

    def should_not_run():
        called["n"] += 1
        return "ok"

    try:
        call_with_resilience(should_not_run, breaker=breaker, sleep=lambda _: None)
        raise AssertionError("expected CircuitOpenError")
    except CircuitOpenError:
        pass
    assert called["n"] == 0


class FakeClock:
    """Controllable monotonic clock, so recovery windows are tested without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_breaker_stays_open_until_the_recovery_window_elapses():
    clock = FakeClock()
    breaker = CircuitBreaker("bq", failure_threshold=1, recovery_seconds=30, clock=clock)
    breaker.record_failure()

    assert breaker.state is BreakerState.OPEN
    clock.advance(29)
    assert breaker.state is BreakerState.OPEN
    assert breaker.retry_in == 1.0

    clock.advance(1)
    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.allow(), "half-open must permit a probe"


def test_a_successful_probe_closes_the_breaker():
    clock = FakeClock()
    breaker = CircuitBreaker("bq", failure_threshold=1, recovery_seconds=30, clock=clock)
    breaker.record_failure()
    clock.advance(30)

    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_a_failed_probe_reopens_immediately():
    """The probe was the test, and it said no.

    Note this cannot be tested with recovery_seconds=0: the breaker would re-enter
    half-open on the same tick and the reopen would be unobservable.
    """
    clock = FakeClock()
    breaker = CircuitBreaker("bq", failure_threshold=1, recovery_seconds=30, clock=clock)
    breaker.record_failure()
    clock.advance(30)
    assert breaker.state is BreakerState.HALF_OPEN

    breaker.record_failure()
    assert breaker.state is BreakerState.OPEN
    assert breaker.retry_in == 30.0, "the recovery window restarts from the failed probe"


def test_successes_reset_the_failure_count_while_closed():
    breaker = CircuitBreaker("bq", failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is BreakerState.CLOSED


def test_breakers_are_isolated_per_dependency():
    """BigQuery being down is no reason to stop calling Gemini."""
    registry = BreakerRegistry(failure_threshold=1)
    registry.get("bigquery").record_failure()

    assert registry.get("bigquery").state is BreakerState.OPEN
    assert registry.get("gemini").state is BreakerState.CLOSED
    assert registry.any_open()
    assert registry.states()["gemini"] == "closed"


# --------------------------------------------------------------------------
# User-facing failure messages
# --------------------------------------------------------------------------


def test_open_breaker_message_tells_the_user_when_to_retry():
    message = describe_failure(CircuitOpenError("BigQuery", 25.0), trace_id="abc123")
    assert "BigQuery" in message
    assert "25" in message
    assert "abc123" in message


def test_authorization_failure_points_at_an_administrator():
    """Rephrasing will not fix a permissions problem, so do not imply it might."""
    message = describe_failure(FakeHttpError(403), trace_id="abc123")
    assert "administrator" in message


def test_every_failure_message_carries_the_trace_id():
    for exc in (FakeHttpError(500), FakeHttpError(403), ValueError("x")):
        assert "abc123" in describe_failure(exc, trace_id="abc123")


def test_failure_messages_do_not_leak_raw_exception_text():
    """'403 Forbidden: bigquery.jobs.create denied' is not actionable for an exec."""
    message = describe_failure(FakeHttpError(403), trace_id="abc123")
    assert "HTTP 403" not in message
