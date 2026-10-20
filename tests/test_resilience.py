"""Budgets, failure classification and the circuit breaker.

Requirement 5 asks the agent to self-correct "without inflating costs", and those
clauses pull against each other -- every retry is another call. These tests pin the
boundary.
"""

from __future__ import annotations

import contextlib

from analyst_assistant.resilience.breaker import (
    CircuitBreaker,
    CircuitOpenError,
    call_with_resilience,
    describe_failure,
    is_transient,
)
from analyst_assistant.resilience.budget import Budget, RepairVerdict

# Fakes that mimic SDK exception shapes without importing any SDK.


class FakeHttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeRateLimit(Exception):
    def __init__(self, seconds: float) -> None:
        super().__init__("rate limited")
        self.status_code = 429
        self.retry_after = seconds


class ServiceUnavailable(Exception):
    """Named to match google-api-core, but defined locally."""


class FakeClock:
    """Controllable clock, so recovery windows are tested without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_llm_calls_are_capped():
    budget = Budget(max_llm_calls=2)
    budget.charge_llm(tokens_in=10, tokens_out=5)
    assert budget.can_call_llm()
    budget.charge_llm(tokens_in=10, tokens_out=5)
    assert not budget.can_call_llm()
    assert budget.exhausted_reason() is not None


def test_query_is_checked_against_an_estimate_before_running():
    """Checked before execution; afterwards the bytes are already paid for."""
    budget = Budget(max_bytes_billed=1000)
    assert budget.can_run_query(900)
    assert not budget.can_run_query(1001)


def test_bytes_accumulate_across_a_turn():
    budget = Budget(max_bytes_billed=1000)
    budget.charge_query(600)
    assert budget.can_run_query(400)
    assert not budget.can_run_query(401)
    assert budget.remaining_bytes == 400


def test_first_attempt_does_not_consume_the_repair_allowance():
    budget = Budget(max_repairs=2)
    assert budget.register_attempt("SELECT 1 FROM s_orders") is RepairVerdict.ALLOWED
    assert budget.repairs == 0


def test_repairs_are_capped():
    budget = Budget(max_repairs=2)
    for i in range(3):
        assert budget.register_attempt(f"SELECT {i} FROM s_orders") is RepairVerdict.ALLOWED
    assert budget.register_attempt("SELECT 9 FROM s_orders") is RepairVerdict.EXHAUSTED


def test_a_repair_repeating_a_failed_query_is_rejected_unrun():
    """The real waste: a 'repair' equivalent to what just failed.

    Rejecting costs nothing; executing costs an attempt and quota to learn the same
    thing twice. Cosmetic rewrites count as duplicates.
    """
    budget = Budget(max_repairs=2)
    budget.register_attempt("SELECT revenue FROM s_orders")

    assert budget.register_attempt("select   revenue\nfrom s_orders") is RepairVerdict.DUPLICATE
    assert budget.repairs == 0, "a rejected duplicate must not consume an attempt"


def test_a_genuinely_different_query_is_allowed():
    budget = Budget()
    budget.register_attempt("SELECT revenue FROM s_orders")
    assert (
        budget.register_attempt("SELECT SUM(sale_price) FROM s_order_items")
        is RepairVerdict.ALLOWED
    )


def test_within_budget_there_is_no_reason_to_stop():
    assert Budget().exhausted_reason() is None


def test_exhaustion_is_explained_in_terms_a_user_can_act_on():
    """Exceeding a budget is a degraded answer, not a stack trace."""
    budget = Budget(max_repairs=0)
    budget.register_attempt("SELECT 1")
    budget.register_attempt("SELECT 2")
    assert "stopped" in (budget.exhausted_reason() or "")


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


def test_unknown_exceptions_are_treated_as_permanent():
    """The conservative direction: retrying what we do not understand is a cost risk."""
    assert not is_transient(ValueError("who knows"))


# --------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------


def test_transient_failure_is_retried_until_it_succeeds():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ServiceUnavailable("try later")
        return "ok"

    result = call_with_resilience(flaky, breaker=CircuitBreaker(), sleep=lambda _: None)
    assert result == "ok"
    assert attempts["n"] == 3


def test_permanent_failure_is_not_retried():
    attempts = {"n": 0}

    def broken():
        attempts["n"] += 1
        raise FakeHttpError(400)

    with contextlib.suppress(FakeHttpError):
        call_with_resilience(broken, breaker=CircuitBreaker(), sleep=lambda _: None)
    assert attempts["n"] == 1


def test_server_supplied_retry_after_is_honoured_over_our_backoff():
    """A server that said when to come back knows better than our curve."""
    slept: list[float] = []
    attempts = {"n": 0}

    def rate_limited():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise FakeRateLimit(4.0)
        return "ok"

    call_with_resilience(rate_limited, breaker=CircuitBreaker(), sleep=slept.append)
    assert slept == [4.0]


# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------


def test_breaker_opens_after_the_threshold_and_fails_fast():
    """One outage must not become a slow outage for every user."""
    breaker = CircuitBreaker("BigQuery", failure_threshold=2)
    breaker.record_failure()
    assert not breaker.is_open
    breaker.record_failure()
    assert breaker.is_open

    called = {"n": 0}

    def should_not_run():
        called["n"] += 1

    with contextlib.suppress(CircuitOpenError):
        call_with_resilience(should_not_run, breaker=breaker, sleep=lambda _: None)
    assert called["n"] == 0, "an open breaker must not call the dependency"


def test_breaker_allows_a_probe_once_the_recovery_window_elapses():
    clock = FakeClock()
    breaker = CircuitBreaker("bq", failure_threshold=1, recovery_seconds=30, clock=clock)
    breaker.record_failure()

    assert breaker.is_open
    clock.advance(29)
    assert breaker.is_open
    assert breaker.retry_in == 1.0

    clock.advance(1)
    assert not breaker.is_open


def test_a_success_closes_the_breaker():
    breaker = CircuitBreaker("bq", failure_threshold=1)
    breaker.record_failure()
    breaker.record_success()
    assert not breaker.is_open


# --------------------------------------------------------------------------
# User-facing failure messages
# --------------------------------------------------------------------------


def test_open_breaker_message_tells_the_user_when_to_retry():
    message = describe_failure(CircuitOpenError("BigQuery", 25.0), trace_id="abc123")
    assert "BigQuery" in message and "25" in message and "abc123" in message


def test_authorization_failure_points_at_an_administrator():
    """Rephrasing will not fix a permissions problem, so do not imply it might."""
    assert "administrator" in describe_failure(FakeHttpError(403), trace_id="abc123")


def test_failure_messages_carry_the_trace_id_and_hide_raw_exception_text():
    """'HTTP 403' is not actionable for an executive; the trace id is."""
    message = describe_failure(FakeHttpError(403), trace_id="abc123")
    assert "abc123" in message
    assert "HTTP 403" not in message
