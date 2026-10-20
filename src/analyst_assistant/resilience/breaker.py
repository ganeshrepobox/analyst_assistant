"""Retry and circuit breaking for third-party calls. Requirement 5.

Two separate questions, which are easy to conflate:

**Retry** -- "will trying again help?" Only for transient faults. Retrying a 400 or
a SQL syntax error spends quota to learn nothing.

**The breaker** -- "should we even try?" When a dependency is down, retrying every
request individually turns one outage into a slow outage. The breaker fails fast so
the agent can degrade immediately instead of making each user wait out a backoff.

Faults are classified by HTTP status, by a ``retry_after`` hint and by exception
class name rather than by importing from each SDK. That keeps this module free of
provider imports and testable without network libraries. An unrecognised exception
is treated as **permanent** -- the conservative direction, since retrying what we do
not understand is how a cost incident starts.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")

TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
PERMANENT_STATUS = frozenset({400, 401, 403, 404, 409, 422})

#: Exception class names, across SDKs, that denote a transient fault.
TRANSIENT_NAMES = frozenset(
    {
        "ConnectionError",
        "DeadlineExceeded",
        "InternalServerError",
        "ResourceExhausted",
        "ServiceUnavailable",
        "TimeoutError",
        "TooManyRequests",
        "Unavailable",
    }
)


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def retry_after(exc: BaseException) -> float | None:
    """A server-supplied back-off hint, if present.

    Gemini's free tier returns these under load. A server that has told us when to
    come back knows better than our own curve, and ignoring the hint invites
    harder throttling.
    """
    value = getattr(exc, "retry_after", None)
    return float(value) if isinstance(value, (int, float)) else None


def is_transient(exc: BaseException) -> bool:
    """Whether retrying ``exc`` could plausibly succeed."""
    status = _status_of(exc)
    if status is not None:
        if status in TRANSIENT_STATUS:
            return True
        if status in PERMANENT_STATUS:
            return False
    if retry_after(exc) is not None:
        return True
    names = {base.__name__ for base in type(exc).__mro__}
    return bool(names & TRANSIENT_NAMES)


class CircuitOpenError(RuntimeError):
    """Raised instead of calling a dependency that is presumed down."""

    def __init__(self, name: str, retry_in: float) -> None:
        super().__init__(f"{name} is unavailable; retry in {retry_in:.0f}s")
        self.name = name
        self.retry_in = retry_in


@dataclass
class CircuitBreaker:
    """Opens after repeated failures, then allows a probe once the window elapses.

    ``clock`` is injectable so the recovery window is testable without sleeping.
    Shrinking the window to zero instead does not work -- the breaker would allow a
    probe on the same tick, making the open state unobservable.
    """

    name: str = "dependency"
    failure_threshold: int = 3
    recovery_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic

    _failures: int = field(default=0, repr=False)
    _opened_at: float | None = field(default=None, repr=False)

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        return self.clock() - self._opened_at < self.recovery_seconds

    @property
    def retry_in(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self.recovery_seconds - (self.clock() - self._opened_at))

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = self.clock()


def call_with_resilience(
    fn: Callable[[], T],
    *,
    breaker: CircuitBreaker,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with breaker protection and transient-only retry.

    Backoff is jittered: without it, clients that failed together retry together
    and recreate the spike that caused the outage.

    Raises:
        CircuitOpenError: the breaker is open; callers catch this to degrade.
        Exception: the original error, re-raised at once for permanent faults.
    """
    if breaker.is_open:
        raise CircuitOpenError(breaker.name, breaker.retry_in)

    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
        except Exception as exc:
            if not is_transient(exc) or attempt == max_attempts:
                breaker.record_failure()
                raise
            hinted = retry_after(exc)
            sleep(hinted if hinted is not None else random.uniform(0, base_delay * 2**attempt))
        else:
            breaker.record_success()
            return result

    raise AssertionError("unreachable")  # pragma: no cover


def describe_failure(exc: BaseException, *, trace_id: str) -> str:
    """Turn an exception into something an executive can act on.

    Deliberately not the exception text: "403 Forbidden: bigquery.jobs.create
    denied" tells them nothing. They need to know whether to wait, rephrase, or
    call someone -- and the trace id is what makes the last option work.
    """
    if isinstance(exc, CircuitOpenError):
        return (
            f"{exc.name} is temporarily unavailable. Try again in about "
            f"{exc.retry_in:.0f} seconds. (trace {trace_id})"
        )
    if _status_of(exc) in (401, 403):
        return (
            "I am not authorised to read that data. This needs an administrator "
            f"rather than a different question. (trace {trace_id})"
        )
    if is_transient(exc):
        return (
            "The data service is busy. I retried without luck -- please try again "
            f"shortly. (trace {trace_id})"
        )
    return (
        "Something went wrong answering that, and nothing was changed. Quote "
        f"trace {trace_id} if you report it."
    )
