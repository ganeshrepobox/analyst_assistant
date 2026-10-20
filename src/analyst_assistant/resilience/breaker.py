"""Circuit breakers and transient-failure classification. Requirement 5.

Two distinct concerns that are easy to conflate:

**Retry** answers "will trying again help?" Only for transient faults. Retrying a
400 or a SQL syntax error is pure waste -- it will fail identically, having spent
quota and the user's patience to learn nothing.

**The breaker** answers "should we even try?" When a dependency is down, retrying
*each* request individually turns one outage into a slow, expensive outage. The
breaker fails fast and lets the agent degrade immediately -- serving a labelled
cached answer, or saying plainly that the warehouse is unavailable -- instead of
making every user wait out a backoff ladder.

Classification is duck-typed on purpose
---------------------------------------
Rather than importing exception classes from every SDK, faults are classified by
HTTP status, by a ``retry_after`` hint, and by class name. That keeps this module
free of provider imports, makes it unit-testable without network libraries
installed, and means a new dependency works without modification. The trade-off is
that an unrecognised exception is treated as **permanent** -- the conservative
choice, since retrying something we do not understand is how a cost incident
starts.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar

T = TypeVar("T")

#: Statuses where the same request may succeed shortly afterwards.
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: Statuses that will fail identically no matter how often they are retried.
PERMANENT_STATUS = frozenset({400, 401, 403, 404, 409, 413, 422})

#: Exception class names, across SDKs, that denote a transient fault.
TRANSIENT_NAMES = frozenset(
    {
        "ConnectionError",
        "ConnectionResetError",
        "DeadlineExceeded",
        "GatewayTimeout",
        "InternalServerError",
        "ResourceExhausted",
        "ServerError",
        "ServiceUnavailable",
        "SSLError",
        "Timeout",
        "TimeoutError",
        "TooManyRequests",
        "TransportError",
        "Unavailable",
        "ReadTimeout",
        "RemoteProtocolError",
    }
)


class BreakerState(StrEnum):
    CLOSED = "closed"
    """Normal operation."""

    OPEN = "open"
    """Failing fast. The dependency is presumed down."""

    HALF_OPEN = "half_open"
    """Probing with a single request to see whether it recovered."""


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        # google-api-core wraps the numeric code in an enum-like object.
        inner = getattr(value, "value", None)
        if isinstance(inner, int):
            return inner
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_transient(exc: BaseException) -> bool:
    """Whether retrying ``exc`` could plausibly succeed.

    Unrecognised exceptions are treated as permanent. Retrying what we do not
    understand is how a runaway cost incident begins.
    """
    status = _status_of(exc)
    if status is not None:
        if status in TRANSIENT_STATUS:
            return True
        if status in PERMANENT_STATUS:
            return False

    if retry_after(exc) is not None:
        return True

    names = {type(exc).__name__} | {base.__name__ for base in type(exc).__mro__}
    return bool(names & TRANSIENT_NAMES)


def retry_after(exc: BaseException) -> float | None:
    """Honour a server-supplied back-off hint if there is one.

    Gemini's free tier returns these under load, and they are not hypothetical. A
    server that has told us when to come back knows better than our own backoff
    curve, and ignoring the hint is how a client gets throttled harder.
    """
    for attr in ("retry_after", "retry_delay"):
        value = getattr(exc, attr, None)
        if isinstance(value, (int, float)):
            return float(value)
        seconds = getattr(value, "seconds", None)  # protobuf Duration
        if isinstance(seconds, (int, float)):
            return float(seconds)

    headers = getattr(getattr(exc, "response", None), "headers", None)
    if isinstance(headers, dict):
        raw = headers.get("Retry-After") or headers.get("retry-after")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None
    return None


class CircuitOpenError(RuntimeError):
    """Raised instead of calling a dependency that is presumed down."""

    def __init__(self, name: str, retry_in: float) -> None:
        super().__init__(f"{name} is unavailable; retrying in {retry_in:.0f}s")
        self.name = name
        self.retry_in = retry_in


@dataclass
class CircuitBreaker:
    """One breaker per dependency.

    Per-dependency rather than global: BigQuery being down is no reason to stop
    calling Gemini, and a single shared breaker would couple unrelated outages.
    """

    name: str
    failure_threshold: int = 3
    recovery_seconds: float = 30.0
    half_open_successes: int = 1
    clock: Callable[[], float] = time.monotonic
    """Injectable monotonic clock.

    Present so that recovery-window behaviour is testable without sleeping. Testing
    it by shrinking the window to zero does not work: the breaker then re-enters
    half-open instantly and the open state becomes unobservable."""

    _failures: int = field(default=0, repr=False)
    _successes: int = field(default=0, repr=False)
    _opened_at: float | None = field(default=None, repr=False)

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self.clock() - self._opened_at >= self.recovery_seconds:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    @property
    def retry_in(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self.recovery_seconds - (self.clock() - self._opened_at))

    def allow(self) -> bool:
        """Whether a call may proceed. Half-open permits a single probe."""
        return self.state is not BreakerState.OPEN

    def record_success(self) -> None:
        if self.state is BreakerState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self.half_open_successes:
                self._reset()
            return
        self._failures = 0

    def record_failure(self) -> None:
        """Count a failure, opening the breaker at the threshold.

        A failure while half-open re-opens immediately: the probe was the test, and
        it said no.
        """
        if self.state is BreakerState.HALF_OPEN:
            self._open()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._open()

    def _open(self) -> None:
        self._opened_at = self.clock()
        self._successes = 0

    def _reset(self) -> None:
        self._opened_at = None
        self._failures = 0
        self._successes = 0


def call_with_resilience(
    fn: Callable[[], T],
    *,
    breaker: CircuitBreaker,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Invoke ``fn`` with breaker protection and transient-only retry.

    Exponential backoff with full jitter. Jitter matters because without it a set
    of clients that failed together retries together, re-creating the spike that
    caused the outage.

    Args:
        fn: Zero-argument callable performing the dependency call.
        breaker: The breaker guarding this dependency.
        max_attempts: Total attempts, including the first.
        sleep: Injectable so tests do not actually wait.

    Raises:
        CircuitOpenError: If the breaker is open. Callers catch this to degrade.
        Exception: The last error, if attempts are exhausted or the fault is
            permanent -- permanent faults are re-raised immediately, unretried.
    """
    if not breaker.allow():
        raise CircuitOpenError(breaker.name, breaker.retry_in)

    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
        except Exception as exc:
            last = exc
            if not is_transient(exc):
                breaker.record_failure()
                raise
            if attempt == max_attempts:
                breaker.record_failure()
                raise
            hinted = retry_after(exc)
            delay = (
                hinted
                if hinted is not None
                else random.uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1)))
            )
            sleep(delay)
        else:
            breaker.record_success()
            return result

    assert last is not None  # unreachable: the loop either returns or raises
    raise last


@dataclass
class BreakerRegistry:
    """All breakers, so ``/debug`` and the metrics snapshot can report their state."""

    recovery_seconds: float = 30.0
    failure_threshold: int = 3
    clock: Callable[[], float] = time.monotonic
    _breakers: dict[str, CircuitBreaker] = field(default_factory=dict)

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name,
                failure_threshold=self.failure_threshold,
                recovery_seconds=self.recovery_seconds,
                clock=self.clock,
            )
        return self._breakers[name]

    def states(self) -> dict[str, str]:
        return {name: str(b.state) for name, b in sorted(self._breakers.items())}

    def any_open(self) -> bool:
        return any(b.state is BreakerState.OPEN for b in self._breakers.values())


def describe_failure(exc: BaseException, *, trace_id: str) -> str:
    """Turn an exception into something an executive can act on.

    Deliberately not the exception text: an executive cannot use "403 Forbidden:
    bigquery.jobs.create denied". They need to know whether to wait, rephrase, or
    call someone -- and the trace id is what makes the last option work.
    """
    if isinstance(exc, CircuitOpenError):
        return (
            f"{exc.name} is temporarily unavailable, so I could not check the latest "
            f"figures. I will retry automatically; try again in about "
            f"{exc.retry_in:.0f} seconds. (trace {trace_id})"
        )

    status = _status_of(exc)
    if status in (401, 403):
        return (
            "I am not authorised to read that data. This needs an administrator rather "
            f"than a different question. (trace {trace_id})"
        )
    if status == 429 or is_transient(exc):
        return (
            "The data service is busy right now. I retried a few times without luck -- "
            f"please try again shortly. (trace {trace_id})"
        )
    return (
        "Something went wrong while answering that, and nothing was changed. Quote "
        f"trace {trace_id} if you report it."
    )
