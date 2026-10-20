"""Per-turn budget ceilings. Implements the cost half of requirement 5.

The requirement is "self-correct before giving up, without inflating costs". Those
two clauses pull against each other: every retry is another LLM call, and an agent
that retries until it succeeds has unbounded cost. So the budget is explicit state,
carried through the graph, and checked before each spend rather than measured
afterwards.

Why a budget object rather than a retry counter
-----------------------------------------------
A counter bounds attempts. It does not bound *cost*, because one attempt on a large
query can bill more than a hundred cheap ones. Four independent ceilings are
tracked -- LLM calls, tokens, bytes billed, and repair attempts -- because any one
of them can be the thing that runs away.

The must-change guard
---------------------
The interesting failure is not an infinite loop; it is a model that "repairs" SQL by
returning something equivalent to what just failed. Attempts are consumed, nothing
improves, and quota burns. So every candidate query is fingerprinted, and a repair
that reproduces a fingerprint already tried is rejected without being executed --
which costs nothing and fails fast instead of slowly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum

#: Defaults sized for the Gemini free tier: roughly 10 requests/minute and 250/day.
#: A conventional six-call-per-turn agent exhausts a day's quota in about 40 turns,
#: so the ceiling is deliberately tight enough to notice when a loop misbehaves.
DEFAULT_MAX_LLM_CALLS = 6
DEFAULT_MAX_TOKENS = 60_000
DEFAULT_MAX_BYTES_BILLED = 2_000_000_000  # 2 GB; the free tier allows 1 TB/month
DEFAULT_MAX_REPAIRS = 2
DEFAULT_MAX_PLAN_STEPS = 3


class RepairVerdict(StrEnum):
    """Whether another repair attempt may proceed."""

    ALLOWED = "allowed"
    DUPLICATE = "duplicate"
    """The candidate matches a query already tried. Reject without executing."""
    EXHAUSTED = "exhausted"


class BudgetBreach(StrEnum):
    """Which ceiling was reached."""

    LLM_CALLS = "llm_calls"
    TOKENS = "tokens"
    BYTES_BILLED = "bytes_billed"
    REPAIRS = "repairs"
    PLAN_STEPS = "plan_steps"


def fingerprint_sql(sql: str) -> str:
    """Fingerprint a query so that cosmetic rewrites do not count as progress.

    Whitespace and case are normalised; a model that re-indents its failing query
    and resubmits it has not repaired anything.

    Deliberately *not* a semantic fingerprint. Normalising aliases or clause order
    would catch more no-op repairs, but it risks treating a genuine fix as a
    duplicate -- and wrongly blocking a correct repair is worse than allowing one
    wasted attempt.
    """
    normalised = re.sub(r"\s+", " ", sql.strip().lower())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


@dataclass
class Budget:
    """Ceilings and consumption for a single user turn.

    Lives in the graph state, so it survives checkpointing and a resumed turn
    cannot start spending again from zero.
    """

    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_bytes_billed: int = DEFAULT_MAX_BYTES_BILLED
    max_repairs: int = DEFAULT_MAX_REPAIRS
    max_plan_steps: int = DEFAULT_MAX_PLAN_STEPS

    llm_calls: int = 0
    tokens: int = 0
    bytes_billed: int = 0
    repairs: int = 0
    plan_steps: int = 0
    tried_fingerprints: list[str] = field(default_factory=list)

    # -- LLM ----------------------------------------------------------------

    def can_call_llm(self) -> bool:
        return self.llm_calls < self.max_llm_calls and self.tokens < self.max_tokens

    def charge_llm(self, *, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.llm_calls += 1
        self.tokens += tokens_in + tokens_out

    # -- BigQuery -----------------------------------------------------------

    def can_run_query(self, estimated_bytes: int) -> bool:
        """Check a dry-run estimate against the remaining byte budget.

        Checked *before* execution, which is the only point at which the answer
        still matters -- BigQuery bills on bytes scanned, so discovering the
        overrun afterwards means having already paid for it.
        """
        return self.bytes_billed + max(0, estimated_bytes) <= self.max_bytes_billed

    def charge_query(self, bytes_billed: int) -> None:
        self.bytes_billed += max(0, bytes_billed)

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.max_bytes_billed - self.bytes_billed)

    # -- repair loop --------------------------------------------------------

    def register_attempt(self, sql: str) -> RepairVerdict:
        """Record a candidate query and decide whether it may run.

        The first call for a turn is the original attempt and does not consume the
        repair allowance; subsequent calls do.
        """
        candidate = fingerprint_sql(sql)
        if candidate in self.tried_fingerprints:
            return RepairVerdict.DUPLICATE

        if self.tried_fingerprints:  # not the first attempt, so this is a repair
            if self.repairs >= self.max_repairs:
                return RepairVerdict.EXHAUSTED
            self.repairs += 1

        self.tried_fingerprints.append(candidate)
        return RepairVerdict.ALLOWED

    # -- multi-step planning ------------------------------------------------

    def can_take_step(self) -> bool:
        return self.plan_steps < self.max_plan_steps

    def charge_step(self) -> None:
        self.plan_steps += 1

    # -- reporting ----------------------------------------------------------

    def breach(self) -> BudgetBreach | None:
        """The first ceiling reached, or ``None`` while within budget."""
        if self.llm_calls >= self.max_llm_calls:
            return BudgetBreach.LLM_CALLS
        if self.tokens >= self.max_tokens:
            return BudgetBreach.TOKENS
        if self.bytes_billed >= self.max_bytes_billed:
            return BudgetBreach.BYTES_BILLED
        if self.repairs >= self.max_repairs:
            return BudgetBreach.REPAIRS
        if self.plan_steps >= self.max_plan_steps:
            return BudgetBreach.PLAN_STEPS
        return None

    def explain_breach(self) -> str | None:
        """Why the turn stopped, in terms an executive can act on.

        Exceeding a budget is a *degraded answer*, not an error: the user is told
        what was completed and what was skipped, rather than being handed a stack
        trace or -- worse -- a partial result presented as complete.
        """
        breach = self.breach()
        if breach is None:
            return None
        return {
            BudgetBreach.LLM_CALLS: (
                "I reached this question's processing limit. Here is what I completed; "
                "ask a narrower question for the rest."
            ),
            BudgetBreach.TOKENS: (
                "This question grew too large to finish in one go. Try splitting it."
            ),
            BudgetBreach.BYTES_BILLED: (
                "Answering this fully would scan more data than one question is allowed. "
                "Narrowing the date range would bring it in range."
            ),
            BudgetBreach.REPAIRS: (
                "I could not build a working query for this after two attempts, so I "
                "stopped rather than keep trying."
            ),
            BudgetBreach.PLAN_STEPS: (
                "This needed more analysis steps than one question allows. Here is what "
                "I found in the first few."
            ),
        }[breach]

    def snapshot(self) -> dict[str, int]:
        """Consumption, for the cost line and the event log."""
        return {
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "bytes_billed": self.bytes_billed,
            "repairs": self.repairs,
            "plan_steps": self.plan_steps,
        }
