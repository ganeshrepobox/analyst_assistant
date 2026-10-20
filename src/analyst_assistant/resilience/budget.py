"""Per-turn spending ceilings. The cost half of requirement 5.

The requirement asks the agent to "self-correct before giving up, without inflating
costs". Those clauses pull against each other -- every retry is another call -- so
the budget is explicit state carried through the graph and checked *before* each
spend rather than measured afterwards.

Three ceilings, each guarding a different runaway:

* ``max_llm_calls`` -- the Gemini free tier allows about 10 requests per minute and
  250 per day, so a loop that misbehaves is noticed in one turn rather than one day.
* ``max_bytes_billed`` -- BigQuery bills on bytes scanned, and one large query can
  cost more than a hundred small ones, so an attempt counter alone is no protection.
* ``max_repairs`` -- bounds the self-correction loop.

The must-change guard
---------------------
The interesting waste is not an infinite loop; it is a model that "repairs" SQL by
returning something equivalent to what just failed. Every candidate is therefore
fingerprinted, and a repeat is rejected without being executed -- which costs
nothing, and fails fast instead of slowly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum

#: Sized for the Gemini free tier. See the module docstring.
DEFAULT_MAX_LLM_CALLS = 6
DEFAULT_MAX_BYTES_BILLED = 2_000_000_000  # 2 GB
DEFAULT_MAX_REPAIRS = 2

#: Cap on sub-questions for one multi-step request.
MAX_PLAN_STEPS = 3


class RepairVerdict(StrEnum):
    """Whether another attempt at a query may proceed."""

    ALLOWED = "allowed"
    DUPLICATE = "duplicate"
    """Matches a query already tried. Reject without executing."""
    EXHAUSTED = "exhausted"


def fingerprint_sql(sql: str) -> str:
    """Fingerprint a query so cosmetic rewrites do not count as progress.

    Textual, not semantic. Normalising aliases or clause order would catch more
    no-op repairs but risks treating a genuine fix as a duplicate, and wrongly
    blocking a correct repair is the worse error.
    """
    return hashlib.sha256(re.sub(r"\s+", " ", sql.strip().lower()).encode()).hexdigest()[:16]


@dataclass
class Budget:
    """Ceilings and consumption for a single turn.

    Lives in graph state, so a resumed turn cannot start spending again from zero.
    """

    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    max_bytes_billed: int = DEFAULT_MAX_BYTES_BILLED
    max_repairs: int = DEFAULT_MAX_REPAIRS

    llm_calls: int = 0
    tokens: int = 0  # tracked for the cost line, not a ceiling
    bytes_billed: int = 0
    repairs: int = 0
    plan_steps: int = 0
    tried: list[str] = field(default_factory=list)

    # -- LLM ---------------------------------------------------------------

    def can_call_llm(self) -> bool:
        return self.llm_calls < self.max_llm_calls

    def charge_llm(self, *, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.llm_calls += 1
        self.tokens += tokens_in + tokens_out

    # -- BigQuery ----------------------------------------------------------

    def can_run_query(self, estimated_bytes: int) -> bool:
        """Test a dry-run estimate against the remaining allowance.

        Checked before execution, the only point at which the answer still
        matters: afterwards the bytes are already paid for.
        """
        return self.bytes_billed + max(0, estimated_bytes) <= self.max_bytes_billed

    def charge_query(self, bytes_billed: int) -> None:
        self.bytes_billed += max(0, bytes_billed)

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.max_bytes_billed - self.bytes_billed)

    # -- repair loop -------------------------------------------------------

    def register_attempt(self, sql: str) -> RepairVerdict:
        """Record a candidate query and decide whether it may run.

        The first call of a turn is the original attempt and does not consume the
        repair allowance; later calls do.
        """
        candidate = fingerprint_sql(sql)
        if candidate in self.tried:
            return RepairVerdict.DUPLICATE
        if self.tried:  # not the first attempt, so this is a repair
            if self.repairs >= self.max_repairs:
                return RepairVerdict.EXHAUSTED
            self.repairs += 1
        self.tried.append(candidate)
        return RepairVerdict.ALLOWED

    # -- multi-step --------------------------------------------------------

    def can_take_step(self) -> bool:
        return self.plan_steps < MAX_PLAN_STEPS

    def charge_step(self) -> None:
        self.plan_steps += 1

    # -- reporting ---------------------------------------------------------

    def exhausted_reason(self) -> str | None:
        """Why the turn stopped, in terms a user can act on, or ``None``.

        Exceeding a budget is a *degraded answer*, not an error: the user is told
        what was completed and what was skipped, never handed a partial result
        presented as complete.
        """
        if self.llm_calls >= self.max_llm_calls:
            return (
                "I reached this question's processing limit. Here is what I completed; "
                "ask a narrower question for the rest."
            )
        if self.bytes_billed >= self.max_bytes_billed:
            return (
                "Answering this fully would scan more data than one question is "
                "allowed. A narrower date range would bring it in range."
            )
        if self.repairs >= self.max_repairs:
            return (
                "I could not build a working query after two attempts, so I stopped "
                "rather than keep trying."
            )
        return None

    def snapshot(self) -> dict[str, int]:
        """Consumption, for the cost line and the event log."""
        return {
            "llm_calls": self.llm_calls,
            "tokens": self.tokens,
            "bytes_billed": self.bytes_billed,
            "repairs": self.repairs,
            "plan_steps": self.plan_steps,
        }
