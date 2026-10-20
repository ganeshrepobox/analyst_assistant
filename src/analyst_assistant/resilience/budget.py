"""Per-turn spending limits: LLM calls, bytes billed, repair attempts.

Held in graph state so a resumed turn does not start spending from zero again.

The duplicate check is the useful part. What actually wastes quota is a model
that "fixes" a query by returning something equivalent to what just failed, so
every candidate is fingerprinted and a repeat is rejected without running.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum

# Sized for the Gemini free tier. See the module docstring.
DEFAULT_MAX_LLM_CALLS = 6
DEFAULT_MAX_BYTES_BILLED = 2_000_000_000  # 2 GB
DEFAULT_MAX_REPAIRS = 2

# Cap on sub-questions for one multi-step request.
MAX_PLAN_STEPS = 3


class RepairVerdict(StrEnum):
    """Whether another attempt at a query may proceed."""

    ALLOWED = "allowed"
    DUPLICATE = "duplicate"
    """Matches a query already tried. Reject without executing."""
    EXHAUSTED = "exhausted"


def fingerprint_sql(sql: str) -> str:
    """Fingerprint a query so cosmetic rewrites do not count as progress.

    Textual, not semantic: flagging a real fix as a duplicate is the worse error.
    """
    return hashlib.sha256(re.sub(r"\s+", " ", sql.strip().lower()).encode()).hexdigest()[:16]


@dataclass
class Budget:
    """Ceilings and consumption for one turn. Held in graph state."""

    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS
    max_bytes_billed: int = DEFAULT_MAX_BYTES_BILLED
    max_repairs: int = DEFAULT_MAX_REPAIRS

    llm_calls: int = 0
    tokens: int = 0  # tracked for the cost line, not a ceiling
    bytes_billed: int = 0
    repairs: int = 0
    plan_steps: int = 0
    tried: list[str] = field(default_factory=list)

    def can_call_llm(self) -> bool:
        return self.llm_calls < self.max_llm_calls

    def charge_llm(self, *, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.llm_calls += 1
        self.tokens += tokens_in + tokens_out

    def can_run_query(self, estimated_bytes: int) -> bool:
        """Check a dry-run estimate against what is left.

        Checked before execution; afterwards the bytes are already paid for.
        """
        return self.bytes_billed + max(0, estimated_bytes) <= self.max_bytes_billed

    def charge_query(self, bytes_billed: int) -> None:
        self.bytes_billed += max(0, bytes_billed)

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.max_bytes_billed - self.bytes_billed)

    # repair loop
    def register_attempt(self, sql: str) -> RepairVerdict:
        """Record a candidate query and decide whether it may run.

        The first call of a turn is the original attempt and does not use the repair
        allowance.
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

    # multi-step
    def can_take_step(self) -> bool:
        return self.plan_steps < MAX_PLAN_STEPS

    def charge_step(self) -> None:
        self.plan_steps += 1

    def exhausted_reason(self) -> str | None:
        """Why the turn stopped, phrased for a user, or None.

        Running out of budget is a degraded answer, not an error.
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
