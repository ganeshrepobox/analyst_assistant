"""Check that the numbers in a report came from the query results.

Deterministic and free. It catches the failure that matters most for this audience:
a figure in the prose that is not in the data. In testing, a report that correctly
tabulated 2504 customers wrote "2,5004 customers" in the sentence above it, which
no amount of prompting reliably prevents and no reader would catch.

Derived figures (percentages, differences, growth rates) legitimately do not appear
in the results, so an unmatched number is not proof of error. This records a count
and the offending values rather than rejecting the answer, which is why it is a
signal for the metrics and /debug rather than a gate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Numbers as a report writes them: 1,284,900 / $84.20 / 12.4% / 2504
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")

# Small integers are usually counts, years, list positions or "top 3", so matching
# them against the data produces noise rather than signal.
MIN_INTERESTING = 100.0

#: Relative tolerance, so a report rounding 1284900.37 to 1,284,900 still matches.
TOLERANCE = 0.015

#: Bare four-digit integers in this range are read as years. Reports say "in 2024"
#: constantly and the year is never a figure from the results.
YEAR_RANGE = (1900, 2100)


def _is_year(token: str, value: float) -> bool:
    plain = token.strip("$%,")
    return (
        plain.isdigit()
        and len(plain) == 4
        and YEAR_RANGE[0] <= value <= YEAR_RANGE[1]
    )


@dataclass
class FaithfulnessReport:
    checked: int = 0
    unverified: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unverified

    def summary(self) -> str:
        if self.checked == 0:
            return "no figures to check"
        if self.ok:
            return f"all {self.checked} figures traced to query results"
        return (
            f"{len(self.unverified)} of {self.checked} figures not found in the results: "
            + ", ".join(self.unverified[:5])
        )


def _parse(token: str) -> float | None:
    cleaned = token.replace("$", "").replace(",", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _result_numbers(frames: Sequence[Mapping[str, Any]]) -> set[float]:
    """Every number in the results, plus the rounded forms a report might print."""
    found: set[float] = set()
    for frame in frames:
        for row in frame.get("rows", []):
            for value in row.values():
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, (int, float)):
                    number = float(value)
                elif isinstance(value, str):
                    parsed = _parse(value)
                    if parsed is None:
                        continue
                    number = parsed
                else:
                    continue
                found.update({number, round(number), round(number, 1), round(number, 2)})
    return found


def check(answer: str, frames: Sequence[Mapping[str, Any]]) -> FaithfulnessReport:
    """Compare the figures in an answer against the data it was written from."""
    report = FaithfulnessReport()
    if not answer or not frames:
        return report

    available = _result_numbers(frames)
    if not available:
        return report

    seen: set[str] = set()
    for token in _NUMBER.findall(answer):
        value = _parse(token)
        if value is None or abs(value) < MIN_INTERESTING or token in seen:
            continue
        if _is_year(token, value):
            continue
        seen.add(token)
        report.checked += 1

        # A percentage is almost always derived, so check it against the data only
        # loosely: if it matches nothing, that is expected rather than suspicious.
        if token.endswith("%"):
            report.checked -= 1
            continue

        if not any(
            abs(value - candidate) <= TOLERANCE * max(abs(value), abs(candidate), 1.0)
            for candidate in available
        ):
            report.unverified.append(token)
    return report
