"""Pseudonymising ids, masking result columns, and a regex sweep before output.

The regex is not the main PII control. The scoped views are, since they do not
contain the columns at all. This catches a failure of those, so a hit is counted
as a metric.

Two orderings matter: masking runs before the report is written, so raw values
never reach the model, and questions and SQL are scrubbed before being logged.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .schema import TABLES, PiiClass

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

PSEUDONYM_PREFIX = "CUST"
_PSEUDONYM_LENGTH = 6

# Narrow by design. Each pattern targets a format that is unambiguously an
# identifier, because a false positive here corrupts a legitimate answer - product
# SKUs, order ids and prices must survive untouched.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    "phone": re.compile(
        r"(?<![\w.])(?:\+\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}\b"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "street_address": re.compile(
        r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b"
    ),
}

REDACTION = "[REDACTED:{kind}]"


@dataclass
class ScrubResult:
    """Outcome of a tripwire sweep."""

    text: str
    findings: dict[str, int] = field(default_factory=dict)

    @property
    def fired(self) -> bool:
        """True if anything was redacted, which means an earlier layer leaked."""
        return bool(self.findings)

    @property
    def total(self) -> int:
        return sum(self.findings.values())


def pseudonymize(value: object, *, salt: str = "", prefix: str = PSEUDONYM_PREFIX) -> str:
    """Map an id to a stable, non-reversible label.

    Stable so the same customer keeps the same label across a report. The salt should
    be a per-deployment secret; unsalted, this dataset's id range is small enough to
    enumerate.
    """
    digest = hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    return f"{prefix}-{digest[:_PSEUDONYM_LENGTH]}"


def scrub(text: str) -> ScrubResult:
    """Redact identifier-shaped text. Used on output and before logging."""
    if not text:
        return ScrubResult(text=text)

    findings: dict[str, int] = {}
    out = text
    for kind, pattern in _PATTERNS.items():
        out, n = pattern.subn(REDACTION.format(kind=kind), out)
        if n:
            findings[kind] = n
    return ScrubResult(text=out, findings=findings)


def _columns_by_class(pii_class: PiiClass) -> frozenset[str]:
    return frozenset(
        col
        for cols in TABLES.values()
        for col, kind in cols.items()
        if kind is pii_class
    )


def mask_dataframe(df: pd.DataFrame, *, salt: str = "") -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask a result frame before it is rendered or sent to the model.

    Direct identifiers are dropped. They should never be here, so it is counted as a
    tripwire hit. Join keys are replaced in place with pseudonyms, so ranking still
    works.
    """
    findings: dict[str, int] = {}
    out = df.copy()

    direct = _columns_by_class(PiiClass.DIRECT_IDENTIFIER)
    join_keys = _columns_by_class(PiiClass.JOIN_KEY)

    for column in list(out.columns):
        base = str(column).lower().split(".")[-1]
        if base in direct:
            out = out.drop(columns=[column])
            findings[f"dropped:{base}"] = len(df)
        elif base in join_keys:
            out[column] = out[column].map(lambda v: pseudonymize(v, salt=salt))
            findings[f"pseudonymized:{base}"] = len(df)

    return out, findings


def scrub_for_log(text: str) -> str:
    """Scrub for the logging path. Named separately so call sites read clearly."""
    return scrub(text).text
