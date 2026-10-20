"""Output masking and the egress tripwire.

Three jobs, in order of how much they are relied upon:

1. **Pseudonymisation** -- turn a customer id into a stable, non-reversible label so
   that "who are my top customers?" stays answerable without disclosing identity.
2. **DataFrame masking** -- defence in depth over query results.
3. **The text tripwire** -- a deliberately narrow regex sweep over anything about to
   be shown, logged, or sent to the model.

The tripwire is not the PII control. The real control is structural: direct
identifiers are absent from the scoped views (see ``entitlements``), so they cannot
be selected. The tripwire exists to *detect the failure of that control*, which is
why a firing is recorded as a defect and alerted on rather than quietly cleaned up.
See ``docs/01-architecture.md`` section 9.

Ordering matters more than the rules
------------------------------------
* Masking runs **before** report synthesis, so raw personal data never enters the
  model's context. Masking at print time would leave the LLM having already seen it.
* The user's question and the generated SQL are scrubbed **before** being written to
  the event log, or a predicate such as ``WHERE email = 'x@y.com'`` is persisted into
  the trace that was meant to make the system auditable.
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
# identifier, because a false positive here corrupts a legitimate answer -- product
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
        """True if anything was redacted.

        A true value is a **defect signal**, not a success. It means personal data
        reached a layer where it should already have been impossible.
        """
        return bool(self.findings)

    @property
    def total(self) -> int:
        return sum(self.findings.values())


def pseudonymize(value: object, *, salt: str = "", prefix: str = PSEUDONYM_PREFIX) -> str:
    """Map an identifier to a stable, non-reversible label.

    Stable so that the same customer carries the same label within and across
    reports, which is what makes ranking and follow-up questions coherent.

    The salt should be a per-deployment secret. Without one, the space of customer
    ids in this dataset is small enough to enumerate by brute force, so an unsalted
    hash is pseudonymisation in name only.
    """
    digest = hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    return f"{prefix}-{digest[:_PSEUDONYM_LENGTH]}"


def scrub(text: str) -> ScrubResult:
    """Redact identifier-shaped substrings from free text.

    Applied to model output before display, and to questions and SQL before logging.
    """
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
    """Mask a query result before it is rendered or sent to the model.

    * Direct-identifier columns are dropped outright. They should never be present --
      the scoped views omit them -- so their presence is logged as a tripwire hit.
    * Join-key columns are replaced in place by pseudonyms, preserving groupings and
      ordering so that "top customers" still ranks correctly.

    Returns:
        The masked frame and a mapping of what was acted on, for metrics.
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
    """Convenience wrapper for the logging path.

    Separate from :func:`scrub` so that the intent is explicit at call sites: this
    one protects the audit trail, not the user-facing output.
    """
    return scrub(text).text
