"""The confirmation gate for destructive operations. Implements requirement 3.

Four ideas, each earning its complexity
---------------------------------------

**1. Resolve, then confirm, then act on the resolved set.**
The ids presented in the preview are the ids deleted. The model never gets a second
chance to reinterpret "the ones I said" after approval. This is the central failure
mode of agentic destructive actions, and it is closed by passing an explicit id set
as *data* to deterministic code -- never by giving the model a delete tool it can
re-parameterise.

**2. Friction scales with blast radius.**
Requirement 3 asks for a strict confirmation flow "without breaking UX". Uniform
maximum friction is the lazy reading and it backfires: forcing a typed phrase to
delete one report trains people to type it without reading, which destroys the
value of the gate on the day it matters. So one or two reports take a natural
"yes"; a bulk delete requires a typed phrase that states the count.

**3. The token binds to the resolved set (TOCTOU).**
Between preview and confirmation the world can change -- a new report may now match
"mentioning Client X", or the process may have restarted. The token carries a hash
of the exact id set that was shown, so if the set has shifted the confirmation is
refused and the user is re-prompted with the new preview. Consent is to a specific
list, not to a sentence.

**4. Tokens expire.**
A stale approval sitting in a checkpoint is not consent. Five minutes.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from dataclasses import dataclass
from enum import StrEnum

#: Above this many reports, a natural-language "yes" is no longer sufficient.
BULK_THRESHOLD = 3

#: Seconds a confirmation token stays valid.
TOKEN_TTL_SECONDS = 300

_AFFIRMATIVE = frozenset(
    {"y", "yes", "yeah", "yep", "ok", "okay", "confirm", "confirmed", "do it", "go ahead"}
)
_NEGATIVE = frozenset({"n", "no", "nope", "cancel", "stop", "abort", "nevermind", "never mind"})


class Decision(StrEnum):
    """What the user's reply to a confirmation prompt meant."""

    CONFIRM = "confirm"
    CANCEL = "cancel"
    SUBSET = "subset"
    """The user picked specific items, e.g. "delete 1 and 3"."""
    UNCLEAR = "unclear"
    """Re-prompt rather than guess. Guessing here deletes the wrong thing."""


class TokenError(StrEnum):
    """Why a confirmation token was refused."""

    EXPIRED = "expired"
    WRONG_USER = "wrong_user"
    SET_CHANGED = "set_changed"


def _hash_ids(report_ids: list[str]) -> str:
    """Order-independent fingerprint of an id set.

    Sorted so that a differently ordered but identical set still matches -- the
    guard is about *which* reports, not the order they were listed in.
    """
    joined = "\n".join(sorted(report_ids))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ConfirmationToken:
    """Proof that a specific user approved a specific set of deletions."""

    user_id: str
    id_set_hash: str
    nonce: str
    issued_at: float
    count: int
    ttl_seconds: int = TOKEN_TTL_SECONDS

    @property
    def requires_typed_phrase(self) -> bool:
        return self.count >= BULK_THRESHOLD

    @property
    def required_phrase(self) -> str:
        """The phrase a bulk delete must be confirmed with.

        States the count, so the user is re-confronted with the blast radius at the
        moment of consent rather than only in the preview above it.
        """
        return f"DELETE {self.count} REPORTS"

    def expired(self, *, now: float | None = None) -> bool:
        return (now or time.time()) - self.issued_at > self.ttl_seconds

    def check(
        self, *, user_id: str, report_ids: list[str], now: float | None = None
    ) -> TokenError | None:
        """Validate this token against the live state. ``None`` means valid."""
        if self.expired(now=now):
            return TokenError.EXPIRED
        if user_id != self.user_id:
            return TokenError.WRONG_USER
        if _hash_ids(report_ids) != self.id_set_hash:
            return TokenError.SET_CHANGED
        return None


def issue_token(
    *, user_id: str, report_ids: list[str], ttl_seconds: int = TOKEN_TTL_SECONDS
) -> ConfirmationToken:
    """Issue a token for the set about to be previewed."""
    return ConfirmationToken(
        user_id=user_id,
        id_set_hash=_hash_ids(report_ids),
        nonce=secrets.token_hex(8),
        issued_at=time.time(),
        count=len(report_ids),
        ttl_seconds=ttl_seconds,
    )


def parse_reply(reply: str, *, token: ConfirmationToken) -> tuple[Decision, list[int]]:
    """Interpret a free-text reply to a confirmation prompt.

    Returns the decision and, for :attr:`Decision.SUBSET`, the 1-based indices the
    user named.

    Ambiguity resolves to :attr:`Decision.UNCLEAR`, never to confirmation. The cost
    of re-asking is one turn; the cost of a false positive is deleted work.
    """
    text = reply.strip().lower()
    if not text:
        return Decision.UNCLEAR, []

    if text in _NEGATIVE or any(text.startswith(f"{n} ") for n in _NEGATIVE):
        return Decision.CANCEL, []

    # Bulk deletes demand the exact phrase, checked before anything else so that a
    # bare "yes" can never satisfy a bulk confirmation.
    if token.requires_typed_phrase:
        if reply.strip().upper() == token.required_phrase:
            return Decision.CONFIRM, []
        if text in _AFFIRMATIVE:
            # Affirmative but insufficient: the caller re-prompts with the phrase.
            return Decision.UNCLEAR, []

    # A subset selection such as "delete 1 and 3" or "just 2".
    #
    # A bare number is ALWAYS an index, never a count. For a two-item preview,
    # "2" is ambiguous between "item 2" and "all 2" -- and the two readings differ
    # by a deleted report. Resolving that ambiguity toward *less* deletion is the
    # only safe direction, so "all" must be said explicitly to mean everything.
    indices = [int(n) for n in re.findall(r"\b(\d{1,3})\b", reply)]
    if indices:
        valid = sorted({i for i in indices if 1 <= i <= token.count})
        if len(valid) != len(set(indices)):
            # Some index was out of range; re-prompt rather than silently drop it.
            return Decision.UNCLEAR, []
        if valid == list(range(1, token.count + 1)) and token.requires_typed_phrase:
            # Enumerating every item is still a bulk delete.
            return Decision.UNCLEAR, []
        return Decision.SUBSET, valid

    if "all" in text.split() and not token.requires_typed_phrase:
        return Decision.CONFIRM, []

    if text in _AFFIRMATIVE:
        return Decision.CONFIRM, []

    return Decision.UNCLEAR, []


def prompt_for(report_ids: list[str], token: ConfirmationToken) -> str:
    """The instruction line shown beneath a deletion preview."""
    n = len(report_ids)
    if token.requires_typed_phrase:
        return (
            f"This will delete {n} reports. Type \"{token.required_phrase}\" to confirm, "
            f'or "cancel" to abort. You can also delete only some, e.g. "1 and 3".'
        )
    if n == 1:
        return 'Delete this report? ("yes" / "cancel")'
    return f'Delete these {n} reports? ("yes" / "cancel", or pick some, e.g. "1").'
