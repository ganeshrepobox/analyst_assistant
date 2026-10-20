"""Tests for the destructive-operation confirmation gate.

Two themes:

* **TOCTOU** -- consent is to a specific list of reports, not to a sentence, so a
  token stops being valid the moment the matching set changes.
* **Ambiguity resolves toward less deletion** -- every unclear reply must re-prompt
  rather than proceed.
"""

from __future__ import annotations

import time

from analyst_assistant.reports.confirmation import (
    BULK_THRESHOLD,
    Decision,
    TokenError,
    issue_token,
    parse_reply,
    prompt_for,
)

ALICE = "exec_women_apparel"
BOB = "exec_outerwear"


# --------------------------------------------------------------------------
# Token validity
# --------------------------------------------------------------------------


def test_token_validates_against_the_set_it_was_issued_for():
    ids = ["a", "b"]
    token = issue_token(user_id=ALICE, report_ids=ids)
    assert token.check(user_id=ALICE, report_ids=ids) is None


def test_token_is_order_independent():
    """The guard is about which reports, not the order they were listed."""
    token = issue_token(user_id=ALICE, report_ids=["a", "b", "c"])
    assert token.check(user_id=ALICE, report_ids=["c", "a", "b"]) is None


def test_token_rejects_a_different_user():
    token = issue_token(user_id=ALICE, report_ids=["a"])
    assert token.check(user_id=BOB, report_ids=["a"]) is TokenError.WRONG_USER


def test_token_expires():
    """A stale approval sitting in a checkpoint is not consent."""
    token = issue_token(user_id=ALICE, report_ids=["a"], ttl_seconds=1)
    future = time.time() + 5
    assert token.check(user_id=ALICE, report_ids=["a"], now=future) is TokenError.EXPIRED


# --------------------------------------------------------------------------
# TOCTOU
# --------------------------------------------------------------------------


def test_an_added_match_invalidates_the_token():
    """A new report started matching between preview and confirmation.

    Confirming now would delete something the user was never shown, so the gate
    refuses and the caller re-previews.
    """
    token = issue_token(user_id=ALICE, report_ids=["a", "b"])
    assert (
        token.check(user_id=ALICE, report_ids=["a", "b", "c"]) is TokenError.SET_CHANGED
    )


def test_a_removed_match_invalidates_the_token():
    token = issue_token(user_id=ALICE, report_ids=["a", "b"])
    assert token.check(user_id=ALICE, report_ids=["a"]) is TokenError.SET_CHANGED


def test_a_swapped_match_invalidates_the_token():
    """Same count, different contents -- a count check alone would miss this."""
    token = issue_token(user_id=ALICE, report_ids=["a", "b"])
    assert token.check(user_id=ALICE, report_ids=["a", "z"]) is TokenError.SET_CHANGED


# --------------------------------------------------------------------------
# Friction scales with blast radius
# --------------------------------------------------------------------------


def test_small_delete_accepts_a_natural_yes():
    """Uniform maximum friction trains people to confirm reflexively."""
    token = issue_token(user_id=ALICE, report_ids=["a"])
    assert not token.requires_typed_phrase
    assert parse_reply("yes", token=token)[0] is Decision.CONFIRM


def test_bulk_delete_requires_the_typed_phrase():
    ids = [str(i) for i in range(BULK_THRESHOLD + 2)]
    token = issue_token(user_id=ALICE, report_ids=ids)
    assert token.requires_typed_phrase

    decision, _ = parse_reply(token.required_phrase, token=token)
    assert decision is Decision.CONFIRM


def test_bare_yes_is_insufficient_for_a_bulk_delete():
    ids = [str(i) for i in range(BULK_THRESHOLD + 2)]
    token = issue_token(user_id=ALICE, report_ids=ids)
    assert parse_reply("yes", token=token)[0] is Decision.UNCLEAR


def test_required_phrase_states_the_count():
    """Re-confronts the user with the blast radius at the moment of consent."""
    token = issue_token(user_id=ALICE, report_ids=["a"] * 7)
    assert token.required_phrase == "DELETE 7 REPORTS"
    assert "7" in prompt_for(["a"] * 7, token)


def test_enumerating_every_item_is_still_a_bulk_delete():
    ids = ["a", "b", "c", "d"]
    token = issue_token(user_id=ALICE, report_ids=ids)
    assert parse_reply("1 2 3 4", token=token)[0] is Decision.UNCLEAR


# --------------------------------------------------------------------------
# Reply parsing
# --------------------------------------------------------------------------


def test_cancellation_is_recognised():
    token = issue_token(user_id=ALICE, report_ids=["a"])
    for reply in ("no", "cancel", "stop", "nevermind", "abort"):
        assert parse_reply(reply, token=token)[0] is Decision.CANCEL, reply


def test_subset_selection():
    token = issue_token(user_id=ALICE, report_ids=["a", "b", "c"])
    decision, picks = parse_reply("delete 1 and 3", token=token)
    assert decision is Decision.SUBSET
    assert picks == [1, 3]


def test_a_bare_number_is_an_index_never_a_count():
    """For a two-item preview, "2" must mean item 2, not "all 2".

    The two readings differ by a deleted report, so ambiguity resolves toward less
    deletion.
    """
    token = issue_token(user_id=ALICE, report_ids=["a", "b"])
    decision, picks = parse_reply("2", token=token)
    assert decision is Decision.SUBSET
    assert picks == [2]


def test_out_of_range_index_re_prompts_rather_than_dropping_it():
    token = issue_token(user_id=ALICE, report_ids=["a", "b"])
    assert parse_reply("1 and 5", token=token)[0] is Decision.UNCLEAR


def test_duplicate_indices_are_collapsed():
    token = issue_token(user_id=ALICE, report_ids=["a", "b", "c"])
    decision, picks = parse_reply("1, 1, 2", token=token)
    assert decision is Decision.SUBSET
    assert picks == [1, 2]


def test_explicit_all_confirms_a_small_delete():
    token = issue_token(user_id=ALICE, report_ids=["a", "b"])
    assert parse_reply("delete all", token=token)[0] is Decision.CONFIRM


def test_explicit_all_is_not_enough_for_a_bulk_delete():
    token = issue_token(user_id=ALICE, report_ids=["a"] * 6)
    assert parse_reply("delete all", token=token)[0] is Decision.UNCLEAR


def test_empty_and_waffling_replies_are_unclear():
    token = issue_token(user_id=ALICE, report_ids=["a"])
    for reply in ("", "   ", "hmm", "what do you think?", "maybe"):
        assert parse_reply(reply, token=token)[0] is Decision.UNCLEAR, repr(reply)


def test_unclear_never_resolves_to_confirm():
    """The cost of re-asking is one turn; a false positive deletes work."""
    token = issue_token(user_id=ALICE, report_ids=["a", "b"])
    for reply in ("probably", "go on then maybe", "i guess?"):
        assert parse_reply(reply, token=token)[0] is not Decision.CONFIRM, reply
