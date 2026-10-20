"""Tests for the Saved Reports library.

The ownership tests matter most: they are the reason "delete all my reports" can
never reach another user's work, regardless of what any layer above decides.
"""

from __future__ import annotations

import pytest

from analyst_assistant.reports.library import ReportLibrary

ALICE = "exec_women_apparel"
BOB = "exec_outerwear"


@pytest.fixture()
def lib(tmp_path):
    library = ReportLibrary(tmp_path / "test.sqlite")
    yield library
    library.close()


def _save(lib, owner, title, *, thread="t1", body=None, trace="tr1"):
    return lib.save(
        owner_id=owner,
        thread_id=thread,
        title=title,
        body=body or {"summary": title},
        trace_id=trace,
    )


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------


def test_saved_report_can_be_read_back(lib):
    saved = _save(lib, ALICE, "Q1 revenue review")
    fetched = lib.get(saved.id, owner_id=ALICE)
    assert fetched is not None
    assert fetched.title == "Q1 revenue review"
    assert fetched.body == {"summary": "Q1 revenue review"}
    assert not fetched.is_deleted


def test_list_live_is_newest_first_and_excludes_deleted(lib):
    first = _save(lib, ALICE, "older")
    second = _save(lib, ALICE, "newer")
    lib.soft_delete(owner_id=ALICE, report_ids=[first.id])

    live = lib.list_live(owner_id=ALICE)
    assert [r.id for r in live] == [second.id]


# --------------------------------------------------------------------------
# Ownership isolation
# --------------------------------------------------------------------------


def test_one_user_cannot_read_anothers_report(lib):
    saved = _save(lib, ALICE, "Alice private analysis")
    assert lib.get(saved.id, owner_id=BOB) is None


def test_not_found_and_not_owned_are_indistinguishable(lib):
    """Revealing that a report exists but belongs to someone else leaks its existence."""
    saved = _save(lib, ALICE, "Alice analysis")
    assert lib.get(saved.id, owner_id=BOB) is None
    assert lib.get("nonexistent-id", owner_id=BOB) is None


def test_resolve_never_returns_another_users_reports(lib):
    _save(lib, ALICE, "Brand X performance")
    _save(lib, BOB, "Brand X performance")

    alice_hits = lib.resolve(owner_id=ALICE, text="brand x")
    assert len(alice_hits) == 1
    assert all(r.owner_id == ALICE for r in alice_hits)


def test_delete_cannot_touch_another_users_report(lib):
    """The core guarantee behind 'delete all reports'."""
    victim = _save(lib, BOB, "Bob's work")
    deleted = lib.soft_delete(owner_id=ALICE, report_ids=[victim.id])

    assert deleted == []
    assert lib.get(victim.id, owner_id=BOB) is not None


# --------------------------------------------------------------------------
# Resolution: both shapes requirement 3 names
# --------------------------------------------------------------------------


def test_resolve_by_text_matches_title(lib):
    _save(lib, ALICE, "Allbirds margin analysis")
    _save(lib, ALICE, "Calvin Klein returns")
    hits = lib.resolve(owner_id=ALICE, text="allbirds")
    assert [r.title for r in hits] == ["Allbirds margin analysis"]


def test_resolve_by_text_matches_body_not_just_title(lib):
    """'Mentioning Client X' is about content, not only the heading."""
    _save(lib, ALICE, "Q1 summary", body={"summary": "Allbirds drove the decline"})
    hits = lib.resolve(owner_id=ALICE, text="allbirds")
    assert len(hits) == 1


def test_resolve_by_thread_scopes_to_one_conversation(lib):
    _save(lib, ALICE, "in this chat", thread="thread-a")
    _save(lib, ALICE, "in this chat too", thread="thread-a")
    _save(lib, ALICE, "a different chat", thread="thread-b")

    hits = lib.resolve(owner_id=ALICE, thread_id="thread-a")
    assert len(hits) == 2


def test_resolve_filters_combine_as_and(lib):
    _save(lib, ALICE, "Allbirds report", thread="thread-a")
    _save(lib, ALICE, "Allbirds report", thread="thread-b")
    hits = lib.resolve(owner_id=ALICE, text="allbirds", thread_id="thread-a")
    assert len(hits) == 1


def test_resolve_excludes_already_deleted(lib):
    saved = _save(lib, ALICE, "Allbirds report")
    lib.soft_delete(owner_id=ALICE, report_ids=[saved.id])
    assert lib.resolve(owner_id=ALICE, text="allbirds") == []


def test_resolve_with_empty_id_list_matches_nothing(lib):
    """Guards against an empty list degenerating into 'match everything'."""
    _save(lib, ALICE, "something")
    assert lib.resolve(owner_id=ALICE, report_ids=[]) == []


# --------------------------------------------------------------------------
# Soft delete and undo
# --------------------------------------------------------------------------


def test_soft_delete_hides_but_retains(lib):
    saved = _save(lib, ALICE, "to delete")
    assert lib.soft_delete(owner_id=ALICE, report_ids=[saved.id]) == [saved.id]

    assert lib.list_live(owner_id=ALICE) == []
    still_there = lib.get(saved.id, owner_id=ALICE)
    assert still_there is not None and still_there.is_deleted


def test_repeated_delete_is_idempotent(lib):
    """Safe to retry after an interrupted confirmation."""
    saved = _save(lib, ALICE, "to delete")
    assert lib.soft_delete(owner_id=ALICE, report_ids=[saved.id]) == [saved.id]
    assert lib.soft_delete(owner_id=ALICE, report_ids=[saved.id]) == []


def test_undo_restores_the_last_batch(lib):
    a = _save(lib, ALICE, "a")
    b = _save(lib, ALICE, "b")
    lib.soft_delete(owner_id=ALICE, report_ids=[a.id, b.id])

    restored = lib.undo_last_delete(owner_id=ALICE)
    assert sorted(restored) == sorted([a.id, b.id])
    assert len(lib.list_live(owner_id=ALICE)) == 2


def test_undo_restores_only_the_most_recent_batch(lib):
    """'Undo' means one user action, not every row deleted recently."""
    first = _save(lib, ALICE, "first")
    second = _save(lib, ALICE, "second")
    lib.soft_delete(owner_id=ALICE, report_ids=[first.id])
    lib.soft_delete(owner_id=ALICE, report_ids=[second.id])

    restored = lib.undo_last_delete(owner_id=ALICE)
    assert restored == [second.id]
    assert [r.id for r in lib.list_live(owner_id=ALICE)] == [second.id]


def test_undo_with_nothing_deleted_is_a_no_op(lib):
    _save(lib, ALICE, "live")
    assert lib.undo_last_delete(owner_id=ALICE) == []


def test_empty_delete_list_is_a_no_op(lib):
    assert lib.soft_delete(owner_id=ALICE, report_ids=[]) == []


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def test_every_action_is_audited_with_actor_and_trace(lib):
    saved = _save(lib, ALICE, "audited", trace="trace-abc")
    lib.soft_delete(owner_id=ALICE, report_ids=[saved.id], trace_id="trace-def")
    lib.undo_last_delete(owner_id=ALICE, trace_id="trace-ghi")

    trail = lib.audit_trail(report_id=saved.id)
    assert [row["action"] for row in trail] == ["save", "delete", "undelete"]
    assert all(row["actor_id"] == ALICE for row in trail)
    assert [row["trace_id"] for row in trail] == ["trace-abc", "trace-def", "trace-ghi"]


def test_audit_survives_the_delete_it_records(lib):
    """The audit row must outlive the content, for later hard-delete compliance."""
    saved = _save(lib, ALICE, "gone")
    lib.soft_delete(owner_id=ALICE, report_ids=[saved.id])
    assert any(row["action"] == "delete" for row in lib.audit_trail(report_id=saved.id))


def test_refused_delete_writes_no_audit_row(lib):
    """A no-op must not look like an action in the trail."""
    victim = _save(lib, BOB, "Bob's")
    lib.soft_delete(owner_id=ALICE, report_ids=[victim.id])
    actions = [r["action"] for r in lib.audit_trail(report_id=victim.id)]
    assert actions == ["save"]


# --------------------------------------------------------------------------
# Preview rendering
# --------------------------------------------------------------------------


def test_summary_line_shows_title_and_date(lib):
    saved = _save(lib, ALICE, "Q1 revenue review")
    line = saved.summary_line(index=1)
    assert line.startswith("1. Q1 revenue review")
    assert saved.created_at[:10] in line
