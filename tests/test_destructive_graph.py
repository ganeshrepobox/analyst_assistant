"""Tests for the destructive-operation subgraph.

The centrepiece is :func:`test_confirmation_survives_a_process_restart`, which
proves the pause is durable rather than merely in-memory -- the property that
justifies LangGraph over a hand-rolled loop.
"""

from __future__ import annotations

import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from analyst_assistant.graph.destructive import build_destructive_graph
from analyst_assistant.reports.library import ReportLibrary

ALICE = "exec_women_apparel"
BOB = "exec_outerwear"


@pytest.fixture()
def lib(tmp_path):
    library = ReportLibrary(tmp_path / "reports.sqlite")
    yield library
    library.close()


@pytest.fixture()
def checkpoint_path(tmp_path):
    return tmp_path / "checkpoints.sqlite"


def _graph(lib, checkpoint_path):
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    return build_destructive_graph(lib, SqliteSaver(conn))


def _seed(lib, owner, titles, *, thread="thread-a"):
    return [
        lib.save(owner_id=owner, thread_id=thread, title=t, body={"summary": t})
        for t in titles
    ]


def _cfg(thread_id="run-1"):
    return {"configurable": {"thread_id": thread_id}}


def _start(graph, *, user_id=ALICE, request, thread_id="thread-a", cfg=None):
    return graph.invoke(
        {"user_id": user_id, "thread_id": thread_id, "request": request},
        cfg or _cfg(),
    )


# --------------------------------------------------------------------------
# The pause
# --------------------------------------------------------------------------


def test_graph_pauses_with_a_preview_before_deleting_anything(lib, checkpoint_path):
    _seed(lib, ALICE, ["Allbirds Q1", "Allbirds Q2"])
    graph = _graph(lib, checkpoint_path)

    out = _start(graph, request={"text": "allbirds"})

    assert "__interrupt__" in out
    payload = out["__interrupt__"][0].value
    assert payload["kind"] == "confirm_delete"
    assert payload["count"] == 2
    assert len(payload["preview"]) == 2
    # Crucially, nothing is gone yet.
    assert len(lib.list_live(owner_id=ALICE)) == 2


def test_preview_lists_the_actual_titles(lib, checkpoint_path):
    """The preview is the entire basis on which the user consents."""
    _seed(lib, ALICE, ["Allbirds margin analysis"])
    graph = _graph(lib, checkpoint_path)

    payload = _start(graph, request={"text": "allbirds"})["__interrupt__"][0].value
    assert any("Allbirds margin analysis" in line for line in payload["preview"])


def test_no_matches_short_circuits_without_a_confirmation_prompt(lib, checkpoint_path):
    """Asking someone to confirm an empty deletion is pure friction."""
    _seed(lib, ALICE, ["Something else"])
    graph = _graph(lib, checkpoint_path)

    out = _start(graph, request={"text": "nonexistent brand"})
    assert "__interrupt__" not in out
    assert out["outcome"] == "nothing_matched"


# --------------------------------------------------------------------------
# Durability -- the reason for LangGraph
# --------------------------------------------------------------------------


def test_confirmation_survives_a_process_restart(lib, checkpoint_path):
    """Preview, lose the process, come back, confirm, and the delete completes.

    A new connection, a new SqliteSaver and a freshly compiled graph object stand
    in for a restarted CLI. Nothing is carried over in memory.
    """
    reports = _seed(lib, ALICE, ["Allbirds Q1", "Allbirds Q2"])
    cfg = _cfg("durable-run")

    first = _graph(lib, checkpoint_path)
    out = _start(first, request={"text": "allbirds"}, cfg=cfg)
    assert "__interrupt__" in out
    del first  # the process is gone

    revived = _graph(lib, checkpoint_path)
    snapshot = revived.get_state(cfg)
    assert snapshot.next == ("await_confirmation",)
    assert snapshot.interrupts, "the pending decision did not survive the restart"

    final = revived.invoke(Command(resume="yes"), cfg)
    assert final["outcome"] == "deleted"
    assert sorted(final["deleted_ids"]) == sorted(r.id for r in reports)
    assert lib.list_live(owner_id=ALICE) == []


def test_resolution_is_not_repeated_on_resume(lib, checkpoint_path):
    """A report that starts matching mid-decision must not be swept in.

    The node holding interrupt() re-executes on resume, so if resolution lived
    there the new report would join a set the user already approved. It is a
    separate upstream node precisely to prevent that.
    """
    _seed(lib, ALICE, ["Allbirds Q1"])
    graph = _graph(lib, checkpoint_path)
    cfg = _cfg("race")

    _start(graph, request={"text": "allbirds"}, cfg=cfg)

    # A new match appears while the user is deciding.
    intruder = lib.save(
        owner_id=ALICE, thread_id="thread-a", title="Allbirds Q3", body={"summary": "x"}
    )

    final = graph.invoke(Command(resume="yes"), cfg)

    # Consent was to the original single report, so the approval is refused
    # outright rather than partially applied.
    assert final["outcome"] == "set_changed"
    assert final["deleted_ids"] == []
    assert lib.get(intruder.id, owner_id=ALICE) is not None


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


def test_cancelling_deletes_nothing(lib, checkpoint_path):
    _seed(lib, ALICE, ["Allbirds Q1", "Allbirds Q2"])
    graph = _graph(lib, checkpoint_path)
    cfg = _cfg("cancel")

    _start(graph, request={"text": "allbirds"}, cfg=cfg)
    final = graph.invoke(Command(resume="cancel"), cfg)

    assert final["outcome"] == "cancelled"
    assert len(lib.list_live(owner_id=ALICE)) == 2


def test_subset_selection_deletes_only_the_chosen_rows(lib, checkpoint_path):
    reports = _seed(lib, ALICE, ["Allbirds A", "Allbirds B"])
    graph = _graph(lib, checkpoint_path)
    cfg = _cfg("subset")

    payload = _start(graph, request={"text": "allbirds"}, cfg=cfg)["__interrupt__"][0].value
    # Reports list newest-first, so index 1 is the most recent.
    expected = reports[1].id if payload["preview"][0].startswith("1. Allbirds B") else reports[0].id

    final = graph.invoke(Command(resume="1"), cfg)
    assert final["outcome"] == "deleted"
    assert final["deleted_ids"] == [expected]
    assert len(lib.list_live(owner_id=ALICE)) == 1


def test_a_bare_yes_cannot_confirm_a_bulk_delete(lib, checkpoint_path):
    """Friction scales with blast radius.

    Uniform maximum friction is the lazy reading and it backfires: demanding a
    typed phrase to delete one report trains people to type it unread.
    """
    _seed(lib, ALICE, [f"Allbirds {i}" for i in range(6)])
    graph = _graph(lib, checkpoint_path)
    cfg = _cfg("bulk-rejected")

    payload = _start(graph, request={"text": "allbirds"}, cfg=cfg)["__interrupt__"][0].value
    assert payload["requires_phrase"] is True
    assert payload["required_phrase"] == "DELETE 6 REPORTS"

    final = graph.invoke(Command(resume="yes"), cfg)
    assert final["outcome"] == "unclear"
    assert len(lib.list_live(owner_id=ALICE)) == 6


def test_the_typed_phrase_confirms_a_bulk_delete(lib, checkpoint_path):
    _seed(lib, ALICE, [f"Allbirds {i}" for i in range(6)])
    graph = _graph(lib, checkpoint_path)
    cfg = _cfg("bulk-confirmed")

    _start(graph, request={"text": "allbirds"}, cfg=cfg)
    final = graph.invoke(Command(resume="DELETE 6 REPORTS"), cfg)

    assert final["outcome"] == "deleted"
    assert lib.list_live(owner_id=ALICE) == []


def test_an_unclear_reply_deletes_nothing_and_says_so(lib, checkpoint_path):
    """Ambiguity resolves toward less deletion.

    Reported as unclear rather than as a cancellation, so the user knows to ask
    again rather than assuming they declined.
    """
    _seed(lib, ALICE, ["Allbirds Q1"])
    graph = _graph(lib, checkpoint_path)
    cfg = _cfg("unclear")

    _start(graph, request={"text": "allbirds"}, cfg=cfg)
    final = graph.invoke(Command(resume="hmm maybe"), cfg)

    assert final["outcome"] == "unclear"
    assert len(lib.list_live(owner_id=ALICE)) == 1


# --------------------------------------------------------------------------
# Scope shapes the brief names explicitly
# --------------------------------------------------------------------------


def test_delete_everything_from_this_conversation(lib, checkpoint_path):
    """'Delete all the reports we made in this conversation' -- a scope predicate."""
    _seed(lib, ALICE, ["One", "Two"], thread="this-chat")
    _seed(lib, ALICE, ["Elsewhere"], thread="other-chat")
    graph = _graph(lib, checkpoint_path)
    cfg = _cfg("by-thread")

    _start(graph, request={"thread_id": "this-chat"}, cfg=cfg)
    final = graph.invoke(Command(resume="yes"), cfg)

    assert final["outcome"] == "deleted"
    assert [r.title for r in lib.list_live(owner_id=ALICE)] == ["Elsewhere"]


# --------------------------------------------------------------------------
# Authorization
# --------------------------------------------------------------------------


def test_another_users_reports_are_never_candidates(lib, checkpoint_path):
    """Ownership is enforced before resolution, so bulk delete cannot cross users."""
    _seed(lib, BOB, ["Allbirds confidential"])
    _seed(lib, ALICE, ["Allbirds mine"])
    graph = _graph(lib, checkpoint_path)
    cfg = _cfg("isolation")

    payload = _start(graph, request={"text": "allbirds"}, cfg=cfg)["__interrupt__"][0].value
    assert payload["count"] == 1

    graph.invoke(Command(resume="yes"), cfg)
    assert len(lib.list_live(owner_id=BOB)) == 1


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def test_deletions_are_audited_and_reversible(lib, checkpoint_path):
    reports = _seed(lib, ALICE, ["Allbirds Q1"])
    graph = _graph(lib, checkpoint_path)
    cfg = _cfg("audit")

    _start(graph, request={"text": "allbirds"}, cfg=cfg)
    graph.invoke(Command(resume="yes"), cfg)

    trail = lib.audit_trail(report_id=reports[0].id)
    assert [row["action"] for row in trail] == ["save", "delete"]

    assert lib.undo_last_delete(owner_id=ALICE) == [reports[0].id]
    assert len(lib.list_live(owner_id=ALICE)) == 1
