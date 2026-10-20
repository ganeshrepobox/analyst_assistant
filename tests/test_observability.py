"""Tests for the event log, metrics and trace retrieval.

The important one is :func:`test_pii_is_scrubbed_before_being_logged`: the event log
exists to make the system auditable, and writing personal data into it would make
that artifact itself a leak.
"""

from __future__ import annotations

import json

import pytest

from analyst_assistant.obs.events import EventLog, Metrics, Recorder, new_trace_id


@pytest.fixture()
def recorder(tmp_path):
    return Recorder(EventLog(tmp_path / "logs"), Metrics(), user_id="exec_women_apparel")


# --------------------------------------------------------------------------
# Trace ids
# --------------------------------------------------------------------------


def test_trace_ids_are_unique_and_short():
    """Short because users quote them to support; a full UUID is hostile to retype."""
    ids = {new_trace_id() for _ in range(500)}
    assert len(ids) == 500
    assert all(len(i) == 12 for i in ids)


# --------------------------------------------------------------------------
# Spans
# --------------------------------------------------------------------------


def test_span_records_an_event_and_its_latency(recorder):
    with recorder.span("generate_sql") as m:
        m["model"] = "gemini-3.8-flash"
        m["tokens_in"] = 120
        m["tokens_out"] = 45

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.node == "generate_sql"
    assert event.status == "ok"
    assert event.model == "gemini-3.8-flash"
    assert event.tokens_in == 120
    assert event.latency_ms >= 0


def test_span_records_an_error_and_re_raises(recorder):
    """The event must be written even when the node fails.

    Failures are precisely what the log exists to capture, so recording cannot
    depend on the happy path completing.
    """
    with pytest.raises(ValueError), recorder.span("execute_bq"):
        raise ValueError("bad query")

    event = recorder.events[0]
    assert event.status == "error"
    assert event.error_class == "ValueError"
    assert "bad query" in event.attrs["error"]


def test_span_allows_an_explicit_non_error_status(recorder):
    """'Empty result' is a distinct outcome, not a failure."""
    with recorder.span("execute_bq") as m:
        m["status"] = "empty"
    assert recorder.events[0].status == "empty"


def test_events_accumulate_in_order(recorder):
    for node in ("triage", "generate_sql", "execute_bq"):
        with recorder.span(node):
            pass
    assert [e.node for e in recorder.events] == ["triage", "generate_sql", "execute_bq"]


def test_all_events_in_a_turn_share_one_trace_id(recorder):
    with recorder.span("a"):
        pass
    with recorder.span("b"):
        pass
    assert {e.trace_id for e in recorder.events} == {recorder.trace_id}


# --------------------------------------------------------------------------
# PII in the audit trail
# --------------------------------------------------------------------------


def test_pii_is_scrubbed_before_being_logged(tmp_path):
    """A WHERE email = '...' predicate must not be persisted into the trace."""
    log = EventLog(tmp_path / "logs")
    rec = Recorder(log, Metrics(), user_id="u1")

    with rec.span("generate_sql") as m:
        m["attrs"] = {
            "sql": "SELECT * FROM s_users WHERE email = 'dana.levi@example.com'",
            "question": "what did dana.levi@example.com spend?",
        }

    written = log.path.read_text(encoding="utf-8")
    assert "dana.levi@example.com" not in written
    assert "REDACTED" in written


def test_structural_fields_are_not_scrubbed(tmp_path):
    """Scrubbing is confined to free-text fields; measurements must survive intact."""
    log = EventLog(tmp_path / "logs")
    rec = Recorder(log, Metrics(), user_id="u1")
    with rec.span("execute_bq") as m:
        m["bytes_billed"] = 123456
        m["attrs"] = {"row_count": 42}

    record = json.loads(log.path.read_text(encoding="utf-8").strip())
    assert record["bytes_billed"] == 123456
    assert record["attrs"]["row_count"] == 42


# --------------------------------------------------------------------------
# Trace retrieval
# --------------------------------------------------------------------------


def test_trace_can_be_read_back_by_id(tmp_path):
    log = EventLog(tmp_path / "logs")
    rec = Recorder(log, Metrics(), user_id="u1")
    for node in ("triage", "generate_sql"):
        with rec.span(node):
            pass

    trace = log.read_trace(rec.trace_id)
    assert [e["node"] for e in trace] == ["triage", "generate_sql"]


def test_reading_an_unknown_trace_returns_empty(tmp_path):
    assert EventLog(tmp_path / "logs").read_trace("deadbeef") == []


def test_a_truncated_final_line_does_not_break_trace_reading(tmp_path):
    """A crash mid-write must not make /trace unusable afterwards."""
    log = EventLog(tmp_path / "logs")
    rec = Recorder(log, Metrics(), user_id="u1")
    with rec.span("triage"):
        pass
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"trace_id": "broken", "nod')

    assert [e["node"] for e in log.read_trace(rec.trace_id)] == ["triage"]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_node_outcomes_are_counted(recorder):
    with recorder.span("execute_bq"):
        pass
    with pytest.raises(RuntimeError), recorder.span("execute_bq"):
        raise RuntimeError("boom")

    assert recorder.metrics.counter("node.execute_bq.ok") == 1
    assert recorder.metrics.counter("node.execute_bq.error") == 1


def test_cost_signals_are_aggregated(recorder):
    with recorder.span("execute_bq") as m:
        m["bytes_billed"] = 1_000_000
        m["cache_hit"] = True
    with recorder.span("synthesize") as m:
        m["tokens_in"] = 100
        m["tokens_out"] = 50

    metrics = recorder.metrics
    assert metrics.counter("bq.bytes_billed") == 1_000_000
    assert metrics.counter("llm.calls") == 1
    assert metrics.counter("llm.tokens") == 150
    assert metrics.counter("cache.hits") == 1


def test_pii_tripwire_is_recorded_as_an_alarm(recorder):
    """Should never fire. A firing means an upstream control failed."""
    recorder.record_pii_tripwire({"email": 2})
    assert recorder.metrics.counter("safety.pii_tripwire.email") == 2
    assert recorder.metrics.counter("safety.pii_tripwire.total") == 2


def test_tripwire_with_no_findings_records_nothing(recorder):
    recorder.record_pii_tripwire({})
    assert recorder.metrics.counter("safety.pii_tripwire.total") == 0


def test_snapshot_exposes_counters_and_percentiles(recorder):
    with recorder.span("triage"):
        pass
    snap = recorder.metrics.snapshot()
    assert "node.triage.ok" in snap["counters"]
    assert snap["latency"]["node.triage.latency_ms"]["n"] == 1


def test_percentiles_on_empty_samples_do_not_crash():
    assert Metrics().snapshot() == {"counters": {}, "latency": {}}


# --------------------------------------------------------------------------
# User-facing output
# --------------------------------------------------------------------------


def test_cost_line_reports_calls_tokens_bytes_and_trace(recorder):
    """Requirement 5's cost clause should be observable, not merely true."""
    with recorder.span("synthesize") as m:
        m["tokens_in"] = 800
        m["tokens_out"] = 200
    with recorder.span("execute_bq") as m:
        m["bytes_billed"] = 2_500_000

    line = recorder.cost_line()
    assert "1 LLM call" in line
    assert "1000 tokens" in line
    assert "2.5 MB billed" in line
    assert recorder.trace_id in line


def test_debug_timeline_lists_every_node(recorder):
    with recorder.span("triage"):
        pass
    with pytest.raises(RuntimeError), recorder.span("execute_bq"):
        raise RuntimeError("x")

    timeline = "\n".join(recorder.timeline())
    assert "triage" in timeline
    assert "execute_bq" in timeline
    assert "RuntimeError" in timeline
