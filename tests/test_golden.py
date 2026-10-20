"""Golden Bucket: the trios, the metric registry and retrieval.

The test that matters most is test_every_trio_passes_our_own_validator. The trios
are few-shot examples, so a trio the validator would reject is an example teaching
the model to write SQL we then refuse to run.
"""

from __future__ import annotations

import json

import pytest
import sqlglot

from analyst_assistant.golden.bucket import GoldenBucket, MetricRegistry, Trio
from analyst_assistant.safety.entitlements import Entitlement, compose
from analyst_assistant.safety.sql_guard import validate

RESTRICTED = Entitlement(
    "exec_women_apparel", "VP", categories=frozenset({"Women's Apparel", "Accessories"})
)


@pytest.fixture(scope="module")
def bucket():
    return GoldenBucket("golden_bucket")


# The seeded corpus


def test_trios_are_loaded(bucket):
    assert len(bucket.trios) >= 5
    assert all(t.question and t.sql for t in bucket.trios)


def test_every_trio_passes_our_own_validator(bucket):
    """An example the validator rejects teaches the model to write rejected SQL."""
    failures = [(t.id, validate(t.sql).reason) for t in bucket.trios if not validate(t.sql).ok]
    assert not failures, failures


def test_every_trio_still_composes_under_entitlement_scoping(bucket):
    """Scoping rewrites the query, so the examples must survive it."""
    for trio in bucket.trios:
        composed = compose(validate(trio.sql).sql, RESTRICTED)
        sqlglot.parse_one(composed, dialect="bigquery")  # raises if malformed


def test_trio_metrics_all_exist_in_the_registry(bucket):
    """A trio citing an undefined metric would make the footnote wrong."""
    unknown = {
        (t.id, m) for t in bucket.trios for m in t.metrics if m not in bucket.registry.definitions
    }
    assert not unknown, unknown


# Metric registry


def test_registry_loads_definitions(bucket):
    assert "net_revenue" in bucket.registry.definitions


def test_net_revenue_excludes_cancelled_and_returned(bucket):
    """The whole point of the registry.

    Omitting this filter overstates revenue by 32.7% on this dataset, and nothing
    in the schema tells a model which is meant.
    """
    spec = bucket.registry.definitions["net_revenue"]
    assert "Cancelled" in spec["filter"]
    assert "Returned" in spec["filter"]


def test_gross_revenue_is_available_but_marked_as_not_the_default(bucket):
    spec = bucket.registry.definitions["gross_revenue"]
    assert not spec.get("filter")
    assert "default" in spec["note"].lower()


def test_prompt_block_carries_filters_not_just_expressions():
    registry = MetricRegistry(
        {"net_revenue": {"sql": "SUM(x)", "filter": "status != 'Returned'", "plain": "net"}}
    )
    block = registry.prompt_block()
    assert "SUM(x)" in block
    assert "status != 'Returned'" in block


def test_footnote_is_plain_english_for_the_report():
    registry = MetricRegistry(
        {"net_revenue": {"sql": "SUM(x)", "plain": "Revenue excludes returns"}}
    )
    assert registry.footnote(["net_revenue"]) == "Revenue excludes returns"


def test_footnote_ignores_unknown_metrics():
    assert MetricRegistry({}).footnote(["nonexistent"]) == ""


# Retrieval


def test_lexical_retrieval_finds_the_obvious_match(bucket):
    """Works with no index, so the agent still functions offline."""
    hits = bucket.retrieve("why do Texas customers spend less than California")
    assert hits
    assert hits[0].id == "trio_002"


def test_retrieval_respects_k(bucket):
    assert len(bucket.retrieve("revenue", k=2)) <= 2


def test_retrieval_on_an_unrelated_question_returns_nothing(bucket):
    """Better to send no example than a misleading one."""
    assert bucket.retrieve("qwertyuiop zxcvbnm") == []


def test_low_quality_trios_are_not_retrieved(tmp_path):
    folder = tmp_path / "trios"
    folder.mkdir()
    (folder / "bad.json").write_text(
        json.dumps(
            {
                "id": "bad",
                "question": "revenue by state",
                "sql": "SELECT 1 FROM s_orders",
                "metadata": {"quality_score": 0.1},
            }
        ),
        encoding="utf-8",
    )
    assert GoldenBucket(tmp_path).retrieve("revenue by state") == []


def test_a_malformed_trio_file_is_skipped_not_fatal(tmp_path):
    folder = tmp_path / "trios"
    folder.mkdir()
    (folder / "broken.json").write_text("{ not json", encoding="utf-8")
    assert GoldenBucket(tmp_path).trios == []


# The prompt block


def test_context_always_includes_the_definitions(bucket):
    """Examples are the nearest few; definitions go in every time."""
    context = bucket.context("anything at all")
    assert "net_revenue" in context
    assert "Cancelled" in context


def test_context_includes_examples_when_one_matches(bucket):
    context = bucket.context("why are Texas customers underspending")
    assert "Queries analysts wrote" in context
    assert "spend_per_customer" in context


def test_question_catalog_answers_what_can_i_ask(bucket):
    catalog = bucket.question_catalog()
    assert catalog and all(isinstance(q, str) for q in catalog)


def test_trio_example_shows_question_and_sql():
    trio = Trio(id="t", question="How much?", sql="SELECT 1", report="")
    example = trio.as_example()
    assert "How much?" in example and "SELECT 1" in example
