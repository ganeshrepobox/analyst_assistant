"""Tests for pseudonymisation, DataFrame masking and the egress tripwire.

False positives are tested as deliberately as false negatives. A masker that
redacts order totals and SKUs destroys the answer it was protecting.
"""

from __future__ import annotations

import pandas as pd

from analyst_assistant.safety.masking import (
    mask_dataframe,
    pseudonymize,
    scrub,
)

# --------------------------------------------------------------------------
# Pseudonymisation
# --------------------------------------------------------------------------


def test_pseudonym_is_stable_for_the_same_input():
    """Stability is what makes ranking and follow-up questions coherent."""
    assert pseudonymize(4211, salt="s") == pseudonymize(4211, salt="s")


def test_pseudonym_differs_across_customers():
    assert pseudonymize(1, salt="s") != pseudonymize(2, salt="s")


def test_salt_changes_the_mapping():
    """Unsalted, this dataset's id space is small enough to brute-force."""
    assert pseudonymize(1, salt="a") != pseudonymize(1, salt="b")


def test_pseudonym_is_recognisable_and_short():
    label = pseudonymize(99, salt="s")
    assert label.startswith("CUST-")
    assert len(label) == len("CUST-") + 6


# --------------------------------------------------------------------------
# Tripwire: true positives
# --------------------------------------------------------------------------


def test_email_is_redacted_and_recorded():
    r = scrub("Contact dana.levi@example.com about the Q1 dip.")
    assert "dana.levi@example.com" not in r.text
    assert r.findings == {"email": 1}
    assert r.fired


def test_phone_is_redacted():
    r = scrub("Call 415-555-0134 for details.")
    assert "415-555-0134" not in r.text
    assert "phone" in r.findings


def test_multiple_findings_are_counted_per_kind():
    r = scrub("a@b.com and c@d.com")
    assert r.findings["email"] == 2
    assert r.total == 2


# --------------------------------------------------------------------------
# Tripwire: false positives must not occur
# --------------------------------------------------------------------------


def test_clean_analytical_text_is_untouched():
    text = (
        "Revenue in Women's Apparel fell 12.4% QoQ to $1,284,900 across 18,322 orders. "
        "Return rate rose to 9.1% and AOV fell from $84.20 to $79.10."
    )
    r = scrub(text)
    assert r.text == text
    assert not r.fired, f"false positive: {r.findings}"


def test_product_identifiers_and_dates_survive():
    text = "SKU 7741290 shipped 2024-03-14; order 55512345 totalled $421.00."
    r = scrub(text)
    assert not r.fired, f"false positive: {r.findings}"


def test_empty_text_is_handled():
    assert scrub("").text == ""
    assert not scrub("").fired


# --------------------------------------------------------------------------
# DataFrame masking
# --------------------------------------------------------------------------


def test_direct_identifier_columns_are_dropped():
    """These should never arrive -- the scoped views omit them.

    Arrival means an upstream control failed, so masking records it.
    """
    df = pd.DataFrame({"email": ["a@b.com"], "state": ["CA"], "revenue": [10.0]})
    out, findings = mask_dataframe(df)
    assert "email" not in out.columns
    assert "state" in out.columns and "revenue" in out.columns
    assert any(k.startswith("dropped:email") for k in findings)


def test_join_keys_are_pseudonymized_in_place():
    """Replaced rather than dropped, so 'top customers' still ranks correctly."""
    df = pd.DataFrame({"user_id": [1, 2, 1], "spend": [10.0, 20.0, 5.0]})
    out, findings = mask_dataframe(df, salt="s")

    assert "user_id" in out.columns
    assert all(str(v).startswith("CUST-") for v in out["user_id"])
    # The same customer keeps the same label, so grouping is preserved.
    assert out["user_id"].iloc[0] == out["user_id"].iloc[2]
    assert out["user_id"].iloc[0] != out["user_id"].iloc[1]
    assert any(k.startswith("pseudonymized:user_id") for k in findings)


def test_qualified_column_names_are_recognised():
    df = pd.DataFrame({"u.email": ["a@b.com"], "oi.sale_price": [1.0]})
    out, _ = mask_dataframe(df)
    assert "u.email" not in out.columns
    assert "oi.sale_price" in out.columns


def test_analytic_columns_are_left_alone():
    df = pd.DataFrame({"state": ["CA", "TX"], "revenue": [1.0, 2.0]})
    out, findings = mask_dataframe(df)
    assert out.equals(df)
    assert not findings


def test_masking_does_not_mutate_the_input():
    df = pd.DataFrame({"user_id": [1], "email": ["a@b.com"]})
    before = df.copy()
    mask_dataframe(df)
    assert df.equals(before)
