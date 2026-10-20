"""Tests for the static SQL validator.

Structured as a red-team suite: most cases assert that something is *rejected*.
The validator is the single enforcement point for requirements 2 and 3, so these
tests are the evidence that it holds rather than a claim that it does.
"""

from __future__ import annotations

import pytest

from analyst_assistant.safety.sql_guard import MAX_ROW_LIMIT, validate

# Queries that must be accepted


def test_simple_aggregate_is_allowed():
    r = validate("SELECT SUM(sale_price) AS revenue FROM s_order_items")
    assert r.ok, r.reason


def test_join_across_scoped_views_is_allowed():
    r = validate(
        """
        SELECT u.state, SUM(oi.sale_price) AS revenue
        FROM s_order_items AS oi
        JOIN s_users AS u ON u.id = oi.user_id
        GROUP BY u.state
        ORDER BY revenue DESC
        """
    )
    assert r.ok, r.reason


def test_model_may_declare_its_own_ctes():
    r = validate(
        """
        WITH monthly AS (
            SELECT DATE_TRUNC(created_at, MONTH) AS m, SUM(sale_price) AS rev
            FROM s_order_items GROUP BY m
        )
        SELECT m, rev FROM monthly ORDER BY m
        """
    )
    assert r.ok, r.reason


def test_set_operations_are_allowed():
    r = validate("SELECT 1 AS n FROM s_orders UNION ALL SELECT 2 AS n FROM s_users")
    assert r.ok, r.reason


# Read-only enforcement


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM s_orders",
        "UPDATE s_orders SET status = 'x'",
        "DROP TABLE s_orders",
        "INSERT INTO s_orders (order_id) VALUES (1)",
        "CREATE TABLE evil AS SELECT * FROM s_users",
        "TRUNCATE TABLE s_orders",
    ],
)
def test_write_statements_are_rejected(sql):
    r = validate(sql)
    assert not r.ok
    assert r.violations


def test_bigquery_scripting_is_rejected():
    """EXECUTE IMMEDIATE parses to a generic Command node.

    Rejecting Command closes the whole 'syntax sqlglot does not model' class, rather
    than enumerating individual dangerous constructs.
    """
    r = validate("EXECUTE IMMEDIATE 'SELECT 1'")
    assert not r.ok


def test_stacked_statements_are_rejected():
    r = validate("SELECT 1 FROM s_orders; DROP TABLE s_orders")
    assert not r.ok
    assert "single statement" in r.reason


def test_semicolon_inside_a_string_literal_is_not_a_second_statement():
    """Guards against a naive split(';') implementation."""
    r = validate("SELECT 'a;b' AS s FROM s_orders")
    assert r.ok, r.reason


# Table whitelist


def test_direct_base_table_reference_is_rejected():
    r = validate(
        "SELECT email FROM `bigquery-public-data.thelook_ecommerce.users`"
    )
    assert not r.ok
    assert any("not permitted" in v for v in r.violations)


def test_unscoped_bare_table_name_is_rejected():
    """'products' is the base table; only 's_products' is reachable."""
    r = validate("SELECT category FROM products")
    assert not r.ok
    assert "unknown table" in r.reason


def test_cte_may_not_shadow_a_scoped_view():
    """The most direct attack: redefine the scoped view over the base table."""
    r = validate(
        """
        WITH s_products AS (
            SELECT * FROM `bigquery-public-data.thelook_ecommerce.products`
        )
        SELECT category FROM s_products
        """
    )
    assert not r.ok
    assert any("may not reuse a scoped view name" in v for v in r.violations)


# PII


@pytest.mark.parametrize(
    "column", ["email", "first_name", "last_name", "street_address", "postal_code", "latitude"]
)
def test_direct_identifier_columns_are_rejected(column):
    r = validate(f"SELECT {column} FROM s_users")
    assert not r.ok
    assert "personal data" in r.reason


def test_analytic_geography_is_still_allowed():
    """The brief's own example question needs state-level geography.

    A blanket geographic block would make the client's headline use case
    unanswerable, so the policy is field-level rather than categorical.
    """
    r = validate("SELECT state, city, country FROM s_users")
    assert r.ok, r.reason


# Fail-closed behaviour


def test_unparseable_sql_is_rejected_not_passed_through():
    r = validate("SELECT FROM WHERE ((")
    assert not r.ok
    assert "could not be parsed" in r.reason


def test_empty_query_is_rejected():
    assert not validate("").ok


# Row limits


def test_missing_limit_is_injected():
    r = validate("SELECT order_id FROM s_orders", default_limit=500)
    assert r.ok
    assert "LIMIT 500" in r.sql.upper()
    assert any("row limit" in n for n in r.notes)


def test_excessive_limit_is_clamped_and_reported():
    r = validate(f"SELECT order_id FROM s_orders LIMIT {MAX_ROW_LIMIT * 10}")
    assert r.ok
    assert str(MAX_ROW_LIMIT) in r.sql
    assert any("reduced" in n for n in r.notes)


def test_reasonable_explicit_limit_is_preserved():
    r = validate("SELECT order_id FROM s_orders LIMIT 25")
    assert r.ok
    assert "LIMIT 25" in r.sql.upper()
    assert not r.notes
