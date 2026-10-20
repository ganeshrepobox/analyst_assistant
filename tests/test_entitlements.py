"""Tests for entitlement scoping.

The headline test is test_revenue_query_is_scoped_without_naming_products,
which is a regression test for a real bypass in an earlier design, see the module
docstring of safety/entitlements.py.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from analyst_assistant.safety.entitlements import Entitlement, build_scoped_ctes, compose
from analyst_assistant.safety.schema import blocked_columns

RESTRICTED = Entitlement(
    user_id="exec_women_apparel",
    display_name="VP Women's Apparel",
    categories=frozenset({"Women's Apparel", "Accessories"}),
)

UNRESTRICTED = Entitlement(user_id="ceo", display_name="CEO", categories=None)


def _cte_names(sql: str) -> list[str]:
    tree = sqlglot.parse_one(sql, dialect="bigquery")
    return [c.alias_or_name for c in tree.find_all(exp.CTE)]


# Composition mechanics


def test_all_four_scoped_views_are_prepended_in_dependency_order():
    out = compose("SELECT 1 AS n FROM s_orders", RESTRICTED)
    assert _cte_names(out) == ["s_products", "s_order_items", "s_orders", "s_users"]


def test_model_ctes_are_preserved_and_come_after_ours():
    """Regression: an earlier implementation silently discarded the model's CTEs.

    BigQuery requires define-before-use, so ours must come first, but the model's
    must survive, or the query returned is not the query that was validated.
    """
    out = compose(
        "WITH monthly AS (SELECT SUM(sale_price) AS r FROM s_order_items) SELECT r FROM monthly",
        RESTRICTED,
    )
    names = _cte_names(out)
    assert names == ["s_products", "s_order_items", "s_orders", "s_users", "monthly"]


def test_composed_sql_is_valid_bigquery():
    out = compose(
        "SELECT u.state, SUM(oi.sale_price) AS rev FROM s_order_items oi "
        "JOIN s_users u ON u.id = oi.user_id GROUP BY u.state",
        RESTRICTED,
    )
    sqlglot.parse_one(out, dialect="bigquery")  # raises if malformed


# The bypass this design exists to close


def test_revenue_query_is_scoped_without_naming_products():
    """The regression test for the bypass described in entitlements.py.

    An earlier design filtered references to products. This query names no products
    table, so no filter was applied and the caller saw company-wide revenue. Since
    sale_price is the only revenue column here, that was the common case.

    Now the scope travels through s_order_items' join to the filtered s_products, so
    leaving a table out does not sidestep it.
    """
    out = compose("SELECT SUM(sale_price) AS revenue FROM s_order_items", RESTRICTED)

    assert "Women\\'s Apparel" in out or "Women's Apparel" in out, "category filter missing"
    assert "JOIN s_products" in out, "order_items is not joined to the entitled products"


def test_unrestricted_user_gets_no_category_filter_but_still_gets_scoped_views():
    out = compose("SELECT SUM(sale_price) AS revenue FROM s_order_items", UNRESTRICTED)
    assert "category IN" not in out
    # PII removal is independent of entitlement breadth.
    assert "email" not in out
    assert _cte_names(out) == ["s_products", "s_order_items", "s_orders", "s_users"]


# Structural PII removal


def test_direct_identifiers_are_absent_from_the_scoped_views():
    """PII is structural: the columns are not forbidden, they are not there."""
    ctes = dict(build_scoped_ctes(RESTRICTED))
    users_view = ctes["s_users"]
    for column in blocked_columns("users"):
        assert column not in users_view, f"{column} leaked into s_users"


def test_analytic_geography_survives():
    ctes = dict(build_scoped_ctes(RESTRICTED))
    for column in ("state", "city", "country"):
        assert column in ctes["s_users"]


def test_num_of_item_is_withheld_from_scoped_orders():
    """Not PII, but it counts items outside the caller's entitlements.

    Exposing it would leak the size of orders in categories the caller cannot see.
    """
    ctes = dict(build_scoped_ctes(RESTRICTED))
    assert "num_of_item" not in ctes["s_orders"]


# SQL literal safety


def test_category_with_apostrophe_is_escaped_not_broken():
    """"Women's Apparel" is a real category value.

    Built through the sqlglot AST rather than string interpolation, because
    hand-rolled quoting is how injection bugs are introduced.
    """
    out = compose("SELECT 1 AS n FROM s_products", RESTRICTED)
    sqlglot.parse_one(out, dialect="bigquery")  # would raise on a broken literal


def test_category_values_cannot_inject_sql():
    hostile = Entitlement(
        user_id="attacker",
        display_name="hostile config",
        categories=frozenset({"x') OR 1=1 --"}),
    )
    out = compose("SELECT 1 AS n FROM s_products", hostile)
    tree = sqlglot.parse_one(out, dialect="bigquery")
    # The payload must survive as a single string literal, not as syntax.
    literals = [literal.this for literal in tree.find_all(exp.Literal) if literal.is_string]
    assert "x') OR 1=1 --" in literals


# Scope reporting


def test_scope_is_described_for_the_user():
    """A silently narrowed aggregate looks like a complete answer, so say the scope."""
    assert "2 of 26" in RESTRICTED.describe_scope()
    assert "all product categories" in UNRESTRICTED.describe_scope()
