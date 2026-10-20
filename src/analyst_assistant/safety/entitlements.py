"""Builds the set of views a given user is allowed to query.

Rather than rewriting the model's SQL, we prepend four scoped CTEs and let
sql_guard reject anything that references something else.

The rewrite approach has a hole. "SELECT SUM(sale_price) FROM order_items"
mentions no products table, so a filter keyed on products never fires and the
user sees company-wide revenue. Since sale_price is the only revenue column in
this dataset, that is the common case rather than an edge case. A rewrite that
misses a case fails open; a validator that misses one fails closed.

Production equivalent: BigQuery authorized views plus row-level access policies.
Not possible here, since row-level security needs table ownership and
bigquery-public-data is read-only to us.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from .schema import (
    DATASET,
    SCOPED_VIEW_NAMES,
    selectable_columns,
)

# sqlglot 30 names these args with a trailing underscore to avoid clashing with
# Python keywords. Resolved dynamically so a future rename fails loudly at import
# rather than silently writing to an argument the SQL generator ignores - which is
# exactly the bug this constant exists to prevent.
_WITH_ARG = "with_" if "with_" in exp.Select.arg_types else "with"


@dataclass(frozen=True)
class Entitlement:
    """What one user is allowed to analyze.

    categories=None means unrestricted, which still gets the PII-stripped views.
    user_scope "all" exposes every customer; "purchasers" only those who bought
    something in an entitled category. See docs/ASSUMPTIONS.md A2.
    """

    user_id: str
    display_name: str
    categories: frozenset[str] | None = None
    user_scope: str = "all"
    persona: str = "default"
    preferences: dict[str, str] = field(default_factory=dict)

    @property
    def is_unrestricted(self) -> bool:
        return self.categories is None

    def describe_scope(self, total_categories: int = 26) -> str:
        """Scope line attached to every response.

        We answer within scope rather than refusing, but a silently narrowed total looks
        like a complete answer, so we always say what was applied.
        """
        if self.is_unrestricted:
            return "Scope: all product categories"
        cats = sorted(self.categories or ())
        shown = ", ".join(cats[:3]) + (f" +{len(cats) - 3} more" if len(cats) > 3 else "")
        return f"Scope: {shown} ({len(cats)} of {total_categories} categories)"


def _qualified(table: str) -> str:
    return f"`{DATASET}.{table}`"


def _category_filter(categories: frozenset[str]) -> exp.Expression:
    """Build category IN (...) through the AST so literals are escaped properly.

    Category names here contain apostrophes (Women's Apparel).
    """
    literals = [exp.Literal.string(c) for c in sorted(categories)]
    return exp.column("category").isin(*literals)


def build_scoped_ctes(ent: Entitlement) -> list[tuple[str, str]]:
    """Return (view_name, sql_body) pairs in dependency order.

    s_order_items references s_products, and BigQuery needs CTEs defined before use.
    """
    # NOTE: dialect="bigquery" is required on from_(). Without it sqlglot parses the
    # table string with its DEFAULT dialect, where BigQuery's backtick-quoted
    # `project.dataset.table` is a syntax error.
    products = exp.select(*selectable_columns("products")).from_(
        _qualified("products"), dialect="bigquery"
    )
    if not ent.is_unrestricted:
        products = products.where(_category_filter(ent.categories or frozenset()))

    # order_items is scoped by an inner join to the entitled products, so an item
    # belonging to a non-entitled product is unreachable rather than filtered later.
    oi_cols = ", ".join(f"oi.{c}" for c in selectable_columns("order_items"))
    order_items = (
        f"SELECT {oi_cols} FROM {_qualified('order_items')} AS oi "
        f"JOIN s_products AS sp ON sp.id = oi.product_id"
    )

    # Orders are restricted to those containing at least one entitled item.
    # NOTE: orders.num_of_item is withheld (see schema.SCOPE_LEAKING_COLUMNS) because
    # it counts every item on the order including non-entitled ones, which would leak
    # information about products outside the caller's scope.
    o_cols = ", ".join(f"o.{c}" for c in selectable_columns("orders"))
    orders = (
        f"SELECT {o_cols} FROM {_qualified('orders')} AS o "
        f"WHERE o.order_id IN (SELECT order_id FROM s_order_items)"
    )

    u_cols = ", ".join(f"u.{c}" for c in selectable_columns("users"))
    if ent.user_scope == "purchasers":
        users = (
            f"SELECT {u_cols} FROM {_qualified('users')} AS u "
            f"WHERE u.id IN (SELECT user_id FROM s_order_items)"
        )
    else:
        users = f"SELECT {u_cols} FROM {_qualified('users')} AS u"

    return [
        (SCOPED_VIEW_NAMES["products"], products.sql(dialect="bigquery")),
        (SCOPED_VIEW_NAMES["order_items"], order_items),
        (SCOPED_VIEW_NAMES["orders"], orders),
        (SCOPED_VIEW_NAMES["users"], users),
    ]


def _make_cte(name: str, body: str) -> exp.CTE:
    return exp.CTE(
        this=sqlglot.parse_one(body, dialect="bigquery"),
        alias=exp.TableAlias(this=exp.to_identifier(name)),
    )


def compose(model_sql: str, ent: Entitlement) -> str:
    """Prepend the scoped CTEs to the model's SQL.

    The model's own CTEs are kept and placed after ours, since they may reference the
    scoped views. Raises sqlglot.ParseError, which callers must treat as a rejection.
    """
    tree = sqlglot.parse_one(model_sql, dialect="bigquery")

    if _WITH_ARG not in tree.arg_types:
        # Set operations and other roots that cannot carry a WITH: wrap so the
        # scoped CTEs still apply. The wrapper preserves row semantics.
        tree = exp.select("*").from_(exp.Subquery(this=tree, alias=exp.to_identifier("_scoped_q")))

    ours = [_make_cte(name, body) for name, body in build_scoped_ctes(ent)]
    existing = tree.args.get(_WITH_ARG)
    if existing is not None:
        existing.set("expressions", ours + existing.expressions)
    else:
        tree.set(_WITH_ARG, exp.With(expressions=ours))

    return tree.sql(dialect="bigquery", pretty=True)
