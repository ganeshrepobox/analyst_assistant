"""Entitlement scoping: build the queryable surface a given user is allowed to see.

The mechanism
-------------
Rather than inspecting and *rewriting* model-authored SQL, this module **prepends** a
set of scoped CTEs and the validator (``sql_guard``) then requires that the query
reference nothing else. The model writes SQL against ``s_products``,
``s_order_items``, ``s_orders`` and ``s_users``; it has no reachable path to a base
table.

Why not rewriting
-----------------
The obvious design is "find references to ``products`` and add a category filter".
It has a bypass that covers the *common* case rather than an edge case::

    SELECT SUM(sale_price) FROM order_items WHERE created_at >= '2024-01-01'

That query names no ``products`` table, so no filter is injected, and the caller sees
company-wide revenue. Since ``order_items.sale_price`` is the only revenue column in
this dataset, most real executive questions have exactly this shape.

The deeper reason is asymmetry of failure: **a rewrite that misses a case fails open;
a validator that misses a case fails closed.** For a security control that settles
the design.

A useful side effect is that PII becomes *structural*: ``email`` is not
"present but forbidden", it is absent from the view. A prompt injection cannot argue
its way to a column that does not exist.

Production equivalent
---------------------
In the client's own warehouse this same shape is expressed as BigQuery authorized
views plus row-level access policies keyed on ``SESSION_USER()``, with the agent's
service account holding no grant on the base tables. That is strictly better, because
the warehouse enforces it rather than the application. It is impossible here only
because row-level security requires table ownership and ``bigquery-public-data`` is
read-only to us -- so the scoped-CTE design was chosen to mirror the production shape
as closely as the constraint allows.
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
# rather than silently writing to an argument the SQL generator ignores -- which is
# exactly the bug this constant exists to prevent.
_WITH_ARG = "with_" if "with_" in exp.Select.arg_types else "with"


@dataclass(frozen=True)
class Entitlement:
    """What one user is permitted to analyze.

    Attributes:
        user_id: Stable identifier, used for report ownership and audit records.
        display_name: Shown in the CLI.
        categories: Entitled ``products.category`` values. ``None`` means
            unrestricted (e.g. a CEO persona), which still receives the scoped views
            so that PII removal continues to apply.
        user_scope: ``"all"`` exposes every customer with PII stripped;
            ``"purchasers"`` restricts to customers who bought an entitled product.
            Default ``"all"`` because the brief's own example question -- comparing
            spend across states -- needs customers who *did not* buy as the
            denominator. See ``docs/ASSUMPTIONS.md`` A2.
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
        """Human-readable scope line attached to every response.

        Requirement 2 is satisfied either by refusing out-of-scope questions or by
        answering within scope. We answer within scope -- but a silently narrowed
        aggregate is dangerous, because it looks like a complete answer and an
        executive may act on it. So the scope is always stated.
        """
        if self.is_unrestricted:
            return "Scope: all product categories"
        cats = sorted(self.categories or ())
        shown = ", ".join(cats[:3]) + (f" +{len(cats) - 3} more" if len(cats) > 3 else "")
        return f"Scope: {shown} ({len(cats)} of {total_categories} categories)"


def _qualified(table: str) -> str:
    return f"`{DATASET}.{table}`"


def _category_filter(categories: frozenset[str]) -> exp.Expression:
    """Build ``category IN (...)`` with literals escaped by sqlglot.

    Built through the AST rather than string interpolation: category names in this
    dataset contain apostrophes ("Women's Apparel"), and hand-rolled quoting is how
    injection bugs get introduced.
    """
    literals = [exp.Literal.string(c) for c in sorted(categories)]
    return exp.column("category").isin(*literals)


def build_scoped_ctes(ent: Entitlement) -> list[tuple[str, str]]:
    """Return ``(view_name, sql_body)`` pairs, in dependency order.

    Order matters: BigQuery requires a CTE to be defined before it is referenced, and
    ``s_order_items`` depends on ``s_products``.
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
    """Prepend the scoped CTEs to model-authored SQL.

    The model's own CTEs are **preserved** and placed after ours, since they may
    reference the scoped views and BigQuery requires define-before-use.

    Raises:
        sqlglot.ParseError: if ``model_sql`` cannot be parsed. Callers must treat
            this as a rejection -- never as "pass it through unmodified".
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
