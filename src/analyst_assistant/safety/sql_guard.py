"""Static checks on model-generated SQL, run before the scoped CTEs are added.

Whitelist rather than blacklist: a query may only reference the four scoped views
or CTEs it declares itself. Unparseable SQL is rejected rather than passed
through.

No LLM involved, so it is cheap and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from .schema import ALLOWED_TABLE_NAMES, all_direct_identifiers

# Node types that must never appear anywhere in the tree. exp.Command is the
# important one: sqlglot parses constructs it does not model - including BigQuery
# scripting such as EXECUTE IMMEDIATE - into a generic Command node, so
# rejecting Command closes the "unsupported syntax" hole as a class rather than
# enumerating members of it.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Merge,
    exp.Command,
    exp.Grant,
    exp.TruncateTable,
)

# Functions that reach outside the dataset or execute dynamic SQL.
FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {"external_query", "session_user", "net.host", "execute_immediate"}
)

DEFAULT_ROW_LIMIT = 1000
MAX_ROW_LIMIT = 10_000


@dataclass
class ValidationResult:
    """Outcome of validating one candidate query."""

    ok: bool
    sql: str
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.violations)


def _limit_value(tree: exp.Expression) -> int | None:
    limit = tree.args.get("limit")
    if limit is None:
        return None
    try:
        return int(limit.expression.name)
    except (AttributeError, ValueError):
        return None


def validate(
    model_sql: str,
    *,
    default_limit: int = DEFAULT_ROW_LIMIT,
    max_limit: int = MAX_ROW_LIMIT,
) -> ValidationResult:
    """Check model SQL against the scoped-view whitelist.

    Returns a ValidationResult. When ok is false, violations is phrased so it can be
    fed back to the model for one regeneration attempt.
    """
    violations: list[str] = []
    notes: list[str] = []

    # Exactly one statement. Stacked statements are the classic injection shape and
    # BigQuery will run a script happily. Counting parsed statements is the only
    # reliable test; splitting on ';' breaks on a semicolon inside a string literal.
    try:
        statements = sqlglot.parse(model_sql, dialect="bigquery")
    except sqlglot.ParseError as e:
        return ValidationResult(
            ok=False,
            sql=model_sql,
            violations=[f"SQL could not be parsed, so it is rejected: {e}"],
        )

    statements = [s for s in statements if s is not None]
    if len(statements) == 0:
        return ValidationResult(ok=False, sql=model_sql, violations=["empty query"])
    if len(statements) > 1:
        return ValidationResult(
            ok=False,
            sql=model_sql,
            violations=[f"expected a single statement, found {len(statements)}"],
        )

    tree = statements[0]

    # Read-only shape: SELECT and set operations only.
    if not isinstance(tree, (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)):
        return ValidationResult(
            ok=False,
            sql=model_sql,
            violations=[f"only SELECT queries are allowed, got {type(tree).__name__}"],
        )

    for node_type in FORBIDDEN_NODES:
        if isinstance(tree, node_type) or any(True for _ in tree.find_all(node_type)):
            violations.append(
                f"{node_type.__name__.upper()} is not permitted; the connection is read-only"
            )

    # Only the scoped views, or CTEs the query declares itself.
    declared_ctes = {c.alias_or_name for c in tree.find_all(exp.CTE)}

    # A CTE named s_products would shadow the scoped view and could be defined over
    # the base table, which is the most direct way to defeat the scoping.
    shadowed = declared_ctes & ALLOWED_TABLE_NAMES
    if shadowed:
        violations.append(
            f"CTE may not reuse a scoped view name: {', '.join(sorted(shadowed))}"
        )

    permitted = ALLOWED_TABLE_NAMES | declared_ctes
    for table in tree.find_all(exp.Table):
        if table.catalog or table.db:
            violations.append(
                f"direct reference to `{table.catalog}.{table.db}.{table.name}` is not "
                f"permitted; query the scoped views instead"
            )
            continue
        if table.name and table.name not in permitted:
            violations.append(
                f"unknown table '{table.name}'; allowed: "
                f"{', '.join(sorted(ALLOWED_TABLE_NAMES))}"
            )

    # Defence in depth. The scoped views already omit these, so a reference means
    # either a hallucinated column or an attempt at a base table. Either way this is
    # a clearer message than BigQuery's "field not found".
    blocked = all_direct_identifiers()
    seen_pii = {
        col.name.lower()
        for col in tree.find_all(exp.Column)
        if col.name and col.name.lower() in blocked
    }
    if seen_pii:
        violations.append(
            f"personal data columns cannot be queried: {', '.join(sorted(seen_pii))}"
        )

    # Functions that reach outside the dataset.
    for func in tree.find_all(exp.Anonymous):
        if (func.name or "").lower() in FORBIDDEN_FUNCTIONS:
            violations.append(f"function {func.name} is not permitted")

    if violations:
        return ValidationResult(ok=False, sql=model_sql, violations=violations)

    # Clamped rather than rejected. An unbounded result is a cost and memory risk,
    # not a safety breach, and failing a valid question outright would be worse than
    # trimming it and saying so.
    current = _limit_value(tree)
    if current is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(default_limit)))
        notes.append(f"row limit of {default_limit} applied")
    elif current > max_limit:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(max_limit)))
        notes.append(f"row limit reduced from {current} to {max_limit}")

    return ValidationResult(
        ok=True, sql=tree.sql(dialect="bigquery", pretty=True), notes=notes
    )
