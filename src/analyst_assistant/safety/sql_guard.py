"""Static SQL validation: a fail-closed whitelist over the parsed AST.

This is the single enforcement point for requirements 2 and 3. It runs on
model-authored SQL *before* the scoped CTEs are prepended, and it uses no LLM --
which makes it cheap, fast, deterministic, unit-testable, and immune to persuasion.

Design rules
------------
**Whitelist, never blacklist.** The query must reference only the four scoped view
names or CTEs it declares itself. Anything else is rejected. A blacklist of forbidden
patterns is a losing game against a generative model that can phrase the same access
a dozen ways.

**Fail closed.** If the SQL cannot be parsed, it is rejected. "Unparseable" is never
treated as "probably fine" -- a validator that waves through what it does not
understand is not a validator.

**Reject, never repair.** The validator returns violations and the caller decides
(regenerate once with the violation as feedback, then refuse). It never silently
edits the query to make it safe, because a silent edit changes the meaning of the
answer without telling anyone. The single exception is the row limit, which is
clamped rather than rejected, and that clamp is reported to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from .schema import ALLOWED_TABLE_NAMES, all_direct_identifiers

#: Node types that must never appear anywhere in the tree. ``exp.Command`` is the
#: important one: sqlglot parses constructs it does not model -- including BigQuery
#: scripting such as ``EXECUTE IMMEDIATE`` -- into a generic Command node, so
#: rejecting Command closes the "unsupported syntax" hole as a class rather than
#: enumerating members of it.
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

#: Functions that reach outside the dataset or execute dynamic SQL.
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
    """Validate model-authored SQL against the scoped-view whitelist.

    Args:
        model_sql: SQL as written by the model, referencing only scoped view names.
        default_limit: Row limit applied when the query specifies none.
        max_limit: Ceiling; a larger explicit limit is clamped down to this.

    Returns:
        A :class:`ValidationResult`. When ``ok`` is false, ``violations`` explains
        why in terms suitable for feeding back to the model for one regeneration
        attempt.
    """
    violations: list[str] = []
    notes: list[str] = []

    # --- 1. Exactly one statement -------------------------------------------
    # Stacked statements are the classic injection shape, and BigQuery will happily
    # run a script. Parsing the whole string and counting is the only reliable test;
    # splitting on ';' is defeated by a semicolon inside a string literal.
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

    # --- 2. Read-only shape --------------------------------------------------
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

    # --- 3. Table whitelist --------------------------------------------------
    declared_ctes = {c.alias_or_name for c in tree.find_all(exp.CTE)}

    # A CTE named s_products would shadow the scoped view and could be defined over
    # the base table -- the most direct way to defeat entitlement scoping.
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

    # --- 4. PII columns (defence in depth) -----------------------------------
    # The scoped views already omit these, so a reference means either a
    # hallucinated column or an attempt to reach a base table. Either way it is a
    # clearer error message than BigQuery's "field not found".
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

    # --- 5. Dangerous functions ----------------------------------------------
    for func in tree.find_all(exp.Anonymous):
        if (func.name or "").lower() in FORBIDDEN_FUNCTIONS:
            violations.append(f"function {func.name} is not permitted")

    if violations:
        return ValidationResult(ok=False, sql=model_sql, violations=violations)

    # --- 6. Row limit --------------------------------------------------------
    # Clamped rather than rejected: an unbounded result is a cost and memory risk,
    # not a safety breach, and silently failing a valid question would be worse UX
    # than trimming it and saying so.
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
