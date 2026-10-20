"""Table and column definitions for thelook_ecommerce, with a privacy class per column.

Everything else reads its rules from here, so the scoped views and the output
masker cannot drift apart.

Classification is per column rather than per table because the client's own
example question ("why are users in state X underspending vs state Y") needs
geography. State and city describe a population; street address, postal code and
lat/long identify a household.
"""

from __future__ import annotations

from enum import StrEnum

DATASET = "bigquery-public-data.thelook_ecommerce"


class PiiClass(StrEnum):
    """Privacy class of a column, which determines how it may be used."""

    DIRECT_IDENTIFIER = "direct_identifier"
    """Identifies a specific person. Never selectable, never printed, never logged.

    Enforced structurally: these columns are omitted from the scoped views, so they
    are *absent* from the queryable surface rather than present-but-forbidden. A
    prompt injection cannot talk its way past a column that does not exist.
    """

    JOIN_KEY = "join_key"
    """A direct identifier that is nonetheless required to join tables.

    Usable in JOIN and GROUP BY, but surfaced only as a stable pseudonym.
    This is what keeps "who are my top customers?" - an explicitly required
    capability - answerable without disclosing identity.
    """

    QUASI_IDENTIFIER = "quasi_identifier"
    """Not identifying alone, but identifying in combination.

    Safe in aggregate; bucketed when it appears in a narrow cross-tab alongside
    other quasi-identifiers.
    """

    ANALYTIC = "analytic"
    """Describes a population rather than a person. Freely usable."""


# 
# Table definitions. Column order matches the BigQuery schema so that generated
# view definitions read naturally against the real table.
# 
TABLES: dict[str, dict[str, PiiClass]] = {
    "orders": {
        "order_id": PiiClass.ANALYTIC,
        "user_id": PiiClass.JOIN_KEY,
        "status": PiiClass.ANALYTIC,
        "gender": PiiClass.ANALYTIC,
        "created_at": PiiClass.ANALYTIC,
        "returned_at": PiiClass.ANALYTIC,
        "shipped_at": PiiClass.ANALYTIC,
        "delivered_at": PiiClass.ANALYTIC,
        # num_of_item is deliberately withheld from the scoped view - see
        # entitlements.py. It counts every item on the order, including items from
        # categories the caller is not entitled to see.
        "num_of_item": PiiClass.ANALYTIC,
    },
    "order_items": {
        "id": PiiClass.ANALYTIC,
        "order_id": PiiClass.ANALYTIC,
        "user_id": PiiClass.JOIN_KEY,
        "product_id": PiiClass.ANALYTIC,
        "inventory_item_id": PiiClass.ANALYTIC,
        "status": PiiClass.ANALYTIC,
        "created_at": PiiClass.ANALYTIC,
        "shipped_at": PiiClass.ANALYTIC,
        "delivered_at": PiiClass.ANALYTIC,
        "returned_at": PiiClass.ANALYTIC,
        "sale_price": PiiClass.ANALYTIC,
    },
    "products": {
        "id": PiiClass.ANALYTIC,
        "cost": PiiClass.ANALYTIC,
        "category": PiiClass.ANALYTIC,
        "name": PiiClass.ANALYTIC,
        "brand": PiiClass.ANALYTIC,
        "retail_price": PiiClass.ANALYTIC,
        "department": PiiClass.ANALYTIC,
        "sku": PiiClass.ANALYTIC,
        "distribution_center_id": PiiClass.ANALYTIC,
    },
    "users": {
        "id": PiiClass.JOIN_KEY,
        "first_name": PiiClass.DIRECT_IDENTIFIER,
        "last_name": PiiClass.DIRECT_IDENTIFIER,
        "email": PiiClass.DIRECT_IDENTIFIER,
        "age": PiiClass.QUASI_IDENTIFIER,
        "gender": PiiClass.ANALYTIC,
        "state": PiiClass.ANALYTIC,
        "street_address": PiiClass.DIRECT_IDENTIFIER,
        "postal_code": PiiClass.DIRECT_IDENTIFIER,
        "city": PiiClass.ANALYTIC,
        "country": PiiClass.ANALYTIC,
        "latitude": PiiClass.DIRECT_IDENTIFIER,
        "longitude": PiiClass.DIRECT_IDENTIFIER,
        # A GEOGRAPHY point of the customer's location, so the same class as
        # lat/long. Found by the doctor schema-drift check rather than by reading
        # the docs, which is the reason that check exists.
        "user_geom": PiiClass.DIRECT_IDENTIFIER,
        "traffic_source": PiiClass.ANALYTIC,
        "created_at": PiiClass.ANALYTIC,
    },
}

# Columns withheld from a scoped view even though they are not PII, because their
# value is computed across the *whole* order and would therefore leak information
# about products outside the caller's entitlements.
SCOPE_LEAKING_COLUMNS: dict[str, frozenset[str]] = {
    "orders": frozenset({"num_of_item"}),
}

# Name of the scoped view for each base table. Model-authored SQL may reference
# only these names; the validator rejects everything else.
SCOPED_VIEW_NAMES: dict[str, str] = {
    "orders": "s_orders",
    "order_items": "s_order_items",
    "products": "s_products",
    "users": "s_users",
}

ALLOWED_TABLE_NAMES: frozenset[str] = frozenset(SCOPED_VIEW_NAMES.values())


def blocked_columns(table: str) -> frozenset[str]:
    """Columns that must never appear in a scoped view for table.

    Direct identifiers (privacy) plus scope-leaking columns (entitlement integrity).
    """
    cols = TABLES[table]
    direct = {c for c, k in cols.items() if k is PiiClass.DIRECT_IDENTIFIER}
    return frozenset(direct | set(SCOPE_LEAKING_COLUMNS.get(table, frozenset())))


def selectable_columns(table: str) -> list[str]:
    """Columns that a scoped view for table exposes, in schema order."""
    blocked = blocked_columns(table)
    return [c for c in TABLES[table] if c not in blocked]


def all_direct_identifiers() -> frozenset[str]:
    """Every direct-identifier column name across the dataset.

    Used by the SQL validator as defence in depth: the scoped views already omit
    these, so a reference to one means either a hallucinated column or an attempt to
    reach a base table.
    """
    return frozenset(
        col
        for cols in TABLES.values()
        for col, kind in cols.items()
        if kind is PiiClass.DIRECT_IDENTIFIER
    )


def schema_card() -> str:
    """A compact, human-readable schema summary.

    Serves two purposes: it is injected into the SQL-generation prompt, and it is
    the entire answer to "what data is available?" - a capability the brief requires
    and which should cost no BigQuery query and no LLM-generated SQL at all.
    """
    lines: list[str] = []
    for table, cols in TABLES.items():
        view = SCOPED_VIEW_NAMES[table]
        lines.append(f"{view}  (scoped view over {table})")
        for col in selectable_columns(table):
            kind = cols[col]
            note = ""
            if kind is PiiClass.JOIN_KEY:
                note = "  - join key; shown only as a pseudonym"
            elif kind is PiiClass.QUASI_IDENTIFIER:
                note = "  - quasi-identifier; aggregate only"
            lines.append(f"    {col}{note}")
        withheld = sorted(blocked_columns(table))
        if withheld:
            lines.append(f"    [withheld: {', '.join(withheld)}]")
        lines.append("")
    return "\n".join(lines).rstrip()
