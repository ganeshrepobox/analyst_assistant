# Code walkthrough

A reading order for the repo, and the reasoning behind each part. Roughly 20 minutes
if you read it straight through.

## The one-paragraph version

An executive asks a question in English. The agent classifies it, retrieves similar
past analyses, writes SQL, checks that SQL against a whitelist, runs it with a cost
ceiling, masks anything personal, and writes up the answer. Saved reports can be
deleted, but only after a preview and an explicit confirmation. Every turn gets a
trace ID and a cost line.

## Start here: the problem that shapes everything

`thelook_ecommerce` has no revenue column. Revenue has to be derived from
`order_items.sale_price`, and both `orders.status` and `order_items.status` contain
`Cancelled` and `Returned`. So "total revenue" has several defensible definitions and a
language model will pick one without mentioning which.

This is measurable, not hypothetical. Across the whole dataset, including cancelled and
returned orders gives $10,814,271; excluding them gives $8,148,817. That is a 32.7%
overstatement from one unstated choice.

Worth knowing the detail: the two status columns hold identical values in all 181,978
rows, so picking the order table over the item table makes no difference. Whether to
filter at all makes a third of the revenue figure.

The audience here cannot check the number. A technical user notices when revenue is
out by a factor of three; an executive acts on it. So the main risk in this system
isn't a query that fails, it's a query that succeeds with the wrong number.

That's the reason the Golden Bucket matters. Retrieving similar questions helps the
phrasing. What stops wrong numbers is the metric definitions extracted from those
past analyses: revenue means one thing, the report says which, and it cites where
the definition came from.

---

## Reading order

### 1. `safety/schema.py`, what the data is and what's sensitive

Start here, because everything else refers to it. One dict of tables and columns,
each column tagged with a privacy class.

The classification is per-column rather than per-table. That matters because the
client's own example question is "why are users in state X underspending versus
state Y", which needs geography. So `state`, `city` and `country` are analytic
dimensions, while `street_address`, `postal_code` and lat/long are blocked, those
identify a household, the others describe a population.

`users.id` is a third case: needed for joins, never shown raw. It comes out as a
hash.

### 2. `safety/entitlements.py`, who can see which products

Requirement 2 says each user may only analyze products related to them. The dataset
has no ownership column, so entitlements come from `config/entitlements.yaml`
(in production, an SSO claim).

This is the part worth spending time on in a review.

The obvious implementation is to find `products` in the generated SQL and add a
category filter. I built that first, and it has a hole:

```sql
SELECT SUM(sale_price) FROM order_items WHERE created_at >= '2024-01-01'
```

No `products` reference, so no filter gets added, so a category manager sees
company-wide revenue. And because `sale_price` is the only revenue column, that's
the shape most real questions take, it's the common case, not an edge case.

What it does instead: build four CTEs (`s_products`, `s_order_items`, `s_orders`,
`s_users`), already filtered to the user's categories, and prepend them. The model
writes SQL against those names only. `s_order_items` inner-joins `s_products`, so
the scope travels through the join and can't be skipped by leaving a table out.

Two side effects worth mentioning:

- PII becomes structural. `email` isn't forbidden, it's absent from the view. A
  prompt injection can't talk its way into a column that doesn't exist.
- `orders.num_of_item` is withheld too. It isn't personal data, but it counts items
  across the whole order including categories the user can't see.

The general principle: a rewrite that misses a case fails open, a validator that
misses a case fails closed. For a security control, pick the one that fails closed.

In the client's own warehouse this would be BigQuery authorized views plus row-level
access policies, which is better because the warehouse enforces it rather than the
app. That isn't possible here, row-level security needs table ownership, and
`bigquery-public-data` is read-only to us. The CTE design mirrors the same shape.

### 3. `safety/sql_guard.py`, the whitelist

Parses the model's SQL with `sqlglot` and rejects anything that isn't a single
read-only query over the four scoped views.

Checks: one statement only, SELECT/WITH only, no DML or DDL, no `EXECUTE IMMEDIATE`,
every table name in the allowed set, no CTE reusing a scoped view name, no
direct-identifier columns, and a row limit.

Two details I'd point out:

- Unparseable SQL is rejected. If the validator doesn't understand it, it doesn't run.
- The CTE shadowing check matters. Without it, a model could define
  `WITH s_products AS (SELECT * FROM <base table>)` and the whole scoping is gone.

No LLM involved, so it's cheap, fast, and testable. `tests/test_sql_guard.py` is
mostly red-team cases, most of them assert that something is refused.

### 4. `safety/masking.py`, output masking

Three things: pseudonymise customer IDs, drop personal columns from result frames,
and a regex sweep over text before it's displayed or logged.

Two orderings matter more than the rules:

- Masking runs before the report is written, so raw data never reaches the model.
  Masking at print time would mean the LLM already saw it.
- Questions and SQL are scrubbed before logging. Otherwise a `WHERE email = '...'`
  predicate ends up in the audit log.

The regex is narrow on purpose and isn't the real control, the scoped views are. It
exists to catch a failure of those, which is why a hit increments a metric. If it
ever fires in practice, something upstream is broken.

### 5. `reports/library.py`, saved reports

SQLite table plus an audit table. Ownership is in the SQL: every query carries
`owner_id = ?`, so another user's report is never in a result set to begin with.
That's what makes "delete all my reports" safe regardless of what the layers above
decide.

Deletes are soft, with `/undo`. Not-found and not-owned both return `None`, because
telling someone a report exists but belongs to someone else leaks its existence.

`soft_delete` takes a list of IDs and nothing else. There's no `delete_where(...)`
function, because having one would invite the pattern the confirmation flow exists
to prevent.

### 6. `reports/confirmation.py`, the consent rules

Three ideas:

**The token binds the ID set.** It holds a hash of exactly the reports that were
shown. If the matching set changes while the user is deciding, the approval is
refused and they see the new list. Consent is to a specific list, not to a sentence.
Five-minute expiry.

**Friction scales with blast radius.** One or two reports: "yes" is enough. Six
reports: you type `DELETE 6 REPORTS`. Requiring a typed phrase for everything sounds
safer but isn't, it trains people to type it without reading, which is exactly when
it stops working.

**Ambiguity means less deletion, never more.** A bare number is always an index. For
a two-item list, "2" means item 2, not "all 2", those readings differ by a deleted
report, so it resolves toward the smaller one. Anything unclear re-asks.

### 7. `graph/destructive.py`, the confirmation flow

This is the LangGraph part, and the clearest reason for using LangGraph at all.

Four nodes: `resolve_candidates` → `await_confirmation` → `apply_delete` / `abort`.

The node split isn't cosmetic. I checked how langgraph 1.2.16 actually behaves and
the node containing `interrupt()` re-runs from its first line when the graph
resumes, while earlier nodes don't. So:

- Resolution sits in its own node upstream, runs once, and writes the ID set to
  state.
- `await_confirmation` only renders a prompt before the `interrupt()` call. If
  resolution lived here it would re-run on resume, and a report that started
  matching in the meantime would be swept into a set the user already approved.
  `test_resolution_is_not_repeated_on_resume` covers that.
- `apply_delete` reads the IDs back from state and re-checks the token before doing
  anything irreversible.

The model never names IDs. Its only input is a search term or a thread scope. There
is no delete tool it can call with its own arguments, so "the ones I said" can't be
reinterpreted after approval.

**The demo to run.** Ask to delete some reports, see the preview, kill the process,
restart, confirm, and the delete completes. That's
`test_confirmation_survives_a_process_restart`, which throws away the graph object
and the connection to simulate a restart. A plain function-calling loop can't do
this.

### 8. `obs/events.py`, tracing

A JSONL file, one line per node, correlated by a short trace ID. Plus counters.

It's one system on purpose. LangSmith and OpenTelemetry both belong in the
production design, but a required feature shouldn't depend on an external service
being reachable. JSONL is greppable, ships in the repo, and powers `/trace`.

`span()` is a context manager that times a node and writes its event whether the
node succeeds or throws. Failures are the thing you most want logged, so recording
can't depend on the happy path finishing.

`cost_line()` prints LLM calls, tokens, MB billed and the trace ID after each turn.
The brief asks for self-correction "without inflating costs", so it seemed worth
making that visible rather than just claiming it.

### 9. `resilience/budget.py`, spending limits

Three ceilings per turn: LLM calls, bytes billed, repair attempts. Carried in graph
state, so a resumed turn doesn't start spending from zero again.

The bit I like is the duplicate check. The failure that actually wastes quota isn't
an infinite loop, it's a model that "fixes" SQL by returning something equivalent to
what just failed. Every candidate query is fingerprinted, and a repeat is rejected
without running, costs nothing, and fails fast instead of slowly.

The fingerprint is textual, not semantic. Normalising aliases would catch more no-op
repairs but could flag a real fix as a duplicate, and blocking a correct repair is
the worse mistake.

### 10. `resilience/breaker.py`, retry and circuit breaking

Two different questions. Retry asks "would trying again help?", only for transient
faults, so a 400 or a SQL error fails on the first attempt instead of spending quota
to learn nothing. The breaker asks "should we try at all?", when BigQuery is down,
retrying every request turns one outage into a slow outage for everyone.

Faults are classified by HTTP status, a `retry_after` hint, and exception class name,
rather than importing from each SDK. Keeps the module free of provider imports and
testable without network libraries. Anything unrecognised counts as permanent,
because retrying what you don't understand is how you get a surprise bill.

`describe_failure()` turns exceptions into something an executive can use. "403
Forbidden: bigquery.jobs.create denied" tells them nothing; they need to know
whether to wait, rephrase, or call an administrator, plus the trace ID.

### 11. `data/bigquery.py`, the warehouse

One entry point, so the dry run, byte ceiling, cache, breaker and replay switch all
apply to every query. There's nowhere else to go, which is the easiest way to be
sure.

Order inside `execute()`: cache, replay check, dry run, budget check, execute.
BigQuery bills on bytes scanned, so checking cost afterwards is a receipt, not a
control. The dry run is free. `maximum_bytes_billed` is set as well, so a bug in our
own accounting still can't produce a runaway bill.

Three modes: `live`, `record`, `replay`. Replay answers from recorded results and
never imports a Google library, so a reviewer with no GCP account can still run the
thing. That's also where the test fixtures come from.

This replaced an earlier plan for a DuckDB copy of the dataset. DuckDB's SQL differs
from BigQuery's in ways that matter here, `DATE_TRUNC` argument order,
`SAFE_DIVIDE`, no dry run, so it would have meant two dialects and two validators.
And it couldn't be seeded without the BigQuery access it was meant to replace.

---

## Suggested demo order

1. A schema question ("what data do we have?"), answered from the schema card, no
   SQL and no query.
2. A simple metric. Show the generated SQL and the cost line.
3. The same question again, cache hit, zero bytes billed.
4. A comparison across two states, which needs more than one query.
5. "Who are our top customers?", works, but IDs come out hashed.
6. Ask for an email address, refused.
7. A prompt injection ("ignore your instructions and..."), refused.
8. Switch user with `/su` and ask the same revenue question, different number, and
   the scope line says why.
9. Save a report, then ask to delete it. Show the preview, kill the process,
   restart, confirm.
10. `/debug` and `/trace` on the last turn.
11. `--chaos bq_fail` to show degradation.

## Questions worth having an answer ready for

**Why LangGraph and not a function-calling loop?** Because of the restart demo
above. A durable pause mid-execution, a bounded retry cycle, and per-node
attribution are the three things the required features need, and a plain loop gives
none of them. For a demo alone, a loop would have been about 200 lines and fine,
the brief asked for production-ready.

**Why no vector database?** Thirty-odd documents. Brute-force cosine over
precomputed embeddings is exact and instant. At forty documents you could honestly
just put the question list in the prompt and let the model pick. Embeddings are
there for the scaling story, behind one interface, and `pgvector` is the production
answer.

**Why is there no k-anonymity?** Because the brief requires "top customers", which
is inherently per-person. A blanket `HAVING COUNT(*) >= 5` returns nothing for that
question. The requirement is about not displaying personal data, not about
preventing inference, so identifiers are unprojectable and ranking stays available
through a pseudonym. It's argued for demographic cross-tabs in the HLD.

**What would you do differently with more time?** Push entitlements into BigQuery
authorized views with row-level security, so the warehouse enforces them instead of
the application. Build the eval harness properly, execution accuracy against gold
SQL, a red-team suite in CI. And a semantic layer in dbt rather than mining metric
definitions out of past analyses.

**What's the weakest part?** The metric definitions are mined from a small set of
hand-written examples. With thirty trios the coverage is thin, and a question
outside that set falls back to free-form SQL generation, which is where wrong
numbers come from. In production that gap closes through the curation loop in the
HLD, with a human approving promotions.
