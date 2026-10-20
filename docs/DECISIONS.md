# Decision log

Running log kept from hour zero. Each entry is *decision → why → what was rejected*.
This is the raw material for the technical explanation (deliverable 2), so it is written to be
quotable rather than as scratch notes.

---

## D1. Framework: LangGraph 1.0

All four mandated requirements are control-flow problems, and LangGraph makes control flow
explicit rather than emergent:

| Requirement | Control-flow need | LangGraph primitive |
|---|---|---|
| High-stakes oversight | durable pause mid-execution, survives restart | `interrupt()` + checkpointer |
| Resilience | a *bounded* cycle (a free-running loop inflates cost) | conditional edge + counter in state |
| Observability | per-step attribution | one event/span per node, from the topology |
| Extensibility | add a capability without a rewrite | new node/subgraph + edge |

**Rejected:** a single Gemini function-calling loop. It would be ~200 lines and fine for a demo,
but the assignment asks for *production-ready*, and an LLM-driven loop cannot bound its own
retries or expose where a turn failed. Rejected: Google ADK, strong Gemini/Vertex story,
but learning it during a time-boxed assignment trades away control over the interrupt and retry
semantics that the graded requirements hinge on.

## D2. Entitlements: scoped-CTE prepend + fail-closed whitelist, NOT SQL rewriting

The first design rewrote `products` references into a category-filtered view. **It had a
bypass.** Enforcement only fires if the generated SQL mentions `products`, and in this dataset
revenue lives in `order_items.sale_price`:

```sql
SELECT SUM(sale_price) FROM order_items WHERE created_at >= '2024-01-01'
```

No `products` reference, so no filter, so the caller sees company-wide revenue. That is the
*common* question shape, not an edge case.

**Adopted instead:** the system prepends a generated `WITH` block defining `s_orders`,
`s_order_items`, `s_products`, `s_users`, each pre-joined to `products`, filtered to the
caller's entitled categories, with PII columns structurally absent. `sqlglot` then enforces
a whitelist: every table identifier must be one of those four; no base-table references; no CTE
may shadow a scoped name; single SELECT/WITH statement only.

The governing principle: a rewrite fails open, a validator fails closed. For a security
control that asymmetry decides the design. Unparseable SQL is rejected, never waved through.

This is also isomorphic to the production answer, authorized views plus row-level access
policies keyed on SESSION_USER(), with the agent service account holding no base-table grant,
so the HLD-to-prototype mapping is honest rather than aspirational. Note that row-level security
is *impossible* in the prototype by construction: it requires table ownership, and
`bigquery-public-data` is read-only to us.

Side benefit: entitlements and PII collapse into one mechanism, which is less code than the
rewrite approach and provable by unit test.

## D3. PII: schema-driven masking, not NER; and masking before synthesis

We *know* the schema, so a typed registry beats guessing. `users` columns, verified:
`id, first_name, last_name, email, age, gender, state, street_address, postal_code, city,
country, latitude, longitude, traffic_source, created_at`.

| Class | Columns | Rule |
|---|---|---|
| Direct identifier | `first_name`, `last_name`, `email`, `street_address`, `postal_code`, `latitude`, `longitude` | absent from scoped views; unprojectable |
| Join key | `users.id`, `orders.user_id` | joinable; surfaced only as a stable hash |
| Analytic dimension | `state`, `city`, `country`, `gender`, `traffic_source`, `created_at` | permitted |
| Quasi-identifier | `age` | permitted aggregated; bucketed in narrow cross-tabs |

Two orderings that matter more than the rules themselves:

1. **`mask_pii` runs before `synthesize_report`**, so raw PII never enters the model context.
   Masking at print time silently exfiltrates PII to the LLM, the common mistake.
2. **The user question and the generated SQL are scrubbed before logging.** A predicate such as
   `WHERE email = ...` otherwise lands in the JSONL event log and the traces.

A narrow email/phone regex sits at egress as a tripwire, not a filter: it should never fire,
so a firing emits a metric and is treated as a defect in an upstream layer.

**Rejected:** spaCy/transformers NER, adds a model download to a "must run on another machine"
deliverable and false-positives on brand and product names.

**Rejected:** a blanket `HAVING COUNT(*) >= 5`. The brief explicitly requires *top customers*,
which is inherently per-individual; a global k-anonymity floor returns nothing and would have
failed a named capability in service of a control nobody asked for. The requirement is identity
*disclosure*, not inference: direct identifiers are unprojectable, and ranking stays answerable
through a stable pseudonym. k-anonymity is argued in the HLD for demographic cross-tabs only.

## D4. Destructive ops: resolve, preview, interrupt, apply

Never delete by predicate. Resolve to an explicit ID set, persist it to state, show a numbered
preview, interrupt, then apply.

- **The decision an LLM must never make:** deletion executes against the *previewed, persisted*
  ID set passed as data to deterministic code, never a tool call the model can re-parametrise
  after confirmation. An owner predicate at the data-access layer is defence in depth.
- Friction scales with blast radius: this is the "without breaking UX" clause. A natural
  "yes" suffices for one or two reports; bulk requires a typed token.
- **TOCTOU-safe:** the confirmation token binds user, a hash of the resolved ID set, a nonce,
  and a 5-minute TTL. If the matching set changed between preview and confirm, abort and
  re-preview.
- Authorization precedes resolution, so another user's reports are never even candidates.
- Both named cases must demo: text match ("mentioning Client X") and session scope ("the reports
  we made in this conversation"), the latter needs thread provenance on every saved report.

## D5. Cost control as a design driver, not an afterthought

Gemini's free tier is roughly 10 requests/minute, 250 requests/day, 250k tokens/minute. A
conventional six-LLM-calls-per-turn agent exhausts the daily quota in about 40 turns, during
development, not production.

- Target 3 or fewer LLM calls per turn: one fused triage call (intent + scope + plan), one
  SQL generation, one synthesis.
- The safety gate, entitlement prepend, PII masking and the report faithfulness check use
  **no LLM at all**, cheaper, faster, and far more reliable than asking a model to police
  itself.
- Follow-up questions are answered from the prior result frame in state: zero BigQuery bytes.
- Dry-run plus a billed-bytes ceiling before every execute; repair capped at two attempts with a
  must-change-hash guard; schema and results cached.
- A per-turn budget object in state (calls, tokens, bytes). Exceeding it degrades with an
  explanation rather than silently spending.
- **Visible to the reviewer:** the CLI prints a cost line (bytes billed, tokens, cache hit).
  "Without inflating costs" is a graded clause, so it should be observable, not merely true.

## D6. Record/replay query cache instead of a DuckDB offline replica

**Rejected DuckDB.** The dialect divergence is load-bearing: backtick-quoted
project.dataset.table names, `DATE_TRUNC(d, MONTH)` versus DuckDB's inverted argument order,
SAFE_DIVIDE, APPROX_QUANTILES, FARM_FINGERPRINT, no dry-run, no billed-bytes ceiling. Two
dialects means two prompt variants and two validators, for zero grading credit, and the replica
cannot be seeded without the BigQuery access it was meant to replace.

**Adopted:** a cached BigQuery client with three modes, live, record, replay, keyed on a hash
of the normalized SQL. One dialect, one code path, around 60 lines, and it pays for four
requirements at once: a review without a Google Cloud project (a Gemini key is still
needed), the cost cache, deterministic fixtures for
the eval harness, and network-free tests. The help text states plainly that replay only answers
pre-recorded questions.

## D7. Observability: JSONL is the owned source of truth

One system, not four. A JSONL event log is greppable, ships in the repo as an artifact, and
powers the trace command. LangSmith is a single optional environment variable for the
message-correspondence view, but a *mandated* requirement must not depend on a third party.
OpenTelemetry to Cloud Trace is described in the HLD as the production exporter and deliberately
not wired into the prototype.

**Rejected:** half-wired OTel spans alongside JSONL alongside LangSmith, overlapping systems
read as indecision.

## D8. Retrieval right-sized: precomputed embeddings + numpy cosine

At 30 to 50 trios, brute-force cosine is exact, instant, and dependency-free. Embeddings are
precomputed at build time so no API calls are spent on the corpus at query time.

Worth stating in the write-up: at roughly 40 documents, stuffing a compact catalog of trio
*questions* into one prompt and letting the model select by ID would also work. Embeddings were
chosen for the scaling path, not because the corpus demands them. Rejected:
FAISS/Chroma/Vertex Vector Search, knowing when RAG infrastructure is overkill is the more
senior signal.

The substantive payload is not the trios themselves but the **metric definitions mined from
them**, for example, revenue as a sum of `order_items.sale_price` excluding cancelled and
returned orders. This dataset is a trap for free-form text-to-SQL: there is no revenue column,
and both `orders.status` and `order_items.status` carry Cancelled and Returned values, so
"revenue" has several plausible-looking wrong definitions that the model will pick between
silently. For non-technical executives the worst failure mode is a confidently wrong number, so
the metric registry is what makes the Golden Bucket load-bearing instead of decorative.

## D9. Synchronous throughout

Async buys nothing in a single-user CLI and costs Windows event-loop and gRPC debugging time,
with BigQuery and Gemini both on gRPC.

## D10. Model IDs in config, never in source

Gemini 2.5 became access-restricted; current stable models are `gemini-3.8-flash`,
`gemini-3.5-flash-lite`, `gemini-4-argon`, and `gemini-embedding-001`. A `doctor` command lists
what the key can *actually* reach before any code depends on it. This also serves requirement 8:
changing models is a config edit, not a redeploy.

## D11. flash-lite everywhere, measured rather than assumed

On the free-tier key used to build this, `gemini-3.8-flash` took 154 seconds to
answer "reply with one word", and 170 seconds with thinking lowered, so the latency
is queuing rather than reasoning. `gemini-3.5-flash-lite` answers the same prompt in
11-39 seconds and a realistic SQL prompt in about 27.

So flash-lite handles triage, SQL and synthesis. A key with better quota can change
that in `.env` without touching code. A 90-second request timeout was added at the
same time, so a queued call fails and degrades rather than hanging the CLI.

Related: `gemini-2.5-flash` is still returned by the models list and 404s when
called. Anything hardcoded from the brief would already be broken, which is the
practical argument for model ids living in config.

## D12. Temperature 0 for classification and SQL

Two runs of "what was our total revenue in 2024" produced two different
formulations: one filtering on `order_items.status`, one on `orders.status`. They
agreed numerically, but only because those columns happen to be identical in this
dataset.

Triage and SQL now run at temperature 0, synthesis stays warmer for prose. Stable
SQL also means a recorded fixture is reusable, which is what makes replay mode
worth having.

## D13. A faithfulness check on the written answer

A live run tabulated 2504 customers correctly and wrote "2,5004 customers" in the
sentence above the table. Prompting does not reliably prevent that and no reader
would catch it.

`safety/faithfulness.py` extracts the figures from the answer and checks them
against the query results. Deterministic, no model call, so it costs nothing.

It flags rather than blocks. Percentages, differences and growth rates legitimately
do not appear in the results, so an unmatched number is a signal rather than proof.
Unverified figures become a metric and show in `/debug`.

The same run also footnoted "revenue after returns and discounts". Discounts were
invented. The metric registry now supplies that line verbatim with an instruction to
reproduce it unchanged, rather than asking the model to describe its own definition.

## D14. Report commands end after triage

A delete request originally ran the whole analysis path. Triage classified it
correctly, but the router had no route for it, so it fell through to SQL generation
and spent a model call and 25 seconds writing a query for "delete all reports
mentioning Allbirds" before the CLI noticed and took over. It now ends at a
`defer_to_reports` node: one model call and under 10 seconds.

Worth noting as a class of bug. The classification was right and the handler was
right; the routing table silently had no entry, and a missing route defaulted to the
most expensive path.

## D15. What replay mode actually removes

Replay serves recorded query results, so a reviewer does not need a Google Cloud
project. It does not remove Gemini: triage, SQL generation and the written answer
are all model calls, and the agent refuses to start without a key. An earlier draft
of the setup guide claimed "no GCP project and no API key", which was wrong, and
the docs now say so plainly.

It is also only as complete as the recordings. The lookup is keyed on the generated
SQL, so a question whose SQL does not match a fixture reports a miss rather than an
answer. Temperature 0 on SQL generation makes that much less likely but does not
guarantee it.

For a reviewer who wants to see a full session without running anything, the
transcript in the README is the honest artifact.
