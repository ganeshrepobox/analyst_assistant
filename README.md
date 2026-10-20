# Executive Data Analysis Agent

A conversational data-analysis agent for non-technical retail executives, over the
`bigquery-public-data.thelook_ecommerce` dataset. Ask questions in natural language, discuss the
answers, and generate reports with action items — with PII protection, entitlement scoping,
confirmed destructive operations, self-correcting SQL, and full per-turn tracing.

Built with **LangGraph** on **Gemini** against **BigQuery**.

---

## Documentation

| Document | What it covers |
|---|---|
| **[docs/01-architecture.md](docs/01-architecture.md)** | **High-level design** — architecture and flow diagrams, service choices with reasoning, safety model, resilience matrix, observability metrics, and how each of the 8 requirements is solved |
| [docs/SETUP.md](docs/SETUP.md) | Credentials, install, verification, and the no-credentials replay mode |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log — each choice with its reasoning *and the alternatives rejected* |
| [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) | Every ambiguity in the brief, resolved, with what would change if the assumption is wrong |

---

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env     # add your Gemini key + GCP project
python -m analyst_assistant doctor    # verify credentials and model access
python -m analyst_assistant chat
```

No Google Cloud account? `BQ_MODE=replay` runs the full agent against recorded query results —
enough to exercise every safety, confirmation, resilience and observability path.
See [docs/SETUP.md](docs/SETUP.md).

---

## The core design idea

This dataset has **no revenue column**. Revenue must be derived from `order_items.sale_price`,
and *both* `orders.status` and `order_items.status` carry `Cancelled` and `Returned` values. So
"total revenue" has several plausible-looking definitions that differ by percent-level amounts,
and a language model will silently pick one.

For an audience that cannot sanity-check the number, **a confidently wrong answer is the dominant
risk** — worse than a failed query. Two mechanisms address it directly:

1. **The Golden Bucket pins metric definitions down.** Retrieving similar past analyses improves
   phrasing; the *metric registry mined from them* is what makes "revenue" mean one thing, cite
   the analyst work it came from, and footnote its own definition.
2. **Safety is structural, not instructional.** PII columns are *absent* from the queryable
   surface rather than present-but-forbidden, and entitlements are enforced by prepending scoped
   CTEs that the SQL validator then requires. A rewrite that misses a case fails open; a
   validator that misses a case fails closed.

Full reasoning in [docs/01-architecture.md](docs/01-architecture.md).

---

## Requirement coverage

Four requirements are implemented in code as mandated; four are design deliverables.

| # | Requirement | Code | Design |
|---|---|---|---|
| 1 | Hybrid Intelligence (Golden Bucket) | ✅ retrieval + mined metric layer | ✅ full curation lifecycle |
| 2 | **Safety & PII Masking** | ✅ **mandated** | ✅ |
| 3 | **High-Stakes Oversight** | ✅ **mandated** | ✅ |
| 4 | Continuous Improvement | ✅ user preferences | ✅ system-level loop |
| 5 | **Resilience & Error Handling** | ✅ **mandated** | ✅ |
| 6 | Quality Assurance | ◐ eval harness | ✅ |
| 7 | **Observability** | ✅ **mandated** | ✅ |
| 8 | Agility (Persona Management) | ✅ hot-reload | ✅ |

---

## Status

Design complete. Prototype in progress — this README will carry a verbatim end-to-end transcript
(a safety refusal, a self-repair, a confirmed deletion with restart-resume, and a `/trace`) once
the implementation lands.
