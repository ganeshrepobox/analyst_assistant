# Assumptions

The brief invites questions about assumptions. Rather than block on answers, each ambiguity is
resolved here with the reasoning, so a reviewer can see *what* was assumed, *why*, and what
would change if the assumption is wrong.

Format: **assumption** → why it is the reasonable reading → what changes if the client decides
otherwise.

---

## A1 — "Users" in the dataset are customers, not the agent's users

`thelook_ecommerce.users` holds ~100k retail *customers*. The agent's users are a handful of
**executives**. These are different populations and conflating them breaks both the entitlement
model and the PII model.

So the system carries its own identity concept: an `ActiveUser` with an ID, a display name, a
persona, preferences, and an **entitlement set**. In the prototype this comes from
`config/entitlements.yaml`; in production it comes from an SSO token claim.

*If wrong:* if executives were to be matched to dataset rows, the entitlement join changes — but
the enforcement mechanism (scoped views) does not.

## A2 — Entitlements are product **category** scoped

Requirement 2: *"Each user should only be able to analyze data on products related to him."*
`thelook_ecommerce` has no product-ownership column, so ownership must be imposed externally.
`products.category` (~26 values) and `products.department` (2 values) are the natural grains;
category is specific enough to be meaningful and coarse enough to configure by hand.

Modelled as: user → set of entitled categories. A "CEO" role with all categories exists to
demonstrate the unrestricted case, which also makes the restricted case visibly different.

*If wrong:* brand- or product-ID-level scoping changes only the `WHERE` clause inside the
generated scoped views. The architecture is unaffected — this is the payoff of enforcing at a
single choke point.

## A3 — Out-of-scope questions are answered **within** scope, with the scope stated

The sharpest judgment call in the assignment. Two options:

- **Refuse** — safe, but hostile UX; an exec asking a reasonable question gets nothing.
- **Silently filter** — smooth, but **dangerous**: it returns a smaller number that still looks
  like a complete answer. An executive could act on a partial figure believing it to be total.

Chosen: **answer within scope and always state the scope applied.** Every response carries a
scope line (e.g. *"Scope: Women's Apparel, Accessories — 2 of 26 categories"*). The number is
never silently partial.

*Rationale:* for a decision-support tool the failure mode that matters is not a blocked
question, it is a believed-wrong number. This makes partiality impossible to miss while keeping
the tool usable.

*If wrong:* a config flag flips the behaviour to hard refusal; the scope resolver already
computes everything needed for either.

## A4 — PII is classified field-by-field, not blanket-blocked

The brief's own flagship example — *"why are users for state X underspending vs state Y"* —
**requires** geography. A blanket geographic block would make the client's headline question
unanswerable, so precision matters more than severity.

| Class | Columns | Treatment |
|---|---|---|
| Direct identifier | `first_name`, `last_name`, `email`, `street_address`, `postal_code`, `latitude`, `longitude` | never selectable, never printed |
| Join key | `users.id`, `orders.user_id` | joinable; surfaced only as a stable pseudonym |
| Analytic dimension | `state`, `city`, `country`, `gender`, `traffic_source`, `created_at` | permitted |
| Quasi-identifier | `age` | permitted aggregated; bucketed in narrow cross-tabs |

`postal_code` and lat/long are blocked because they re-identify an individual household;
`state` and `city` do not.

## A5 — Top-customer analysis is permitted via pseudonym

*"Customer behavior (e.g., top customers, total spend)"* is an explicitly required capability,
and it is inherently per-individual. Blocking it to satisfy PII rules would fail a named
requirement.

Resolution: customers are ranked and shown as a **stable hash** (`CUST-3f9a2b`) plus
non-identifying attributes. The exec gets the distribution and can act on it; nobody learns who
the person is. Identity disclosure is prevented; inference is not the stated concern.

## A6 — Saved reports are private to their owner; delete is soft

Requirement 3 says *"users are allowed to delete their own reports"*, which implies ownership.
Assumed: private to owner, non-owned reports invisible — so "delete all my reports" cannot reach
anyone else's library.

Delete is **soft** (timestamp + audit row) with an `/undo`. Nothing in the brief requires
irreversible erasure, and reversibility is the safer default for a destructive action.

*If wrong:* GDPR-style erasure needs a hard-delete path. The audit log is designed to retain the
*event* without the content, so this is additive.

## A7 — "Client X" maps to a brand or category

Requirement 3's example is *"Delete all reports mentioning Client X"*, but `thelook_ecommerce` is
B2C and has no clients. Substituted: a **brand** or **category** name as the text-match target.
The mechanism being tested — resolve a predicate to an explicit set, preview, confirm — is
unchanged.

## A8 — Two destructive-op shapes must both work

Requirement 3 names *"Delete all reports mentioning Client X"* (content match) and *"Delete all
the reports we made in this conversation"* (session scope). These are different resolvers:
text search versus thread provenance. Every saved report therefore records its originating
thread ID. Both are demonstrated.

## A9 — Stale data during an outage is served, loudly labelled

Requirement 5 asks for resilience to third-party downtime. When BigQuery or Gemini is
unavailable and a cached result exists, it is served **with its age stated prominently** rather
than refused.

*Rationale:* an executive with a number from 40 minutes ago and a clear timestamp is better off
than one with no answer. The risk is a stale number being mistaken for live, so the label — not
the cache — is the control. Degradation always announces itself.

*If wrong:* a config flag disables stale-serving for clients who prefer refusal.

## A10 — Personas control tone, never safety

Requirement 8 wants non-developers to change instructions without redeployment. Assumed: the
editable surface is **tone, verbosity, structure and vocabulary**. The safety contract — PII
rules, SQL constraints, entitlement enforcement — is code-owned and **not** reachable from a
persona file.

*Rationale:* "non-developers can change the prompt" and "the guardrails are prompt-defined"
together mean a non-developer can delete the guardrails. Splitting the layers is what makes
hot-reload safe to offer.

## A11 — The Golden Bucket is seeded synthetically

The brief marks it *(Theoretical)*. ~30 trios are authored covering the question archetypes the
brief names, each with the metric definitions a human analyst would apply. The **schema and
lifecycle** are the deliverable; the contents are illustrative.

## A12 — Single-tenant, single-user CLI session

The prototype runs one user at a time, selected by config, with a `/su` command to switch and
demonstrate entitlement differences. Production multi-tenancy is addressed in the HLD. Built
synchronously: async buys nothing for one interactive user and costs debugging time with two
gRPC clients on Windows.

## A13 — BigQuery remains the source of truth, with no governed metrics layer

The brief says *"raw transaction logs"*, so no dbt/semantic layer is assumed to exist. This is
why the agent must own its metric definitions — and why those definitions come from the Golden
Bucket rather than being invented per query. Were a governed layer available, the agent should
defer to it instead; that is the better production end-state and is noted in the HLD.

---

## Open questions that would materially change the design

Documented for honesty — each has a working default above, so none blocks delivery:

1. **A3** (refuse vs. scope-and-state) is the only one with a *correctness* consequence for the
   executive rather than a UX consequence.
2. **A6** — whether reports are team-visible changes the delete blast radius.
3. **A11** — whether a human analyst approves Golden Bucket promotions, or curation must be
   automatic. Auto-promoting accepted answers risks the system learning its own errors, so this
   changes a review queue into an eval-gated auto-promote.
