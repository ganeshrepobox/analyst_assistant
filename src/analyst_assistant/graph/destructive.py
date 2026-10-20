"""The destructive-operation subgraph: resolve, preview, interrupt, apply.

Implements the control-flow half of requirement 3. This is also the clearest
justification for choosing LangGraph: the confirmation is a *durable* pause. The
process can exit between the preview and the approval, and the pending decision
survives in the checkpointer.

Node boundaries are load-bearing
--------------------------------
Verified against langgraph 1.2.16 rather than assumed: **the node containing
``interrupt()`` re-executes from its top when the graph resumes.** Earlier nodes do
not. That single fact dictates the shape here:

* ``resolve_candidates`` is a separate node, so resolution happens exactly once and
  the id set is written to state before the pause.
* ``await_confirmation`` holds the ``interrupt()`` and everything above that call is
  pure -- it only renders a prompt. If resolution lived here it would re-run on
  resume, and a report that began matching in the meantime would be silently
  swept into a set the user already approved.
* ``apply_delete`` acts on the ids read back from state, never on a fresh
  resolution.

Why the token exists on top of that
-----------------------------------
Persisting the id set stops *accidental* drift. The token stops *consent* drift: it
binds a hash of the exact set shown, so if the world changed while the user was
deciding, the approval is refused and they are shown the new set. Consent is to a
specific list of reports, not to a sentence.

What the model is not allowed to do
-----------------------------------
The model's only influence is upstream: it may propose a ``text`` or thread scope to
resolve. It never names the ids to delete, and there is no delete tool it can call
with its own arguments. Deletion executes against the previewed, persisted set via
deterministic code -- which is what prevents "the ones I said" from being
reinterpreted after approval.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..obs.events import Recorder
from ..reports.confirmation import (
    ConfirmationToken,
    Decision,
    TokenError,
    issue_token,
    parse_reply,
    prompt_for,
)
from ..reports.library import ReportLibrary

#: How many times an unclear reply is re-prompted before giving up. Bounded so an
#: uncooperative or automated client cannot hold a turn open indefinitely.
MAX_CLARIFICATIONS = 2


class DeleteRequest(TypedDict, total=False):
    """What the model proposed as the scope of the deletion."""

    text: str
    thread_id: str


class DestructiveState(TypedDict, total=False):
    """State for one destructive request.

    Everything here must survive JSON round-tripping through the checkpointer, so
    the token is held as a plain dict rather than as a dataclass.
    """

    user_id: str
    thread_id: str
    request: DeleteRequest
    candidate_ids: list[str]
    preview: list[str]
    token: dict[str, Any] | None
    decision: str
    picks: list[int]
    clarifications: int
    deleted_ids: list[str]
    outcome: str
    message: str


Outcome = Literal["deleted", "cancelled", "nothing_matched", "set_changed", "unclear", "expired"]


def build_destructive_graph(
    library: ReportLibrary,
    checkpointer: Any,
    *,
    recorder: Recorder | None = None,
) -> Any:
    """Compile the destructive-operation subgraph.

    Args:
        library: The reports store. Ownership is enforced inside it, so this graph
            never needs to reason about whose reports these are.
        checkpointer: A LangGraph checkpointer. Required -- ``interrupt()`` cannot
            work without one, and an in-memory saver would defeat the restart
            demonstration this graph exists to provide.
        recorder: Optional observability recorder. Nodes are instrumented when it is
            supplied and run unchanged when it is not, so the graph is testable
            without a log directory.
    """

    def _span(node: str):
        if recorder is None:
            from contextlib import nullcontext

            return nullcontext({})
        return recorder.span(node)

    # -- resolve ------------------------------------------------------------

    def resolve_candidates(state: DestructiveState) -> DestructiveState:
        """Turn the requested scope into an explicit, owned id set.

        Runs exactly once per request: it is deliberately upstream of the interrupt
        so that resuming cannot re-resolve.
        """
        with _span("resolve_candidates") as m:
            request = state.get("request", {})
            matches = library.resolve(
                owner_id=state["user_id"],
                text=request.get("text"),
                thread_id=request.get("thread_id"),
            )
            m["attrs"] = {"matched": len(matches)}

            if not matches:
                scope = request.get("text") or "this conversation"
                return {
                    "candidate_ids": [],
                    "preview": [],
                    "token": None,
                    "outcome": "nothing_matched",
                    "message": f'No reports of yours match "{scope}". Nothing to delete.',
                }

            ids = [r.id for r in matches]
            token = issue_token(user_id=state["user_id"], report_ids=ids)
            return {
                "candidate_ids": ids,
                "preview": [r.summary_line(i) for i, r in enumerate(matches, start=1)],
                "token": asdict(token),
                "clarifications": 0,
            }

    # -- confirm ------------------------------------------------------------

    def await_confirmation(state: DestructiveState) -> DestructiveState:
        """Pause for human approval.

        Everything before ``interrupt()`` must stay side-effect-free, because this
        node's body re-executes when the graph resumes.
        """
        token = ConfirmationToken(**(state["token"] or {}))
        ids = state.get("candidate_ids", [])

        payload = {
            "kind": "confirm_delete",
            "count": len(ids),
            "preview": state.get("preview", []),
            "prompt": prompt_for(ids, token),
            "requires_phrase": token.requires_typed_phrase,
            "required_phrase": token.required_phrase if token.requires_typed_phrase else None,
        }

        # Durable pause. On resume, execution continues from here with the reply.
        reply = interrupt(payload)

        with _span("await_confirmation") as m:
            decision, picks = parse_reply(str(reply), token=token)
            m["attrs"] = {"decision": str(decision), "picks": picks}
            m["status"] = "ok" if decision is not Decision.UNCLEAR else "degraded"
            return {"decision": str(decision), "picks": picks}

    def route_decision(state: DestructiveState) -> str:
        decision = state.get("decision")
        if decision in (Decision.CONFIRM, Decision.SUBSET):
            return "apply_delete"
        if decision == Decision.UNCLEAR:
            if state.get("clarifications", 0) < MAX_CLARIFICATIONS:
                return "clarify"
            return "abort"
        return "abort"

    def clarify(state: DestructiveState) -> DestructiveState:
        """Count a re-prompt and loop back.

        Re-prompting in place rather than aborting is the "without breaking UX"
        half of the requirement: a user who typed something ambiguous should not
        have to restate the whole command.
        """
        return {"clarifications": state.get("clarifications", 0) + 1}

    # -- apply --------------------------------------------------------------

    def apply_delete(state: DestructiveState) -> DestructiveState:
        """Delete the previewed set, after re-validating consent.

        The TOCTOU check lives here, immediately before the only irreversible step,
        and compares the live match set against the hash captured at preview time.
        """
        with _span("apply_delete") as m:
            token = ConfirmationToken(**(state["token"] or {}))
            approved_ids = state.get("candidate_ids", [])
            request = state.get("request", {})

            # Re-resolve to see the world as it is now, not as it was.
            current = [
                r.id
                for r in library.resolve(
                    owner_id=state["user_id"],
                    text=request.get("text"),
                    thread_id=request.get("thread_id"),
                )
            ]

            error = token.check(user_id=state["user_id"], report_ids=current)
            if error is not None:
                m["status"] = "refused"
                m["attrs"] = {"token_error": str(error)}
                return {
                    "deleted_ids": [],
                    "outcome": "set_changed" if error is TokenError.SET_CHANGED else str(error),
                    "message": _explain_token_error(error),
                }

            # A subset selection narrows the approved set by 1-based index.
            targets = approved_ids
            if state.get("decision") == Decision.SUBSET:
                picks = state.get("picks", [])
                targets = [approved_ids[i - 1] for i in picks if 1 <= i <= len(approved_ids)]

            deleted = library.soft_delete(
                owner_id=state["user_id"],
                report_ids=targets,
                trace_id=recorder.trace_id if recorder else None,
            )
            m["attrs"] = {"deleted": len(deleted)}

            noun = "report" if len(deleted) == 1 else "reports"
            return {
                "deleted_ids": deleted,
                "outcome": "deleted",
                "message": (
                    f"Deleted {len(deleted)} {noun}. Say \"undo\" to restore them."
                    if deleted
                    else "Nothing was deleted."
                ),
            }

    def abort(state: DestructiveState) -> DestructiveState:
        with _span("abort") as m:
            decision = state.get("decision")
            m["status"] = "refused"
            if decision == Decision.UNCLEAR:
                return {
                    "deleted_ids": [],
                    "outcome": "unclear",
                    "message": (
                        "I could not tell whether you wanted to go ahead, so nothing was "
                        "deleted. Ask again when you are ready."
                    ),
                }
            return {
                "deleted_ids": [],
                "outcome": "cancelled",
                "message": "Cancelled. Nothing was deleted.",
            }

    # -- wiring -------------------------------------------------------------

    graph = StateGraph(DestructiveState)
    graph.add_node("resolve_candidates", resolve_candidates)
    graph.add_node("await_confirmation", await_confirmation)
    graph.add_node("clarify", clarify)
    graph.add_node("apply_delete", apply_delete)
    graph.add_node("abort", abort)

    graph.add_edge(START, "resolve_candidates")
    graph.add_conditional_edges(
        "resolve_candidates",
        # Nothing matched is a complete answer, not a failure: skip straight to the
        # end rather than asking the user to confirm an empty deletion.
        lambda s: END if not s.get("candidate_ids") else "await_confirmation",
        {END: END, "await_confirmation": "await_confirmation"},
    )
    graph.add_conditional_edges(
        "await_confirmation",
        route_decision,
        {"apply_delete": "apply_delete", "clarify": "clarify", "abort": "abort"},
    )
    graph.add_edge("clarify", "await_confirmation")
    graph.add_edge("apply_delete", END)
    graph.add_edge("abort", END)

    return graph.compile(checkpointer=checkpointer)


def _explain_token_error(error: TokenError) -> str:
    """Turn a refusal into something a non-technical user can act on."""
    if error is TokenError.SET_CHANGED:
        return (
            "The set of matching reports changed while you were deciding, so nothing was "
            "deleted. Here is the up-to-date list -- please take another look."
        )
    if error is TokenError.EXPIRED:
        return (
            "That confirmation expired, so nothing was deleted. Ask again and I will "
            "re-check which reports match."
        )
    return "I could not verify that confirmation, so nothing was deleted."
