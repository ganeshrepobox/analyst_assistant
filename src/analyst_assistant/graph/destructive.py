"""LangGraph flow for deleting reports: resolve, preview, interrupt, apply.

The node split matters. In langgraph 1.2.x the node holding interrupt() re-runs
from its first line when the graph resumes, while earlier nodes do not. So
resolution lives in its own upstream node and writes the id set to state, and
await_confirmation only builds a prompt. If resolution ran inside that node, a
report that started matching while the user was deciding would join a set they
had already approved.

The model only supplies a search term or a thread scope. It never names ids, and
there is no delete tool it can call with its own arguments.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, TypedDict

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


class DestructiveState(TypedDict, total=False):
    """State for one destructive request.

    Everything must survive JSON round-tripping through the checkpointer, so the
    token is held as a plain dict rather than a dataclass.
    """

    user_id: str
    request: dict[str, str]
    candidate_ids: list[str]
    preview: list[str]
    token: dict[str, Any] | None
    decision: str
    picks: list[int]
    deleted_ids: list[str]
    outcome: str
    message: str


def build_destructive_graph(
    library: ReportLibrary, checkpointer: Any, *, recorder: Recorder | None = None
) -> Any:
    """Compile the delete flow.

    checkpointer is required: interrupt() needs one, and an in-memory saver would
    defeat the restart behaviour. recorder is optional so the graph is testable
    without a log directory.
    """

    def _span(node: str):
        if recorder is None:
            from contextlib import nullcontext

            return nullcontext({})
        return recorder.span(node)

    def resolve_candidates(state: DestructiveState) -> DestructiveState:
        """Turn the requested scope into an explicit, owned id set. Runs once."""
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
                    "outcome": "nothing_matched",
                    "message": f'No reports of yours match "{scope}". Nothing to delete.',
                }

            ids = [r.id for r in matches]
            return {
                "candidate_ids": ids,
                "preview": [r.summary_line(i) for i, r in enumerate(matches, start=1)],
                "token": asdict(issue_token(user_id=state["user_id"], report_ids=ids)),
            }

    def await_confirmation(state: DestructiveState) -> DestructiveState:
        """Pause for approval.

        Everything before interrupt() must be side-effect-free, since this body re-runs
        on resume.
        """
        token = ConfirmationToken(**(state["token"] or {}))
        ids = state.get("candidate_ids", [])

        reply = interrupt(
            {
                "kind": "confirm_delete",
                "count": len(ids),
                "preview": state.get("preview", []),
                "prompt": prompt_for(ids, token),
                "requires_phrase": token.requires_typed_phrase,
                "required_phrase": token.required_phrase if token.requires_typed_phrase else None,
            }
        )

        with _span("await_confirmation") as m:
            decision, picks = parse_reply(str(reply), token=token)
            m["attrs"] = {"decision": str(decision), "picks": picks}
            return {"decision": str(decision), "picks": picks}

    def apply_delete(state: DestructiveState) -> DestructiveState:
        """Delete the previewed set after re-checking the token."""
        with _span("apply_delete") as m:
            token = ConfirmationToken(**(state["token"] or {}))
            approved = state.get("candidate_ids", [])
            request = state.get("request", {})

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

            targets = approved
            if state.get("decision") == Decision.SUBSET:
                picks = state.get("picks", [])
                targets = [approved[i - 1] for i in picks if 1 <= i <= len(approved)]

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
                    f'Deleted {len(deleted)} {noun}. Say "undo" to restore them.'
                    if deleted
                    else "Nothing was deleted."
                ),
            }

    def abort(state: DestructiveState) -> DestructiveState:
        """Nothing deleted.

        Reports "unclear" separately from "cancelled", so the user knows to ask again
        rather than assuming they declined.
        """
        with _span("abort") as m:
            m["status"] = "refused"
            if state.get("decision") == Decision.UNCLEAR:
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

    graph = StateGraph(DestructiveState)
    graph.add_node("resolve_candidates", resolve_candidates)
    graph.add_node("await_confirmation", await_confirmation)
    graph.add_node("apply_delete", apply_delete)
    graph.add_node("abort", abort)

    graph.add_edge(START, "resolve_candidates")
    graph.add_conditional_edges(
        "resolve_candidates",
        # No matches is a complete answer, not a failure: skip to the end rather
        # than asking the user to confirm an empty deletion.
        lambda s: END if not s.get("candidate_ids") else "await_confirmation",
        {END: END, "await_confirmation": "await_confirmation"},
    )
    graph.add_conditional_edges(
        "await_confirmation",
        lambda s: (
            "apply_delete"
            if s.get("decision") in (Decision.CONFIRM, Decision.SUBSET)
            else "abort"
        ),
        {"apply_delete": "apply_delete", "abort": "abort"},
    )
    graph.add_edge("apply_delete", END)
    graph.add_edge("abort", END)

    return graph.compile(checkpointer=checkpointer)


def _explain_token_error(error: TokenError) -> str:
    """Turn a refused token into something a user can act on."""
    if error is TokenError.SET_CHANGED:
        return (
            "The set of matching reports changed while you were deciding, so nothing "
            "was deleted. Please ask again to see the up-to-date list."
        )
    if error is TokenError.EXPIRED:
        return (
            "That confirmation expired, so nothing was deleted. Ask again and I will "
            "re-check which reports match."
        )
    return "I could not verify that confirmation, so nothing was deleted."
