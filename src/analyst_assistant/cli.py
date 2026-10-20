"""The chat interface.

Two commands: `doctor` checks the setup, `chat` runs the REPL.

Every turn is wrapped so a failure prints a message and a trace id rather than a
traceback, and the session carries on. An analytics assistant that dies on a bad
question is worse than one that says it could not answer.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config.persona import ConfigStore, Persona
from .data.bigquery import BigQueryGateway, CacheMode, ResultCache
from .golden.bucket import GoldenBucket
from .graph.agent import (
    Deps,
    ReportAction,
    ResultFrame,
    build_agent,
    parse_report_command,
    run_turn,
)
from .graph.destructive import build_destructive_graph
from .llm import Gemini
from .obs.events import EventLog, Metrics, Recorder
from .reports.library import ReportLibrary
from .resilience.budget import Budget
from .safety.entitlements import Entitlement

# The BigQuery client warns on every query that the faster Storage API is absent and
# it is using REST. True, harmless for aggregate results, and noise in a demo.
warnings.filterwarnings("ignore", message=".*BigQuery Storage.*")

app = typer.Typer(add_completion=False, help="Data analysis assistant for retail executives.")
console = Console()

DB_PATH = "analyst_assistant.sqlite"
CHECKPOINT_PATH = "checkpoints.sqlite"


class ChaosError(Exception):
    """Injected failure. status_code makes it look transient to the retry logic."""

    def __init__(self, what: str) -> None:
        super().__init__(f"injected failure: {what}")
        self.status_code = 503


@dataclass
class Chaos:
    """Fault injection, so degradation can be demonstrated rather than described."""

    mode: str | None = None

    def maybe_fail(self, dependency: str) -> None:
        if self.mode == f"{dependency}_fail":
            raise ChaosError(dependency)


class ChaosGateway:
    """Wraps the gateway to fail on demand. Used only by --chaos."""

    def __init__(self, inner: BigQueryGateway, chaos: Chaos) -> None:
        self._inner = inner
        self._chaos = chaos

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self._chaos.maybe_fail("bq")
        return self._inner.execute(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ChaosLLM:
    def __init__(self, inner: Gemini, chaos: Chaos) -> None:
        self._inner = inner
        self._chaos = chaos

    def structured(self, *args: Any, **kwargs: Any) -> Any:
        self._chaos.maybe_fail("llm")
        return self._inner.structured(*args, **kwargs)

    def text(self, *args: Any, **kwargs: Any) -> Any:
        self._chaos.maybe_fail("llm")
        return self._inner.text(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class Session:
    """Everything that lives for the length of a conversation."""

    store: ConfigStore
    llm: Any
    gateway: Any
    library: ReportLibrary
    log: EventLog
    metrics: Metrics
    user: Entitlement
    persona: Persona
    thread_id: str
    bucket: GoldenBucket | None = None
    graph: Any = None
    delete_graph: Any = None
    prior: list[ResultFrame] = field(default_factory=list)
    last_recorder: Recorder | None = None
    last_answer: str = ""
    last_question: str = ""

    def rebuild(self) -> None:
        """Recompile with the current persona.

        Called after /persona or /su, which is what makes a tone change take effect
        without a restart.
        """
        deps = Deps(
            llm=self.llm,
            gateway=self.gateway,
            persona_style=self.persona.style_instructions(),
            global_note=self.store.global_note(),
        )
        self.graph = build_agent(deps)


def _make_session(user_id: str, mode: str, chaos_mode: str | None) -> Session:
    load_dotenv(Path(".env"))
    store = ConfigStore("config")

    users = store.users()
    if user_id not in users:
        raise typer.BadParameter(f"unknown user {user_id!r}; known: {', '.join(users)}")
    user = users[user_id]

    chaos = Chaos(chaos_mode)
    llm: Any = Gemini()
    gateway: Any = BigQueryGateway(
        project_id=os.environ.get("GCP_PROJECT_ID"),
        mode=CacheMode(mode),
        cache=ResultCache("fixtures"),
    )
    if chaos_mode:
        llm, gateway = ChaosLLM(llm, chaos), ChaosGateway(gateway, chaos)

    session = Session(
        store=store,
        llm=llm,
        gateway=gateway,
        library=ReportLibrary(DB_PATH),
        log=EventLog("logs"),
        metrics=Metrics(),
        user=user,
        persona=store.persona(user.persona),
        thread_id=uuid.uuid4().hex[:12],
        bucket=GoldenBucket("golden_bucket"),
    )
    session.rebuild()
    conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
    session.delete_graph = build_destructive_graph(session.library, SqliteSaver(conn))
    return session


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def _show_help() -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    for command, what in [
        ("ask anything", "a question about sales, products, customers or time periods"),
        ("/save <title>", "keep the last answer as a report"),
        ("/reports", "list your saved reports"),
        ("/undo", "restore the reports you just deleted"),
        ("/debug", "node-by-node timeline of the last turn"),
        ("/trace <id>", "replay a stored turn from the event log"),
        ("/metrics", "counters for this session"),
        ("/persona [name]", "show or change the reporting style"),
        ("/su [user]", "switch user, to see entitlements change"),
        ("/help, /quit", ""),
    ]:
        table.add_row(f"[bold cyan]{command}[/]", what)
    console.print(Panel(table, title="commands", border_style="dim"))


def _show_debug(session: Session) -> None:
    recorder = session.last_recorder
    if recorder is None:
        console.print("[dim]Nothing to debug yet.[/]")
        return
    console.print(f"[bold]trace[/] {recorder.trace_id}")
    for row in recorder.timeline():
        console.print(row)
    console.print(f"[dim]{recorder.cost_line()}[/]")
    for event in recorder.events:
        note = (event.attrs or {}).get("faithfulness")
        if note:
            console.print(f"[dim]figures: {note}[/]")


def _show_trace(session: Session, trace_id: str) -> None:
    events = session.log.read_trace(trace_id)
    if not events:
        console.print(f"[dim]No events for trace {trace_id}.[/]")
        return
    table = Table("node", "status", "ms", "model", "detail", box=None)
    for event in events:
        detail = ", ".join(
            f"{k}={v}" for k, v in (event.get("attrs") or {}).items() if k != "sql"
        )
        table.add_row(
            event.get("node", ""),
            event.get("status", ""),
            f"{event.get('latency_ms', 0):.0f}",
            event.get("model") or "",
            detail[:60],
        )
    console.print(table)


def _show_metrics(session: Session) -> None:
    snapshot = session.metrics.snapshot()
    if not snapshot:
        console.print("[dim]No metrics yet.[/]")
        return
    table = Table("metric", "value", box=None)
    for name, value in snapshot.items():
        table.add_row(name, str(value))
    console.print(table)


def _show_reports(session: Session) -> None:
    reports = session.library.list_live(owner_id=session.user.user_id)
    if not reports:
        console.print("[dim]No saved reports.[/]")
        return
    for index, report in enumerate(reports, start=1):
        console.print(f"  {report.summary_line(index)}")


def _switch_persona(session: Session, name: str | None) -> None:
    available = session.store.personas()
    if not name:
        console.print(f"persona: [bold]{session.persona.name}[/] ({session.persona.label})")
        console.print(f"[dim]available: {', '.join(available)}[/]")
        return
    if name not in available:
        console.print(f"[red]Unknown persona {name!r}.[/] Available: {', '.join(available)}")
        return
    session.persona = session.store.persona(name)
    session.rebuild()
    console.print(f"Persona is now [bold]{session.persona.name}[/] ({session.persona.label}).")


def _switch_user(session: Session, user_id: str | None) -> None:
    users = session.store.users()
    if not user_id:
        console.print(f"you are [bold]{session.user.display_name}[/]")
        console.print(f"[dim]{session.user.describe_scope()}[/]")
        console.print(f"[dim]available: {', '.join(users)}[/]")
        return
    if user_id not in users:
        console.print(f"[red]Unknown user {user_id!r}.[/] Available: {', '.join(users)}")
        return
    session.user = users[user_id]
    session.persona = session.store.persona(session.user.persona)
    session.prior = []  # a different user must not inherit another's figures
    session.rebuild()
    console.print(f"Now acting as [bold]{session.user.display_name}[/].")
    console.print(f"[dim]{session.user.describe_scope()}[/]")


def _save_report(session: Session, title: str) -> None:
    if not session.last_answer:
        console.print("[dim]Nothing to save yet.[/]")
        return
    report = session.library.save(
        owner_id=session.user.user_id,
        thread_id=session.thread_id,
        title=title or session.last_question[:60] or "Untitled analysis",
        body={"question": session.last_question, "answer": session.last_answer},
        trace_id=session.last_recorder.trace_id if session.last_recorder else None,
    )
    console.print(f"Saved as [bold]{report.title}[/].")


# ---------------------------------------------------------------------------
# Destructive flow
# ---------------------------------------------------------------------------


def _run_delete(session: Session, question: str, budget: Budget) -> None:
    """Resolve, preview, confirm, apply.

    The loop re-invokes with Command(resume=...) for as long as the graph keeps
    interrupting, so the confirmation can be re-prompted without leaving the REPL.
    """
    command = parse_report_command(session.llm, question, budget=budget)

    if command.action is ReportAction.LIST:
        _show_reports(session)
        return
    if command.action is ReportAction.SAVE:
        _save_report(session, session.last_question[:60])
        return
    if command.action is ReportAction.UNDO:
        restored = session.library.undo_last_delete(owner_id=session.user.user_id)
        console.print(
            f"Restored {len(restored)} report(s)." if restored else "[dim]Nothing to restore.[/]"
        )
        return

    request: dict[str, str] = {}
    if command.this_conversation:
        request["thread_id"] = session.thread_id
    if command.match_text:
        request["text"] = command.match_text
    if not request:
        console.print("[dim]I was not sure which reports you meant, so nothing was deleted.[/]")
        return

    state = session.delete_graph.invoke(
        {"user_id": session.user.user_id, "request": request}, _delete_config(session)
    )
    _confirm_loop(session, state)


def _delete_config(session: Session) -> dict[str, Any]:
    """Checkpointer thread for this user's pending confirmation.

    Keyed on the user rather than a fresh uuid, so a confirmation left unanswered
    can be found again after a restart. One pending delete per user at a time,
    which is also the only sensible number.
    """
    return {"configurable": {"thread_id": f"delete-{session.user.user_id}"}}


def _confirm_loop(session: Session, state: dict[str, Any]) -> None:
    """Drive the interrupt until the graph stops asking."""
    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        console.print()
        for line in payload["preview"]:
            console.print(f"  {line}")
        console.print(f"\n[bold yellow]{payload['prompt']}[/]")
        try:
            reply = console.input("[bold]> [/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Left unconfirmed. Nothing was deleted.[/]")
            return
        state = session.delete_graph.invoke(Command(resume=reply), _delete_config(session))

    console.print(state.get("message", "Done."))


def _resume_pending_delete(session: Session) -> None:
    """On startup, pick up a confirmation that was never answered.

    This is what makes the pause durable in practice rather than only in the
    checkpointer: preview, kill the process, come back, confirm.
    """
    try:
        snapshot = session.delete_graph.get_state(_delete_config(session))
    except Exception:
        return
    if not getattr(snapshot, "interrupts", None):
        return
    console.print("[yellow]You left a deletion unconfirmed last time.[/]")
    _confirm_loop(session, {"__interrupt__": snapshot.interrupts})
    console.print()


# ---------------------------------------------------------------------------
# A turn
# ---------------------------------------------------------------------------


def _handle_question(session: Session, question: str) -> None:
    recorder = Recorder(
        session.log, session.metrics, user_id=session.user.user_id, persona=session.persona.name
    )
    session.last_recorder = recorder
    budget = Budget()

    bucket = session.bucket
    golden_context = ""
    suggestions: list[str] = []
    footnote = ""
    if bucket is not None:
        # Retrieval needs the question, so this is per turn rather than per session.
        # The metric definitions are the part that must always be present.
        golden_context = bucket.context(question, llm=session.llm)
        suggestions = bucket.question_catalog()
        hits = bucket.retrieve(question, llm=session.llm)
        used = [m for trio in hits for m in trio.metrics] or ["net_revenue"]
        footnote = bucket.registry.footnote(list(dict.fromkeys(used))) or ""

    with console.status("[dim]thinking...[/]", spinner="dots"):
        state = run_turn(
            session.graph,
            question=question,
            entitlement=session.user,
            deps=Deps(
                llm=session.llm,
                gateway=session.gateway,
                persona_style=session.persona.style_instructions(),
                global_note=session.store.global_note(),
                golden_context=golden_context,
                suggestions=suggestions,
                metric_footnote=footnote + "." if footnote else "",
            ),
            recorder=recorder,
            budget=budget,
            prior=session.prior,
        )

    if state.get("intent") == "report_crud":
        _run_delete(session, question, budget)
        console.print(f"[dim]{recorder.cost_line()}[/]")
        return

    answer = state.get("answer") or "I could not answer that."
    console.print()
    console.print(Markdown(answer))
    console.print(f"\n[dim]{session.user.describe_scope()}[/]")
    console.print(f"[dim]{recorder.cost_line()}[/]")

    if state.get("results"):
        session.prior = state["results"]
    session.last_answer = answer
    session.last_question = question


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def chat(
    user: str = typer.Option("exec_women_apparel", help="Which executive to act as."),
    mode: str = typer.Option(
        None, help="live, record or replay. Defaults to BQ_MODE, else live."
    ),
    chaos: str = typer.Option(
        None, help="Inject failures: bq_fail or llm_fail. For demonstrating degradation."
    ),
) -> None:
    """Start the chat session."""
    load_dotenv(Path(".env"))
    resolved_mode = mode or os.environ.get("BQ_MODE", "live")

    try:
        session = _make_session(user, resolved_mode, chaos)
    except Exception as exc:
        console.print(f"[red]Could not start:[/] {exc}")
        console.print("[dim]Run `python -m analyst_assistant doctor` to check the setup.[/]")
        raise typer.Exit(1) from exc

    console.print(
        Panel(
            f"[bold]{session.user.display_name}[/]\n"
            f"{session.user.describe_scope()}\n"
            f"persona: {session.persona.name}   mode: {resolved_mode}"
            + ("   [yellow]chaos: " + chaos + "[/]" if chaos else ""),
            title="data analysis assistant",
            border_style="cyan",
        )
    )
    if resolved_mode == "replay":
        console.print("[yellow]Replay mode: only questions recorded earlier can be answered.[/]")
    console.print("[dim]/help for commands, /quit to leave.[/]\n")
    _resume_pending_delete(session)

    while True:
        try:
            line = console.input("[bold cyan]you >[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line[1:].split(maxsplit=1)
            name = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if name in {"quit", "exit", "q"}:
                console.print("Bye.")
                break
            if name == "help":
                _show_help()
            elif name == "debug":
                _show_debug(session)
            elif name == "trace":
                _show_trace(session, arg)
            elif name == "metrics":
                _show_metrics(session)
            elif name == "reports":
                _show_reports(session)
            elif name == "undo":
                restored = session.library.undo_last_delete(owner_id=session.user.user_id)
                console.print(
                    f"Restored {len(restored)} report(s)."
                    if restored
                    else "[dim]Nothing to restore.[/]"
                )
            elif name == "persona":
                _switch_persona(session, arg or None)
            elif name == "su":
                _switch_user(session, arg or None)
            elif name == "save":
                _save_report(session, arg)
            else:
                console.print(f"[dim]Unknown command /{name}. Try /help.[/]")
            console.print()
            continue

        # The boundary that keeps the session alive. run_turn already converts most
        # failures into answers; this catches anything it could not.
        try:
            _handle_question(session, line)
        except Exception as exc:  # noqa: BLE001
            trace = session.last_recorder.trace_id if session.last_recorder else "n/a"
            console.print(f"[red]That turn failed[/] ({type(exc).__name__}). Nothing changed.")
            console.print(f"[dim]trace {trace} - /debug for the timeline[/]")
        console.print()


@app.command()
def index() -> None:
    """Embed the golden trios so retrieval can use cosine rather than word overlap."""
    load_dotenv(Path(".env"))
    bucket = GoldenBucket("golden_bucket")
    console.print(
        f"{len(bucket.trios)} trios, {len(bucket.registry.definitions)} metric definitions"
    )
    if not bucket.trios:
        console.print("[red]No trios found in golden_bucket/trios.[/]")
        raise typer.Exit(1)
    try:
        count = bucket.build_index(Gemini())
    except Exception as exc:
        console.print(f"[red]Could not build the index:[/] {type(exc).__name__}: {exc}")
        console.print("[dim]Retrieval falls back to word overlap, so the agent still runs.[/]")
        raise typer.Exit(1) from exc
    console.print(f"[green]Embedded {count} trios[/] into golden_bucket/index.npz")


@app.command()
def doctor() -> None:
    """Check credentials, models and data access before anything else."""
    load_dotenv(Path(".env"))
    ok = True

    def report(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        mark = "[green]ok[/]" if good else "[red]FAIL[/]"
        console.print(f"  {mark}  {label}" + (f"  [dim]{detail}[/]" if detail else ""))

    console.print("\n[bold]config[/]")
    project = os.environ.get("GCP_PROJECT_ID", "")
    report("GCP_PROJECT_ID", bool(project), project or "not set in .env")
    report("GOOGLE_API_KEY", bool(os.environ.get("GOOGLE_API_KEY")), "from .env")

    store = ConfigStore("config")
    users = store.users()
    report("entitlements.yaml", bool(users), f"{len(users)} users: {', '.join(users)}")
    report("personas", bool(store.personas()), ", ".join(store.personas()))

    console.print("\n[bold]gemini[/]")
    try:
        llm = Gemini()
        available = set(llm.available_models())
        report("API reachable", True, f"{len(available)} models")
        for job, model in llm.models.items():
            if job == "embedding":
                continue
            report(f"model for {job}", model in available, model)
    except Exception as exc:
        report("API reachable", False, f"{type(exc).__name__}: {exc}")

    console.print("\n[bold]bigquery[/]")
    try:
        gateway = BigQueryGateway(project_id=project or None, cache=ResultCache("fixtures"))
        probe = "SELECT COUNT(*) AS n FROM `bigquery-public-data.thelook_ecommerce.orders`"
        estimated = gateway.dry_run(probe)
        report("dry run", True, f"~{estimated / 1e6:.1f} MB for a count over orders")

        # Catches schema drift: a new column nobody has classified could be PII.
        from .safety.schema import TABLES

        for table, columns in TABLES.items():
            live = {field["name"] for field in gateway.table_schema(table)}
            unknown = live - set(columns)
            report(f"schema {table}", not unknown, f"unclassified columns: {sorted(unknown)}"
                   if unknown else f"{len(live)} columns match")
    except Exception as exc:
        report("dry run", False, f"{type(exc).__name__}: {str(exc)[:120]}")

    console.print("\n[bold]local state[/]")
    fixtures = ResultCache("fixtures").count()
    report("fixtures for replay", True, f"{fixtures} recorded queries")
    report("reports database", True, DB_PATH)

    console.print()
    if ok:
        console.print("[green]Ready.[/] Start with: python -m analyst_assistant chat\n")
    else:
        console.print("[red]Not ready.[/] See docs/SETUP.md\n")
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
