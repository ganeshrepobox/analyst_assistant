"""The Saved Reports library: storage, ownership, soft delete and audit.

Implements the data half of requirement 3. The confirmation flow that guards
deletion lives in :mod:`.confirmation`; the graph wiring lives in the agent.

Three properties this module guarantees
---------------------------------------
**Ownership is enforced in SQL, not in application logic.** Every query carries
``owner_id = ?``. A report belonging to someone else is not "filtered out later" --
it is never in a result set in the first place, so "delete all my reports" cannot
reach another user's library even if every layer above this one is wrong.

**Deletion is soft and audited.** A ``deleted_at`` timestamp plus an append-only
audit row. Nothing in the brief requires irreversible erasure, and reversibility is
the right default for a destructive action taken on a model's interpretation of a
sentence. ``/undo`` restores the most recent delete batch.

**Deletes are resolved to explicit ids, never executed as a predicate.** The caller
resolves candidates, shows them, obtains confirmation, and only then deletes *those
ids*. ``soft_delete`` therefore takes a list of ids and nothing else -- there is
deliberately no ``delete_where(text=...)`` entry point, because its existence would
invite the unsafe pattern.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id          TEXT PRIMARY KEY,
    owner_id    TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    deleted_at  TEXT,
    delete_batch TEXT
);
CREATE INDEX IF NOT EXISTS idx_reports_owner ON reports(owner_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_reports_thread ON reports(owner_id, thread_id);

CREATE TABLE IF NOT EXISTS audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    actor_id   TEXT NOT NULL,
    action     TEXT NOT NULL,
    report_id  TEXT,
    trace_id   TEXT,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_report ON audit(report_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Report:
    """A saved analysis."""

    id: str
    owner_id: str
    thread_id: str
    title: str
    body: dict[str, Any]
    created_at: str
    deleted_at: str | None = None

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    def summary_line(self, index: int | None = None) -> str:
        """One row of the deletion preview.

        The preview is the entire basis on which a user consents, so it shows the
        title and the date -- enough to recognise a report -- rather than an opaque
        id.
        """
        stamp = self.created_at[:10]
        prefix = f"{index}. " if index is not None else ""
        return f"{prefix}{self.title}  ({stamp}, {self.id[:8]})"


class ReportLibrary:
    """SQLite-backed report store.

    SQLite in the prototype, Cloud SQL Postgres in production; the SQL is
    intentionally portable between them. The audit insert shares a transaction with
    the delete it records, which is the main reason this is a relational store
    rather than a document one.
    """

    def __init__(self, db_path: str | Path = "analyst_assistant.sqlite") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- internal -----------------------------------------------------------

    def _row_to_report(self, row: sqlite3.Row) -> Report:
        return Report(
            id=row["id"],
            owner_id=row["owner_id"],
            thread_id=row["thread_id"],
            title=row["title"],
            body=json.loads(row["body"]),
            created_at=row["created_at"],
            deleted_at=row["deleted_at"],
        )

    def _audit(
        self,
        actor_id: str,
        action: str,
        *,
        report_id: str | None = None,
        trace_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO audit (ts, actor_id, action, report_id, trace_id, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), actor_id, action, report_id, trace_id, detail),
        )

    # -- writes -------------------------------------------------------------

    def save(
        self,
        *,
        owner_id: str,
        thread_id: str,
        title: str,
        body: dict[str, Any],
        trace_id: str | None = None,
    ) -> Report:
        """Persist a report.

        ``thread_id`` is recorded because requirement 3 names "delete all the
        reports we made in this conversation" -- a *scope* predicate rather than a
        text match, which is unanswerable without provenance on every row.
        """
        report = Report(
            id=uuid.uuid4().hex,
            owner_id=owner_id,
            thread_id=thread_id,
            title=title,
            body=body,
            created_at=_now(),
        )
        self._conn.execute(
            "INSERT INTO reports (id, owner_id, thread_id, title, body, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                report.id,
                report.owner_id,
                report.thread_id,
                report.title,
                json.dumps(report.body),
                report.created_at,
            ),
        )
        self._audit(owner_id, "save", report_id=report.id, trace_id=trace_id, detail=title)
        self._conn.commit()
        return report

    def soft_delete(
        self, *, owner_id: str, report_ids: list[str], trace_id: str | None = None
    ) -> list[str]:
        """Soft-delete specific reports owned by ``owner_id``.

        Takes explicit ids only. There is no predicate-based delete by design: the
        caller must resolve, preview and confirm first, and the ids deleted here are
        the ones that were shown.

        Returns:
            The ids actually deleted. Ids not owned by the caller, already deleted,
            or non-existent are silently absent -- which is what makes the operation
            idempotent and safe to retry after an interrupted confirmation.
        """
        if not report_ids:
            return []

        batch = uuid.uuid4().hex
        placeholders = ",".join("?" for _ in report_ids)
        cursor = self._conn.execute(
            f"SELECT id FROM reports WHERE owner_id = ? AND deleted_at IS NULL "
            f"AND id IN ({placeholders})",
            (owner_id, *report_ids),
        )
        deletable = [row["id"] for row in cursor.fetchall()]
        if not deletable:
            self._conn.commit()
            return []

        stamp = _now()
        marks = ",".join("?" for _ in deletable)
        self._conn.execute(
            f"UPDATE reports SET deleted_at = ?, delete_batch = ? "
            f"WHERE owner_id = ? AND id IN ({marks})",
            (stamp, batch, owner_id, *deletable),
        )
        for report_id in deletable:
            self._audit(
                owner_id, "delete", report_id=report_id, trace_id=trace_id, detail=batch
            )
        self._conn.commit()
        return deletable

    def undo_last_delete(
        self, *, owner_id: str, trace_id: str | None = None
    ) -> list[str]:
        """Restore the caller's most recent delete batch.

        Batch-scoped rather than time-windowed, so ``/undo`` reverses exactly one
        user action -- which is what a user means by "undo" -- instead of an
        arbitrary set of rows that happen to share a timestamp.
        """
        cursor = self._conn.execute(
            "SELECT delete_batch FROM reports "
            "WHERE owner_id = ? AND deleted_at IS NOT NULL AND delete_batch IS NOT NULL "
            "ORDER BY deleted_at DESC LIMIT 1",
            (owner_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return []
        batch = row["delete_batch"]

        cursor = self._conn.execute(
            "SELECT id FROM reports WHERE owner_id = ? AND delete_batch = ?",
            (owner_id, batch),
        )
        restored = [r["id"] for r in cursor.fetchall()]
        self._conn.execute(
            "UPDATE reports SET deleted_at = NULL, delete_batch = NULL "
            "WHERE owner_id = ? AND delete_batch = ?",
            (owner_id, batch),
        )
        for report_id in restored:
            self._audit(
                owner_id, "undelete", report_id=report_id, trace_id=trace_id, detail=batch
            )
        self._conn.commit()
        return restored

    # -- reads --------------------------------------------------------------

    def get(self, report_id: str, *, owner_id: str) -> Report | None:
        """Fetch one report. Returns ``None`` if the caller does not own it.

        Not-found and not-owned are deliberately indistinguishable: telling a user
        that a report exists but belongs to someone else leaks the existence of
        other people's work.
        """
        cursor = self._conn.execute(
            "SELECT * FROM reports WHERE id = ? AND owner_id = ?", (report_id, owner_id)
        )
        row = cursor.fetchone()
        return self._row_to_report(row) if row else None

    def list_live(self, *, owner_id: str) -> list[Report]:
        cursor = self._conn.execute(
            "SELECT * FROM reports WHERE owner_id = ? AND deleted_at IS NULL "
            "ORDER BY created_at DESC",
            (owner_id,),
        )
        return [self._row_to_report(r) for r in cursor.fetchall()]

    def resolve(
        self,
        *,
        owner_id: str,
        text: str | None = None,
        thread_id: str | None = None,
        report_ids: list[str] | None = None,
    ) -> list[Report]:
        """Resolve a deletion request to an explicit candidate set.

        Supports both shapes requirement 3 names:

        * ``text`` -- "delete all reports mentioning Client X" (title and body)
        * ``thread_id`` -- "delete all the reports we made in this conversation"

        Only live reports owned by the caller are ever candidates. Combining
        filters narrows (AND), so a thread-scoped text match is expressible.
        """
        clauses = ["owner_id = ?", "deleted_at IS NULL"]
        params: list[Any] = [owner_id]

        if text:
            # Matched against the serialised body as well as the title, because
            # "mentioning Client X" is about content, not just the heading.
            clauses.append("(LOWER(title) LIKE ? OR LOWER(body) LIKE ?)")
            needle = f"%{text.lower()}%"
            params.extend([needle, needle])
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if report_ids is not None:
            if not report_ids:
                return []
            clauses.append(f"id IN ({','.join('?' for _ in report_ids)})")
            params.extend(report_ids)

        cursor = self._conn.execute(
            f"SELECT * FROM reports WHERE {' AND '.join(clauses)} ORDER BY created_at DESC",
            params,
        )
        return [self._row_to_report(r) for r in cursor.fetchall()]

    def audit_trail(self, *, report_id: str | None = None) -> list[dict[str, Any]]:
        """Read the audit log.

        Not owner-scoped: the audit log is an operator and compliance artifact, and
        access to it is an infrastructure-level decision rather than one the agent
        makes per request.
        """
        if report_id:
            cursor = self._conn.execute(
                "SELECT * FROM audit WHERE report_id = ? ORDER BY id", (report_id,)
            )
        else:
            cursor = self._conn.execute("SELECT * FROM audit ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]
