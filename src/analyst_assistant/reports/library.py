"""Saved reports: storage, ownership, soft delete and an audit trail.

Ownership is enforced in SQL. Every query carries owner_id, so another user's
report is never in a result set to begin with.

soft_delete takes explicit ids and nothing else. There is no delete_where(), on
purpose: the caller has to resolve and confirm first.
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
    deleted_at  TEXT
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
        """One row of the deletion preview: title and date, not an opaque id."""
        stamp = self.created_at[:10]
        prefix = f"{index}. " if index is not None else ""
        return f"{prefix}{self.title}  ({stamp}, {self.id[:8]})"


class ReportLibrary:
    """SQLite report store.

    Postgres in production; the SQL is kept portable. The audit insert shares a
    transaction with the delete it records.
    """

    def __init__(self, db_path: str | Path = "analyst_assistant.sqlite") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

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

        thread_id is stored so "delete the reports from this conversation" is answerable.
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
        """Soft-delete specific reports owned by owner_id.

        Explicit ids only; the caller resolves, previews and confirms first. Returns the
        ids actually deleted, so ids that are not owned or already deleted are simply
        absent, which makes this safe to retry.
        """
        if not report_ids:
            return []

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

        # All rows in one call share a timestamp, which is what /undo keys on.
        stamp = _now()
        marks = ",".join("?" for _ in deletable)
        self._conn.execute(
            f"UPDATE reports SET deleted_at = ? WHERE owner_id = ? AND id IN ({marks})",
            (stamp, owner_id, *deletable),
        )
        for report_id in deletable:
            self._audit(owner_id, "delete", report_id=report_id, trace_id=trace_id)
        self._conn.commit()
        return deletable

    def undo_last_delete(
        self, *, owner_id: str, trace_id: str | None = None
    ) -> list[str]:
        """Restore the caller's most recent deletion.

        Keyed on the newest deleted_at: everything deleted in one call shares that
        timestamp, so this reverses exactly one action.
        """
        cursor = self._conn.execute(
            "SELECT MAX(deleted_at) AS last FROM reports "
            "WHERE owner_id = ? AND deleted_at IS NOT NULL",
            (owner_id,),
        )
        last = cursor.fetchone()["last"]
        if last is None:
            return []

        cursor = self._conn.execute(
            "SELECT id FROM reports WHERE owner_id = ? AND deleted_at = ?", (owner_id, last)
        )
        restored = [row["id"] for row in cursor.fetchall()]
        self._conn.execute(
            "UPDATE reports SET deleted_at = NULL WHERE owner_id = ? AND deleted_at = ?",
            (owner_id, last),
        )
        for report_id in restored:
            self._audit(owner_id, "undelete", report_id=report_id, trace_id=trace_id)
        self._conn.commit()
        return restored

    def get(self, report_id: str, *, owner_id: str) -> Report | None:
        """Fetch one report, or None if the caller does not own it.

        Not-found and not-owned are indistinguishable on purpose.
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

        text matches title and body; thread_id scopes to one conversation. Filters
        combine with AND. Only live reports owned by the caller are candidates.
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
        """Read the audit log. Not owner-scoped: it is an operator artifact."""
        if report_id:
            cursor = self._conn.execute(
                "SELECT * FROM audit WHERE report_id = ? ORDER BY id", (report_id,)
            )
        else:
            cursor = self._conn.execute("SELECT * FROM audit ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]
