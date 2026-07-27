"""SQLite audit log: one row per compression and per extraction/dedup
event, so every automatic memory-shaping decision is inspectable after the
fact (not just the current state of `facts`)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from memagent.config import CONFIG


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditEvent:
    id: int
    user_id: str
    event_type: str
    timestamp: datetime
    details: dict


class AuditLog:
    def __init__(self, sqlite_path: Path = CONFIG.sqlite_path) -> None:
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(sqlite_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                details TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def log(self, user_id: str, event_type: str, details: dict) -> None:
        self._conn.execute(
            "INSERT INTO audit_log (user_id, event_type, timestamp, details) VALUES (?, ?, ?, ?)",
            (user_id, event_type, _now_iso(), json.dumps(details)),
        )
        self._conn.commit()

    def log_compression(self, user_id: str, compressed_turn_count: int, summary: str) -> None:
        self.log(
            user_id,
            "compression",
            {"compressed_turn_count": compressed_turn_count, "summary": summary},
        )

    def log_extraction(self, user_id: str, fact_text: str, action: str, previous_text: str | None) -> None:
        self.log(
            user_id,
            "extraction",
            {"fact_text": fact_text, "action": action, "previous_text": previous_text},
        )

    def log_transcript(self, user_id: str, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO transcripts (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (user_id, role, content, _now_iso()),
        )
        self._conn.commit()

    def events(self, user_id: str, event_type: str | None = None) -> list[AuditEvent]:
        query = "SELECT * FROM audit_log WHERE user_id = ?"
        params: tuple = (user_id,)
        if event_type:
            query += " AND event_type = ?"
            params += (event_type,)
        query += " ORDER BY timestamp ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            AuditEvent(
                id=r["id"],
                user_id=r["user_id"],
                event_type=r["event_type"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
                details=json.loads(r["details"]),
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
