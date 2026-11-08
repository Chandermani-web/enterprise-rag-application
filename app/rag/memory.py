"""Persistent conversation memory (SQLite, WAL). Swap for Postgres by re-implementing this small class."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class ChatMemory:
    def __init__(self, db_path: Path):
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL,"
            " role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT, created_at REAL NOT NULL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_thread ON messages(thread_id, id)")
        self._db.commit()
        self._lock = threading.Lock()

    def add(self, thread_id: str, role: str, content: str, meta: dict | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO messages(thread_id, role, content, meta, created_at) VALUES (?,?,?,?,?)",
                (thread_id, role, content, json.dumps(meta or {}), time.time()),
            )
            self._db.commit()

    def history(self, thread_id: str, limit: int = 12) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE thread_id=? ORDER BY id DESC LIMIT ?", (thread_id, limit)
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def messages(self, thread_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content, meta, created_at FROM messages WHERE thread_id=? ORDER BY id", (thread_id,)
            ).fetchall()
        return [{"role": r, "content": c, "meta": json.loads(m or "{}"), "created_at": t} for r, c, m, t in rows]

    def threads(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT thread_id, MAX(created_at), COUNT(*),"
                " (SELECT content FROM messages m2 WHERE m2.thread_id=m.thread_id AND m2.role='user' ORDER BY id LIMIT 1)"
                " FROM messages m GROUP BY thread_id ORDER BY MAX(created_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"thread_id": t, "updated_at": u, "messages": n, "title": (title or "")[:60]} for t, u, n, title in rows]

    def delete(self, thread_id: str) -> int:
        with self._lock:
            cur = self._db.execute("DELETE FROM messages WHERE thread_id=?", (thread_id,))
            self._db.commit()
            return cur.rowcount
