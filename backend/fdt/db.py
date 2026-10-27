"""SQLite persistence for saved scenarios."""
import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.environ.get("FINTWIN_DB", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fintwin.db"))


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            inputs TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL)""")


def save(name: str, kind: str, inputs: dict, summary: dict) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO scenarios(name, kind, inputs, summary, created_at) VALUES (?,?,?,?,?)",
                        (name[:120], kind, json.dumps(inputs), json.dumps(summary, default=float),
                         datetime.now(timezone.utc).isoformat(timespec="seconds")))
        return cur.lastrowid


def list_all(limit: int = 50):
    with _conn() as c:
        rows = c.execute("SELECT * FROM scenarios ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "inputs": json.loads(r["inputs"]), "summary": json.loads(r["summary"])} for r in rows]


def delete(sid: int):
    with _conn() as c:
        c.execute("DELETE FROM scenarios WHERE id = ?", (sid,))
