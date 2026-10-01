"""Room/message storage. SQLite by default (a local file, zero setup); Postgres when
DATABASE_URL is set — needed on hosts like Render's free tier, whose disk is wiped on
every deploy/restart. Both backends speak the same SQL below: `?` placeholders (rewritten
for Postgres), ON CONFLICT and RETURNING all work in SQLite >= 3.35 and Postgres."""

import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("DEVHIVE_DB_PATH", "data/devhive.db"))
DATABASE_URL = os.environ.get("DATABASE_URL")

ROOMS_LIST_LIMIT = 200


class SqliteDatabase:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def execute(self, sql, params=()):
        cursor = self._conn.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        self._conn.commit()
        return rows

    def _create_schema(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL NOT NULL
            )
        """)
        self.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                sender_type TEXT NOT NULL,
                text TEXT NOT NULL,
                ts REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'ok'
            )
        """)
        # databases created before the status column existed: add it, and backfill the
        # failures that used to be marked only by their "(failed to ...)" text
        columns = {row["name"] for row in self.execute("PRAGMA table_info(messages)")}
        if "status" not in columns:
            self.execute("ALTER TABLE messages ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'")
            self.execute("UPDATE messages SET status = 'failed' WHERE text LIKE '(failed to %'")
        self.execute("CREATE INDEX IF NOT EXISTS idx_messages_room ON messages (room_id, id)")

    def close(self):
        self._conn.close()


class PostgresDatabase:
    def __init__(self, url):
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(
            url,
            min_size=1,
            max_size=4,
            # hosted free tiers (Neon, Supabase) close idle connections — check each one
            # before handing it out instead of failing the first query after a quiet spell
            check=ConnectionPool.check_connection,
            # prepare_threshold=None: server-side prepared statements break behind
            # transaction-mode poolers like Supabase's pgbouncer
            kwargs={"row_factory": dict_row, "autocommit": True, "prepare_threshold": None},
            open=True,
        )
        self._create_schema()

    def execute(self, sql, params=()):
        with self._pool.connection() as conn:
            cursor = conn.execute(sql.replace("?", "%s"), params or None)
            return cursor.fetchall() if cursor.description else []

    def _create_schema(self):
        self.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at DOUBLE PRECISION NOT NULL
            )
        """)
        self.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                room_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                sender_type TEXT NOT NULL,
                text TEXT NOT NULL,
                ts DOUBLE PRECISION NOT NULL,
                status TEXT NOT NULL DEFAULT 'ok'
            )
        """)
        self.execute("CREATE INDEX IF NOT EXISTS idx_messages_room ON messages (room_id, id)")

    def close(self):
        self._pool.close()


def get_connection(path=None, url=None):
    """An explicit path or url wins; otherwise DATABASE_URL if set, else the SQLite file."""
    if path is not None:
        return SqliteDatabase(path)
    url = url or DATABASE_URL
    if url:
        return PostgresDatabase(url)
    return SqliteDatabase(DB_PATH)


def create_room(conn, room_id, created_at):
    conn.execute(
        "INSERT INTO rooms (id, title, created_at) VALUES (?, NULL, ?) ON CONFLICT (id) DO NOTHING",
        (room_id, created_at),
    )


def list_rooms(conn):
    # a room row exists as soon as "+ New war room" is clicked, before anyone's sent
    # anything — hide the ones nobody ever actually used instead of listing dead rows
    return conn.execute("""
        SELECT r.id, r.title, r.created_at, COUNT(m.id) AS message_count
        FROM rooms r
        LEFT JOIN messages m ON m.room_id = r.id
        GROUP BY r.id
        HAVING COUNT(m.id) > 0
        ORDER BY r.created_at DESC
        LIMIT ?
    """, (ROOMS_LIST_LIMIT,))


def room_exists(conn, room_id):
    return bool(conn.execute("SELECT 1 FROM rooms WHERE id = ?", (room_id,)))


def set_room_title(conn, room_id, title):
    conn.execute(
        "UPDATE rooms SET title = ? WHERE id = ? AND title IS NULL",
        (title, room_id),
    )


def load_messages(conn, room_id):
    return conn.execute(
        "SELECT id, sender, sender_type, text, ts, status FROM messages WHERE room_id = ? ORDER BY id",
        (room_id,),
    )


def insert_message(conn, room_id, sender, sender_type, text, ts, status="ok"):
    rows = conn.execute(
        "INSERT INTO messages (room_id, sender, sender_type, text, ts, status) "
        "VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
        (room_id, sender, sender_type, text, ts, status),
    )
    return rows[0]["id"]
