import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("DEVHIVE_DB_PATH", "data/devhive.db"))


def get_connection(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id TEXT NOT NULL,
            sender TEXT NOT NULL,
            sender_type TEXT NOT NULL,
            text TEXT NOT NULL,
            ts REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_room ON messages (room_id, id)")
    conn.commit()
    return conn


def create_room(conn, room_id, created_at):
    conn.execute(
        "INSERT OR IGNORE INTO rooms (id, title, created_at) VALUES (?, NULL, ?)",
        (room_id, created_at),
    )
    conn.commit()


def list_rooms(conn):
    rows = conn.execute("""
        SELECT r.id, r.title, r.created_at, COUNT(m.id) AS message_count
        FROM rooms r
        LEFT JOIN messages m ON m.room_id = r.id
        GROUP BY r.id
        ORDER BY r.created_at DESC
    """).fetchall()
    return [dict(row) for row in rows]


def set_room_title(conn, room_id, title):
    conn.execute(
        "UPDATE rooms SET title = ? WHERE id = ? AND title IS NULL",
        (title, room_id),
    )
    conn.commit()


def load_messages(conn, room_id):
    rows = conn.execute(
        "SELECT id, sender, sender_type, text, ts FROM messages WHERE room_id = ? ORDER BY id",
        (room_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def insert_message(conn, room_id, sender, sender_type, text, ts):
    cursor = conn.execute(
        "INSERT INTO messages (room_id, sender, sender_type, text, ts) VALUES (?, ?, ?, ?, ?)",
        (room_id, sender, sender_type, text, ts),
    )
    conn.commit()
    return cursor.lastrowid
