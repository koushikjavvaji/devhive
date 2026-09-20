import sqlite3
from pathlib import Path

DB_PATH = Path("data/devhive.db")


def get_connection(path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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
