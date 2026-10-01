import sqlite3

from app import db


def test_old_database_gets_a_status_column_with_failures_backfilled(tmp_path):
    """Databases from before the status column marked failures only by their text."""
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE rooms (id TEXT PRIMARY KEY, title TEXT, created_at REAL NOT NULL)")
    old.execute("""CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, room_id TEXT NOT NULL,
        sender TEXT NOT NULL, sender_type TEXT NOT NULL, text TEXT NOT NULL, ts REAL NOT NULL)""")
    old.execute("INSERT INTO messages (room_id, sender, sender_type, text, ts) VALUES ('r', 'bot', 'teammate', 'fine', 1)")
    old.execute("INSERT INTO messages (room_id, sender, sender_type, text, ts) "
                "VALUES ('r', 'bot', 'teammate', '(failed to respond: boom)', 2)")
    old.commit()
    old.close()

    conn = db.get_connection(path)

    assert [m["status"] for m in db.load_messages(conn, "r")] == ["ok", "failed"]
