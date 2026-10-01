import os
import sqlite3

import pytest

from app import db

TEST_DATABASE_URL = os.environ.get("DEVHIVE_TEST_DATABASE_URL")


@pytest.fixture(params=["sqlite", "postgres"])
def conn(request, tmp_path):
    """Every test here runs against both backends. Postgres needs a server, so it only
    runs when DEVHIVE_TEST_DATABASE_URL points at one (CI provides it)."""
    if request.param == "sqlite":
        database = db.get_connection(tmp_path / "test.db")
    else:
        if not TEST_DATABASE_URL:
            pytest.skip("DEVHIVE_TEST_DATABASE_URL not set")
        database = db.get_connection(url=TEST_DATABASE_URL)
        database.execute("TRUNCATE rooms, messages RESTART IDENTITY")
    yield database
    database.close()


def test_rooms_round_trip(conn):
    db.create_room(conn, "r1", 1.0)
    db.create_room(conn, "r1", 2.0)  # creating twice is a no-op, not an error
    db.create_room(conn, "r2", 3.0)

    assert db.room_exists(conn, "r1")
    assert not db.room_exists(conn, "nope")

    db.set_room_title(conn, "r1", "first title")
    db.set_room_title(conn, "r1", "second title")  # only the first one sticks

    db.insert_message(conn, "r1", "you", "human", "hi", 4.0)

    assert db.list_rooms(conn) == [
        {"id": "r1", "title": "first title", "created_at": 1.0, "message_count": 1},
    ]  # r2 has no messages, so it's hidden


def test_messages_round_trip(conn):
    db.create_room(conn, "r1", 1.0)
    first = db.insert_message(conn, "r1", "you", "human", "it's broken? 100%", 2.0)
    second = db.insert_message(conn, "r1", "bot", "teammate", "(failed to respond: timed out)", 3.0, "failed")

    assert second > first
    assert db.load_messages(conn, "r1") == [
        {"id": first, "sender": "you", "sender_type": "human", "text": "it's broken? 100%", "ts": 2.0, "status": "ok"},
        {"id": second, "sender": "bot", "sender_type": "teammate", "text": "(failed to respond: timed out)",
         "ts": 3.0, "status": "failed"},
    ]
    assert db.load_messages(conn, "other") == []


def test_schema_creation_is_idempotent(conn, tmp_path):
    db.create_room(conn, "r1", 1.0)
    again = db.get_connection(tmp_path / "test.db") if isinstance(conn, db.SqliteDatabase) \
        else db.get_connection(url=TEST_DATABASE_URL)
    assert db.room_exists(again, "r1")
    again.close()


def test_old_sqlite_database_gets_a_status_column_with_failures_backfilled(tmp_path):
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


def test_database_url_selects_postgres(monkeypatch):
    if not TEST_DATABASE_URL:
        pytest.skip("DEVHIVE_TEST_DATABASE_URL not set")
    monkeypatch.setattr(db, "DATABASE_URL", TEST_DATABASE_URL)
    conn = db.get_connection()
    assert isinstance(conn, db.PostgresDatabase)
    conn.close()
