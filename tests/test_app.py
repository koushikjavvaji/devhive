from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.teammates.heuristic import HeuristicTeammate


def make_client(tmp_path):
    # isolated db + a fixed teammate list, so these tests don't touch the real
    # database or depend on a trained checkpoint being present on disk
    conn = db.get_connection(tmp_path / "test.db")
    fastapi_app = create_app(conn=conn, teammates=[HeuristicTeammate()])
    return TestClient(fastapi_app)


def test_health(tmp_path):
    resp = make_client(tmp_path).get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "teammates": ["triage-bot"]}


def test_create_and_list_rooms(tmp_path):
    client = make_client(tmp_path)

    created = client.post("/api/rooms").json()
    assert "id" in created

    rooms = client.get("/api/rooms").json()
    assert len(rooms) == 1
    assert rooms[0]["id"] == created["id"]
    assert rooms[0]["title"] is None
    assert rooms[0]["message_count"] == 0


def test_websocket_flow_sets_title_and_gets_teammate_reply(tmp_path):
    client = make_client(tmp_path)
    room_id = client.post("/api/rooms").json()["id"]

    with client.websocket_connect(f"/ws/{room_id}") as ws:
        history = ws.receive_json()
        assert history == {
            "type": "history",
            "messages": [],
            "teammates": [{"name": "triage-bot", "color": "#2b9348"}],
        }

        ws.send_json({"type": "message", "text": "ZeroDivisionError: division by zero"})

        human_msg = ws.receive_json()
        assert human_msg["message"]["sender_type"] == "human"

        typing = ws.receive_json()
        assert typing == {"type": "typing", "sender": "triage-bot"}

        reply = ws.receive_json()
        assert reply["message"]["sender"] == "triage-bot"
        assert "ZeroDivisionError" in reply["message"]["text"]

    rooms = client.get("/api/rooms").json()
    assert rooms[0]["title"] == "ZeroDivisionError: division by zero"
    assert rooms[0]["message_count"] == 2


def test_blank_message_is_ignored(tmp_path):
    client = make_client(tmp_path)
    room_id = client.post("/api/rooms").json()["id"]

    with client.websocket_connect(f"/ws/{room_id}") as ws:
        ws.receive_json()  # history
        ws.send_json({"type": "message", "text": "   "})

        # nothing should come back for a blank message; a real one proves the
        # blank one didn't silently queue something up behind it
        ws.send_json({"type": "message", "text": "real message"})
        human_msg = ws.receive_json()
        assert human_msg["message"]["text"] == "real message"
