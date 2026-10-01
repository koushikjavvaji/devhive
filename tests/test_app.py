import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import db, limits
from app.main import create_app, display_name
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


def test_created_room_is_hidden_from_the_list_until_it_has_a_message(tmp_path):
    """A room row exists as soon as "+ New war room" is clicked — that shouldn't clutter
    the list with dead "untitled, 0 messages" rows nobody ever actually used."""
    client = make_client(tmp_path)

    created = client.post("/api/rooms").json()
    assert "id" in created
    assert client.get("/api/rooms").json() == []


def test_websocket_flow_sets_title_and_gets_teammate_reply(tmp_path):
    client = make_client(tmp_path)
    room_id = client.post("/api/rooms").json()["id"]

    with client.websocket_connect(f"/ws/{room_id}") as ws:
        history = ws.receive_json()
        assert history == {
            "type": "history",
            "messages": [],
            "teammates": [{"name": "triage-bot", "color": "#2b9348"}],
            "busy": False,
        }

        ws.send_json({"type": "message", "text": "ZeroDivisionError: division by zero"})

        human_msg = ws.receive_json()
        assert human_msg["message"]["sender_type"] == "human"

        typing = ws.receive_json()
        assert typing == {"type": "typing", "sender": "triage-bot"}

        delta = ws.receive_json()
        assert delta["type"] == "delta"

        reply = ws.receive_json()
        assert reply["message"]["sender"] == "triage-bot"
        assert reply["message"]["status"] == "ok"
        assert "ZeroDivisionError" in reply["message"]["text"]

        assert ws.receive_json() == {"type": "idle"}

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


def test_unknown_room_is_rejected_instead_of_created(tmp_path):
    client = make_client(tmp_path)

    with client.websocket_connect("/ws/made-up-id") as ws:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_json()
    assert exc_info.value.code == 4404


def test_oversized_message_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "MAX_MESSAGE_CHARS", 10)
    client = make_client(tmp_path)
    room_id = client.post("/api/rooms").json()["id"]

    with client.websocket_connect(f"/ws/{room_id}") as ws:
        ws.receive_json()  # history
        ws.send_json({"type": "message", "text": "x" * 11})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "too long" in error["text"]


def test_messages_are_rate_limited_per_ip(tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "MESSAGES_PER_IP", 1)
    client = make_client(tmp_path)
    room_id = client.post("/api/rooms").json()["id"]

    with client.websocket_connect(f"/ws/{room_id}") as ws:
        ws.receive_json()  # history
        ws.send_json({"type": "message", "text": "first"})
        while ws.receive_json()["type"] != "idle":
            pass

        ws.send_json({"type": "message", "text": "second"})
        error = ws.receive_json()
        assert error["type"] == "error"
        assert "Slow down" in error["text"]


def test_room_creation_is_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "ROOMS_PER_IP_PER_HOUR", 1)
    client = make_client(tmp_path)

    assert client.post("/api/rooms").status_code == 200
    assert client.post("/api/rooms").status_code == 429


def test_invalid_json_is_ignored_without_dropping_the_connection(tmp_path):
    client = make_client(tmp_path)
    room_id = client.post("/api/rooms").json()["id"]

    with client.websocket_connect(f"/ws/{room_id}") as ws:
        ws.receive_json()  # history
        ws.send_text("{not json")
        ws.send_json({"type": "message", "text": "still here", "name": "ada"})
        human_msg = ws.receive_json()["message"]
        assert human_msg["text"] == "still here"
        assert human_msg["sender"] == "ada"


def test_display_name():
    assert display_name(None, ["triage-bot"]) == "you"
    assert display_name("   ", ["triage-bot"]) == "you"
    assert display_name("  ada \n lovelace ", ["triage-bot"]) == "ada lovelace"
    assert display_name("x" * 100, []) == "x" * 24
    # a human can't pass themselves off as an AI teammate
    assert display_name("Triage-Bot", ["triage-bot"]) == "Triage-Bot (human)"


def test_client_ip_trusts_only_the_proxy_appended_entry():
    class Conn:
        def __init__(self, xff):
            self.headers = {"x-forwarded-for": xff} if xff else {}
            self.client = type("C", (), {"host": "10.0.0.1"})()

    assert limits.client_ip(Conn("6.6.6.6, 1.2.3.4"), trusted_hops=1) == "1.2.3.4"
    assert limits.client_ip(Conn("6.6.6.6, 1.2.3.4"), trusted_hops=0) == "10.0.0.1"
    assert limits.client_ip(Conn(None), trusted_hops=1) == "10.0.0.1"


def test_rate_limiter_window_slides():
    now = [0.0]
    limiter = limits.RateLimiter(2, 10, clock=lambda: now[0])

    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")  # per key

    now[0] = 10.0
    assert limiter.allow("a")


def test_pages_get_a_content_security_policy(tmp_path):
    resp = make_client(tmp_path).get("/")
    csp = resp.headers["content-security-policy"]
    assert "script-src 'self' https://cdn.jsdelivr.net;" in csp
    assert "object-src 'none'" in csp
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_importing_the_app_module_has_no_side_effects():
    """No module-level app: importing must not open the real db or load the checkpoint."""
    import app.main
    assert not hasattr(app.main, "app")
