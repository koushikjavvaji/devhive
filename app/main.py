import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import db, limits
from app.rooms import RoomManager
from app.teammates.heuristic import HeuristicTeammate
from app.teammates.llm_teammate import build_llm_teammates
from app.teammates.local_model import build_local_model_teammate

STATIC_DIR = Path(__file__).parent / "static"

MAX_NAME_CHARS = 24

# app-level loggers (teammate failures, crashed debates) default to WARNING with no
# handler of their own under uvicorn — give them one so they actually show up
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:     %(name)s: %(message)s")


def build_teammates():
    teammates = [HeuristicTeammate()]

    local_model = build_local_model_teammate()
    if local_model:
        teammates.append(local_model)

    teammates.extend(build_llm_teammates())

    return teammates


def display_name(raw, teammate_names):
    """Who a human message is from. Free text from the client, so: collapse whitespace,
    cap the length, and don't let a human pass themselves off as one of the AI teammates."""
    name = " ".join(str(raw or "").split())[:MAX_NAME_CHARS]
    if not name:
        return "you"
    if name.lower() in {t.lower() for t in teammate_names}:
        return f"{name} (human)"
    return name


def create_app(conn=None, teammates=None):
    """Factory so tests can inject an isolated db connection and a fixed teammate list
    instead of hitting the real database and (possibly absent) trained checkpoint."""
    fastapi_app = FastAPI(title="devhive")
    fastapi_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    conn = conn or db.get_connection()
    room_manager = RoomManager(teammates if teammates is not None else build_teammates(), conn)
    fastapi_app.state.room_manager = room_manager
    teammate_names = [t.name for t in room_manager.teammates]

    message_limiter = limits.RateLimiter(limits.MESSAGES_PER_IP, limits.MESSAGES_WINDOW_SECONDS)
    daily_limiter = limits.RateLimiter(limits.DAILY_MESSAGE_LIMIT, 24 * 60 * 60)
    room_limiter = limits.RateLimiter(limits.ROOMS_PER_IP_PER_HOUR, 60 * 60)

    @fastapi_app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @fastapi_app.get("/r/{room_id}")
    async def room_page(room_id: str):
        return FileResponse(STATIC_DIR / "index.html")

    @fastapi_app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "teammates": [t.name for t in room_manager.teammates],
        }

    @fastapi_app.get("/api/rooms")
    async def list_rooms():
        return db.list_rooms(conn)

    @fastapi_app.post("/api/rooms")
    async def create_room(request: Request):
        if not room_limiter.allow(limits.client_ip(request)):
            raise HTTPException(429, "Too many new rooms — try again later.")
        room_id = uuid.uuid4().hex[:12]
        db.create_room(conn, room_id, time.time())
        return {"id": room_id}

    @fastapi_app.websocket("/ws/{room_id}")
    async def websocket_endpoint(websocket: WebSocket, room_id: str):
        await websocket.accept()

        room = room_manager.get(room_id)
        if room is None:
            await websocket.close(code=4404, reason="room not found")
            return

        ip = limits.client_ip(websocket)
        try:
            await room.connect(websocket)
            while True:
                try:
                    data = json.loads(await websocket.receive_text())
                except ValueError:
                    continue  # not JSON — ignore it rather than dropping the connection
                if not isinstance(data, dict) or data.get("type") != "message":
                    continue
                text = str(data.get("text") or "")
                if not text.strip():
                    continue

                error = None
                if len(text) > limits.MAX_MESSAGE_CHARS:
                    error = f"That's too long — keep it under {limits.MAX_MESSAGE_CHARS:,} characters."
                elif not message_limiter.allow(ip):
                    error = "Slow down — too many messages. Try again in a few minutes."
                elif not daily_limiter.allow():
                    error = "The demo has hit its daily message limit. Try again tomorrow."

                if error:
                    await websocket.send_json({"type": "error", "text": error})
                    continue

                await room.handle_human_message(text, display_name(data.get("name"), teammate_names))
        except WebSocketDisconnect:
            pass
        finally:
            room.disconnect(websocket)

    return fastapi_app


app = create_app()
