import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import db
from app.rooms import RoomManager
from app.teammates.heuristic import HeuristicTeammate
from app.teammates.local_model import build_local_model_teammate
from app.teammates.openai_teammate import build_openai_teammate

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="devhive")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def build_teammates():
    teammates = [HeuristicTeammate()]

    local_model = build_local_model_teammate()
    if local_model:
        teammates.append(local_model)

    openai_teammate = build_openai_teammate()
    if openai_teammate:
        teammates.append(openai_teammate)

    return teammates


conn = db.get_connection()
room_manager = RoomManager(build_teammates(), conn)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/r/{room_id}")
async def room_page(room_id: str):
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/rooms")
async def list_rooms():
    return db.list_rooms(conn)


@app.post("/api/rooms")
async def create_room():
    room_id = uuid.uuid4().hex[:12]
    db.create_room(conn, room_id, time.time())
    return {"id": room_id}


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    room = room_manager.get(room_id)
    await room.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "message" and data.get("text", "").strip():
                await room.handle_human_message(data["text"])
    except WebSocketDisconnect:
        room.disconnect(websocket)
