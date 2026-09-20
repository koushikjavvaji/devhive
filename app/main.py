from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.rooms import Room
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


room = Room(build_teammates())


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await room.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "message" and data.get("text", "").strip():
                await room.handle_human_message(data["text"])
    except WebSocketDisconnect:
        room.disconnect(websocket)
