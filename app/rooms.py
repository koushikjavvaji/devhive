import asyncio
import time

from app import db


class Room:
    """A single war room: message history, connected clients, and the teammates in it.

    Each teammate responds independently and concurrently as soon as it's ready, so
    faster teammates (the heuristic one) don't wait on slower ones (model inference).
    Messages are persisted to sqlite, so history survives a server restart."""

    def __init__(self, teammates, room_id="main", conn=None):
        self.teammates = teammates
        self.room_id = room_id
        self.conn = conn or db.get_connection()
        self.messages = db.load_messages(self.conn, self.room_id)
        self.connections = []

    def _record(self, sender, sender_type, text):
        ts = time.time()
        message_id = db.insert_message(self.conn, self.room_id, sender, sender_type, text, ts)
        message = {
            "id": message_id,
            "sender": sender,
            "sender_type": sender_type,
            "text": text,
            "ts": ts,
        }
        self.messages.append(message)
        return message

    async def connect(self, websocket):
        await websocket.accept()
        self.connections.append(websocket)
        await websocket.send_json({
            "type": "history",
            "messages": self.messages,
            "teammates": [{"name": t.name, "color": t.color} for t in self.teammates],
        })

    def disconnect(self, websocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, payload):
        dead = []
        for websocket in self.connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)

    async def handle_human_message(self, text):
        message = self._record("you", "human", text)
        await self.broadcast({"type": "message", "message": message})

        for teammate in self.teammates:
            asyncio.create_task(self._get_teammate_reply(teammate))

    async def _get_teammate_reply(self, teammate):
        await self.broadcast({"type": "typing", "sender": teammate.name})

        try:
            reply = await teammate.respond(self.messages)
        except Exception as exc:
            reply = f"(failed to respond: {exc})"

        message = self._record(teammate.name, "teammate", reply)
        await self.broadcast({"type": "message", "message": message})
