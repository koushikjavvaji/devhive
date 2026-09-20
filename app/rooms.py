import asyncio
import time

from app import db


class Room:
    """A single war room: message history, connected clients, and the teammates in it.

    A message triggers round 1 (everyone answers the human independently and concurrently,
    so faster teammates like the heuristic one don't wait on slower model inference), then
    `reaction_rounds` more rounds where any teammate that can_react sees the whole room —
    including every prior reaction round, not just round 1 — and actually pushes back on
    it. That's what makes it a conversation between reactors instead of each one just
    reacting to round 1 in isolation. Messages are persisted to sqlite, so history survives
    a server restart."""

    REACTION_ROUNDS = 2

    def __init__(self, teammates, room_id, conn, reaction_rounds=REACTION_ROUNDS):
        self.teammates = teammates
        self.room_id = room_id
        self.conn = conn
        self.reaction_rounds = reaction_rounds
        self.messages = db.load_messages(self.conn, self.room_id)
        self.connections = []
        self._has_title = any(m["sender_type"] == "human" for m in self.messages)

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

        if not self._has_title:
            db.set_room_title(self.conn, self.room_id, text.splitlines()[0][:60])
            self._has_title = True

        # runs as one background task so this doesn't block the websocket loop, but the
        # two rounds inside it still happen in order (react needs round 1 to exist first)
        asyncio.create_task(self._run_rounds())

    async def _run_rounds(self):
        await asyncio.gather(*(
            self._get_teammate_reply(teammate) for teammate in self.teammates
        ))

        reactors = [t for t in self.teammates if t.can_react]
        if reactors and len(self.teammates) > 1:
            for _ in range(self.reaction_rounds):
                await asyncio.gather(*(
                    self._get_teammate_reaction(teammate) for teammate in reactors
                ))

    async def _get_teammate_reply(self, teammate):
        await self.broadcast({"type": "typing", "sender": teammate.name})

        try:
            reply = await teammate.respond(self.messages)
        except Exception as exc:
            reply = f"(failed to respond: {exc})"

        message = self._record(teammate.name, "teammate", reply)
        await self.broadcast({"type": "message", "message": message})

    async def _get_teammate_reaction(self, teammate):
        await self.broadcast({"type": "typing", "sender": teammate.name})

        try:
            reply = await teammate.react(self.messages)
        except Exception as exc:
            reply = f"(failed to react: {exc})"

        message = self._record(teammate.name, "teammate", reply)
        await self.broadcast({"type": "message", "message": message})


class RoomManager:
    """Rooms are cheap and lazy: teammates (incl. the loaded model) are shared across
    all of them, only per-room state (messages, connections) differs."""

    def __init__(self, teammates, conn):
        self.teammates = teammates
        self.conn = conn
        self._rooms = {}

    def get(self, room_id):
        if room_id not in self._rooms:
            db.create_room(self.conn, room_id, time.time())
            self._rooms[room_id] = Room(self.teammates, room_id, self.conn)
        return self._rooms[room_id]
