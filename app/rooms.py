import asyncio
import logging
import re
import time

from app import db

log = logging.getLogger(__name__)

# hard ceiling on one teammate's turn, on top of the LLM client's own timeout — covers
# teammates that don't have one (the local model) and streams that trickle forever
TEAMMATE_TIMEOUT_SECONDS = 90

VERDICT_RE = re.compile(r"\s*VERDICT:\s*(AGREE|DISAGREE)\W*\s*$", re.IGNORECASE)


def split_verdict(text):
    """Reactors end with a 'VERDICT: AGREE/DISAGREE' line (see REACTION_SYSTEM_PROMPT).
    Returns the text without it, and True/False/None (None = no verdict given)."""
    match = VERDICT_RE.search(text)
    if not match:
        return text, None
    return text[:match.start()].rstrip(), match.group(1).upper() == "AGREE"


def describe_failure(exc):
    """What the room gets to see about a failure — short, and without dumping whatever
    an arbitrary exception's repr happens to contain. Full details go to the server log."""
    if isinstance(exc, TimeoutError):
        return "timed out"
    status_code = getattr(exc, "status_code", None)
    if status_code:
        return f"provider returned HTTP {status_code}"
    if isinstance(exc, RuntimeError):
        # raised by our own teammate code with a message meant to be shown
        return str(exc)[:200]
    return "unexpected error"


class Room:
    """A single war room: message history, connected clients, and the teammates in it.

    A message triggers round 1 (everyone answers the human independently and concurrently,
    so faster teammates like the heuristic one don't wait on slower model inference), then
    up to `reaction_rounds` more rounds where any teammate that can_react sees the whole
    room — including every prior reaction round, not just round 1 — and actually pushes
    back on it. That's what makes it a conversation between reactors instead of each one
    just reacting to round 1 in isolation. The debate stops early once every reactor says
    it agrees. Messages that arrive mid-debate queue up and get their own full debate,
    in order — each debate only sees the human messages up to the one it's answering, so
    a queued one never gets answered early. Messages are persisted to sqlite."""

    REACTION_ROUNDS = 2

    def __init__(self, teammates, room_id, conn, reaction_rounds=REACTION_ROUNDS, on_idle=None):
        self.teammates = teammates
        self.room_id = room_id
        self.conn = conn
        self.reaction_rounds = reaction_rounds
        self.on_idle = on_idle
        self.messages = db.load_messages(self.conn, self.room_id)
        self.connections = []
        self._has_title = any(m["sender_type"] == "human" for m in self.messages)
        # one debate at a time; asyncio.Lock wakes waiters in FIFO order, so queued
        # messages are answered in the order they were sent
        self._debate_lock = asyncio.Lock()
        # asyncio only keeps weak references to running tasks — hold them here
        self._tasks = set()

    @property
    def busy(self):
        return bool(self._tasks)

    def _record(self, sender, sender_type, text, status="ok"):
        ts = time.time()
        message_id = db.insert_message(self.conn, self.room_id, sender, sender_type, text, ts, status)
        message = {
            "id": message_id,
            "sender": sender,
            "sender_type": sender_type,
            "text": text,
            "ts": ts,
            "status": status,
        }
        self.messages.append(message)
        return message

    async def connect(self, websocket):
        """Expects an already-accepted websocket: registering it happens before the first
        await, so the room can't be evicted between RoomManager.get() and this."""
        self.connections.append(websocket)
        await websocket.send_json({
            "type": "history",
            "messages": self.messages,
            "teammates": [{"name": t.name, "color": t.color} for t in self.teammates],
            "busy": self.busy,
        })

    def disconnect(self, websocket):
        if websocket in self.connections:
            self.connections.remove(websocket)
        self._maybe_idle()

    def _maybe_idle(self):
        if self.on_idle and not self.connections and not self.busy:
            self.on_idle(self)

    async def broadcast(self, payload):
        dead = []
        for websocket in self.connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            if websocket in self.connections:
                self.connections.remove(websocket)

    async def handle_human_message(self, text, sender="you"):
        message = self._record(sender, "human", text)
        await self.broadcast({"type": "message", "message": message})

        if not self._has_title:
            db.set_room_title(self.conn, self.room_id, text.splitlines()[0][:60])
            self._has_title = True

        # runs as a background task so this doesn't block the websocket loop; the rounds
        # inside it still happen in order (reactions need round 1 to exist first)
        task = asyncio.create_task(self._queued_debate(message["id"]))
        self._tasks.add(task)
        task.add_done_callback(self._on_debate_done)

    async def _queued_debate(self, anchor_id):
        async with self._debate_lock:
            await self._run_rounds(anchor_id)

    def _on_debate_done(self, task):
        self._tasks.discard(task)
        if not task.cancelled() and task.exception():
            log.error("room %s: debate crashed", self.room_id, exc_info=task.exception())
        self._maybe_idle()

    def _view(self, anchor_id):
        """The room as the debate answering human message anchor_id should see it.
        Debates run one at a time, so everything non-human after the anchor belongs to
        this debate — only human messages queued behind it need hiding."""
        if anchor_id is None:
            return list(self.messages)
        return [m for m in self.messages if m["sender_type"] != "human" or m["id"] <= anchor_id]

    async def _run_rounds(self, anchor_id=None):
        try:
            await asyncio.gather(*(
                self._take_turn(teammate, "respond", anchor_id) for teammate in self.teammates
            ))

            reactors = [t for t in self.teammates if t.can_react]
            if reactors and len(self.teammates) > 1:
                for _ in range(self.reaction_rounds):
                    verdicts = await asyncio.gather(*(
                        self._take_turn(teammate, "react", anchor_id) for teammate in reactors
                    ))
                    if all(verdict is True for verdict in verdicts):
                        break  # converged — another round would just be "still agree"
        finally:
            if not asyncio.current_task().cancelling():
                await self.broadcast({"type": "idle"})

    async def _take_turn(self, teammate, kind, anchor_id=None):
        """One teammate's reply for this round, streamed to the room as it's produced.
        Returns its verdict (True/False/None) for reaction turns."""
        await self.broadcast({"type": "typing", "sender": teammate.name})

        verdict = None
        status = "ok"
        snapshot = self._view(anchor_id)
        try:
            chunks = []
            async with asyncio.timeout(TEAMMATE_TIMEOUT_SECONDS):
                async for chunk in teammate.stream(kind, snapshot):
                    chunks.append(chunk)
                    await self.broadcast({"type": "delta", "sender": teammate.name, "text": chunk})
            reply = "".join(chunks)
            if kind == "react":
                reply, verdict = split_verdict(reply)
        except Exception as exc:
            log.warning("room %s: %s failed to %s", self.room_id, teammate.name, kind, exc_info=exc)
            reply = f"(failed to {kind}: {describe_failure(exc)})"
            status = "failed"

        message = self._record(teammate.name, "teammate", reply, status)
        await self.broadcast({"type": "message", "message": message})
        return verdict


class RoomManager:
    """Rooms are cheap and lazy: teammates (incl. the loaded model) are shared across
    all of them, only per-room state (messages, connections) differs. A room is only
    held in memory while someone's connected or a debate is still running."""

    def __init__(self, teammates, conn):
        self.teammates = teammates
        self.conn = conn
        self._rooms = {}

    def get(self, room_id):
        """The live room, or None if no such room was ever created."""
        if room_id not in self._rooms:
            if not db.room_exists(self.conn, room_id):
                return None
            self._rooms[room_id] = Room(self.teammates, room_id, self.conn, on_idle=self._evict)
        return self._rooms[room_id]

    def _evict(self, room):
        if self._rooms.get(room.room_id) is room:
            del self._rooms[room.room_id]
