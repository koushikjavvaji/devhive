import asyncio

from app import db, rooms
from app.rooms import Room, RoomManager, split_verdict
from app.teammates.base import Teammate


class FixedTeammate(Teammate):
    name = "fixed-bot"
    can_react = False

    async def respond(self, messages):
        return f"round1 reply to: {self.last_human_message(messages)}"


class ReactiveTeammate(Teammate):
    can_react = True

    def __init__(self, name):
        self.name = name

    async def respond(self, messages):
        return f"{self.name}'s round-1 take"

    async def react(self, messages):
        others = [f"{m['sender']}:{m['text']}" for m in messages if m["sender"] != self.name]
        return f"reacting to: {others}"


def make_room(tmp_path, teammates, reaction_rounds=1):
    conn = db.get_connection(tmp_path / "test.db")
    return Room(teammates, "test-room", conn, reaction_rounds=reaction_rounds)


def test_round_2_lets_a_reactor_see_round_1_answers(tmp_path):
    """The actual point of the two-round design: reactive-bot's reaction should
    reference fixed-bot's round-1 answer, not just re-answer the human in isolation."""
    room = make_room(tmp_path, [FixedTeammate(), ReactiveTeammate("reactive-bot")])
    room._record("you", "human", "something broke")

    asyncio.run(room._run_rounds())

    senders = [m["sender"] for m in room.messages]
    assert senders == ["you", "fixed-bot", "reactive-bot", "reactive-bot"]

    reaction = room.messages[-1]["text"]
    assert "round1 reply to: something broke" in reaction
    assert "round-1 take" not in reaction  # excludes its own earlier reply, not just anyone's


def test_round_2_skipped_when_no_teammate_can_react(tmp_path):
    room = make_room(tmp_path, [FixedTeammate(), FixedTeammate()])
    room._record("you", "human", "something broke")

    asyncio.run(room._run_rounds())

    senders = [m["sender"] for m in room.messages]
    assert senders == ["you", "fixed-bot", "fixed-bot"]


def test_round_2_skipped_when_the_reactor_is_the_only_teammate(tmp_path):
    """Nothing for it to react to but itself, so there's no second round."""
    room = make_room(tmp_path, [ReactiveTeammate("reactive-bot")])
    room._record("you", "human", "something broke")

    asyncio.run(room._run_rounds())

    senders = [m["sender"] for m in room.messages]
    assert senders == ["you", "reactive-bot"]


def test_reaction_rounds_chain_instead_of_each_one_only_seeing_round_1(tmp_path):
    """The actual point of reaction_rounds > 1: a later reaction round should be able to
    reference what the other reactor said in an *earlier reaction round*, not just its
    round-1 answer — that's what makes it a back-and-forth instead of two isolated takes."""
    room = make_room(
        tmp_path,
        [ReactiveTeammate("bot-a"), ReactiveTeammate("bot-b")],
        reaction_rounds=2,
    )
    room._record("you", "human", "something broke")

    asyncio.run(room._run_rounds())

    senders = [m["sender"] for m in room.messages]
    assert senders == [
        "you",
        "bot-a", "bot-b",  # round 1
        "bot-a", "bot-b",  # reaction round 1
        "bot-a", "bot-b",  # reaction round 2
    ]

    # bot-a's final (reaction-round-2) message must be able to see bot-b's
    # reaction-round-1 message, not just bot-b's round-1 initial answer
    bot_a_final_reaction = room.messages[5]["text"]
    assert "bot-b:reacting to:" in bot_a_final_reaction


class AgreeingTeammate(ReactiveTeammate):
    async def react(self, messages):
        return "Looks right to me.\nVERDICT: AGREE"


class BrokenTeammate(Teammate):
    name = "broken-bot"

    async def respond(self, messages):
        raise ConnectionError("secret-internal-host:5432 refused")


class SlowTeammate(Teammate):
    name = "slow-bot"

    async def respond(self, messages):
        await asyncio.sleep(10)
        return "too late"


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


def test_split_verdict():
    assert split_verdict("fine.\nVERDICT: AGREE") == ("fine.", True)
    assert split_verdict("nope\n\nverdict: disagree.") == ("nope", False)
    assert split_verdict("no verdict here") == ("no verdict here", None)


def test_debate_stops_early_once_every_reactor_agrees(tmp_path):
    room = make_room(tmp_path, [AgreeingTeammate("bot-a"), AgreeingTeammate("bot-b")], reaction_rounds=3)
    room._record("you", "human", "something broke")

    asyncio.run(room._run_rounds())

    senders = [m["sender"] for m in room.messages]
    assert senders == ["you", "bot-a", "bot-b", "bot-a", "bot-b"]  # one reaction round, not 3
    assert room.messages[-1]["text"] == "Looks right to me."  # verdict line stripped


def test_failed_reply_is_marked_failed_without_leaking_the_raw_error(tmp_path):
    room = make_room(tmp_path, [BrokenTeammate()])
    room._record("you", "human", "something broke")

    asyncio.run(room._run_rounds())

    failure = room.messages[-1]
    assert failure["status"] == "failed"
    assert "secret-internal-host" not in failure["text"]
    assert db.load_messages(room.conn, room.room_id)[-1]["status"] == "failed"


def test_a_hung_teammate_times_out_instead_of_holding_the_round(tmp_path, monkeypatch):
    monkeypatch.setattr(rooms, "TEAMMATE_TIMEOUT_SECONDS", 0.05)
    room = make_room(tmp_path, [SlowTeammate(), FixedTeammate()])
    room._record("you", "human", "something broke")

    asyncio.run(room._run_rounds())

    by_sender = {m["sender"]: m for m in room.messages}
    assert by_sender["slow-bot"]["status"] == "failed"
    assert "timed out" in by_sender["slow-bot"]["text"]
    assert by_sender["fixed-bot"]["status"] == "ok"


def test_replies_stream_as_deltas_then_land_as_a_message(tmp_path):
    room = make_room(tmp_path, [FixedTeammate()])
    socket = FakeSocket()
    room.connections.append(socket)
    room._record("you", "human", "something broke")

    asyncio.run(room._run_rounds())

    assert [p["type"] for p in socket.sent] == ["typing", "delta", "message", "idle"]


class EchoTeammate(Teammate):
    """Slow enough that a second message arrives mid-debate; answers whatever it sees
    as the latest human message, so a debate peeking at a queued message would show."""

    name = "echo-bot"

    async def respond(self, messages):
        await asyncio.sleep(0.05)
        return f"answer to: {self.last_human_message(messages)}"


def test_messages_sent_mid_debate_queue_up_and_each_gets_answered_in_order(tmp_path):
    async def scenario():
        room = make_room(tmp_path, [EchoTeammate()])
        await room.handle_human_message("first", sender="ada")
        await asyncio.sleep(0)  # first debate is now underway
        await room.handle_human_message("second", sender="grace")
        await asyncio.gather(*room._tasks)
        return room

    room = asyncio.run(scenario())

    assert [(m["sender"], m["text"]) for m in room.messages] == [
        ("ada", "first"),
        ("grace", "second"),
        # the first debate never saw "second", even though it was already in the room
        ("echo-bot", "answer to: first"),
        ("echo-bot", "answer to: second"),
    ]
    assert not room.busy


def test_room_is_evicted_once_nobody_is_connected_and_nothing_is_running(tmp_path):
    conn = db.get_connection(tmp_path / "test.db")
    db.create_room(conn, "r1", 0)
    manager = RoomManager([FixedTeammate()], conn)

    room = manager.get("r1")
    socket = FakeSocket()
    asyncio.run(room.connect(socket))
    assert manager.get("r1") is room

    room.disconnect(socket)
    assert "r1" not in manager._rooms


def test_unknown_room_is_not_conjured_into_existence(tmp_path):
    conn = db.get_connection(tmp_path / "test.db")
    manager = RoomManager([FixedTeammate()], conn)

    assert manager.get("made-up-id") is None
    assert not db.room_exists(conn, "made-up-id")
