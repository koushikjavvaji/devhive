import asyncio

from app import db
from app.rooms import Room
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
