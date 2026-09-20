import asyncio

from app.teammates.heuristic import HeuristicTeammate

TRACEBACK = """Traceback (most recent call last):
  File "app.py", line 42, in handle_request
    result = process(data)
  File "app.py", line 17, in process
    return data["value"] / count
KeyError: 'value'"""


def respond(text):
    teammate = HeuristicTeammate()
    messages = [{"sender": "you", "sender_type": "human", "text": text}]
    return asyncio.run(teammate.respond(messages))


def test_parses_exception_type_and_message():
    assert "KeyError: 'value'" in respond(TRACEBACK)


def test_gives_hint_for_known_exception():
    assert "dict" in respond(TRACEBACK).lower()


def test_extracts_innermost_frame_and_call_chain():
    reply = respond(TRACEBACK)
    assert "app.py:17" in reply
    assert "app.py:42 -> app.py:17" in reply


def test_no_traceback_falls_back_to_free_form():
    assert "free-form" in respond("something is broken but idk what").lower()


def test_empty_message_prompts_for_input():
    assert "paste" in respond("").lower()
