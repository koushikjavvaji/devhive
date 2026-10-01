import asyncio

from app.teammates.heuristic import HeuristicTeammate, exception_line

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


JS_TRACE = """TypeError: Cannot read properties of undefined (reading 'id')
    at getUser (/srv/app/users.js:14:22)
    at async handler (/srv/app/routes.js:8:5)"""

JAVA_TRACE = """Exception in thread "main" java.lang.RuntimeException: request failed
\tat com.acme.Api.call(Api.java:30)
\tat com.acme.Main.main(Main.java:12)
Caused by: java.lang.NullPointerException: Cannot invoke "String.length()" because "name" is null
\tat com.acme.Names.check(Names.java:7)
\t... 2 more"""

GO_TRACE = """panic: runtime error: invalid memory address or nil pointer dereference
[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x4553a6]

goroutine 1 [running]:
main.load(...)
\t/home/me/proj/main.go:21
main.main()
\t/home/me/proj/main.go:9 +0x1d
exit status 2"""

RUST_TRACE = """thread 'main' panicked at src/main.rs:4:37:
called `Option::unwrap()` on a `None` value
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace"""


def test_js_error_gets_a_message_specific_hint_and_frames_in_order():
    reply = respond(JS_TRACE)
    assert "TypeError: Cannot read properties of undefined" in reply
    assert "optional chaining" in reply
    assert "Raised at /srv/app/users.js:14 in getUser()" in reply
    assert "/srv/app/routes.js:8 -> /srv/app/users.js:14" in reply


def test_java_reports_the_root_cause_not_the_wrapper():
    reply = respond(JAVA_TRACE)
    assert reply.startswith("java.lang.NullPointerException: Cannot invoke")
    assert "null where an object was expected" in reply
    assert "Main.java:12 -> Api.java:30" in reply


def test_go_panic_skips_runtime_frames():
    reply = respond(GO_TRACE)
    assert reply.startswith("panic: runtime error: invalid memory address or nil pointer dereference")
    assert "nil pointer" in reply
    assert "Raised at /home/me/proj/main.go:21 in main.load()" in reply


def test_rust_panic():
    reply = respond(RUST_TRACE)
    assert "panic: called `Option::unwrap()` on a `None` value" in reply
    assert "unwrap() on None" in reply
    assert "Panicked at src/main.rs:4:37" in reply


def test_exception_line():
    assert exception_line(TRACEBACK) == "KeyError: 'value'"
    assert exception_line("no error here") is None
