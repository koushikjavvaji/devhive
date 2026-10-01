import re

from app.teammates.base import Teammate

# stack frames, per language: (file, line, func). Python lists frames outermost-first;
# JS, Java and Go list them innermost-first, so those get reversed in parse_frames.
PY_FRAME_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)')
JS_FRAME_RE = re.compile(
    r"^\s*at (?:(?:async )?(?P<func>[^\s(]+) \()?(?P<file>[^\s()]+?):(?P<line>\d+):\d+\)?\s*$", re.MULTILINE
)
JAVA_FRAME_RE = re.compile(r"^\s*at (?P<func>[\w$.<>]+)\((?P<file>[\w$.-]+):(?P<line>\d+)\)", re.MULTILINE)
GO_FRAME_RE = re.compile(r"^(?P<func>[\w./*()-]+)\(.*\)\n\s+(?P<file>\S+\.go):(?P<line>\d+)", re.MULTILINE)

# "KeyError: msg" (Python), "Uncaught TypeError: msg" (JS), "Exception in thread "main"
# java.lang.NullPointerException" / "Caused by: x.y.FooException: msg" (Java)
EXCEPTION_RE = re.compile(
    r'^(?P<prefix>Uncaught |Exception in thread "[^"]*" |Caused by: )?'
    r"(?P<type>[\w.$]+(?:Error|Exception|Warning))(?::\s*(?P<msg>.*))?$",
    re.MULTILINE,
)
GO_PANIC_RE = re.compile(r"^(?P<type>panic|fatal error): (?P<msg>.+)$", re.MULTILINE)
RUST_PANIC_RE = re.compile(
    r"^thread '[^']*' panicked at (?:'(?P<old_msg>.*)', )?(?P<loc>[^\s:]+:\d+:\d+):?\s*(?:\n(?P<msg>.+))?$",
    re.MULTILINE,
)

COMMON_HINTS = {
    # Python
    "ModuleNotFoundError": "Looks like a missing dependency — check it's installed and the venv/interpreter is right.",
    "ImportError": "An import is failing — check for circular imports or a package that isn't installed.",
    "KeyError": "You're indexing a dict with a key that isn't there — print the dict or use .get() with a default.",
    "IndexError": "List/array index out of range — check the length before indexing, or an off-by-one loop bound.",
    "AttributeError": "Calling something on an object that doesn't have it — check the object's actual type (maybe it's None).",
    "TypeError": "Wrong type passed somewhere — check what's actually being passed vs. what's expected.",
    "ValueError": "A value is out of the expected range or in the wrong shape/format.",
    "NameError": "Using a variable/name before it's defined, or a typo.",
    "ZeroDivisionError": "Dividing by zero somewhere — guard the denominator.",
    "ConnectionError": "A network call failed — check the endpoint, network, and retries/timeouts.",
    "TimeoutError": "Something took too long — check for blocking calls or increase the timeout.",
    "RecursionError": "Infinite or too-deep recursion — check the base case.",
    "AssertionError": "An assertion failed — check the condition against the actual state at that point.",
    # JS
    "ReferenceError": "Using a variable that isn't defined in this scope — a typo, a missing import, or used before declaration.",
    "SyntaxError": "The code (or JSON being parsed) isn't valid syntax — check the reported line and the one before it.",
    "RangeError": "A value is outside the allowed range — often runaway recursion or an invalid array length.",
    # Java
    "NullPointerException": "Something is null where an object was expected — check what's dereferenced on the innermost frame's line.",
    "ArrayIndexOutOfBoundsException": "Index past the end of an array — check the length before indexing, or an off-by-one loop bound.",
    "IndexOutOfBoundsException": "Index past the end of a list — check its size before indexing.",
    "ClassCastException": "Casting an object to a type it isn't — check the actual runtime type before casting.",
    "ClassNotFoundException": "A class isn't on the classpath — check the dependency is declared and packaged.",
    "NumberFormatException": "Parsing a string that isn't a valid number — validate or trim the input first.",
    "ConcurrentModificationException": "A collection was modified while being iterated — use an Iterator's remove() or iterate over a copy.",
    "IllegalArgumentException": "A method got an argument it doesn't accept — check the value against its contract.",
    "IllegalStateException": "Called at the wrong time — the object isn't in the state this operation needs.",
    "OutOfMemoryError": "The heap ran out — look for something growing without bound, or raise -Xmx if it's legitimate.",
    "StackOverflowError": "Infinite or too-deep recursion — check the base case.",
}

# checked before COMMON_HINTS: the same exception type can mean very different things
# (a JS TypeError is usually undefined access; a Go panic says what it is in the message)
MESSAGE_HINTS = [
    ("cannot read propert", "Reading a property of undefined/null — the object on the left of the '.' isn't set yet; check where it comes from or use optional chaining (?.)."),
    ("is not a function", "Calling something that isn't a function — check the import/export, a typo in the name, or that the value is what you think it is."),
    ("nil pointer dereference", "Dereferencing a nil pointer — check the error return before using the value, or that the struct/map was initialized."),
    ("index out of range", "Index past the end of a slice/array — check len() before indexing, or an off-by-one loop bound."),
    ("assignment to entry in nil map", "Writing to a nil map — initialize it with make() first."),
    ("all goroutines are asleep", "Deadlock — every goroutine is blocked, usually on a channel nobody sends to or receives from."),
    ("called `option::unwrap()` on a `none` value", "unwrap() on None — handle the None case with match/if let, or use ? / expect() with context."),
    ("called `result::unwrap()` on an `err` value", "unwrap() on an Err — handle the error with match or ?, or use expect() to say what failed."),
]


def find_exception(text):
    """The error's (type, message), or None. For Java, the root cause (the
    last "Caused by:") is what matters, not the wrapper exception thrown at the top."""
    match = RUST_PANIC_RE.search(text)
    if match:
        return "panic", (match.group("old_msg") or match.group("msg") or "").strip()

    match = GO_PANIC_RE.search(text)
    if match:
        return match.group("type"), match.group("msg").strip()

    matches = list(EXCEPTION_RE.finditer(text))
    if not matches:
        return None
    caused_by = [m for m in matches if m.group("prefix") == "Caused by: "]
    match = caused_by[-1] if caused_by else matches[0]
    return match.group("type"), (match.group("msg") or "").strip()


def exception_line(text):
    """Just "Type: message" — the one line that actually describes the error."""
    found = find_exception(text)
    if not found:
        return None
    exc_type, exc_msg = found
    return f"{exc_type}: {exc_msg}" if exc_msg else exc_type


def parse_frames(text):
    """Stack frames as (file, line, func), outermost first."""
    frames = PY_FRAME_RE.findall(text)
    if frames:
        return frames

    for pattern in (JAVA_FRAME_RE, JS_FRAME_RE, GO_FRAME_RE):
        found = [(m.group("file"), m.group("line"), m.group("func") or "<anonymous>") for m in pattern.finditer(text)]
        if found:
            if pattern is GO_FRAME_RE:
                # the first frames are the Go runtime raising the panic, not your code
                found = [f for f in found if not f[2].startswith("runtime.")] or found
            return list(reversed(found))
    return []


def hint_for(exc_type, exc_msg):
    lowered = exc_msg.lower()
    for needle, hint in MESSAGE_HINTS:
        if needle in lowered:
            return hint
    return COMMON_HINTS.get(exc_type.rsplit(".", 1)[-1])


class HeuristicTeammate(Teammate):
    """No model involved — deterministic error triage (Python, JS/Node, Java, Go, Rust), always available."""

    name = "triage-bot"
    color = "#2b9348"

    async def respond(self, messages):
        text = self.last_human_message(messages)
        if not text.strip():
            return "Paste an error message or traceback and I'll triage it."

        found = find_exception(text)
        frames = parse_frames(text)

        lines = []

        if found:
            exc_type, exc_msg = found
            lines.append(f"{exc_type}: {exc_msg or '(no message)'}")
            hint = hint_for(exc_type, exc_msg)
            if hint:
                lines.append(hint)
        else:
            lines.append("No recognizable exception/panic line found — treating this as free-form.")

        rust = RUST_PANIC_RE.search(text)
        if rust and not frames:
            lines.append(f"Panicked at {rust.group('loc')}.")

        if frames:
            last_file, last_line, last_func = frames[-1]
            lines.append(f"Raised at {last_file}:{last_line} in {last_func}() (innermost frame).")
            if len(frames) > 1:
                chain = " -> ".join(f"{file}:{line}" for file, line, _ in frames)
                lines.append(f"Call chain: {chain}")

        return "\n".join(lines)
