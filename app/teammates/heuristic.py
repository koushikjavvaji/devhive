import re

from app.teammates.base import Teammate

TRACEBACK_LINE_RE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)')
EXCEPTION_RE = re.compile(r"^([\w.]+(?:Error|Exception|Warning)):\s*(.*)$", re.MULTILINE)

COMMON_HINTS = {
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
}


class HeuristicTeammate(Teammate):
    """No model involved — deterministic traceback triage, always available."""

    name = "triage-bot"
    color = "#2b9348"

    async def respond(self, messages):
        text = self.last_human_message(messages)
        if not text.strip():
            return "Paste an error message or traceback and I'll triage it."

        frames = TRACEBACK_LINE_RE.findall(text)
        exc_match = EXCEPTION_RE.search(text)

        lines = []

        if exc_match:
            exc_type, exc_msg = exc_match.group(1), exc_match.group(2)
            lines.append(f"{exc_type}: {exc_msg or '(no message)'}")

            short_type = exc_type.rsplit(".", 1)[-1]
            hint = COMMON_HINTS.get(short_type)
            if hint:
                lines.append(hint)
        else:
            lines.append("No standard Python exception line found — treating this as free-form.")

        if frames:
            last_file, last_line, last_func = frames[-1]
            lines.append(f"Raised at {last_file}:{last_line} in {last_func}() (innermost frame).")
            if len(frames) > 1:
                chain = " -> ".join(f"{f}:{l}" for f, l, _ in frames)
                lines.append(f"Call chain: {chain}")

        return "\n".join(lines)
