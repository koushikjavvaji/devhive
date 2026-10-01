import json
import os

from app.teammates.base import Teammate

SYSTEM_PROMPT = (
    "You are a teammate in a collaborative debugging war room. Another participant just "
    "pasted an error or stack trace. Give a short, concrete diagnosis and a likely fix. "
    "If earlier room context is included, use it — the latest message may be a follow-up. "
    "Keep it under 120 words."
)

REACTION_SYSTEM_PROMPT = (
    "You are a teammate in a collaborative debugging war room. Below is the room's "
    "transcript. Your own earlier message is labeled 'You:'; everyone else is labeled by "
    "their name (which may include a rule-based bot and/or other AI models). Don't just "
    "restate your own answer — react to the others: say what you agree with, call out "
    "anything you think is wrong, incomplete, or made up, and add anything they missed. "
    "If everyone already nailed it, say so briefly instead of padding. Refer to yourself "
    "as 'I', not by name. Keep it under 100 words. End with one final line that is exactly "
    "'VERDICT: AGREE' if you fully agree with where the room has landed and have nothing "
    "left to add, or 'VERDICT: DISAGREE' otherwise."
)

# a hung provider shouldn't hold a round hostage — the openai client's default is 10 min
LLM_TIMEOUT_SECONDS = float(os.environ.get("DEVHIVE_LLM_TIMEOUT", "45"))
RESPOND_MAX_TOKENS = 400
REACT_MAX_TOKENS = 300

# rough cap on how much room history goes into one call (~3k tokens). The current
# exchange (latest human message onwards) always goes in; older messages fill whatever
# budget is left, newest first, so long rooms don't grow the prompt without bound.
MAX_CONTEXT_CHARS = 12000

REQUIRED_PROVIDER_KEYS = ("name", "api_key", "model")


class LLMTeammate(Teammate):
    """Any OpenAI-compatible chat completions API — OpenAI itself, or an OpenAI-compatible
    provider (DeepSeek, Groq, OpenRouter, Together, ...) via a different base_url. This is
    how the room gets more than one real LLM in it at once."""

    color = "#577590"
    can_react = True

    def __init__(self, name, client, model, color=None):
        self.name = name
        self.client = client
        self.model = model
        if color:
            self.color = color

    async def respond(self, messages):
        return "".join([chunk async for chunk in self.stream("respond", messages)])

    async def react(self, messages):
        return "".join([chunk async for chunk in self.stream("react", messages)])

    async def stream(self, kind, messages):
        earlier, current = self._context(messages)

        if kind == "respond":
            latest = self.last_human_message(messages)
            if earlier:
                user_content = (
                    f"Earlier in this room:\n\n{self._transcript(earlier)}\n\n---\n\n"
                    f"Latest message:\n{latest}"
                )
            else:
                user_content = latest
            system_prompt, max_tokens = SYSTEM_PROMPT, RESPOND_MAX_TOKENS
        else:
            user_content = self._transcript(earlier + current)
            system_prompt, max_tokens = REACTION_SYSTEM_PROMPT, REACT_MAX_TOKENS

        async for chunk in self._stream(system_prompt, user_content, max_tokens):
            yield chunk

    def _context(self, messages):
        # a crash isn't an opinion to weigh in on
        usable = [m for m in messages if m.get("status", "ok") == "ok"]

        start = 0
        for i in range(len(usable) - 1, -1, -1):
            if usable[i]["sender_type"] == "human":
                start = i
                break
        current = usable[start:]

        budget = MAX_CONTEXT_CHARS - sum(len(m["text"]) for m in current)
        earlier = []
        for message in reversed(usable[:start]):
            budget -= len(message["text"])
            if budget < 0:
                break
            earlier.insert(0, message)

        return earlier, current

    def _transcript(self, messages):
        return "\n\n".join(f"{self._label(m)}: {m['text']}" for m in messages)

    def _label(self, message):
        if message["sender_type"] == "teammate" and message["sender"] == self.name:
            return "You"
        return message["sender"]

    async def _stream(self, system_prompt, user_content, max_tokens):
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            stream=True,
        )

        produced = False
        async for chunk in stream:
            if not chunk.choices:
                # some OpenAI-compatible providers (OpenRouter, at least) send an "error"
                # field with HTTP 200 instead of an actual error status, so the openai
                # client doesn't raise — the chunk just arrives with no choices
                error = (getattr(chunk, "model_extra", None) or {}).get("error")
                if error:
                    raise RuntimeError(error.get("message") or "upstream returned an error")
                continue

            content = chunk.choices[0].delta.content
            if content:
                produced = True
                yield content

        if not produced:
            raise RuntimeError("upstream returned an empty response")


def _client(api_key, base_url=None):
    from openai import AsyncOpenAI
    kwargs = {"api_key": api_key, "timeout": LLM_TIMEOUT_SECONDS, "max_retries": 1}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def parse_providers(raw):
    """Validate DEVHIVE_LLM_PROVIDERS up front, so a typo in a deploy's env var fails
    with a message that says what's wrong instead of a bare KeyError/JSONDecodeError."""
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DEVHIVE_LLM_PROVIDERS is not valid JSON: {exc}") from None

    if not isinstance(entries, list):
        raise ValueError("DEVHIVE_LLM_PROVIDERS must be a JSON list of provider objects")

    names = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"DEVHIVE_LLM_PROVIDERS[{i}] must be an object")
        missing = [k for k in REQUIRED_PROVIDER_KEYS if not entry.get(k)]
        if missing:
            raise ValueError(f"DEVHIVE_LLM_PROVIDERS[{i}] is missing {', '.join(missing)}")
        if entry["name"] in names:
            raise ValueError(f"DEVHIVE_LLM_PROVIDERS has two teammates named {entry['name']!r}")
        names.add(entry["name"])

    return entries


def build_llm_teammates():
    """OPENAI_API_KEY is a shortcut for a single default OpenAI teammate.
    DEVHIVE_LLM_PROVIDERS (a JSON list) adds any number of OpenAI-compatible providers on
    top of that — that's the knob for "more than one real LLM in the room"."""
    try:
        import openai  # noqa: F401
    except ImportError:
        return []

    teammates = []

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        name = os.environ.get("OPENAI_TEAMMATE_NAME", "gpt")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        teammates.append(LLMTeammate(name, _client(api_key), model))

    providers_json = os.environ.get("DEVHIVE_LLM_PROVIDERS")
    if providers_json:
        for entry in parse_providers(providers_json):
            teammates.append(LLMTeammate(
                name=entry["name"],
                client=_client(entry["api_key"], entry.get("base_url")),
                model=entry["model"],
                color=entry.get("color"),
            ))

    return teammates
