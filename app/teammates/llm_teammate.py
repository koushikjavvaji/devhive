import asyncio
import json
import os

from app.teammates.base import Teammate

SYSTEM_PROMPT = (
    "You are a teammate in a collaborative debugging war room. Another participant just "
    "pasted an error or stack trace. Give a short, concrete diagnosis and a likely fix. "
    "Keep it under 120 words."
)


class LLMTeammate(Teammate):
    """Any OpenAI-compatible chat completions API — OpenAI itself, or an OpenAI-compatible
    provider (DeepSeek, Groq, OpenRouter, Together, ...) via a different base_url. This is
    how the room gets more than one real LLM in it at once."""

    color = "#577590"

    def __init__(self, name, client, model, color=None):
        self.name = name
        self.client = client
        self.model = model
        if color:
            self.color = color

    async def respond(self, messages):
        prompt = self.last_human_message(messages)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._complete, prompt)

    def _complete(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        if not completion.choices:
            # some OpenAI-compatible providers (OpenRouter, at least) return HTTP 200
            # with an "error" field instead of an actual error status, so the openai
            # client doesn't raise — it just leaves choices as None
            error = (completion.model_extra or {}).get("error", {})
            raise RuntimeError(error.get("message") or "upstream returned no choices")

        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("upstream returned an empty response")

        return content


def _client(api_key, base_url=None):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


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
        for entry in json.loads(providers_json):
            teammates.append(LLMTeammate(
                name=entry["name"],
                client=_client(entry["api_key"], entry.get("base_url")),
                model=entry["model"],
                color=entry.get("color"),
            ))

    return teammates
