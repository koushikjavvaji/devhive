import asyncio
import os

from app.teammates.base import Teammate

SYSTEM_PROMPT = (
    "You are a teammate in a collaborative debugging war room. Another participant just "
    "pasted an error or stack trace. Give a short, concrete diagnosis and a likely fix. "
    "Keep it under 120 words."
)


class OpenAITeammate(Teammate):
    name = "gpt"
    color = "#577590"

    def __init__(self, client, model):
        self.client = client
        self.model = model

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
        return completion.choices[0].message.content


def build_openai_teammate():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return OpenAITeammate(OpenAI(api_key=api_key), model)
