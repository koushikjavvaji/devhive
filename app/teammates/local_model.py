import asyncio
import os
import threading

from app.teammates.base import Teammate
from app.teammates.heuristic import exception_line

CHECKPOINT_PATH = os.environ.get("DEVHIVE_CHECKPOINT_PATH", "checkpoints/ckpt.pt")
TOKENIZER_PATH = os.environ.get("DEVHIVE_TOKENIZER_PATH", "tokenizer/vocab.json")


class LocalModelTeammate(Teammate):
    """Our from-scratch GPT. Trained comment -> code on a small corpus, so treat replies as a
    rough first draft, not a diagnosis — it's a demo of the custom model, not the strongest teammate."""

    name = "from-scratch-gpt"
    color = "#f3722c"

    def __init__(self, generator):
        self.generator = generator
        # one shared model, and generation runs in executor threads — rooms debating at
        # the same time would otherwise run it concurrently, which isn't safe on every
        # torch backend (MPS especially)
        self._lock = threading.Lock()

    async def respond(self, messages):
        prompt = self.prompt_for(self.last_human_message(messages))
        loop = asyncio.get_running_loop()
        code = await loop.run_in_executor(None, self._generate, prompt)
        return f"```\n{code}\n```"

    def _generate(self, prompt):
        with self._lock:
            return self.generator.generate_code(prompt)

    @staticmethod
    def prompt_for(text):
        # trained on short docstring-style comments, not stack traces — the exception
        # line is the closest thing in a traceback to "what this code should be about"
        return exception_line(text) or text


def build_local_model_teammate():
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None

    from inference.generator import LocalGenerator

    generator = LocalGenerator(CHECKPOINT_PATH, TOKENIZER_PATH)
    return LocalModelTeammate(generator)
