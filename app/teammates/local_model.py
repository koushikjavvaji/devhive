import asyncio
import os

from app.teammates.base import Teammate

CHECKPOINT_PATH = os.environ.get("DEVHIVE_CHECKPOINT_PATH", "checkpoints/ckpt.pt")
TOKENIZER_PATH = os.environ.get("DEVHIVE_TOKENIZER_PATH", "tokenizer/vocab.json")


class LocalModelTeammate(Teammate):
    """Our from-scratch GPT. Trained comment -> code on a small corpus, so treat replies as a
    rough first draft, not a diagnosis — it's a demo of the custom model, not the strongest teammate."""

    name = "from-scratch-gpt"
    color = "#f3722c"

    def __init__(self, generator):
        self.generator = generator

    async def respond(self, messages):
        prompt = self.last_human_message(messages)
        loop = asyncio.get_running_loop()
        code = await loop.run_in_executor(None, self.generator.generate_code, prompt)
        return f"```\n{code}\n```"


def build_local_model_teammate():
    if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None

    from inference.generator import LocalGenerator

    generator = LocalGenerator(CHECKPOINT_PATH, TOKENIZER_PATH)
    return LocalModelTeammate(generator)
