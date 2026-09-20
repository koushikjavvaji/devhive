from abc import ABC, abstractmethod


class Teammate(ABC):
    """An AI participant in a war room. Can be backed by any model: ours, OpenAI's, a local one, or plain rules."""

    name: str
    color: str = "#888888"

    @abstractmethod
    async def respond(self, messages):
        """messages is the room's ordered list of {sender, sender_type, text}; return a reply string."""

    @staticmethod
    def last_human_message(messages):
        for message in reversed(messages):
            if message["sender_type"] == "human":
                return message["text"]
        return ""
