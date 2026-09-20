from abc import ABC, abstractmethod


class Teammate(ABC):
    """An AI participant in a war room. Can be backed by any model: ours, OpenAI's, a local one, or plain rules."""

    name: str
    color: str = "#888888"

    # can this teammate look at what everyone else just said and push back on it?
    # deterministic/non-conversational teammates (triage-bot, our tiny code-completion
    # model) can't meaningfully do this, so they opt out and only ever answer the human.
    can_react = False

    @abstractmethod
    async def respond(self, messages):
        """messages is the room's ordered list of {sender, sender_type, text}; return a reply string."""

    async def react(self, messages):
        """Only called when can_react is True. messages includes every teammate's round-1
        answer, so the reply can actually agree/disagree/build on them instead of just
        re-answering the original message in isolation."""
        raise NotImplementedError

    @staticmethod
    def last_human_message(messages):
        for message in reversed(messages):
            if message["sender_type"] == "human":
                return message["text"]
        return ""
