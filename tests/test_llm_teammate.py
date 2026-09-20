import asyncio
from types import SimpleNamespace

import pytest

from app.teammates.llm_teammate import LLMTeammate, build_llm_teammates


class FakeCompletions:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeCompletions(response))


def respond(client):
    teammate = LLMTeammate("fake", client, "fake-model")
    messages = [{"sender": "you", "sender_type": "human", "text": "help"}]
    return asyncio.run(teammate.respond(messages))


def test_no_config_means_no_teammates(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEVHIVE_LLM_PROVIDERS", raising=False)
    assert build_llm_teammates() == []


def test_openai_api_key_adds_a_single_default_teammate(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("DEVHIVE_LLM_PROVIDERS", raising=False)

    teammates = build_llm_teammates()

    assert [t.name for t in teammates] == ["gpt"]
    assert teammates[0].model == "gpt-4o-mini"


def test_llm_providers_json_adds_one_teammate_per_entry(monkeypatch):
    """The actual point: several OpenAI-compatible providers (DeepSeek, Groq, ...) can
    all join the same room as real, independent LLM teammates."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEVHIVE_LLM_PROVIDERS", """[
        {"name": "deepseek", "base_url": "https://api.deepseek.com", "api_key": "ds-test", "model": "deepseek-chat"},
        {"name": "groq-llama", "base_url": "https://api.groq.com/openai/v1", "api_key": "gq-test", "model": "llama-3.3-70b-versatile"}
    ]""")

    teammates = build_llm_teammates()

    assert [t.name for t in teammates] == ["deepseek", "groq-llama"]
    assert str(teammates[0].client.base_url) == "https://api.deepseek.com"
    assert teammates[1].model == "llama-3.3-70b-versatile"


def test_openai_key_and_providers_json_combine(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv(
        "DEVHIVE_LLM_PROVIDERS",
        '[{"name": "deepseek", "base_url": "https://api.deepseek.com", "api_key": "ds-test", "model": "deepseek-chat"}]',
    )

    teammates = build_llm_teammates()

    assert [t.name for t in teammates] == ["gpt", "deepseek"]


def test_normal_response_returns_content():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="here's the fix"))],
        model_extra={},
    )
    assert respond(FakeClient(response)) == "here's the fix"


def test_upstream_200_with_error_field_raises_the_real_message():
    """Regression test for a bug hit live: OpenRouter returned HTTP 200 with an "error"
    field instead of an error status when the upstream provider was overloaded. The
    openai client doesn't turn that into an exception — it just leaves choices as None,
    which crashed with a useless 'NoneType is not subscriptable' before this was fixed."""
    response = SimpleNamespace(
        choices=None,
        model_extra={"error": {"message": "Upstream error from Nvidia: Service temporarily overloaded"}},
    )
    with pytest.raises(RuntimeError, match="Service temporarily overloaded"):
        LLMTeammate("fake", FakeClient(response), "fake-model")._complete("system", "help")


def test_empty_content_raises_instead_of_returning_none():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
        model_extra={},
    )
    with pytest.raises(RuntimeError, match="empty response"):
        LLMTeammate("fake", FakeClient(response), "fake-model")._complete("system", "help")


def test_can_react_is_true():
    assert LLMTeammate("fake", FakeClient(None), "fake-model").can_react is True


def test_react_sends_the_whole_room_not_just_the_human_message():
    """The point of react(): it should see every teammate's round-1 answer, not just
    reuse last_human_message like respond() does."""
    captured = {}

    class RecordingCompletions:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="I disagree with triage-bot"))],
                model_extra={},
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=RecordingCompletions()))
    teammate = LLMTeammate("fake", client, "fake-model")

    room_messages = [
        {"sender": "you", "sender_type": "human", "text": "KeyError: 'x'"},
        {"sender": "triage-bot", "sender_type": "teammate", "text": "guard the key"},
        {"sender": "fake", "sender_type": "teammate", "text": "my own earlier take"},
    ]

    reply = asyncio.run(teammate.react(room_messages))

    assert reply == "I disagree with triage-bot"
    transcript = captured["messages"][1]["content"]
    assert "triage-bot: guard the key" in transcript
    assert "my own earlier take" in transcript


def test_react_labels_its_own_earlier_message_as_you_not_its_name():
    """Regression test hit live: nemotron's own round-1 message was labeled with its own
    name in the transcript, so when reacting it referred to itself in third person
    ("Nemotron didn't provide input") instead of understanding that was its own answer."""
    captured = {}

    class RecordingCompletions:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                model_extra={},
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=RecordingCompletions()))
    teammate = LLMTeammate("nemotron", client, "fake-model")

    room_messages = [
        {"sender": "you", "sender_type": "human", "text": "bug"},
        {"sender": "nemotron", "sender_type": "teammate", "text": "my earlier diagnosis"},
        {"sender": "triage-bot", "sender_type": "teammate", "text": "some hint"},
    ]

    asyncio.run(teammate.react(room_messages))

    transcript = captured["messages"][1]["content"]
    assert "You: my earlier diagnosis" in transcript
    assert "nemotron:" not in transcript
    assert "triage-bot: some hint" in transcript


def test_react_skips_a_teammates_failed_round_1_message():
    """A crashed response isn't an opinion — it shouldn't confuse the reactor into
    treating "(failed to respond: ...)" as something to agree or disagree with."""
    captured = {}

    class RecordingCompletions:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                model_extra={},
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=RecordingCompletions()))
    teammate = LLMTeammate("nemotron", client, "fake-model")

    room_messages = [
        {"sender": "you", "sender_type": "human", "text": "bug"},
        {"sender": "nemotron", "sender_type": "teammate", "text": "(failed to respond: upstream overloaded)"},
        {"sender": "triage-bot", "sender_type": "teammate", "text": "some hint"},
    ]

    asyncio.run(teammate.react(room_messages))

    transcript = captured["messages"][1]["content"]
    assert "failed to respond" not in transcript
    assert "triage-bot: some hint" in transcript
