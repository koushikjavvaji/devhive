import asyncio
from types import SimpleNamespace

import pytest

from app.teammates import llm_teammate
from app.teammates.llm_teammate import LLMTeammate, build_llm_teammates, parse_providers


def chunk(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))],
        model_extra={},
    )


def error_chunk(message):
    return SimpleNamespace(choices=[], model_extra={"error": {"message": message}})


class FakeCompletions:
    """Stands in for AsyncOpenAI's chat.completions with stream=True: records each call's
    kwargs and returns an async iterator over the given chunks."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        async def stream():
            for c in self.chunks:
                yield c

        return stream()


class FakeClient:
    def __init__(self, chunks=()):
        self.completions = FakeCompletions(list(chunks))
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def last_user_content(self):
        return self.completions.calls[-1]["messages"][1]["content"]


def respond(client, messages=None):
    teammate = LLMTeammate("fake", client, "fake-model")
    messages = messages or [{"sender": "you", "sender_type": "human", "text": "help"}]
    return asyncio.run(teammate.respond(messages))


def react(client, messages, name="fake"):
    return asyncio.run(LLMTeammate(name, client, "fake-model").react(messages))


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


def test_clients_get_a_timeout_instead_of_the_10_minute_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("DEVHIVE_LLM_PROVIDERS", raising=False)

    client = build_llm_teammates()[0].client

    assert client.timeout == llm_teammate.LLM_TIMEOUT_SECONDS
    assert client.max_retries == 1


@pytest.mark.parametrize("raw, error", [
    ("{not json", "not valid JSON"),
    ('{"name": "x"}', "must be a JSON list"),
    ('["x"]', r"\[0\] must be an object"),
    ('[{"name": "x", "model": "m"}]', r"\[0\] is missing api_key"),
    ('[{"name": "x", "api_key": "k", "model": "m"}, {"name": "x", "api_key": "k", "model": "m"}]',
     "two teammates named 'x'"),
])
def test_bad_providers_config_fails_with_a_clear_message(raw, error):
    with pytest.raises(ValueError, match=error):
        parse_providers(raw)


def test_streamed_chunks_join_into_the_reply():
    client = FakeClient([chunk("here's "), chunk(None), chunk("the fix")])
    assert respond(client) == "here's the fix"


def test_calls_cap_the_reply_length():
    client = FakeClient([chunk("ok")])
    respond(client)
    assert client.completions.calls[0]["max_tokens"] == llm_teammate.RESPOND_MAX_TOKENS
    assert client.completions.calls[0]["stream"] is True


def test_upstream_200_with_error_field_raises_the_real_message():
    """Regression test for a bug hit live: OpenRouter returned HTTP 200 with an "error"
    field instead of an error status when the upstream provider was overloaded. The
    openai client doesn't turn that into an exception — the response just has no
    choices, which crashed with a useless 'NoneType is not subscriptable' before."""
    client = FakeClient([error_chunk("Upstream error from Nvidia: Service temporarily overloaded")])
    with pytest.raises(RuntimeError, match="Service temporarily overloaded"):
        respond(client)


def test_empty_content_raises_instead_of_returning_nothing():
    with pytest.raises(RuntimeError, match="empty response"):
        respond(FakeClient([chunk(None)]))


def test_can_react_is_true():
    assert LLMTeammate("fake", FakeClient(), "fake-model").can_react is True


def test_respond_includes_earlier_room_context_for_follow_ups():
    """A follow-up like "that didn't work" is meaningless without what came before."""
    client = FakeClient([chunk("ok")])
    respond(client, [
        {"sender": "you", "sender_type": "human", "text": "KeyError: 'x'"},
        {"sender": "fake", "sender_type": "teammate", "text": "use .get()"},
        {"sender": "you", "sender_type": "human", "text": "that didn't work"},
    ])

    content = client.last_user_content
    assert "KeyError: 'x'" in content
    assert "You: use .get()" in content
    assert content.endswith("Latest message:\nthat didn't work")


def test_respond_with_no_history_sends_just_the_message():
    client = FakeClient([chunk("ok")])
    respond(client)
    assert client.last_user_content == "help"


def test_context_drops_oldest_messages_past_the_budget(monkeypatch):
    monkeypatch.setattr(llm_teammate, "MAX_CONTEXT_CHARS", 100)
    client = FakeClient([chunk("ok")])
    react(client, [
        {"sender": "you", "sender_type": "human", "text": "ancient " + "x" * 80},
        {"sender": "you", "sender_type": "human", "text": "recent question"},
        {"sender": "bot", "sender_type": "teammate", "text": "recent answer"},
    ])

    content = client.last_user_content
    assert "ancient" not in content
    assert "recent question" in content
    assert "recent answer" in content


def test_react_sends_the_whole_room_not_just_the_human_message():
    """The point of react(): it should see every teammate's round-1 answer, not just
    reuse last_human_message like respond() does."""
    client = FakeClient([chunk("I disagree with triage-bot")])

    reply = react(client, [
        {"sender": "you", "sender_type": "human", "text": "KeyError: 'x'"},
        {"sender": "triage-bot", "sender_type": "teammate", "text": "guard the key"},
        {"sender": "fake", "sender_type": "teammate", "text": "my own earlier take"},
    ])

    assert reply == "I disagree with triage-bot"
    transcript = client.last_user_content
    assert "triage-bot: guard the key" in transcript
    assert "my own earlier take" in transcript


def test_react_labels_its_own_earlier_message_as_you_not_its_name():
    """Regression test hit live: nemotron's own round-1 message was labeled with its own
    name in the transcript, so when reacting it referred to itself in third person
    ("Nemotron didn't provide input") instead of understanding that was its own answer."""
    client = FakeClient([chunk("ok")])
    react(client, [
        {"sender": "you", "sender_type": "human", "text": "bug"},
        {"sender": "nemotron", "sender_type": "teammate", "text": "my earlier diagnosis"},
        {"sender": "triage-bot", "sender_type": "teammate", "text": "some hint"},
    ], name="nemotron")

    transcript = client.last_user_content
    assert "You: my earlier diagnosis" in transcript
    assert "nemotron:" not in transcript
    assert "triage-bot: some hint" in transcript


def test_a_human_who_shares_the_teammates_name_is_not_labeled_you():
    client = FakeClient([chunk("ok")])
    react(client, [
        {"sender": "nemotron", "sender_type": "human", "text": "bug"},
        {"sender": "triage-bot", "sender_type": "teammate", "text": "some hint"},
    ], name="nemotron")

    assert "nemotron: bug" in client.last_user_content


def test_react_skips_a_teammates_failed_message():
    """A crashed response isn't an opinion — it shouldn't confuse the reactor into
    treating "(failed to respond: ...)" as something to agree or disagree with."""
    client = FakeClient([chunk("ok")])
    react(client, [
        {"sender": "you", "sender_type": "human", "text": "bug"},
        {"sender": "nemotron", "sender_type": "teammate", "text": "(failed to respond: timed out)",
         "status": "failed"},
        {"sender": "triage-bot", "sender_type": "teammate", "text": "some hint"},
    ], name="nemotron")

    transcript = client.last_user_content
    assert "failed to respond" not in transcript
    assert "triage-bot: some hint" in transcript
