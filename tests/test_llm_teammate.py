from app.teammates.llm_teammate import build_llm_teammates


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
