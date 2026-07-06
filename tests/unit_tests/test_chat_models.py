"""Unit tests for ChatConcentrate — defaults, env vars, and pass-throughs."""

from __future__ import annotations

import pytest

from langchain_concentrate import ChatConcentrate
from langchain_concentrate.chat_models import DEFAULT_BASE_URL, DEFAULT_MODEL


def test_default_base_url_and_responses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONCENTRATE_API_KEY", "test-key")
    model = ChatConcentrate()
    assert model.openai_api_base == DEFAULT_BASE_URL
    assert model.model_name == DEFAULT_MODEL
    assert model.use_responses_api is True
    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() == "test-key"


def test_explicit_api_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONCENTRATE_API_KEY", "env-key")
    model = ChatConcentrate(api_key="explicit-key")
    assert model.openai_api_key.get_secret_value() == "explicit-key"


def test_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONCENTRATE_API_KEY", "k")
    monkeypatch.setenv("CONCENTRATE_BASE_URL", "https://staging.concentrate.ai/v1")
    model = ChatConcentrate()
    assert model.openai_api_base == "https://staging.concentrate.ai/v1"


def test_explicit_base_url_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONCENTRATE_API_KEY", "k")
    monkeypatch.setenv("CONCENTRATE_BASE_URL", "https://staging.concentrate.ai/v1")
    model = ChatConcentrate(base_url="https://custom.example.com/v1")
    assert model.openai_api_base == "https://custom.example.com/v1"


def test_routing_dict_flows_to_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONCENTRATE_API_KEY", "k")
    model = ChatConcentrate(
        model="auto",
        routing={"strategy": "min", "metric": "cost"},
    )
    assert model.extra_body == {"routing": {"strategy": "min", "metric": "cost"}}


def test_routing_merges_with_existing_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONCENTRATE_API_KEY", "k")
    model = ChatConcentrate(
        model="auto",
        routing={"strategy": "min", "metric": "cost"},
        extra_body={"foo": "bar"},
    )
    assert model.extra_body == {
        "foo": "bar",
        "routing": {"strategy": "min", "metric": "cost"},
    }


def test_use_responses_api_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONCENTRATE_API_KEY", "k")
    model = ChatConcentrate(use_responses_api=False)
    assert model.use_responses_api is False


def test_llm_type_is_concentrate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONCENTRATE_API_KEY", "k")
    assert ChatConcentrate()._llm_type == "concentrate-chat"


def test_lc_namespace() -> None:
    assert ChatConcentrate.get_lc_namespace() == [
        "langchain_concentrate",
        "chat_models",
    ]
