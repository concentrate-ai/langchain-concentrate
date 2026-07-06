"""LangChain standard unit-test suite for ChatConcentrate."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_concentrate import ChatConcentrate


class TestConcentrateStandard(ChatModelUnitTests):
    """Runs LangChain's standard unit-test suite against `ChatConcentrate`."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatConcentrate

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-5.4", "api_key": "test-key"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"CONCENTRATE_API_KEY": "api_key"},
            {},
            {"openai_api_key": "api_key"},
        )
