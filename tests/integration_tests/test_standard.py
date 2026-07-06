"""LangChain standard integration-test suite for `ChatConcentrate`.

This runs the same contract suite that first-party partners (OpenAI, Anthropic)
run, proving `ChatConcentrate` behaves like a first-class LangChain chat model
end-to-end against the live Concentrate.ai gateway. Requires
``CONCENTRATE_API_KEY``.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_concentrate import ChatConcentrate

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONCENTRATE_API_KEY"),
    reason="CONCENTRATE_API_KEY not set; skipping live Concentrate tests.",
)


class TestConcentrateStandard(ChatModelIntegrationTests):
    """Run LangChain's standard integration suite against `ChatConcentrate`."""

    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatConcentrate

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-5.4"}

    @property
    def supports_json_mode(self) -> bool:
        # Concentrate proxies OpenAI's Responses API, which supports JSON mode.
        return True
