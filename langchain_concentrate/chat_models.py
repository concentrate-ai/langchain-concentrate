"""`ChatConcentrate` — a LangChain chat model backed by Concentrate.ai.

Concentrate.ai is an LLM gateway that exposes an OpenAI Responses-compatible
API at ``https://api.concentrate.ai/v1``. This class is a thin subclass of
``langchain_openai.ChatOpenAI`` that pins the base URL, forces the Responses
API path, and reads its key from ``CONCENTRATE_API_KEY``.

Because the wire protocol is OpenAI Responses, everything a caller expects
from a LangChain chat model — ``invoke``/``ainvoke``/``astream``, tool binding
via ``bind_tools``, structured output via ``with_structured_output``, message
history, etc. — works unchanged.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "https://api.concentrate.ai/v1"
DEFAULT_MODEL = "gpt-5.4"


class ChatConcentrate(ChatOpenAI):
    """Chat model that routes through the Concentrate.ai LLM gateway.

    Setup:
        Install the package and set your Concentrate API key:

        ```bash
        pip install langchain-concentrate
        export CONCENTRATE_API_KEY="sk-cn-v1-..."
        ```

    Key init args — completion params:
        model: Model identifier. Bare name (e.g. ``"gpt-5.4"``), provider-prefixed
            (e.g. ``"anthropic/claude-opus-4-6"``), or ``"auto"`` combined with
            ``routing=`` to let Concentrate pick.
        temperature: Sampling temperature.
        max_output_tokens: Max tokens to generate (Responses API name).

    Key init args — client params:
        api_key: Concentrate API key. Defaults to ``CONCENTRATE_API_KEY`` env var.
        base_url: Override the API base URL. Defaults to
            ``https://api.concentrate.ai/v1``. Can also be set via
            ``CONCENTRATE_BASE_URL``.
        routing: Optional auto-routing directive, e.g.
            ``{"strategy": "min", "metric": "cost"}``. Only meaningful when
            ``model="auto"``. Passed through to Concentrate via ``extra_body``.

    Example:
        Basic usage:

        ```python
        from langchain_concentrate import ChatConcentrate

        model = ChatConcentrate(model="gpt-5.4")
        result = model.invoke("Say hi in three words.")
        print(result.text)
        ```

        Auto-routing on cost:

        ```python
        model = ChatConcentrate(
            model="auto",
            routing={"strategy": "min", "metric": "cost"},
        )
        ```
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        routing: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model. Overrides base-URL and forces Responses API."""
        resolved_key = api_key or os.environ.get("CONCENTRATE_API_KEY")
        resolved_base_url = (
            base_url
            or os.environ.get("CONCENTRATE_BASE_URL")
            or DEFAULT_BASE_URL
        )

        if routing is not None:
            extra_body = dict(kwargs.pop("extra_body", None) or {})
            extra_body.setdefault("routing", routing)
            kwargs["extra_body"] = extra_body

        # Concentrate's primary endpoint is /v1/responses, so we force the
        # Responses API path unless the caller has explicitly overridden it.
        kwargs.setdefault("use_responses_api", True)

        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=resolved_base_url,
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        """LangChain callback identifier for this chat model."""
        return "concentrate-chat"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Namespace under which this class is serialized."""
        return ["langchain_concentrate", "chat_models"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Third-party class; not on the langchain-core deserialization allowlist."""
        return False
