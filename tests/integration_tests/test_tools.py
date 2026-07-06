"""Live tests for tool calling and structured output. Needs CONCENTRATE_API_KEY.

The README advertises that ``bind_tools`` and ``with_structured_output`` work
out of the box because ``ChatConcentrate`` inherits from ``ChatOpenAI``. These
tests exercise that claim against the real Concentrate.ai gateway.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_concentrate import ChatConcentrate

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONCENTRATE_API_KEY"),
    reason="CONCENTRATE_API_KEY not set; skipping live Concentrate tests.",
)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city to look up.
    """
    return f"It is 21C and sunny in {city}."


class MathAnswer(BaseModel):
    """A numeric answer to an arithmetic question."""

    number: int = Field(description="The numeric result.")


def test_bind_tools_emits_tool_call() -> None:
    model = ChatConcentrate(model="gpt-5.4").bind_tools([get_weather])
    result = model.invoke("What's the weather in Paris? Use the get_weather tool.")
    assert isinstance(result, AIMessage)
    assert result.tool_calls, "expected the model to request a tool call"
    call = result.tool_calls[0]
    assert call["name"] == "get_weather"
    assert call["args"].get("city", "").lower() == "paris"
    assert call["id"]


def test_tool_call_round_trip() -> None:
    """A full turn: model asks for the tool, we answer, model uses the result."""
    model = ChatConcentrate(model="gpt-5.4").bind_tools([get_weather])
    first = model.invoke(
        [HumanMessage("What's the weather in Paris? Use the get_weather tool.")]
    )
    assert isinstance(first, AIMessage)
    assert first.tool_calls

    call = first.tool_calls[0]
    tool_result = get_weather.invoke(call)
    assert isinstance(tool_result, ToolMessage)

    final = model.invoke(
        [
            HumanMessage("What's the weather in Paris? Use the get_weather tool."),
            first,
            tool_result,
        ]
    )
    assert isinstance(final, AIMessage)
    assert not final.tool_calls, "model should answer, not call the tool again"
    assert "paris" in (final.text or "").lower()


async def test_bind_tools_async() -> None:
    model = ChatConcentrate(model="gpt-5.4").bind_tools([get_weather])
    result = await model.ainvoke(
        "What's the weather in Tokyo? Use the get_weather tool."
    )
    assert isinstance(result, AIMessage)
    assert result.tool_calls
    assert result.tool_calls[0]["name"] == "get_weather"


def test_with_structured_output_pydantic() -> None:
    model = ChatConcentrate(model="gpt-5.4").with_structured_output(MathAnswer)
    result = model.invoke("What is 2 + 2?")
    assert isinstance(result, MathAnswer)
    assert result.number == 4


async def test_with_structured_output_async() -> None:
    model = ChatConcentrate(model="gpt-5.4").with_structured_output(MathAnswer)
    result = await model.ainvoke("What is 10 minus 3?")
    assert isinstance(result, MathAnswer)
    assert result.number == 7


def test_structured_output_with_auto_routing() -> None:
    """Structured output should survive Concentrate's auto-routing path."""
    model = ChatConcentrate(
        model="auto",
        routing={"strategy": "min", "metric": "cost"},
    ).with_structured_output(MathAnswer)
    result = model.invoke("What is 5 + 6?")
    assert isinstance(result, MathAnswer)
    assert result.number == 11
