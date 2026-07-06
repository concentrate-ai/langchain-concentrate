"""Integration smoke tests against real Concentrate.ai. Needs CONCENTRATE_API_KEY."""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import AIMessage

from langchain_concentrate import ChatConcentrate

pytestmark = pytest.mark.skipif(
    not os.environ.get("CONCENTRATE_API_KEY"),
    reason="CONCENTRATE_API_KEY not set; skipping live Concentrate tests.",
)


def test_invoke_roundtrip() -> None:
    model = ChatConcentrate(model="gpt-5.4")
    result = model.invoke("Reply with just the number 42.")
    assert isinstance(result, AIMessage)
    assert "42" in (result.text or "")
    # Concentrate returns a Responses-shaped id like "resp_..."
    assert (result.response_metadata or {}).get("id") is not None


async def test_ainvoke_roundtrip() -> None:
    model = ChatConcentrate(model="gpt-5.4")
    result = await model.ainvoke("Reply with just the word 'hello'.")
    assert isinstance(result, AIMessage)
    assert "hello" in (result.text or "").lower()


async def test_stream_yields_chunks() -> None:
    model = ChatConcentrate(model="gpt-5.4")
    chunks = []
    async for chunk in model.astream("Count from 1 to 3, one number per line."):
        chunks.append(chunk)
    assert chunks, "expected at least one streamed chunk"
    joined = "".join((c.text or "") for c in chunks)
    assert any(digit in joined for digit in ("1", "2", "3"))


def test_auto_routing() -> None:
    model = ChatConcentrate(
        model="auto",
        routing={"strategy": "min", "metric": "cost"},
    )
    result = model.invoke("What is 2 + 2? Reply with just the digit.")
    assert isinstance(result, AIMessage)
    # Concentrate reports which underlying model handled the request.
    routed_to = (result.response_metadata or {}).get("model")
    assert routed_to, "expected response_metadata.model to record the routed provider"
