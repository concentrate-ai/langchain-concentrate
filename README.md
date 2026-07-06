# langchain-concentrate

[![PyPI version](https://img.shields.io/pypi/v/langchain-concentrate.svg)](https://pypi.org/project/langchain-concentrate/)

LangChain integration for [Concentrate.ai](https://concentrate.ai) — a unified
LLM gateway that lets you access OpenAI, Anthropic, Google, DeepSeek and other
providers through a single API, with governance, spend tracking, fallbacks
and auto-routing.

Concentrate exposes an OpenAI **Responses API**–compatible endpoint at
`https://api.concentrate.ai/v1`, so this package is a thin subclass of
`langchain-openai`'s `ChatOpenAI` that pins the base URL, forces the Responses
path, and reads its key from `CONCENTRATE_API_KEY`.

## Install

```bash
pip install langchain-concentrate
# or
uv add langchain-concentrate
```

## Quickstart

```bash
export CONCENTRATE_API_KEY="sk-cn-v1-..."
```

```python
from langchain_concentrate import ChatConcentrate

model = ChatConcentrate(model="gpt-5.4")
print(model.invoke("Say hi in three words.").text)
```

### Streaming

```python
for chunk in model.stream("Count from 1 to 5."):
    print(chunk.text, end="", flush=True)
```

### Auto-routing

Pass `model="auto"` with a `routing` directive to let Concentrate pick a
provider based on your policy:

```python
cheap = ChatConcentrate(
    model="auto",
    routing={"strategy": "min", "metric": "cost"},
)
cheap.invoke("What is 2 + 2?")
```

### Tool calling & structured output

Because `ChatConcentrate` inherits from `ChatOpenAI`, `bind_tools` and
`with_structured_output` work out of the box for models that support them
downstream.

```python
from pydantic import BaseModel

class Answer(BaseModel):
    number: int

structured = ChatConcentrate(model="gpt-5.4").with_structured_output(Answer)
result = structured.invoke("What is 2 + 2?")
assert result.number == 4
```

## Configuration

| Env var                  | Purpose                                             | Default                          |
| ------------------------ | --------------------------------------------------- | -------------------------------- |
| `CONCENTRATE_API_KEY`    | API key sent as `Authorization: Bearer <key>`.      | *(required)*                     |
| `CONCENTRATE_BASE_URL`   | Override the API base URL.                          | `https://api.concentrate.ai/v1`  |

Any keyword accepted by `ChatOpenAI` — `temperature`, `max_output_tokens`,
`timeout`, `default_headers`, `extra_body`, `model_kwargs`, etc. — can be
passed through.

## Development

```bash
uv sync --all-groups
make test              # unit tests (no network)
make integration_tests # against real Concentrate, needs CONCENTRATE_API_KEY
```

## License

MIT

## Disclaimer

Portions of this package were generated with the help of an AI coding
assistant. All code is human-reviewed.
