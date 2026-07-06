# Changelog

All notable changes to `langchain-concentrate` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Live integration coverage for tool calling (`bind_tools`) and structured
  output (`with_structured_output`), including a full tool round-trip, async
  variants, and structured output through the auto-routing path.
- Offline unit coverage for `bind_tools` schema registration and
  `with_structured_output` pipeline construction.
- LangChain standard integration-test suite (`ChatModelIntegrationTests`),
  the same contract suite run by first-party partners — verifying invoke,
  stream, batch, conversation history, usage metadata, tool calling,
  `tool_choice`, agent loops, JSON mode, and structured output (pydantic /
  typeddict / json_schema) against the live gateway.
- GitHub Actions CI running lint, format check, and `mypy`, plus unit tests
  across Python 3.10–3.13, resolving against published PyPI packages.

### Fixed

- `api_key` is now wrapped in `SecretStr` before being passed to `ChatOpenAI`,
  matching its expected type and satisfying `mypy`.

## [0.1.0]

### Added

- Initial release: `ChatConcentrate`, a `ChatOpenAI` subclass that pins the
  Concentrate.ai gateway base URL, forces the OpenAI Responses API, reads
  `CONCENTRATE_API_KEY`, and passes an auto-`routing` directive through
  `extra_body`.
