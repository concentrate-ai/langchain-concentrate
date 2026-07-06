.PHONY: all format lint type test tests integration_test integration_tests help

TEST_FILE ?= tests/unit_tests/
PYTEST_EXTRA ?=

.EXPORT_ALL_VARIABLES:
UV_FROZEN = false

test tests:
	uv run --group test pytest $(PYTEST_EXTRA) --disable-socket --allow-unix-socket $(TEST_FILE)

integration_test integration_tests:
	uv run --group test pytest -v --tb=short tests/integration_tests/

format:
	uv run --group lint ruff format .
	uv run --group lint ruff check --fix .

lint:
	uv run --group lint ruff check .
	uv run --group lint ruff format --check .

type:
	uv run --group typing mypy langchain_concentrate

help:
	@echo "make test              - run unit tests (no network)"
	@echo "make integration_tests - run integration tests against Concentrate (needs CONCENTRATE_API_KEY)"
	@echo "make format            - autoformat with ruff"
	@echo "make lint              - lint check"
	@echo "make type              - mypy"
