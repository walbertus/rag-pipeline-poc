.PHONY: check format lint type-check run

check: lint format type-check

run:
	uv run main.py

lint:
	uvx ruff check --fix

format:
	uvx ruff format

type-check:
	uvx ty check
