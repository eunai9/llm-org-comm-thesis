.RECIPEPREFIX = >
.PHONY: help venv install fmt lint type test check data sim judge results clean

VENV := .venv
PY   := $(VENV)/bin/python

help:
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv
> python3 -m venv $(VENV)

install:  ## Install dev + runtime dependencies, and the package itself
> $(PY) -m pip install --upgrade pip
> $(PY) -m pip install -r requirements-dev.txt
> $(PY) -m pip install -e .

fmt:  ## Format with black and apply ruff autofixes
> $(PY) -m black src tests
> $(PY) -m ruff check --fix src tests

lint:  ## Lint (no changes)
> $(PY) -m ruff check src tests
> $(PY) -m black --check src tests

type:  ## Type-check
> $(PY) -m mypy

test:  ## Run the test suite
> $(PY) -m pytest

check: lint type test  ## Everything CI would run

data:  ## Build the processed message store from the raw corpus
> $(PY) -m thesis.data.build_store

sim:  ## Run the agent simulator
> $(PY) -m thesis.sim.run

judge:  ## Run the LLM-as-a-judge scoring pass
> $(PY) -m thesis.judge.run

results:  ## Regenerate every figure and table from cached responses
> $(PY) -m thesis.analysis.tables
> $(PY) -m thesis.analysis.figures

clean:  ## Remove caches and build artefacts (never touches data/ or runs/)
> find . -type d -name __pycache__ -prune -exec rm -rf {} +
> rm -rf .pytest_cache .mypy_cache .ruff_cache build dist src/*.egg-info
