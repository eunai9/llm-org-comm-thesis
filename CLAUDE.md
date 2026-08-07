# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project Context

This is my AI agent workspace. I use it for my thesis: LLMs for Simulating Organizational Communication and Decision-Making with Email. See `Expose.pdf` for the research plan (questions, dataset, architecture, timeline) — don't duplicate it here.

# About Me

I am a master student majoring Statistics and Data Science. I am writing my master thesis with you. The readers will be professors, my supervisor, and experts in AI and data science.

# Tech Stack

- Python only, developed in an Ubuntu environment.
- Environment: `python3 -m venv .venv` + `source .venv/bin/activate`.
- Dependencies: `pip`, tracked in `requirements.txt`. Install with `pip install -r requirements.txt`;
  after adding a new package, `pip freeze > requirements.txt` to keep it in sync.

# Code Style

- Format with `black`; lint and sort imports with `ruff`; type-check with `mypy`.
  - `black .`
  - `ruff check .`
  - `mypy .`
- Type hint all function signatures.
- Docstrings only on public functions/classes, one-line summary unless the behavior is non-obvious.
- No dead code, no commented-out code, no speculative abstractions for hypothetical future needs.

# Workflow

- Notebooks (`notebooks/`) are for exploration only — trying out preprocessing steps, inspecting
  data, prototyping prompts. Nothing in a notebook is "done."
- Once logic is settled, port it into a reusable module under `src/` (plain function/class, no
  notebook-only state) and call it from the notebook or a script instead of duplicating it.
- Scripts that should be reproducible (a full preprocessing run, a full experiment) belong in `src/`
  as a `python -m thesis.<module>` entry point, not only as notebook cells.
- Treat the Enron dataset as real corporate/personal correspondence even though it's a standard
  public research corpus: analysis only, never commit it, never commit API keys/credentials.
