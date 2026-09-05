# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project Context

This is my AI agent workspace. I use it for my thesis: LLMs for Simulating Organizational Communication and Decision-Making with Email. See `Expose.pdf` for the research plan (questions, dataset, architecture, timeline) — don't duplicate it here.

# About Me

I am a master student majoring Statistics and Data Science. I am writing my master thesis with you. The readers will be professors, my supervisor, and experts in AI and data science.

# Writing Style

This is a rule, not a preference. Follow it in everything you write for me:
chat replies, PROGRESS.md, commit messages, reports, code comments, and docstrings.

The goal is that I understand the text on the first read.

Do this:

- Write short sentences. One idea per sentence.
- Use the easiest word that is still correct.
- Put the result first. Then the reason.
- Use plain numbers and plain facts.

Do not do this:

- Do not write long sentences with many commas.
- Do not use dashes to add extra clauses.
- Do not write rhetorical or dramatic lines. Do not build up to a point.
- Do not say the same thing twice in different words.
- Do not add phrases that carry no information, like "it is worth noting that"
  or "this is exactly the kind of thing that".

I do not want elaborate writing. I dislike it. Being clear is the only goal.

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
