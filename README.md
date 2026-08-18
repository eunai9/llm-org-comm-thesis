# LLMs for Simulating Organizational Communication and Decision-Making with Email

Master's thesis, MSc Statistics & Data Science, LMU Munich.
Defense: end of March 2027.

## Research questions

- **Q1 — Hierarchical influence.** How does hierarchical role influence response
  content, sentiment, and decision attitude?
- **Q2 — Simulation fidelity.** Are LLM-generated responses diverse enough, and
  similar to empirical Enron patterns?
- **Q3 — Evaluator validity.** Can LLM-as-a-judge give consistent, calibrated
  quality scores that agree with human coders?

## Try the simulator (no setup, no key)

Want to see it work without setting up the full thesis environment? See
[RUNNING_THE_DEMO.md](RUNNING_THE_DEMO.md) -- runs a small AI model locally,
free, in about 15 minutes of setup.

## Quick start

```bash
make venv
source .venv/bin/activate
make install
make check
```

## Layout

| Path | Purpose |
|---|---|
| `src/thesis/data/` | Enron parsing, role resolution, threads, power score, sampling |
| `src/thesis/llm/` | Provider abstraction, response cache, batch submission, cost ledger |
| `src/thesis/sim/` | Persona/scenario schemas, memory stream, prompt assembly, experiment grid |
| `src/thesis/judge/` | Rubric, judge prompts, scoring passes, aggregation |
| `src/thesis/analysis/` | Q1/Q2/Q3 analyses; writes to `outputs/` |
| `configs/` | YAML run configuration |
| `data/` | Gitignored, except curated files in `data/external/` |
| `runs/` | Per-run manifests and raw responses (gitignored) |
| `outputs/` | Figures and tables consumed directly by the LaTeX in `thesis/` |

Notebooks are for exploration only. Once logic settles it moves into `src/thesis/`
and is invoked as `python -m thesis.<module>`.
