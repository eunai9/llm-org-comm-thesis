# Q1 persona-variant experiment — design

Date: 2026-10-22. Status: approved in chat, awaiting review of this spec.

## Goal

A second Q1 experiment that runs next to the existing 10-persona design and
does not replace it. It fixes two problems of the current design:

- One persona per role cell. A rank effect could come from that one persona's
  prompt, not from rank.
- Only 10 clusters. Persona-clustered standard errors are too small with so
  few clusters, so p-values look better than they are.

Success: 40 personas (40 clusters), with every rank effect estimated across
four persona variants per rank, for all four free-tier models.

## Decisions made with the user (2026-10-22)

| Question | Decision |
|---|---|
| How variants differ | One trait line each, on traits that are not outcome measures. Corpus style numbers stay as they are. |
| Receiver ranks | One step up, same rank, one step down. Ranks 1 and 5 write in two directions. |
| Size | 4 variants per cell (40 personas). Pilot first (8 scenarios per direction), then the full set (48). |
| Models | All four: Llama 3.2 3B, gpt-oss-20b, gpt-oss-120b, qwen3.8-27b. |

Earlier decisions (2026-10-14) still hold: one reply per email, no threads,
and new measures are audited by the user before they are used.

## 1. Personas

- Base: the 10 personas in `data/external/personas_snapshot.json` (5 ranks x
  Trading/Legal). Their corpus style numbers are not changed.
- Variants: the same 2x2 trait set in every cell.
  - Tenure: about 2 years vs. about 12 years at the company.
  - Origin: promoted from within vs. hired from another energy firm.
- Rendering: the base persona block plus one trait line, for example
  "You joined from another energy firm about 2 years ago."
- IDs: `<base_id>_t<2|12>_<internal|external>`, for example
  `r3_trading_t2_external`.
- Memory: a variant uses its base persona's memory store.

Why these traits: they are plausible personal history, and neither one is an
instruction about orders, hedges, "you", gratitude or length. The same four
variants in every cell keep traits balanced across ranks, so a trait cannot
stand in for rank. The 2x2 also allows a check that traits do not move the
outcomes much.

Known risk: the persona block already states the rank's corpus rate of direct
instructions ("Gives direct instructions in roughly X% of sentences"). This
is unchanged from the 10-persona design, so the two designs stay comparable.

## 2. Direction framing

- The current sentence (`_DIRECTION_FRAMING`, e.g. "You are writing to
  someone senior to you.") is replaced by a named receiver: "You are writing
  to a Vice President in Trading."
- The receiver is in the sender's own department, at rank +1 (up), the same
  rank (lateral) or rank -1 (down).
- Allowed pairs: rank 1 writes lateral and up; rank 5 writes down and
  lateral; ranks 2-4 write all three. That is 13 pairs per department and 26
  in total, each written by all 4 variants.
- The existing framing stays the default. The named framing is selected
  explicitly, so the 10-persona grids and their cache entries are untouched.
- The prompt text changes, so the new grid records its own prompt hash.

## 3. Grid and generation

- Scenarios: the existing sets. Pilot = the Q1 pilot scenarios (2 task types
  x 4 tones, 8 per direction). Full = all 48 per direction. The pilot is a
  subset of the full set, so the full run reuses every pilot reply from the
  cache.
- Size per model: pilot 832 replies (104 persona-direction pairs x 8); full
  4,992 replies (x 48).
- Generation goes through the existing `run_grid` path: same prompt assembly,
  cache, validation and cost ledger. One resumable tmux runner per model, as
  for the existing grids.
- Providers: Llama local (WSL Ollama), gpt-oss-20b on NVIDIA, gpt-oss-120b
  and qwen3.8-27b on Groq. All free tier.
- Output: `data/interim/q1_persona_grid_<model>_<pilot|full>.parquet`. Each
  row adds to the usual grid columns: `base_persona_id`, `tenure`, `origin`,
  `receiver_rank`, `direction`, `prompt_text_hash`.
- Entry point: `python -m thesis.analysis.q1_personas --<provider> MODEL
  --design pilot|full --out PATH` (generation), and an analysis mode that
  reads saved grids and calls no model.

## 4. Analysis

- Outcomes: the existing Q1 measures (share of sentences that give orders,
  reply-level orders and hedges) and the three kept measures ("you" rate,
  Gilbert hierarchy-phrase score, gratitude).
- Main estimate: writing-down and writing-up contrasts against lateral,
  persona fixed effects with standard errors clustered by the 40 personas.
  Same estimator as the 10-persona design, so the two compare directly.
- Trait check: do tenure or origin shift the outcomes, and do they change
  the direction contrasts? Reported, not a headline.
- Comparisons: each model against its own 10-persona result and against real
  email (the robust real-email fit).
- Pilot gate: after the pilot, report the valid-reply rate per model and a
  first look at the contrasts. The user decides whether to run the full set.

## 5. Testing

Unit tests, written first:

- Variant building: 4 variants per base persona, correct IDs and traits,
  style numbers unchanged.
- Trait line rendering.
- Named framing: correct receiver label for each rank and direction, and the
  default framing unchanged.
- Allowed pairs: rank 1 and rank 5 get two directions, ranks 2-4 get three;
  26 pairs in total.
- Grid counts: 832 cells for the pilot and 4,992 for the full design.
- The prompt hash is stored on every row.

## Out of scope

- Changing or rerunning the 10-persona experiment.
- Multi-turn threads.
- New outcome measures beyond the three already kept.
- A larger receiver gap (two or more ranks).
