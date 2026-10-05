# Session handoff — 2026-10-05, 2004

For a fresh agent continuing this Claude Code session on the LLM-org-comm
thesis project. Read this first. It does not repeat CLAUDE.md, the pinned
memories, or `PROGRESS_llms.md` itself — it points to them.

## Where things live

Same as every prior handoff: this chat's working folder is the Windows
OneDrive folder; the real repo is in WSL at `~/projects/thesis`, pushed to
`github.com/eunai9/llm-org-comm-thesis`, branch `main`. Reach it via
`wsl.exe -e bash -c "cd ~/projects/thesis && <command>"`. `PROGRESS.md` and
`HANDOVER.md` are deprecated — don't touch them (see pinned memories).
`PROGRESS_llms.md` is the active log; this session's work is sections 9-13
and the "Next steps" updates inside it. No peer Claude session was running
at any point this session (checked via `ListAgents`).

Two new/updated memory files from this session, both still accurate as of
this writing — read them, don't re-derive this from scratch:

- `progress-section-titles-tag-research-question.md` — the exact title
  format every `PROGRESS_llms.md` section must follow
  (`## N. Title (Q1)`, sequential numbering through the whole file,
  research-question tag only where one applies), and that sections are
  grouped by topic/research-question, not by session date. This was tested
  the other way once (grouping a new Q2 section next to that day's other
  new work) and reverted at the user's request — don't redo that move.
- `handoff-docs-location.md` (pre-existing) — handoff docs go in this
  repo's `docs/` folder, not OS temp. This file follows that.

## What this session did, in order

1. Resumed from `docs/general_handoff_2026-10-05_1353.md`. That handoff's
   top priority was closing out Q1's "build a robust-standard-error
   cross-check for the real-email fit" — the `is_imperative` sentence
   model's variational-Bayes (VB) fit understates its own uncertainty, and
   nothing used the sender/persona-clustered fit that already existed as a
   cross-check. Fixed via TDD in `src/thesis/analysis/q1_real.py`
   (`robust_contrasts`) and `src/thesis/analysis/q1_models.py`
   (`real_row`, `summarize_model`, `compare_with_length_matched_real`),
   old VB numbers kept alongside under `_vb`/`"vb"` keys, not deleted.
   Verified by the `results-verifier` agent (one stale docstring number
   found and fixed). `outputs/manifests/q1_models_full.json` regenerated.
   Written up as what is now `PROGRESS_llms.md` section 13. Commit
   `04aebb4`. Headline: gpt-oss-120b's "significantly bigger than real
   email" claim (old p=.009) is not significant under the honest fit
   (p=.084).
2. User asked to move on to Q2 and Q3 (the project's other two research
   questions — Q2: are generated replies realistic/diverse; Q3: can an
   LLM judge give calibrated scores). Asked which to start with; user
   picked Q2.
3. Ran the existing-but-never-used-on-these-models embedding-map check
   (`src/thesis/analysis/embedding_map.py`) on the three free-tier models
   (gpt-oss-20b, gpt-oss-120b, DeepSeek), using pairs files that already
   existed (`data/interim/pairs_{gpt_oss,gpt_oss_120b,deepseek}.parquet`,
   183 real-thread-matched pairs each) — no new model generation, only
   free local embedding calls (`nomic-embed-text` via Ollama). Also built
   (but did not commit — gitignored, contains Enron text) review packets
   for a human coder on the same three models
   (`outputs/tables/{gpt_oss,gpt_oss_120b,deepseek}/`). Written up as what
   is now `PROGRESS_llms.md` section 9. Commit `33a3b71`. Headline: the
   free-tier models are *not* closer to real email in embedding space than
   the small local Llama model — if anything slightly further (AUC
   0.90-0.93 vs Llama's 0.89), agreeing with an earlier independent
   word-choice check.
4. User gave three rounds of formatting feedback on `PROGRESS_llms.md`,
   each acted on and pushed: (a) the new section was too long — cut
   roughly in half; (b) section titles should show which research question
   they answer; (c) titles should follow the exact format
   `N. Title (Q1/Q2/Q3)`, numbered sequentially through the whole file.
   Applied to all 17 sections, including fixing every internal "see '...'
   above" cross-reference this broke (there were over a dozen). Commits
   `e9dd268`, `8ddcaf1`.
5. User questioned whether section 13 (the robust-SE fix) was really Q1,
   not Q2, given it was done right after they'd asked about Q2 — resolved
   by checking commit timestamps: section 13 (`04aebb4`, 14:34) predates
   the user's Q2 request entirely, it was leftover Q1 work from the
   resumed handoff. No file change needed, just an explanation.
6. User asked to move section 9 (the Q2 embedding section) to sit next to
   the other Oct-5-dated sections instead of the older Q2 group. Done,
   full renumbering, commit `86c3813`. User then asked whether the
   original and moved versions differed in substance (no — move only) and
   asked to revert. Reverted cleanly with `git revert --no-edit 86c3813`
   (verified `git diff` against the pre-move commit is empty), commit
   `4503c10`. Memory file updated to reflect topic-grouping as the
   standing convention, not the date-grouping that was tried and undone.
7. User asked to scale Q2 up the way Q1 was scaled (pilot -> full design).
   Investigated and explained why this doesn't apply: Q2's checks
   (mirroring, fidelity, embedding map) all run on `pairs.py`'s fixed set
   of real-thread-plus-matched-persona pairs (capped at 183 of 200 real
   threads, by how many real senders have a rank/department a persona
   covers), not on the Q1 synthetic direction grid. The Q1 grid's replies
   answer invented scenarios, not real emails, so using it as a Q2
   comparison would reintroduce the exact stimulus-content confound the
   paired design exists to avoid. No code or doc change; user accepted
   this and dropped the "full-grid Q2" idea for now. Two real options
   for a future session if this comes up again: expand the persona set
   (more departments/ranks, raises the ceiling toward 200), or accept 183
   as the right ceiling for this kind of check.
8. Checked the qwen3.8-27b generation job (see "Current live state"
   below) twice during idle moments; no intervention needed either time.

## Current live state

- **`q1_qwen_runner` tmux session is alive**, still generating
  qwen3.8-27b's full grid
  (`data/interim/q1_direction_grid_qwen_full.parquet`, 1,440 cells).
  **350/1,440 (24%) as of 19:52** this session, up from 300/1,440 at the
  last handoff. Still hitting Groq's daily token cap repeatedly; the
  exponential-backoff script (`~/q1_qwen_runner.sh`, WSL home directory,
  not in git) is working as designed, currently sitting at its 30-minute
  ceiling between retries. Progress comes in trickles of ~50 cells per
  ~10 hours once the cap briefly lifts — a rough rate, not a hard number,
  implying several more days at minimum. No action needed; it resumes
  from cache on its own. Restart command if the tmux session is gone
  (laptop sleep has killed it before):
  `tmux new-session -d -s q1_qwen_runner -c ~/projects/thesis 'bash ~/q1_qwen_runner.sh 2>&1 | tee -a logs/q1_qwen_full_run.log'`
- **`wsl_ollama` tmux session is alive** (local Ollama server) — used this
  session for the embedding-map calls (`nomic-embed-text`). Still useful,
  don't kill it.
- **Nothing of this session's own work is uncommitted.** `git status`
  shows only the long-standing untracked peer artifacts (`mirroring_*.png`
  figures, `q1_full_grid_openai_gpt-oss-120b_low.*` pair, and
  `outputs/manifests/cost_ledger.csv` modified by the qwen job's own
  cache re-logging) — leave those alone, not this session's to touch.
- Latest commit: `4503c10`, pushed to `origin/main`.

## Next steps

Read `PROGRESS_llms.md`'s own "17. Next steps" section for the current,
maintained priority list (6 items as of this session: a larger sample for
the NVIDIA/Groq models, hand-coding a sample of replies — review packets
are ready, nothing left but a person, a run without the "act" instruction,
the Q3 judge study across four models, moving the line-removal rule into
the corpus cleaner, and backing up `runs/_cache`). Don't duplicate that
list here; it changes faster than this handoff would stay accurate.

One thing *not* on that list, raised but not resolved this session: Q2
has no pilot/full-design scale-up path the way Q1 does (see item 7 above).
If the user raises "bigger Q2 sample" again, start from that explanation
rather than re-investigating it.

## Suggested skills

- **superpowers:test-driven-development** — used for the robust-SE fix
  (item 1 above). Keep using it for anything touching
  `src/thesis/analysis/q1*.py`; this codebase has a real history of
  subtle statistical bugs that only a failing test first reliably catches.
- **results-verifier agent** (not a skill, same idea) — used once this
  session, caught a stale number in a docstring. Use again before any new
  number from `q1*.py` or the Q2 validation modules reaches
  `PROGRESS_llms.md`.
- No other skill was load-bearing this session beyond standard git/testing
  workflow; most of the work was direct investigation and writing, not
  skill-driven.
