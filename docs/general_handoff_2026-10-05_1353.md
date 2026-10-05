# Session handoff — 2026-10-05, 1353

For a fresh agent continuing this Claude Code session on the LLM-org-comm
thesis project. Read this first. It does not duplicate `PROGRESS_llms.md`
or git history — it points to them.

## Where things live, and two standing changes this session made

- This chat's working folder is the Windows OneDrive folder (`Expose.pdf`
  and docs only). The real repo is in WSL at `~/projects/thesis`, pushed to
  `https://github.com/eunai9/llm-org-comm-thesis`, branch `main`. Reach it
  via `wsl.exe -e bash -c "cd ~/projects/thesis && <command>"`.
- **`PROGRESS.md` is no longer maintained — do not update it.** The user
  explicitly reversed a standing instruction from an earlier session
  ("I won't update PROGRESS.md anymore"). This is saved in memory
  (`progress-report-updates.md`).
- **`HANDOVER.md` is no longer used — do not read or update it.** Same
  session, same kind of reversal ("I won't use HANDOVER.md anymore").
  Saved in memory (`handover-md-deprecated.md`). If you've read an older
  handoff that references `HANDOVER.md` heavily, ignore those references.
- `PROGRESS_llms.md` is the one log that is still active. Everything this
  session did is written up there, dated. Read it, not this file, for the
  actual numbers and findings.
- Handoff docs go in `docs/general_handoff_YYYY-MM-DD_HHMM.md` in this
  repo (not OS temp, not the OneDrive folder) — a standing project-specific
  override of this skill's own default, saved in memory
  (`handoff-docs-location.md`).
- No peer Claude session is running right now (checked via `ListAgents`
  just before writing this). One may start again later in the same
  repo — if so, check `ListAgents` for its current name before assuming
  it's gone; this session's own peer-facing name changes across turns too
  (currently `thesis-4d`).

## What this session did, in order

1. Resumed from `docs/general_handoff_2026-10-03_2200.md`. Housekeeping:
   added `logs/` to `.gitignore`, deleted the superseded
   `scripts/role_inference_llama_only.py` (commit `6d2620f`).
2. **Closed out the Layer-1 judge-model question.** All 3 local judge
   models had already failed the self-consistency gate (prior session).
   The user confirmed no bigger free-tier model exists to try, so the
   judge-model idea is dropped outright — not "paused," a closed decision.
   Written up in `PROGRESS_llms.md`, "The new Q1 judge plan, and where
   Layer 1 failed" section (commits `40aa1e4`, `bc03319`).
3. **Went back to the original Q1 approach** (the pre-judge-model
   `is_imperative` grammar check, `src/thesis/data/features.py:180` — a
   spaCy dependency-parse rule, not an LLM call at all) and ran it against
   two models' full-design grids that existed but had never been combined
   into a cross-model comparison: gpt-oss-20b and gpt-oss-120b, alongside
   Llama's existing 3-draw full-design result.
4. **Found and fixed a real bug** while doing this:
   `q1_models.summarize_model` built one model's row from a draw-1-only
   persona-fixed-effects fit for `coefficient`/`p`, while
   `coefficient_vb`/`vs_real` in the same row came from the pooled fit —
   so a 3-draw grid's row silently mixed two different fits. Fixed with a
   new `sentence_model_persona_fe_pooled` field (TDD, tests in
   `test_q1.py`/`test_q1_models.py`), commit `561a2cb`. Verified by the
   `results-verifier` agent before and after.
5. **Wrote up the full-design cross-model comparison** in
   `PROGRESS_llms.md`, "Q1 across models, at full design scale (Oct 4)"
   (commits `cc3f122`, then expanded to the pilot section's full depth —
   reply/sentence-level tables both directions, probability table,
   decision-field chi-square, hedging, persona-variance, all six vs-real
   contrasts per model — in `cd403df`, at the user's request to match the
   older pilot section's detail). Headline finding: gpt-oss-20b showed no
   effect at pilot scale but matches real email's own effect size at full
   scale; gpt-oss-120b still overshoots real email at full scale, half as
   much as at pilot scale.
6. **Built the length-matched Q1-vs-real comparison** (user-approved
   plan): a short model reply inflates the `is_imperative` share for the
   same underlying behavior, so comparing a short-reply model against
   real email's full-length replies confounds direction with length.
   Added `truncate_sentences` (`q1_real.py`) and
   `compare_with_length_matched_real` (`q1_models.py`), wired into
   `summarize_model`/`main()` as an addition alongside the existing
   full-length comparison, not a replacement (TDD throughout, commits
   `91b9166`, `5a7c5fe`).
7. **The length-matched result changed the headline story, verified by
   `results-verifier` before being written up:** gpt-oss-120b's
   full-length "overshoot" (p=.009) is no longer detectable once real
   email is cut to the same reply length (p=.681) — though this is "not
   detectable," not proven equivalence. The verifier also found a bigger,
   separate problem: real email's variational-Bayes standard error looks
   about half the size of a sender-clustered robust one, so even the
   original full-length p=.009 is itself probably overstated (robust
   check: p≈.14). Written up in `PROGRESS_llms.md`, "Length-matching the
   Q1-vs-real comparison (Oct 4)" (commit `bfbfb1d`), with a new Next
   Steps item (now item 1, top priority) to build a robust-SE cross-check
   for the real-email side — nothing has this yet, the way
   `sentence_model_persona_fe` is this cross-check on the simulator side.
8. **Fixed the qwen3.8-27b generation job's retry loop.** It had crashed
   (Groq rate-limit exhaustion with no recovery loop) and needed restarting
   once; then, once it hit Groq's daily token cap and got stuck retrying
   every 5 minutes with zero progress for ~5.5 hours (172 wasted attempts),
   rewrote `~/q1_qwen_runner.sh` (lives in the WSL home directory, **not**
   part of the git repo) to use exponential backoff: 5 → 10 → 20 → 30 min
   (capped), resetting to 5 min the moment an attempt actually generates a
   new cell. Restarted under the new script; confirmed the backoff
   escalation is working (observed going 300s → 600s → 1200s as the cap
   stayed hit).
9. **Cleaned up two stale/small items in `PROGRESS_llms.md`'s Next Steps
   list:** marked the hand-coding page and its codebook fix as already
   committed back in Sep (commit `50bd3aa`, just never checked off —
   commit `4efd4ac`), renamed `mirroring.py`'s wrongly-named
   `compared_with_previous_prompt` summary key to `compared_with_run`
   (commit `6aa2667`), and copied `Q1Grid.prompt_text_hash` into
   `q1_models.py`'s per-model manifest rows, closing a gap `5ee4b46` left
   (commit `dea192b`). Regenerated `outputs/manifests/q1_models_full.json`
   after each code change that touched its contents.

## Current live state

- **`q1_qwen_runner` tmux session is alive**, generating qwen3.8-27b's
  full grid (`data/interim/q1_direction_grid_qwen_full.parquet`, 1,440
  cells). **Stuck at 300/1440 on Groq's daily token cap** as of this
  writing, same cap that stalled the gpt-oss-120b grid before it (that
  one's longest stall was 22 hours; also saw 9.7h and 1.9h stalls). The
  new exponential-backoff script (`~/q1_qwen_runner.sh`) is running and
  confirmed working — no action needed, it will resume on its own once
  the cap lifts. If the tmux session is gone (laptop sleep/shutdown kills
  it — confirmed to have happened at least once already this session),
  restart with:
  `tmux new-session -d -s q1_qwen_runner -c ~/projects/thesis 'bash ~/q1_qwen_runner.sh 2>&1 | tee -a logs/q1_qwen_full_run.log'`
  — it resumes instantly from cache, nothing is lost.
- **`wsl_ollama` tmux session is alive** (judge-model Ollama server) — now
  unused, since the judge-model idea is dropped. Safe to leave running or
  kill; nothing depends on it anymore. If it's gone and something later
  needs it: `tmux new-session -d -s wsl_ollama -c ~ '~/.local/bin/ollama serve 2>&1 | tee -a ~/ollama_serve.log'`.
- **A known, noisy-but-harmless side effect:** every time the qwen retry
  loop restarts, it re-logs its ~300 already-cached cells into
  `outputs/manifests/cost_ledger.csv` at $0 cost each (`from_cache=1`).
  This bloated one commit's diff to ~47,000 lines. Not a correctness bug,
  not new spend, just log duplication — flagged, not fixed, since it
  wasn't what was asked this session.
- **Nothing of this session's own work is uncommitted.** `git status`
  shows only the long-standing untracked peer artifacts (the
  `mirroring_*.png` figures and `q1_full_grid_openai_gpt-oss-120b_low.*`
  pair) — leave those alone, they are not this session's to touch.
- Latest commit: `dea192b`.

## Immediate next steps

Taken from `PROGRESS_llms.md`'s own "Next steps" section — read it there
for full context, this is just the order of priority:

1. **Build a robust-standard-error cross-check for the real-email fit**
   (new top item from this session's length-matching work — see point 7
   above). Until this exists, treat every p-value in the log that compares
   a model against real email as possibly overstated.
2. A larger sample for the remaining under-powered models (DeepSeek is
   blocked — its NVIDIA access is dead; qwen3.8-27b is in progress, see
   above).
3. Hand-code a sample of replies by a person — fully ready, needs a human
   (the user), not more code.
4. A generation run without the "act more directive" instruction, to see
   if a bigger model shows the hierarchy effect unprompted.
5. Embedding map and review pack on the newer models (no new generation
   needed).
6. The Q3 judge study (four models, one writes, one judges, swap roles).
7. Move the line-removal rule into the corpus cleaner (this one needs a
   re-run of earlier results that used the real replies — not small).
8. Back up `runs/_cache` (needs the user to decide where).

Layer 2 (the indicator panel, digging into *why* the hierarchy signal
shows up) is not on this list — it was designed to run only after Layer 1
(the judge model) found a real signal, and Layer 1 is now closed as a
dropped idea, not a finding to build on. Whether Layer 2 still makes sense
on a different basis is an open question the user hasn't been asked yet.

## Suggested skills

- **superpowers:test-driven-development** — every code change this
  session (the `summarize_model` bug fix, the length-matching feature, the
  prompt-hash fix, the key rename) went through red-green TDD. Keep doing
  that for anything touching `src/thesis/analysis/q1*.py` — this codebase
  has a real history of subtle statistical bugs (draw-pooling, fit-mixing)
  that only a failing test first reliably catches.
- **results-verifier agent** (not a skill, but use it the same way) —
  used twice this session, both times caught something real (a
  mislabeled fit, an understated standard error). Use it again before any
  new number from `q1*.py` reaches `PROGRESS_llms.md`, especially anything
  involving a p-value comparison against real email.
- **superpowers:systematic-debugging** — if the qwen job's behavior ever
  looks wrong in a new way (not just "still capped"), use this before
  guessing at a fix, the same way bugs were run down earlier this session
  via `git log -L` and fresh one-off reproductions rather than trusting
  old write-ups at face value.

## Not included here (see the artifacts directly)

- All actual numbers, tables and findings: `PROGRESS_llms.md`, sections
  dated Oct 3-4 (search for "Oct 4" — there are three such sections now).
- Full commit history for this session: `git log --oneline dea192b...95f0a3f`
  in the repo (everything from this session is between those two commits).
- The qwen retry script itself: `~/q1_qwen_runner.sh` (WSL home directory,
  not in git — read it directly if you need the exact backoff logic).
