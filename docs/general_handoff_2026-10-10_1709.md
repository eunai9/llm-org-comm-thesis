# Session handoff — 2026-10-10, 1709

For a fresh agent continuing this Claude Code session on the LLM-org-comm
thesis project. Read this first. It does not repeat CLAUDE.md, the pinned
memories, or `PROGRESS_llms.md` itself — it points to them.

## Where things live

Same as every prior handoff: this chat's working folder is the Windows
OneDrive folder; the real repo is in WSL at `~/projects/thesis`, pushed to
`github.com/eunai9/llm-org-comm-thesis`, branch `main`. Reach it via
`wsl.exe -e bash -c "cd ~/projects/thesis && <command>"`. `PROGRESS.md` and
`HANDOVER.md` are deprecated — don't touch them (see pinned memories).
`PROGRESS_llms.md` is the active log; this session's work is section 16 and
the "Next steps" updates inside it. No peer Claude session is running
(checked via `ListAgents`). No memory files were added or updated this
session.

## What this session did, in order

1. Resumed from `docs/general_handoff_2026-10-05_2004.md`. That handoff's
   top priority was next-steps item 4, "Judge study (Q3)": run the
   judge-swap self-preference design on the project's real four free-tier
   models instead of the two local 3B stand-ins sections 23/41/52
   (`PROGRESS.md`) used.
2. Generalized `src/thesis/analysis/judge_swap.py` (TDD) from a hardcoded
   2-generator/2-judge design to any number of each, added a
   `client_for_model` dispatch helper (NVIDIA/Groq/Ollama), and
   `effect_estimates_multi`/`fit_subset_multi` (self-preference per
   non-reference model). Commit `083a592`.
3. Live-testing that surfaced two real problems, handled in order:
   - **DeepSeek V4 Flash was retired by NVIDIA mid-session**
     (`deepseek-v4-flash-0731`, end of life 2026-09-21, a live call
     returned HTTP 410). User's decision: keep it as a generator only (its
     240 replies were already fully cached, zero new cost), drop it as
     judge (can't be called live any more).
   - **A real billing bug**: `judge/run.py`'s cost check named the local
     model case but not NVIDIA/Groq, so the first-ever NVIDIA/Groq judge
     call crashed with `UnknownModelPriceError`. Fixed in the same commit,
     regression-tested per provider.
4. Ran the judge-swap pilot (4 generators, 3 judges — DeepSeek excluded
   from judging) through a resumable, exponential-backoff tmux script
   (`~/q3_judge_swap_runner.sh`, not in git, same pattern as the
   pre-existing `~/q1_qwen_runner.sh`). Finished cleanly after fighting
   Groq's daily cap for about 10 hours. Verified by the `results-verifier`
   agent (provenance and arithmetic confirmed; flagged that the
   interaction term is a difference-in-differences, not one model's bias
   read alone, and that only exact-model self-preference is tested, not
   family-level — both now stated in the write-up) and cross-checked with
   a cluster-robust OLS refit (same coefficients, same or tighter SE) since
   the mixed model logged a singular-covariance warning. Written up as
   `PROGRESS_llms.md` section 16, commit `0488a44`. Headline: gpt-oss-120b
   and gpt-oss-20b both show self-preference against llama3.2:3b as
   reference (+0.68 and +0.59, both p<.0001) — stronger than the
   two-local-model pilot (+0.40, p=.005).
5. Two follow-up edits to `PROGRESS_llms.md` on request, both committed and
   pushed:
   - Removed a now-stale bullet in section 12 that described DeepSeek's
     missing full-design Q1 grid as still possibly resolvable — it isn't;
     qwen3.8-27b is the confirmed, permanent replacement there. Commit
     `f2a8e51`.
   - Added a **Recommendation** paragraph to section 16: pair gpt-oss-20b
     as generator with llama3.2:3b as judge going forward, with reasoning
     (llama is the only tested judge not caught favoring its own writing;
     both gpt-oss models are rated above llama by llama itself, a judge
     with no stake in the comparison; gpt-oss-20b avoids Groq's rate cap).
     Commit `50ac521`.
6. Built a PowerPoint progress-report deck for the user's supervisor,
   matching the visual style of an existing deck
   (`Q1_progress_update.pptx`, in the OneDrive folder, not in git) that the
   user pointed to by filename. Built with `pptxgenjs` (see **PPT tooling
   notes** below — LibreOffice is not available in this environment, so
   visual QA used PowerPoint COM automation via PowerShell instead of the
   `pptx` skill's usual `soffice` path). Went through two full rounds of
   content feedback (simplify a slide, use exact model names instead of
   vague groupings, keep a table rather than replacing it with bullets,
   remove all DeepSeek references from two slides, surface llama's
   reference-level status in the Q3 table, add the generator/judge
   recommendation with reasoning). Final file: 10 slides, saved to
   `C:\Users\Acer\OneDrive\문서\2. 대학원\LMU S&DS\Thesis\Progress_update_Oct9.pptx`
   — **this file is not tracked in git**; it lives only in the OneDrive
   folder, outside the WSL repo. Structural validation
   (`scripts/office/validate.py`) and full visual QA (all 10 slides
   rendered and inspected) both passed on the final version.
7. Cleaned up the scratch build directory (`~/ppt_work`, ~8.7MB:
   `node_modules`, the `pptxgenjs` build script, intermediate `.pptx`
   files) — deleted, nothing from it is needed again unless another deck
   is built (see **PPT tooling notes**).

## Current live state

- **`q1_qwen_runner` tmux session is alive**, generating qwen3.8-27b's
  full Q1 grid (`data/interim/q1_direction_grid_qwen_full.parquet`, 1,440
  cells). **900/1,440 (62.5%)** as of this session, up from 450 at the
  last handoff. Still hitting Groq's daily token cap repeatedly; the
  exponential-backoff script (`~/q1_qwen_runner.sh`, WSL home directory,
  not in git) is working as designed.
- **This tmux session has died twice since the last handoff**, both times
  because the laptop went to sleep or rebooted (WSL's own uptime reset
  under it). Each time it was restarted manually this session with:
  `tmux new-session -d -s q1_qwen_runner -c ~/projects/thesis 'bash ~/q1_qwen_runner.sh 2>&1 | tee -a logs/q1_qwen_full_run.log'`
  The user was told: keep the laptop plugged in with sleep disabled (or
  accept that the job pauses — no progress is lost, cache persists — but
  needs a manual restart after each sleep/reboot).
- **`wsl_ollama` tmux session is no longer listed** — it was not checked
  or restarted this session since no local-model work was needed after the
  PPT work began. Start it again with `tmux new-session -d -s wsl_ollama
  'ollama serve'` (on the Windows side — Ollama itself runs as a Windows
  app/process, not inside WSL; WSL reaches it via `127.0.0.1:11434`
  through WSL2's localhost forwarding) before any script that needs a
  local Llama/qwen call.
- **Nothing of this session's own work is uncommitted.** `git status`
  shows only the long-standing untracked peer artifacts (`mirroring_*.png`
  figures, the `openai_gpt-oss-120b_low` pair, `outputs/manifests/
  cost_ledger.csv` modified by the qwen job's own cache re-logging) —
  leave those alone, not this session's to touch.
- Latest commit: `50ac521`, pushed to `origin/main`.

## PPT tooling notes (for next time a deck is needed)

- **No `soffice`/LibreOffice anywhere** in this setup — not in WSL, not on
  Windows. The `pptx` skill's usual `scripts/office/soffice.py` /
  `scripts/thumbnail.py` path (which both shell out to `soffice`) will
  fail here.
- **Markitdown, python-pptx, Pillow, lxml, defusedxml are all already
  installed** in WSL's system `python3` (not the thesis venv — a plain
  `python3 -m markitdown file.pptx` works from any directory). The skill's
  pure-Python scripts (`validate.py`, `add_slide.py`, `clean.py`) work
  fine.
- **For visual QA, use PowerPoint COM automation via PowerShell instead**:
  copy the `.pptx` to a plain local Windows path (PowerPoint's COM `Open`
  rejects `\\wsl.localhost\...` UNC paths), then a short PowerShell script
  with `New-Object -ComObject PowerPoint.Application`,
  `$pres.Export($outDir, "PNG", 1600, 900)` exports every slide as a PNG.
  Confirmed working this session; PowerPoint is installed at
  `C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE`.
- **`pptxgenjs` is not preinstalled here** (unlike the skill's assumed
  sandbox) — `npm install pptxgenjs` in a scratch WSL directory first.
  WSL's node is v18.19.1; that's fine for pptxgenjs.
- The existing supervisor-deck style (if asked to match it again): navy
  `1E2761` / dark `262626` / gray `6B6B6B` / footer-gray `A6A6A6` /
  table-border `E3E3E3` / row-highlight `F4F5F8`, Cambria for titles (28pt
  content slides, 40pt title slide), Calibri for body, `LAYOUT_WIDE`
  (13.333in × 7.5in / 12192000×6858000 EMU), colors hardcoded per-run
  (no theme/`apply_theme.js` applied in the source deck, so none was
  applied here either, to match exactly).

## Next steps

Read `PROGRESS_llms.md`'s own "18. Next steps" section for the current,
maintained priority list (5 items as of this session: a larger sample for
the two remaining NVIDIA/Groq models — note its own text still says
"three", stale since DeepSeek dropped out, worth a small fix; hand-coding
a sample of replies — review packets are ready, nothing left but a person;
a run without the "act" instruction; moving the line-removal rule into the
corpus cleaner; and backing up `runs/_cache`). Don't duplicate that list
here; it changes faster than this handoff would stay accurate.

Nothing about the PPT is outstanding — the user confirmed it was ready to
send to their supervisor as of this session's end.

## Suggested skills

- **superpowers:test-driven-development** — if any further work touches
  `src/thesis/analysis/*.py`; this codebase has a real history of subtle
  statistical bugs that only a failing test first reliably catches (true
  again this session: the billing bug in item 3 above was only exposed by
  a live run, not by the existing test suite, and was fixed test-first
  once found).
- **results-verifier agent** (not a skill, same idea) — use before any new
  number from an analysis module reaches `PROGRESS_llms.md`. Used once
  this session on the Q3 self-preference numbers; it caught a real
  statistical-framing issue (see item 4 above), not just a provenance
  check.
- **anthropic-skills:pptx** — only if another deck or deck edit is needed.
  Load it, but expect to deviate from its default tooling exactly as
  described in **PPT tooling notes** above (no `soffice`; PowerPoint COM
  for visual QA instead).
