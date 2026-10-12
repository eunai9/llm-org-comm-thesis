# Session handoff — 2026-10-12, 1939

For a fresh agent continuing this Claude Code session on the LLM-org-comm
thesis project. Read this first. It does not repeat CLAUDE.md, the pinned
memories, or `PROGRESS_llms.md` itself — it points to them.

## Where things live

Same as every prior handoff: this chat's working folder is the Windows
OneDrive folder; the real repo is in WSL at `~/projects/thesis`, pushed to
`github.com/eunai9/llm-org-comm-thesis`, branch `main`. Reach it via
`wsl.exe -e bash -c "cd ~/projects/thesis && <command>"`. `PROGRESS.md` and
`HANDOVER.md` are deprecated — don't touch them (see pinned memories).
`PROGRESS_llms.md` is the active log; this session's work spans sections
12, 15, 16, 17 and the "19. Next steps" updates inside it. No peer Claude
session is running (checked via `ListAgents`).

One more location matters this session: the user's supervisor deck,
`C:\Users\Acer\OneDrive\문서\2. 대학원\LMU S&DS\Thesis\Progress_update_Oct9.pptx`
— **not in git**, lives only in the OneDrive folder. See its own section
below; it needs careful handling.

## What this session did, in order

1. Resumed from `docs/general_handoff_2026-10-10_1709.md`.
2. Investigated and added the **Q1 writing-up effect** next to the
   existing writing-down effect. `direction` is a three-level factor
   (down/lateral/up); the persona-clustered fixed-effects fit already used
   for the writing-down headline table (`PROGRESS_llms.md` section 12)
   estimates both non-reference contrasts in one fit — the up contrast
   just was never pulled out and reported. Re-fit directly from the three
   already-cached grids (no new API calls), added as a companion table in
   section 12. Finding: real email itself shows no writing-up effect
   (p=.404) — the hierarchy effect in this corpus is asymmetric, not
   model-specific. gpt-oss-120b is the one model with a significant
   writing-up effect on its own (p=.046) but not distinguishably different
   from real email's own non-effect (p=.704). Also flagged: this
   persona-clustered p-value disagrees with the pre-existing VB-based fit
   further down the same section for gpt-oss-120b (p=.046 vs p=.160) —
   looks like the same VB-understates-uncertainty issue sections 11/14
   already found and fixed for the down contrast, not yet checked here.
   Commit `a2f43ac`.
3. User noticed Q2 has two different measurements (an AUC/discrimination
   classifier, and the six-item LLM rubric from Q3) and asked why. Added a
   cross-reference in both directions (`PROGRESS_llms.md` sections 8 and
   16, and the deck's slides 7/8) noting they actively **disagree**: the
   same models a word-choice classifier flags instantly (0.92–0.97 AUC)
   score 4.0–4.9 out of 5 on plausibility/fit when an LLM judge reads the
   same replies. Flagged as a real tension, not smoothed over. Commit
   `b29209b`.
4. Added a plain Q2 "rubric scores" section (16) — six-item scores, no
   self-preference statistic, judged by llama3.2:3b only (the one judge
   not caught favoring itself). First version used DeepSeek as one of the
   three models; the user corrected this (DeepSeek is excluded from every
   *new* comparison per standing project decision, even though its
   already-cached results elsewhere, e.g. section 17, stay valid) — swapped
   in Llama's own scores instead, matching the exact three models section
   9's embedding classifier already compares. Commits `add277d`, `e885a59`.
5. User asked whether the two-layer Q1 judge plan (dropped Oct 4, section
   15) was ever tried with the bigger gpt-oss models. Investigated via a
   subagent: **it wasn't** — the Oct 4 decision's stated reason ("no
   bigger free-tier model to try") was already slightly wrong at the time;
   gpt-oss-20b/120b existed as free-tier models by then, but the code to
   call them *as a judge* (not just a generator) wasn't built until Oct 5,
   for Q3's work, one day after the decision.
6. Generalized `role_inference.py` (Layer 1's self-consistency check) to
   judge with an NVIDIA or Groq model, reusing `judge_swap.client_for_model`
   — the same fix Q3 needed. Found and fixed the same billing gap Q3 hit
   on Oct 5 (`is_local_model` named, `is_nim_model`/`is_groq_model` not).
   Commit `291b34a`.
7. Running this live surfaced a second real bug: `judge_self_consistency`
   assumed every item gets a result on every one of three passes, and
   crashed with a `KeyError` the moment one didn't (one item came back as
   non-JSON on one pass, out of 300, when testing gpt-oss-20b). Fixed to
   drop any item missing a result on any pass from the whole comparison,
   with the dropped count logged. Regression-tested first. Commit
   `7f22ca2`.
8. **Ran the self-consistency gate for gpt-oss-20b**: fails decisively,
   kappa −0.016 (n=297, 3 dropped), 13.8% agreement across three passes —
   worse than chance. Same failure pattern as the three smaller models
   tested back on Oct 3. Written into section 15 as a dated correction.
   Commit `ef6a379`.
9. **Ran the self-consistency gate for gpt-oss-120b**: passes clearly,
   kappa 0.542 (n=300, nothing dropped), 65.3% agreement, and the positive
   control also passes (0.952 accuracy), so the harness itself is known to
   detect a real signal when present. This reopens Layer 1 as a real,
   runnable check — every judge tried before this one failed the
   reliability gate before the actual question could even be asked.
   Written into section 15. Commit `b96c33b`. Took two attempts: the first
   background run got killed at its 2-hour limit with zero visible
   progress (stuck on Groq's shared daily cap, same one the qwen job
   fights); moved to a tmux-based resumable runner
   (`~/gate_120b_runner.sh`, `~/gate_check.py`, both in the WSL home
   directory, not in git — same pattern as `~/q1_qwen_runner.sh`), which
   finished cleanly a few hours later and closed its own tmux session on
   success.
10. User asked to update slide 5 of the deck (see below) to propose
    running Layer 1 with gpt-oss-120b as judge, to show their supervisor
    before committing to that work. Done — see the deck section.

## Current live state

- **`q1_qwen_runner` tmux session is alive**, generating qwen3.8-27b's
  full Q1 grid. **1,150/1,440 (80%)** as of this session, still fighting
  Groq's daily token cap, the same resumable backoff script as every prior
  handoff. No action needed unless it dies (laptop sleep/reboot) — restart
  with `tmux new-session -d -s q1_qwen_runner -c ~/projects/thesis 'bash
  ~/q1_qwen_runner.sh 2>&1 | tee -a logs/q1_qwen_full_run.log'`.
- **`gate_120b_runner` tmux session is gone** — it finished successfully
  (see item 9 above) and the session closed itself. Nothing to restart.
- **Nothing of this session's own work is uncommitted.** `git status`
  shows only the long-standing untracked peer artifacts (`mirroring_*.png`
  figures, the `openai_gpt-oss-120b_low` pair, `cost_ledger.csv` modified
  by the qwen job's own cache re-logging) — leave those alone, not this
  session's to touch.
- Latest commit: `b96c33b`, pushed to `origin/main`.
- `runs/_cache` was backed up to the OneDrive folder's
  `backups/runs_cache/` earlier this session (next-steps item 5, done,
  180M, 16,505 files).

## The supervisor deck — read this before touching it again

`Progress_update_Oct9.pptx` has Microsoft 365 co-authoring/AutoSave active
(confirmed: it carries live `ppt/revisionInfo.xml` and
`ppt/changesInfos/` parts with real, recent revision timestamps). The user
has been actively editing several slides themselves throughout this
session (titles like "(mechanical)"/"(semantic)" on slides 7/8, a
mid-edit title on slide 9 as of this writing: "Q3 — Does the model
reliable enough to ?").

**What went wrong once, and the protocol that now avoids it:** opening
this file via PowerPoint COM automation (used for visual screenshot QA)
can silently merge in the user's own live-open-PowerPoint-window content —
this looks exactly like file corruption (stray text, swapped content) but
isn't. Early in this session that led to briefly overwriting a couple of
the user's own in-progress edits before the cause was understood (no
lasting harm — AutoSave had already captured their real edits, confirmed
by reading the live file directly).

**Current safe protocol, used for every edit since:**
1. Copy the live file to a scratch dir, unzip, find the target slide by
   **content search** (`'phrase' in data`), never by assumed filename —
   PowerPoint's own repackaging can renumber internal slide XML files to
   match display order the moment it's opened via COM, independent of
   anything this session does.
2. Edit via `defusedxml.minidom`, never raw string replace (attribute
   order/whitespace varies) — see this session's commits for the
   from-scratch table-rebuild pattern that reliably works, versus
   append-cells-to-an-existing-table, which once rendered a visually blank
   column despite correct-looking XML (cause not fully root-caused; rebuild
   from scratch instead if a table needs new columns).
3. Before saving back: verify every slide this session didn't intend to
   touch is **byte-identical** to the pre-edit copy
   (`zipfile.read(name) == open(unpacked_path, 'rb').read()` — do NOT
   compare via shell `diff` on `print()`'d output, which adds a trailing
   newline and gives a false "different" result).
4. Only use PowerPoint COM (for a screenshot-based visual check) when
   reasonably confident the user doesn't have the file open right now —
   ask if unsure. A plain Python zip-edit-rezip round-trip never touches
   PowerPoint or OneDrive co-authoring at all, so prefer it whenever a
   visual check isn't strictly needed (e.g. a layout already confirmed
   correct in an earlier, near-identical edit).

Current deck state: 11 slides. Slide 5 now proposes running Layer 1 with
gpt-oss-120b (this session's final edit). Slide 9 (Q3 self-preference)
still shows the **original, now-known-to-be-mis-specified** "self-preference
holds" framing (see `PROGRESS_llms.md` section 17's own correction) — the
user has not asked for that slide to be fixed, and appears to be mid-editing
it themselves as of this writing.

## Next steps

1. **Waiting on the user and their supervisor**: whether to actually run
   Layer 1's real comparison (does a generated reply give away hierarchy
   as well as real email does, judged by gpt-oss-120b) — this is new work,
   not yet started, and would be a real multi-hundred-call job across all
   models plus real email. Slide 5 was updated specifically to seek this
   go-ahead before starting it.
2. Slide 9 of the deck still carries the old, corrected-in-the-log
   self-preference framing. Worth asking whether the user wants it updated
   to match section 17's correction, especially since they seem to be
   already touching that slide.
3. `PROGRESS_llms.md`'s own "19. Next steps" section has the other
   standing items (hand-coding review packets, a run without the "act"
   instruction, the corpus-cleaner line-removal rule) — unchanged this
   session, still open.
4. Keep watching the qwen job; no action needed unless it dies.

## Suggested skills

- **superpowers:test-driven-development** — if any further work touches
  `src/thesis/analysis/*.py` or `src/thesis/judge/*.py`. This session hit
  two real bugs (billing, a KeyError on partial data) that only a live run
  surfaced, not the existing test suite — both fixed test-first once found.
- **results-verifier agent** — before any new number from this session's
  work (writing-up effect, the two gate-check kappas) reaches a thesis
  chapter draft, if one gets started. Not used this session since these
  were re-analyses of already-verified cached data and a single pass/fail
  statistic, not a new multi-step statistical claim.
- **anthropic-skills:pptx** — for any further deck work. Load it, but the
  "supervisor deck" section above supersedes its default tooling advice
  for this specific file (no `soffice`, and now also minimize PowerPoint
  COM use given the live co-authoring risk).
