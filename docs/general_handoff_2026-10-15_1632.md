# Session handoff — 2026-10-15, 1632

For a fresh agent continuing this thesis work. Read this first. It does not
repeat CLAUDE.md, the pinned memories, or `PROGRESS_llms.md`. It points to
them.

## Where things live

- The working folder is the Windows OneDrive folder. The real repo is in WSL
  at `~/projects/thesis`, branch `main`, pushed to
  `github.com/eunai9/llm-org-comm-thesis`. Reach it with
  `wsl.exe -e bash -c "cd ~/projects/thesis && <command>"`.
- The active log is `PROGRESS_llms.md`. `PROGRESS.md` and `HANDOVER.md` are
  deprecated (see memories).
- Previous handoff: `docs/general_handoff_2026-10-12_1939.md`.

## Shell pitfalls hit this session

- In `wsl.exe -e bash -c "..."`, backticks and inner double quotes are
  expanded or break the string. This ate file paths in a markdown edit and
  cut a commit message short. **Fix:** write Python edit scripts and commit
  messages to the scratchpad with the Write tool. Then run them via
  `/mnt/c/Users/Acer/AppData/Local/Temp/claude/.../scratchpad/...`, and
  commit with `git commit -F <file>`.
- `pkill -f 'thesis.analysis.role_inference'` inside `bash -c` also kills the
  calling shell, because the pattern is in its own command line. Kill the
  tmux session instead.

## What this session did

All commits are on `main` (`git log 419878c..02360f9`).

1. **Q2 rubric scores with all three judges** (section 17, `5038b57`). The
   narrow 4.0–4.9 band holds only when llama judges. The gpt-oss judges rate
   llama's own replies much lower.
2. **Section 16 (Layer 1) condensed**: one table, one paragraph
   (`6dfb0df`, `b3be1f2`).
3. **New section 11** explains the full design: the 1,440-cell grid vs. the
   pilot. Sections renumbered (`75291ba`, `74a3add`).
4. **Layer 1 now scores a 1,000-email real sample** instead of 2,998. It uses
   a new `--real-sample` flag with a fixed seed and keeps the direction mix
   (`b72b0d2`). The user approved this to cut run time.
5. **Parallel persona-variant Q1 experiment, planned.** The user's decisions
   are in memory `q1-persona-redesign-decisions.md`:
   - It runs next to the existing design and does not replace it.
   - Trait variants inside each fixed role cell, giving 30–50 personas.
   - Named sender and receiver tier pairs.
   - One reply per email, no threads.
   - New measures need the user's audit.
6. **Candidate measures tested on real Enron email** (section 19,
   `f115514`, `02360f9`). Code is in `src/thesis/analysis/q1_measures.py`,
   run with `python -m thesis.analysis.q1_measures`. Manifest:
   `outputs/manifests/q1_measures_real.json`.
   - **Kept:** "you" rate, Gilbert hierarchy-phrase score, gratitude.
   - **Dropped:** "I", "we", apology, please, greeting, style matching.
   - Gilbert's list is at `data/external/gilbert2012_hier_phrases.txt`, a
     Wayback copy. It is git-ignored and must not be committed, because no
     license is stated.

## Current live state

- **The qwen3.8-27b Q1 grid is finished** (Oct 14, 22:33). It is at
  `data/interim/q1_direction_grid_qwen_full.parquet`. It has not been added
  to the four-model analysis yet.
- **The `role_inference_120b_runner` tmux job is running** (Layer 1 with the
  gpt-oss-120b judge). The llama and gpt-oss-20b stages are done. It is on
  the gpt-oss-120b grid, last counted at 748 of 1,440. The 1,000 real emails
  come after that, and none are done.
  - It keeps hitting Groq 429s. About 480 new calls per day go through.
    Expect about 3–4 days.
  - It finishes when `outputs/manifests/role_inference.json` exists.
  - Restart command if WSL reboots:
    `tmux new-session -d -s role_inference_120b_runner -c ~/projects/thesis 'bash ~/role_inference_120b_runner.sh 2>&1 | tee -a logs/role_inference_120b_full_run.log'`
  - To check progress, use the stage lines in the log
    (`grep 'scored,' logs/role_inference_120b_full_run.log`), or count cache
    hits.
  - **Do not count `cost_ledger.csv` rows.** Cached reads are logged again on
    every restart, and Groq calls are always marked `from_cache=1`.
- **Untracked files from other sessions:** `cost_ledger.csv`, the
  `mirroring_*.png` figures, and the `gpt-oss-120b_low` pair. Leave them
  alone.

## Next steps

1. **Persona-variant experiment.** Design the 3–5 trait variants per role
   cell and the sender/receiver tier pairs. Then add the three kept measures
   to the simulator analysis. Start with a design proposal for the user. Do
   not build anything before they approve.
2. When the Layer 1 job finishes, report its accuracy for generated replies
   vs. real email. Put the result in section 16, which has a placeholder for
   it.
3. Add the qwen grid to the Q1 model comparison (section 13).
4. Small-cluster corrections (CR2 or wild cluster bootstrap) for the
   existing 10-persona design. Discussed in chat only, nothing built.
5. The standing items in `PROGRESS_llms.md` section 21 are unchanged.

## User preferences seen this session

- Very short, plain answers. The user pushed back on long explanations.
- In the progress log: no time-series narration, keep backstory short, and
  give room to real results. Each candidate's result and limit sit under
  that candidate.

## Suggested skills

- **superpowers:brainstorming**: before designing the persona variants and
  tier pairs.
- **superpowers:test-driven-development**: for any new code under `src/`.
- **superpowers:verification-before-completion**: before reporting results.
- **results-verifier agent**: before Layer 1 or the persona-variant numbers
  go into a chapter or a supervisor email.
- **progress-writer agent**: for larger edits to `PROGRESS_llms.md`.
