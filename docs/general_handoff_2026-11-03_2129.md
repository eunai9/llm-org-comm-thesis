# Session handoff, 2026-11-03 21:29

## Goal of this period

The user will show all results to the supervisor this month. They will ask
whether the thesis can be written on this basis. Two outputs serve that:

- `Results overview (2026-11).md` in the OneDrive Thesis folder, with figures
  in `results_overview_figures/`. It stays out of git because it quotes Enron
  text.
- The deck `Process (10.26.2026 - ).pptx` in the same folder.

## State now

- **Nothing is running.** The tone pilot finished on Nov 2 at 18:34. Ollama
  and the `wake_watchdog` tmux session are up.
- **Tone pilot results** are recorded in three places:
  - `PROGRESS_llms.md`, section 21 (commit `d2a80d6`, pushed to main).
  - Overview section 2.5, plus rows in 2.6 and section 5.
  - A new deck slide 11, after the user's "New Pilot" slide.
- **Tone judge code** (`src/thesis/analysis/tone_judge.py`) exists only on
  branch `layer1-consensus-tone`, in the worktree `~/projects/thesis`. Main
  holds its manifest (`outputs/manifests/tone_judge_pilot.json`) and the plot
  (`docs/figures/tone_judge_by_direction.png`).
- **Two repo copies.** `~/projects/thesis-main` is on main; commit results
  there. `~/projects/thesis` is on the branch, and the venv lives there: run
  with `PYTHONPATH=src ../thesis/.venv/bin/python -m ...` from thesis-main.
- **The deck has 13 slides.** Slide 10 now reads "A (done, next slide)". The
  tone row was removed from "Next steps". The pre-edit copy is in the old
  session's scratchpad, as `cur/orig.pptx`.

## Open decisions (the user's call)

1. Which background job to start next. My recommendation:
   - StyleDistance style embeddings (Q2): local, free.
   - Self-BLEU (Q2): local, free.
   - Tone judge on more emails (Q1): Groq, can start now.
   - Judge sees the writer's role (Q3): needs a rank-bias test design first.
   - Politeness model: needs a model choice and code.
   - Run without the act instruction: low priority.
2. Whether to merge `tone_judge.py` into main without the consensus code. The
   consensus code is also on `layer1-consensus-tone`, and keeping it is
   undecided.
3. Conclusion boxes on the other deck slides: only when the user asks.
4. The deck's title date: waiting for the meeting date.

## Rules that bit this session (see memory for the full list)

- **Deck edits.** Unpack the current OneDrive file, not an old build copy.
  Check its md5 before copying back. Never change slide titles or order. The
  user was angry when a rebuild overwrote their edits on Nov 2.
- **Ollama after a shutdown.** Restart it with
  `tmux new-session -d -s ollama '~/.local/bin/ollama serve >> ~/ollama_serve.log 2>&1'`.
- **Groq.** One job at a time. No paid APIs.
- **Excluded models.** No DeepSeek and no GLM. gpt-oss-120b is the only judge.
- **Progress log.** Use `PROGRESS_llms.md` only, with sections titled
  `N. Title (Qx, date)`.
- **Windows shell.** Windows `python` is a Store stub. Run Python inside WSL.
- **Writing.** Short, plain sentences. If the user asks a question, answer in
  chat only.

## Suggested skills

- `anthropic-skills:pptx`: any deck change.
- `dataviz`: any new figure (also follow the conventions in `plots.py`).
- `superpowers:verification-before-completion`: before reporting numbers.
- `superpowers:brainstorming`: before designing the role-shown judge test
  (decision 1, job 4).
