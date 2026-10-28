# Handoff, 2026-10-28 20:01

Previous handoff: `docs/general_handoff_2026-10-26_1441.md`. It has the full plan, the pitfalls and the Ollama/tmux restart commands. The restart commands are also in `docs/general_handoff_2026-10-23_1545.md`.
Project log: `PROGRESS_llms.md`. PROGRESS.md and HANDOVER.md are deprecated.

## Live state (Groq queue)

- The tmux sessions `groq_queue` and `wsl_ollama` run inside WSL. Both started Oct 27 07:40.
- The queue script `~/groq_queue.sh` runs three jobs one at a time:
  1. Judge context test with gpt-oss-120b: done (served from cache).
  2. Persona pilot with gpt-oss-120b: running. Log: `logs/q1p_gpt_oss_120b_pilot.log`.
  3. Persona pilot with qwen (`qwen/qwen3.8-27b`): not started. Log: `logs/q1p_qwen_pilot.log`.
- The gpt-oss-120b pilot is at about 660 of 832 cells. That is 300 cells from before Oct 27, plus 364 new cache files since Oct 27 07:40.
- The log prints a line only every 100 cells. On each retry it replays the cached cells first.
- The pilot keeps hitting the Groq daily token limit (HTTP 429). When that happens, the runner `~/q1_personas_runner.sh` sleeps and then retries by itself. The wait grows up to 1800 s. Last 429: Oct 28 19:37.

## Pitfalls found this session

- Laptop sleep freezes the WSL VM. A running `sleep 1800` cooldown then does not advance. The queue stalls for hours while the tmux sessions still look alive.
  - Fix: find the sleep process with `ps -eo pid,etime,cmd | grep sleep` and kill it. The runner retries at once.
- Check progress by counting cache files, not by reading the log. Run `find runs/_cache -newermt '2026-10-27 07:40' -type f | wc -l`, then add 300.
- If WSL shut down completely, no tmux server exists. Restart Ollama and the queue with the commands in the earlier handoffs. Job 1 then finishes from cache in seconds.

## Next steps

1. Watch the gpt-oss-120b pilot until it reaches 832/832. The qwen pilot then starts on its own.
2. When each pilot finishes:
   - Add its rows to the tables in PROGRESS_llms.md section 20, with a short reading for each model.
   - Commit with the run manifests and push.
   - Never commit `runs/_cache`.
3. Ask the user whether to run the full design: `~/q1_personas_runner.sh MODEL SLUG full`.
4. Update slide 4 of the new deck (see below) with the pilot status.

## Decks (OneDrive folder, not in git)

The decks are in `C:\Users\Acer\OneDrive\문서\2. 대학원\LMU S&DS\Thesis\`.

- `Process (10.12.2026 - 10.26.2026).pptx` is finished. It has 12 slides.
- `Process (10.26.2026 - ).pptx` is new and has 5 slides:
  - Slides 1–3 stay the same in every deck.
  - Slide 4 is the open problems table: Open problem | Status | Where it stands now.
  - Slide 5 is the timeline.
- Things to fix in the new deck:
  - The title slide still says "October 21, 2026". Ask the user for the meeting date.
  - On slide 4, the "40-persona check" row says 300 of 832. It is now about 660.
- How to edit:
  - Use python-pptx in WSL: `PYTHONPATH=~/.pptx-libs ~/projects/thesis/.venv/bin/python`.
  - Close PowerPoint first and make a backup.
  - The user also edits the deck by hand. Find shapes by their text, not by id.
  - To check a change, render the slide to PNG with PowerPoint COM in PowerShell.

Small offers that are still open to the user:
- Add "(role task)" after "self-agreement" on the judge slide.
- Add the Q3 judge context test to slide 4 of the old deck.

## Repo notes

- The untracked figures under `docs/figures/` belong to other sessions. So does the change to `outputs/manifests/cost_ledger.csv`. Do not commit them.
- Run Groq jobs one at a time. Use no paid APIs.

## Suggested skills

- `superpowers:verification-before-completion` before reporting pilot results as done.
- `anthropic-skills:pptx` for any deck edit.
- `superpowers:systematic-debugging` if the queue stalls in a way not covered above.
