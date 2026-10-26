# Session handoff — 2026-10-26, 1441

For a fresh agent continuing this thesis work. Read this first. It does not
repeat CLAUDE.md, the pinned memories, or `PROGRESS_llms.md`. It points to
them.

## Where things live

- Real repo: WSL `~/projects/thesis`, branch `main`, pushed to
  `github.com/eunai9/llm-org-comm-thesis`. Last commit `225991d`.
- Active log: `PROGRESS_llms.md`. `PROGRESS.md` and `HANDOVER.md` are
  deprecated.
- Previous handoff: `docs/general_handoff_2026-10-23_1545.md` (shell
  pitfalls, restart commands, deck tooling). Still valid.
- Deck for the supervisor:
  `C:\Users\Acer\OneDrive\문서\2. 대학원\LMU S&DS\Thesis\Process (10.12.2026 - 10.21.2026).pptx`,
  now 12 slides. The user edits it in PowerPoint. Edit in place with
  python-pptx, check `POWERPNT` is not running first, back up first.
- Literature notes the user keeps:
  `C:\Users\Acer\OneDrive\문서\2. 대학원\LMU S&DS\Thesis\Reference Papers\llm-as-judge-communication-emotion-research.md`.
  Part 5 (added this session) maps our judge design to papers, lists five
  gaps, and gives candidate theory per rubric item.

## What this session did

All committed on `main` (`git log a026875..225991d`).

1. **Deck:** new slide 9 "LLM-as-a-judge: how the judge is set up" and
   slide 10 "Test result: the stronger judge needs the incoming message".
   Slide build scripts are in the old session scratchpad and are not
   needed again; reuse the style: navy `1E2761`, card `F2F3F7`, Cambria
   28pt titles, Calibri body.
2. **Judge context test (`PROGRESS_llms.md` section 19, Q3).** The judge is
   shown the incoming message. gpt-oss-120b then separates mirrored from
   sound replies (gap 0.65, p=.001); gpt-oss-20b does not. Code:
   `judge_blindness.py --model` (merged from branch `judge-context-rerun`,
   now deleted).
3. **Bug fix `67da7f0`:** `judge/run.py` skips an item when Groq returns
   "Failed to generate JSON", instead of crashing the run.
4. **qwen dropped as a judge** (about 40% JSON failures on Groq). qwen is a
   writer only. Results logged in sections 7, 8, 9, 17, 18. New Q3 check:
   with qwen as a second neutral reference writer, self-preference is again
   about zero (section 18).
5. **`src/thesis/analysis/word_auc.py`:** rebuild of the lost one-off script
   behind section 8's AUC table. Reproduces the logged numbers closely.
6. **Persona pilots logged (section 20):** Llama and gpt-oss-20b. With 40
   personas the 10-persona rank effect on orders disappears for both.
7. Sections renumbered: new 19 is the judge context test, so the persona
   experiment is now **section 20**, "Where the code lives" 21, "Next
   steps" 22.

## Live state (tmux in WSL)

| Session | What | Done when |
|---|---|---|
| `groq_queue` | `~/groq_queue.sh`: job 2 gpt-oss-120b persona pilot, then job 3 qwen persona pilot | `logs/groq_queue.log` ends with "all finished" |
| `wsl_ollama` | local Ollama server | — |

- Job 2 log: `logs/q1p_gpt_oss_120b_pilot.log`. It has been at 300 of 832
  cells since about 12:15 on Oct 26: each attempt fails on Groq 429 and
  waits 30 minutes. Check if it is moving at all. If it stays stuck, tell
  the user; options are to wait for the Groq daily limit to reset or to
  accept a partial pilot.
- Job 3 (qwen pilot) has not started. Earlier it made no progress at all
  on Groq. Expect the same problem.
- Restart after a reboot: start Ollama first (command in the Oct 23
  handoff), then
  `tmux new-session -d -s groq_queue -c ~/projects/thesis 'bash ~/groq_queue.sh 2>&1 | tee -a logs/groq_queue.log'`.
  Job 1 in that script is already done and finishes instantly from the
  cache.
- Uncommitted: `outputs/manifests/cost_ledger.csv` and long-standing
  untracked peer files. Leave them alone.

## Next steps

1. **Persona pilot gate.** When gpt-oss-120b and qwen finish, add their
   rows to the two tables in section 20 and a per-model reading, commit
   with the manifests, push. Then ask the user whether to run the full
   design (`~/q1_personas_runner.sh MODEL SLUG full`, 4,992 cells per
   model).
2. **Judge gaps (Reference Papers notes, Part 5).** Gap 1 (human check) is
   parked by the user. Gap 3 (irrelevant-attribute test) only matters once
   the role is shown to the judge. Gap 5 is now partly answered by
   section 19: use gpt-oss-120b with the incoming message for future
   judging. Not yet decided: whether to rerun the rubric judging that way.
3. Open items unchanged: small-cluster correction for the 10-persona
   design, hand-coding by a person, `PROGRESS_llms.md` section 22.

## User preferences seen this session

- Very short, plain writing. Per-model interpretation.
- Asks often for "status of all background sessions": answer with one
  table (session, what, status) from a quick tmux and log check.
- Wants urgent work first and other jobs paused when they compete for the
  same API. Keep Groq jobs one at a time.
- Worried about breaking existing code: make changes additive, on a
  branch or worktree, with old defaults and old output files unchanged.
- Keeps the laptop awake for long runs; hibernation pauses WSL jobs.

## Pitfalls hit this session

- Runner scripts must `set -a; source .env; set +a`, or API clients fail
  with "no API key".
- Backticks inside `wsl.exe -e bash -c "..."` run as commands, even inside
  a heredoc. Write Python edit scripts to a file and run them.
- A file-based progress-log edit on `main` while another branch is checked
  out: use `git worktree add`, then remove untracked copies in the main
  tree before merging.
- Groq JSON-schema failures (HTTP 400) are not cached, so every retry asks
  again and burns rate limit.

## Suggested skills

- **superpowers:verification-before-completion**: before reporting any
  pilot number.
- **progress-writer agent**: optional, for larger `PROGRESS_llms.md` edits.
- **anthropic-skills:pptx**: for any deck edit.
- **dataviz**: before any new figure.
- **superpowers:brainstorming**: if the user picks a judge gap to work on
  next, or the full persona design needs changes.
- **superpowers:systematic-debugging**: if a queue job is stuck for a
  reason other than Groq 429.
