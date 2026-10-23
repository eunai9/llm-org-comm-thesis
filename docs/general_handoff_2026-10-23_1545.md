# Session handoff — 2026-10-23, 1545

For a fresh agent continuing this thesis work. Read this first. It does not
repeat CLAUDE.md, the pinned memories, `PROGRESS_llms.md`, or the spec and
plan below. It points to them.

## Where things live

- Real repo: WSL `~/projects/thesis`, branch `main`, pushed to
  `github.com/eunai9/llm-org-comm-thesis`. Reach it with
  `wsl.exe -e bash -c "cd ~/projects/thesis && <command>"`.
- Active log: `PROGRESS_llms.md`. `PROGRESS.md` and `HANDOVER.md` are
  deprecated.
- Previous handoff: `docs/general_handoff_2026-10-15_1632.md`.
- The progress deck for the supervisor:
  `C:\Users\Acer\OneDrive\문서\2. 대학원\LMU S&DS\Thesis\Process (10.12.2026 - 10.21.2026).pptx`
  (10 slides). The user edits it in PowerPoint. Edit it in place with
  python-pptx, never rebuild it, and check it is closed first.

## Shell and tool pitfalls (all hit this session)

- In `wsl.exe -e bash -c "..."`, backticks and inner double quotes break
  the command. Write Python edit scripts and commit messages to the
  scratchpad with the Write tool, run them from WSL, and commit with
  `git commit -F <file>`.
- Windows `python` is a stub. Deck tools live in WSL:
  `PYTHONPATH=~/.pptx-libs ~/projects/thesis/.venv/bin/python` (python-pptx,
  markitdown). Do not name a script `inspect.py`, because it shadows the
  stdlib module.
- No LibreOffice. Render slides through PowerPoint COM in PowerShell. The
  render script is in the session scratchpad (`render.ps1`); it is a
  short `Presentations.Open` plus `Slide.Export` loop.
- `gh` is not installed anywhere. For PRs, push the branch and give the
  user the compare link plus text to paste (memory `no-gh-cli-manual-prs`).
- Ollama: WSL cannot reach Windows Ollama. Use the native WSL server in
  tmux `wsl_ollama` (`~/.local/bin/ollama serve`). It has `llama3.2:3b`
  and `nomic-embed-text`.
- The superpowers skill scripts have CRLF endings. Unix copies are in
  `~/sdd-scripts/`.
- `.superpowers/` is git-ignored through `.git/info/exclude`.

## What this session did (all on `main`, `git log e12a95c..9a3125c`)

1. **qwen3.8-27b added to every full-grid Q1 section** (1, 2, 11, 13, 14,
   15). It has the strongest expected pattern: more orders down (+0.510)
   and fewer up (-0.181), both significant. Real email shows only the
   down half, so qwen overstates the pattern.
2. **Layer 1 finished** (gpt-oss-120b judge, section 16). Real email is at
   chance (35.7%). gpt-oss and qwen replies are easier to read (qwen
   47.9%). Fixed a bug where the interval belonged to plain accuracy, not
   balanced accuracy. Real email is a fixed-seed sample of 1,000; the
   reason is written in section 16.
3. **Layer 2** (`src/thesis/analysis/role_cues.py`, end of section 16).
   Orders are the judge's main cue on every source. "you" flips between
   real and generated email.
4. **Q1 persona-variant experiment built and merged** (`9a3125c`). Spec:
   `docs/superpowers/specs/2026-10-22-q1-persona-variants-design.md`.
   Plan: `docs/superpowers/plans/2026-10-22-q1-persona-variants.md`. Built
   with subagent-driven development; all tasks and the final review are
   clean. Code: `src/thesis/sim/persona_variants.py` and
   `src/thesis/analysis/q1_personas.py`. The user opened a PR; I merged it
   locally and pushed, which closes it. The plan's Task 6 (pilots) is
   running now.
5. `judge_swap` now routes `qwen/qwen3.8-27b` to Groq (`d3d82f9`).
6. The deck was built and then edited many times (see "Where things
   live").

## Current live state (tmux in WSL)

| Session | What | Done when |
|---|---|---|
| `q1p_llama` | persona pilot, llama3.2:3b, 832 cells | `outputs/manifests/q1_personas_llama_pilot.json` exists |
| `q1p_gpt_oss_20b` | persona pilot, NVIDIA | `..._gpt_oss_20b_pilot.json` |
| `q1p_gpt_oss_120b` | persona pilot, Groq | `..._gpt_oss_120b_pilot.json` |
| `q1p_qwen` | persona pilot, Groq; slow (429s and some JSON failures) | `..._qwen_pilot.json` |
| `qwen_q2q3_runner` | qwen Q2 AUC (pairs, mirroring, embedding) and Q3 judge swap | `logs/qwen_q2q3_run.log` ends with "qwen Q2/Q3 runner finished" |
| `wsl_ollama` | local Ollama server, needed by the llama pilot and the embedding step | — |

- All pilots started at 15:41 on Oct 23. Logs: `logs/q1p_<slug>_pilot.log`.
  Runner: `~/q1_personas_runner.sh MODEL SLUG DESIGN`. It retries until it
  exits 0, then runs `--analyse`.
- Restart after a reboot (start Ollama first):
  `tmux new-session -d -s wsl_ollama -c ~ '~/.local/bin/ollama serve 2>&1 | tee -a ~/ollama_serve.log'`
  `tmux new-session -d -s q1p_<slug> -c ~/projects/thesis 'bash ~/q1_personas_runner.sh <MODEL> <slug> pilot 2>&1 | tee -a logs/q1p_<slug>_pilot.log'`
  with (MODEL, slug) = (llama3.2:3b, llama), (openai/gpt-oss-20b@low,
  gpt_oss_20b), (openai/gpt-oss-120b@low, gpt_oss_120b),
  (qwen/qwen3.8-27b, qwen).
  `tmux new-session -d -s qwen_q2q3_runner -c ~/projects/thesis 'bash ~/qwen_q2q3_runner.sh 2>&1 | tee -a logs/qwen_q2q3_run.log'`
- The qwen Q2/Q3 runner has been stuck on step 1 (pairs) since Oct 22,
  19:54. Groq 429s let only a few qwen calls through per 30-minute
  attempt. About 100 or more of the 183 threads are cached. It now shares
  qwen's Groq limit with `q1p_qwen`. Outputs use their own `_qwen` names
  (`pairs_qwen.parquet`, `mirroring_qwen.json`, `embedding_map_qwen.json`,
  `judge_swap_*_free_tier_qwen.*`), so nothing existing is overwritten.
- Uncommitted in the repo: `outputs/manifests/cost_ledger.csv` (re-logged
  by jobs) plus the long-standing untracked peer files. Leave them alone.

## Next steps

1. **Pilot gate.** As each persona pilot finishes, report per model: valid
   replies out of 832, primary down and up contrasts, the trait check.
   Compare each with the model's 10-persona result
   (`outputs/manifests/q1_models_full.json`) and real email's robust fit.
   Write it into `PROGRESS_llms.md` section 19 as short plain blocks,
   commit with the manifests, and push. Then ask the user whether to run
   the full design (same runner with `full`, 4,992 cells per model).
2. **qwen Q2/Q3.** When the runner finishes, add qwen rows to sections 8
   and 9 (AUC), 17 (rubric) and 18 (self-preference). Report how many of
   the 183 pairs were valid.
3. **Deck slide 5:** its title was changed by the user to the Layer 1
   title, which looks accidental. The user was told and has not answered.
4. Open items unchanged: the small-cluster correction (CR2 or wild
   bootstrap) for the 10-persona design, hand-coding by a person, and
   `PROGRESS_llms.md` section 21.

## User preferences seen this session

- Very short, plain writing. The user shouts when text is long ("BE
  SIMPLE"). No date or time-series narration in the log.
- Wants per-model interpretation, not per-effect, and an explicit "which
  model shows the expected pattern most strongly".
- Chose subagent-driven development over native for the persona build.
- Keeps the laptop awake for long runs and asks often whether Groq jobs
  are alive. Answer with a quick tmux and log check.

## Suggested skills

- **superpowers:verification-before-completion**: before reporting any
  pilot or Q2/Q3 number.
- **results-verifier agent**: before pilot numbers go into a chapter or a
  supervisor email (expensive; use sparingly, per memory).
- **progress-writer agent**: for larger `PROGRESS_llms.md` edits.
- **anthropic-skills:pptx**: for any deck edit (edit in place, render
  through PowerPoint COM).
- **dataviz**: before any new figure.
- **superpowers:brainstorming** then **writing-plans**: if the full run
  needs design changes after the pilot.
