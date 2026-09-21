# The agent system

Four agents and three hooks, built from `HANDOVER.md`. This folder is the
source of truth. A copy lives in the OneDrive folder because a Claude session
started there only loads agents and settings from that folder, and the OneDrive
folder is not a git repo.

## Agents

| Agent | Model | What it is for |
|---|---|---|
| `experiment-runner` | Sonnet | Runs the pipeline. Generation, judging, analysis, figures. Reports the numbers. |
| `results-verifier` | Opus | Checks a number before it is published. Provenance and staleness, then statistical correctness. Read-only. |
| `progress-writer` | Sonnet | Writes the progress logs and commit messages. Progress logs only, not thesis chapters. |
| `lit-scout` | Haiku | Finds and summarizes papers. Recent LLM-era work first. |

The split is by role, not by pipeline stage, because most tasks here cross
stages. The normal order for a new result: runner produces it, verifier checks
it, writer records it.

**On cost.** `HANDOVER.md` section 9 says each agent run costs 150,000 to
400,000 tokens, and to prefer Sonnet. The verifier is the one exception. It
runs on Opus because a wrong number in the thesis is the expensive failure, so
its instructions narrow it to numbers on their way into a progress log, a
chapter, or a supervisor email. Routine checks stay inline in the main session.
Small, well-specified work should stay inline too, because an agent starts cold
and re-derives context the main session already holds.

**On long runs.** The runner launches a long job in `tmux`, confirms it is
alive in the same call, reports how to watch it, and returns. It never waits.
A subagent gets no wakeup, so waiting spends its budget and then stalls.

## Hooks

Configured in `settings.json`, implemented in `hooks/`.

**`session_start.sh`** runs at session start and prints repo state: branch, last
commit, uncommitted files, untracked files that exist only on this laptop, the
last section in each progress log, the current prompt hash and which manifests
still match it, cache size, damaged cache entries, and any generation already
running. It exists so a session does not start on stale assumptions, or start a
second copy of a job that is already going.

**`git_guard.sh`** runs before every Bash and PowerShell call. It does two
things:

1. Blocks bulk staging (`git add -A`, `git add .`, `git add -u`). Other
   sessions keep uncommitted files in this repo, so staging must be per path.
2. Runs black, ruff, mypy and pytest before any `git commit`, and blocks the
   commit if one fails. This takes about two minutes, mostly pytest.

**`commit_check.sh`** runs after any call containing `git commit`, and prints
what actually landed: the commit message, its files, and anything still staged.
The git index is one shared file, so a commit can land under another session's
message, and "nothing added to commit" is easy to miss in tool output. Both
happened on Sep 21. This hook never blocks.

## The cache guard

`experiment-runner` must stop and ask before editing anything that changes
rendered prompt text: `sim/prompt.py`, `sim/persona.py`, `sim/memory.py`, the
persona statistics code, or a corpus build step that moves persona numbers. It
first reports how many cached cells the edit invalidates and how many hours
regeneration costs. This is not enforced by a hook, because an edit to those
files is sometimes correct. It is a rule in the agent's instructions.

## Keeping the two copies in sync

After changing anything here:

```
bash .claude/sync-to-onedrive.sh
```

Then commit this folder. The OneDrive copy is not in git.
