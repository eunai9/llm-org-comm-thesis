# The agent system

Four agents and two hooks. This folder is the source of truth. A copy lives in
the OneDrive folder because a Claude session started there only loads agents
and settings from that folder, and the OneDrive folder is not a git repo.

## Agents

| Agent | Model | What it is for |
|---|---|---|
| `experiment-runner` | Sonnet | Runs the pipeline. Generation, judging, analysis, figures. Reports the numbers. |
| `results-verifier` | Opus | Checks a number before it is used. Provenance and staleness, then statistical correctness. Read-only. |
| `progress-writer` | Sonnet | Writes the progress logs and commit messages. Progress logs only, not thesis chapters. |
| `lit-scout` | Haiku | Finds and summarizes papers. Recent LLM-era work first. |

The split is by role, not by pipeline stage, because most tasks here cross
stages. The verifier is on Opus because a wrong number in the thesis is the
expensive failure. The others are on cheaper models to keep usage down.

The normal order for a new result: runner produces it, verifier checks it,
writer records it.

## Hooks

Configured in `settings.json`, implemented in `hooks/`.

**`session_start.sh`** runs at session start and prints repo state: branch, last
commit, uncommitted files, the last section in each progress log, the current
prompt hash and which manifests still match it, and the cache size. It exists
so a session does not start on stale assumptions.

**`git_guard.sh`** runs before every Bash and PowerShell call. It does two
things:

1. Blocks bulk staging (`git add -A`, `git add .`, `git add -u`). Other
   sessions keep uncommitted files in this repo, so staging must be per path.
2. Runs black, ruff, mypy and pytest before any `git commit`, and blocks the
   commit if one fails. This takes about two minutes, mostly pytest.

The hook shells into WSL by absolute path, so it works from a session started
on the Windows side or inside WSL.

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
