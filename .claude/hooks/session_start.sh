#!/usr/bin/env bash
# Prints repo state at session start, so a session does not begin on stale
# assumptions. Output goes into the session context.
set -u
cd "$HOME/projects/thesis" 2>/dev/null || exit 0

echo "## Thesis repo state"
echo
echo "Branch: $(git branch --show-current 2>/dev/null)"
echo "Last commit: $(git log -1 --format='%h %ad %s' --date=short 2>/dev/null)"
echo

dirty=$(git status --porcelain 2>/dev/null)
if [ -n "$dirty" ]; then
  n=$(printf '%s\n' "$dirty" | wc -l)
  echo "Uncommitted files ($n). These may belong to another session. Stage by path, never git add -A:"
  printf '%s\n' "$dirty" | head -20
  [ "$n" -gt 20 ] && echo "  ... and $((n - 20)) more"
else
  echo "Working tree clean."
fi
echo

last_main=$(grep -oP '^### \K[0-9]+' PROGRESS.md 2>/dev/null | tail -1)
last_llms=$(grep -n '^## ' PROGRESS_llms.md 2>/dev/null | tail -1 | cut -d: -f2-)
echo "PROGRESS.md last section: ${last_main:-unknown}"
echo "PROGRESS_llms.md last heading:${last_llms:-unknown}"
echo

hash_now=$(.venv/bin/python -c \
  'from thesis.sim.prompt import prompt_text_hash; print(prompt_text_hash())' 2>/dev/null)
if [ -z "$hash_now" ]; then
  echo "Prompt hash: could not compute (venv or import failed)."
else
  echo "Current prompt_text_hash: $hash_now"
  .venv/bin/python - "$hash_now" <<'PY' 2>/dev/null
import json
import pathlib
import sys

current = sys.argv[1]
stale, fresh, untagged = [], [], 0
for path in sorted(pathlib.Path("outputs/manifests").glob("*.json")):
    try:
        run = json.loads(path.read_text()).get("run", {})
    except Exception:
        continue
    tag = run.get("prompt_text_hash")
    if tag is None:
        untagged += 1
    elif tag == current:
        fresh.append(path.name)
    else:
        stale.append(f"{path.name} ({tag})")

print(f"Manifests on the current prompt: {len(fresh)}")
if stale:
    print("Manifests on an older prompt, results from them are stale:")
    for name in stale:
        print(f"  {name}")
if untagged:
    print(f"Manifests with no prompt hash recorded: {untagged}. Cannot be checked.")
PY
fi

cache_size=$(du -sh runs/_cache 2>/dev/null | cut -f1)
echo
echo "Cache: ${cache_size:-missing} at runs/_cache. Not in git, not backed up."
