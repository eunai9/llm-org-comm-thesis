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

# Files named in HANDOVER.md as existing only on this laptop. Untracked code
# here is a real risk: nothing outside this machine has a copy.
for path in src/thesis/analysis/blind_review.py tests/test_blind_review.py; do
  if [ -f "$path" ] && ! git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    echo "Laptop-only, untracked, no copy anywhere else: $path"
  fi
done
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
echo

cache_size=$(du -sh runs/_cache 2>/dev/null | cut -f1)
echo "Cache: ${cache_size:-missing} at runs/_cache. Not in git, not backed up."

# A crashed write leaves zero-length entries. These killed a 2,880-cell run on
# Sep 19. A damaged entry is treated as a miss now, but say so if any exist.
damaged=$(find runs/_cache -type f -size 0 2>/dev/null | wc -l)
if [ "$damaged" -gt 0 ]; then
  echo "Damaged cache entries (zero-length): $damaged. They are treated as misses and will be regenerated."
fi

running=$(pgrep -af 'thesis\.(analysis|sim|judge)' 2>/dev/null | grep -v pgrep)
if [ -n "$running" ]; then
  echo
  echo "Generation or analysis already running. Do not start a second copy:"
  printf '%s\n' "$running" | cut -c1-120
fi
