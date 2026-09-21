#!/usr/bin/env bash
# PostToolUse check that runs after a shell call containing "git commit".
#
# The git index is one shared file and another session usually works in the
# same tree. On Sep 21 a finished section was swept into a commit titled
# "Untrack HANDOVER.md". A commit that did not land is also easy to miss,
# because "nothing added to commit" is buried in tool output.
#
# So print what actually landed. This never blocks anything; exit 0 always.
set -u

payload=$(cat)
cmd=$(printf '%s' "$payload" | python3 -c \
  'import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("")
    sys.exit(0)
print((d.get("tool_input", {}) or {}).get("command", "") or "")' 2>/dev/null)

printf '%s' "$cmd" | grep -Eq 'git +(-[^ ]+ +)*commit' || exit 0

cd "$HOME/projects/thesis" 2>/dev/null || exit 0

echo "Commit check. What actually landed:"
git log --oneline -1
echo "Files in it:"
git show --stat --format= HEAD | head -15

staged=$(git diff --cached --name-only)
if [ -n "$staged" ]; then
  echo "Still staged after the commit. These did not land, and another session can sweep them up:"
  printf '%s\n' "$staged" | head -10
fi

exit 0
