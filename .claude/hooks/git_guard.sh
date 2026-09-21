#!/usr/bin/env bash
# PreToolUse guard on shell commands.
#
# Two jobs:
#   1. Block bulk staging (git add -A, git add ., git add -u). Other sessions
#      keep uncommitted files in this repo, so staging must be per path.
#   2. Run the quality gates before any git commit, and block the commit if
#      they fail. Gates: black, ruff, mypy, pytest.
#
# Exit 0 allows the call. Exit 2 blocks it and sends stderr back to the model.
set -u

payload=$(cat)
cmd=$(printf '%s' "$payload" | python3 -c \
  'import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("")
    sys.exit(0)
ti = d.get("tool_input", {}) or {}
print(ti.get("command", "") or "")' 2>/dev/null)

case "$cmd" in
  *git\ *) ;;
  *) exit 0 ;;
esac

# --- 1. bulk staging ---------------------------------------------------------
if printf '%s' "$cmd" | grep -Eq 'git +add +(-A\b|--all\b|-u\b|\.($| )|\*)'; then
  {
    echo "Blocked: bulk git add."
    echo
    echo "Other sessions usually have uncommitted files in this repo, and the rule"
    echo "here is to commit only the files from work you actually did."
    echo "Stage them one by one by path instead. Current uncommitted files:"
    echo
    cd "$HOME/projects/thesis" 2>/dev/null && git status --porcelain | head -30
  } >&2
  exit 2
fi

# --- 2. commit gates ---------------------------------------------------------
if printf '%s' "$cmd" | grep -Eq 'git +(-[^ ]+ +)*commit'; then
  repo="$HOME/projects/thesis"
  py="$repo/.venv/bin/python"
  [ -x "$py" ] || exit 0

  log=$(mktemp)
  fail=""
  for stage in \
    "black:$py -m black --check src tests" \
    "ruff:$py -m ruff check src tests" \
    "mypy:$py -m mypy" \
    "pytest:$py -m pytest -q"
  do
    name=${stage%%:*}
    run=${stage#*:}
    if ! (cd "$repo" && eval "$run") >"$log" 2>&1; then
      fail="$name"
      break
    fi
  done

  if [ -n "$fail" ]; then
    {
      echo "Blocked: the $fail gate failed, so this commit did not run."
      echo "All four gates must pass before any commit: black, ruff, mypy, pytest."
      echo
      echo "Last 40 lines of $fail output:"
      tail -40 "$log"
    } >&2
    rm -f "$log"
    exit 2
  fi
  rm -f "$log"
fi

exit 0
