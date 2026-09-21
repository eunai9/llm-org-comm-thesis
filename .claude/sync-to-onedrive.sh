#!/usr/bin/env bash
# Copies the agent definitions and settings to the OneDrive folder.
#
# A Claude session started in the OneDrive folder only loads .claude/ from
# there. That folder is not a git repo, so the repo copy stays the source of
# truth and this script pushes it across. Run after any change in .claude/.
set -eu

src="$HOME/projects/thesis/.claude"
dst="/mnt/c/Users/Acer/OneDrive/문서/2. 대학원/LMU S&DS/Thesis/.claude"

if [ ! -d "$(dirname "$dst")" ]; then
  echo "OneDrive folder not found: $(dirname "$dst")" >&2
  exit 1
fi

mkdir -p "$dst/agents"
cp "$src"/agents/*.md "$dst/agents/"
cp "$src/settings.json" "$dst/settings.json"
cp "$src/README.md" "$dst/README.md"

echo "Synced to $dst"
ls -1 "$dst/agents"
