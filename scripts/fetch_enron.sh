#!/usr/bin/env bash
#
# Download and extract the Enron email corpus.
#
# Source: the canonical CMU/CALO release curated by William Cohen, which is
# what every paper in this literature cites. Kaggle's `emails.csv` is a
# derivative of exactly this tarball, so we take the original: no account
# needed, and the provenance chain is one link shorter.
#
#   https://www.cs.cmu.edu/~enron/
#
# Idempotent: re-running skips work that is already done. The download
# resumes if interrupted.

set -euo pipefail

URL="https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$REPO_ROOT/data/raw"
TARBALL="$RAW_DIR/enron_mail_20150507.tar.gz"
MAILDIR="$RAW_DIR/maildir"
PROVENANCE="$REPO_ROOT/data/external/enron_provenance.md"

mkdir -p "$RAW_DIR"

# ----------------------------------------------------------------- download
if [ -d "$MAILDIR" ]; then
  echo "==> maildir already extracted at $MAILDIR -- nothing to download."
else
  echo "==> Downloading Enron corpus (~423 MB) to $TARBALL"
  echo "    Source: $URL"
  # --continue-at - resumes a partial download rather than starting over.
  curl -fL --continue-at - --progress-bar -o "$TARBALL" "$URL"

  echo "==> Verifying archive integrity"
  if ! gzip -t "$TARBALL"; then
    echo "!! Archive is corrupt. Delete $TARBALL and re-run." >&2
    exit 1
  fi

  echo "==> Extracting (this creates ~2.6 GB of small files; takes a few minutes)"
  tar -xzf "$TARBALL" -C "$RAW_DIR"
fi

# --------------------------------------------------------------- provenance
echo "==> Recording provenance"
SHA=""
if [ -f "$TARBALL" ]; then
  SHA="$(sha256sum "$TARBALL" | awk '{print $1}')"
fi
N_USERS="$(find "$MAILDIR" -maxdepth 1 -mindepth 1 -type d | wc -l)"
N_FILES="$(find "$MAILDIR" -type f | wc -l)"

cat > "$PROVENANCE" <<EOF
# Enron corpus provenance

Recorded automatically by \`scripts/fetch_enron.sh\`. Cite the CMU release,
not a Kaggle mirror.

| Field | Value |
|---|---|
| Source URL | <$URL> |
| Upstream release date | 2015-05-07 |
| Retrieved | $(date -u +"%Y-%m-%d %H:%M:%SZ") |
| Archive sha256 | \`${SHA:-<tarball removed>}\` |
| Archive bytes | $( [ -f "$TARBALL" ] && stat -c%s "$TARBALL" || echo "n/a" ) |
| Mailbox directories | $N_USERS |
| Message files on disk | $N_FILES |

Note: the on-disk file count is **not** the number of unique messages. Each
message is stored once per mailbox folder it appears in, so senders' Sent
items and recipients' Inbox copies duplicate one another. Deduplication
happens in \`thesis.data.ingest\`; the unique count is reported there and
belongs in the thesis rather than this raw figure.
EOF

echo
echo "==> Done."
echo "    maildir:           $MAILDIR"
echo "    mailbox dirs:      $N_USERS"
echo "    message files:     $N_FILES  (NOT unique messages -- see provenance note)"
echo "    provenance:        $PROVENANCE"
